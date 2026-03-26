#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import trange

from make_env import make_mujoco_env
from nn import SAC_Actor, SAC_Critic


@dataclass
class SACConfig:
    env: str
    seed: int
    total_timesteps: int
    learn_start: int
    buffer_size: int
    batch_size: int
    gamma: float
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    target_update_factor: float
    actor_freq: int
    target_freq: int
    eval_freq: int
    eval_num: int
    device: torch.device


class SACReplayBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int):
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)  # [B, obs_dim]
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32)  # [B, act_dim]
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)  # [B, 1]
        self.next_obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32)  # [B, obs_dim]
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)  # [B, 1]
        self.ptr = 0
        self.length = 0
        self.capacity = buffer_size

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: float, nxt_obs: torch.Tensor, done: bool):
        self.obs[self.ptr] = obs.squeeze(0)  # [obs_dim]
        self.actions[self.ptr] = action.squeeze(0)  # [act_dim]
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32)  # [1]
        self.next_obs[self.ptr] = nxt_obs.squeeze(0)  # [obs_dim]
        self.dones[self.ptr] = torch.tensor([float(done)], dtype=torch.float32)  # [1]

        self.ptr = (self.ptr + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.choice(self.length, size=batch_size, replace=False)  # [batch_size]
        b_obs = self.obs[idx].to(device)  # [batch_size, obs_dim]
        b_actions = self.actions[idx].to(device)  # [batch_size, act_dim]
        b_rewards = self.rewards[idx].to(device)  # [batch_size, 1]
        b_next_obs = self.next_obs[idx].to(device)  # [batch_size, obs_dim]
        b_dones = self.dones[idx].to(device)  # [batch_size, 1]
        return b_obs, b_actions, b_rewards, b_next_obs, b_dones


def rollout_random(env_name: str, seed: int, eval_num: int) -> float:
    env = make_mujoco_env(env_name=env_name, seed=seed)
    returns = []
    for ep in range(eval_num):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_re = 0.0
        while not done:
            action = env.action_space.sample()  # [act_dim]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_re += float(reward)
        returns.append(ep_re)
    env.close()
    return float(np.mean(returns))


def evaluate_actor(actor: SAC_Actor, env_name: str, seed: int, eval_num: int, device: torch.device) -> float:
    env = make_mujoco_env(env_name=env_name, seed=seed)
    returns = []
    actor.eval()
    with torch.no_grad():
        for ep in range(eval_num):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_re = 0.0
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # [1, obs_dim]
            while not done:
                action_t, _, _ = actor.get_action(obs_t)  # [1, act_dim], [1,1], [1, act_dim]
                action = action_t.squeeze(0).cpu().numpy()  # [act_dim]
                nxt_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                ep_re += float(reward)
                obs_t = torch.tensor(nxt_obs, dtype=torch.float32).unsqueeze(0).to(device)  # [1, obs_dim]
            returns.append(ep_re)
    env.close()
    actor.train()
    return float(np.mean(returns))


def train_sac_delay0(cfg: SACConfig, log_dir: str) -> float:
    env = make_mujoco_env(env_name=cfg.env, seed=cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_high = float(env.action_space.high[0])
    action_low = float(env.action_space.low[0])

    logger = SummaryWriter(log_dir)
    logger.add_text(
        "config",
        "|param|value|\n|-|-|\n"
        + f"|env|{cfg.env}|\n"
        + f"|seed|{cfg.seed}|\n"
        + f"|total_timesteps|{cfg.total_timesteps}|\n"
        + f"|method|delay0_sac|",
    )

    actor = SAC_Actor(obs_dim, act_dim, action_high=action_high, action_low=action_low).to(cfg.device)
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)

    critic_1 = SAC_Critic(obs_dim, act_dim).to(cfg.device)
    critic_2 = SAC_Critic(obs_dim, act_dim).to(cfg.device)
    target_1 = SAC_Critic(obs_dim, act_dim).to(cfg.device)
    target_2 = SAC_Critic(obs_dim, act_dim).to(cfg.device)
    target_1.load_state_dict(critic_1.state_dict())
    target_2.load_state_dict(critic_2.state_dict())
    target_1.eval()
    target_2.eval()
    critic_opt = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=cfg.critic_lr)

    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(cfg.device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)  # [1]
    alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)
    alpha = float(log_alpha.exp().item())

    rb = SACReplayBuffer(cfg.buffer_size, obs_dim, act_dim)

    obs_np, _ = env.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(cfg.device)  # [1, obs_dim]

    last_eval = None
    for global_step in trange(1, cfg.total_timesteps + 1):
        if global_step <= cfg.learn_start:
            action_np = env.action_space.sample()  # [act_dim]
            action_t = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0).to(cfg.device)  # [1, act_dim]
        else:
            with torch.no_grad():
                action_t, _, _ = actor.get_action(obs_t)  # [1, act_dim], [1,1], [1, act_dim]
            action_np = action_t.squeeze(0).cpu().numpy()  # [act_dim]

        nxt_obs_np, reward, terminated, truncated, _ = env.step(action_np)
        done = bool(terminated or truncated)
        nxt_obs_t = torch.tensor(nxt_obs_np, dtype=torch.float32).unsqueeze(0).to(cfg.device)  # [1, obs_dim]
        rb.store(obs_t, action_t, float(reward), nxt_obs_t, done)

        if done:
            obs_np, _ = env.reset(seed=cfg.seed)
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(cfg.device)  # [1, obs_dim]
        else:
            obs_t = nxt_obs_t

        if global_step >= cfg.learn_start and rb.length >= cfg.batch_size:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(cfg.batch_size, cfg.device)
            with torch.no_grad():
                nxt_actions, nxt_log_pi, _ = actor.get_action(b_next_obs)  # [B,act], [B,1], [B,act]
                q1_nxt = target_1(b_next_obs, nxt_actions)  # [B,1]
                q2_nxt = target_2(b_next_obs, nxt_actions)  # [B,1]
                min_q_nxt = torch.min(q1_nxt, q2_nxt) - alpha * nxt_log_pi  # [B,1]
                target_q = b_rewards + (1.0 - b_dones) * cfg.gamma * min_q_nxt  # [B,1]

            q1 = critic_1(b_obs, b_actions)  # [B,1]
            q2 = critic_2(b_obs, b_actions)  # [B,1]
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)  # scalar
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            if global_step % cfg.actor_freq == 0:
                new_actions, log_pi, _ = actor.get_action(b_obs)  # [B,act], [B,1], [B,act]
                q1_pi = critic_1(b_obs, new_actions)  # [B,1]
                q2_pi = critic_2(b_obs, new_actions)  # [B,1]
                q_pi = torch.min(q1_pi, q2_pi)  # [B,1]
                actor_loss = (alpha * log_pi - q_pi).mean()  # scalar
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                with torch.no_grad():
                    _, log_pi_detach, _ = actor.get_action(b_obs)  # [B,act], [B,1], [B,act]
                alpha_loss = (-(log_alpha.exp()) * (log_pi_detach + target_entropy)).mean()  # scalar
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                alpha = float(log_alpha.exp().item())

            if global_step % cfg.target_freq == 0:
                for p, tp in zip(critic_1.parameters(), target_1.parameters()):
                    tp.data.copy_(cfg.target_update_factor * p.data + (1 - cfg.target_update_factor) * tp.data)
                for p, tp in zip(critic_2.parameters(), target_2.parameters()):
                    tp.data.copy_(cfg.target_update_factor * p.data + (1 - cfg.target_update_factor) * tp.data)

        if global_step % cfg.eval_freq == 0:
            eval_re = evaluate_actor(actor, cfg.env, cfg.seed, cfg.eval_num, cfg.device)
            print(f"global step {global_step}, delay0_sac ep_re {eval_re}")
            logger.add_scalar("eval/delay0_sac", eval_re, global_step=global_step)
            last_eval = eval_re

    env.close()
    logger.close()
    if last_eval is None:
        last_eval = evaluate_actor(actor, cfg.env, cfg.seed, cfg.eval_num, cfg.device)
    return float(last_eval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["delay0_sac", "random"], required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--learn_start", type=int, default=int(5e3))
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--target_update_factor", type=float, default=5e-3)
    parser.add_argument("--actor_freq", type=int, default=2)
    parser.add_argument("--target_freq", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=int(1e4))
    parser.add_argument("--eval_num", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_tag = f'logs/baselines/METHOD_{args.method}/ENV_{args.env}_SEED_{args.seed}'
    os.makedirs(exp_tag, exist_ok=True)

    if args.method == "random":
        value = rollout_random(args.env, args.seed, args.eval_num)
        print(f"global step {args.total_timesteps}, random_policy ep_re {value}")
        print(
            f"FINAL_RESULT method={args.method} env={args.env} seed={args.seed} "
            f"global_step={args.total_timesteps} ep_re={value}"
        )
        return

    cfg = SACConfig(
        env=args.env,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        learn_start=args.learn_start,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        target_update_factor=args.target_update_factor,
        actor_freq=args.actor_freq,
        target_freq=args.target_freq,
        eval_freq=args.eval_freq,
        eval_num=args.eval_num,
        device=device,
    )
    value = train_sac_delay0(cfg, exp_tag)
    print(
        f"FINAL_RESULT method={args.method} env={args.env} seed={args.seed} "
        f"global_step={args.total_timesteps} ep_re={value}"
    )


if __name__ == "__main__":
    main()

