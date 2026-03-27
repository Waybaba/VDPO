#!/usr/bin/env python3
"""
Extract normalized Ret_nor from SLURM logs and write a Markdown table.

Ret_nor is computed as:
  Ret_nor = (Ret_alg - Ret_rand) / (Ret_df - Ret_rand)

Where:
  - Ret_alg: VDPO evaluation return (trans_decision ep_re) at target global step
  - Ret_rand: random policy baseline return
  - Ret_df: delay-free SAC baseline return (delay0_sac)

This script is designed to be robust to partial completion:
it only aggregates over seeds that have all required components available.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Config:
    vdpo_job_id: str
    baselines_job_id: str
    vdpo_slurm_dir: Path
    baselines_slurm_dir: Path
    envs: Tuple[str, ...]
    delays: Tuple[int, ...]
    seeds: Tuple[int, ...]
    target_global_step: int


FINAL_RE = re.compile(
    r"FINAL_RESULT method=(?P<method>\S+) env=(?P<env>\S+) seed=(?P<seed>\d+)"
    r" global_step=(?P<gs>\d+) ep_re=(?P<ep>[-+eE0-9.]+)"
)

VDPO_EVAL_RE = re.compile(
    r"global step (?P<gs>\d+),\s*trans_decision ep_re (?P<ep>[-+eE0-9.]+)"
)


def _read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def load_baselines(
    baselines_dir: Path,
    envs: Iterable[str],
    seeds: Iterable[int],
) -> Dict[Tuple[str, str, int], float]:
    """
    Returns a dict mapping (method, env, seed) -> ep_re.
    """
    baselines: Dict[Tuple[str, str, int], float] = {}
    if not baselines_dir.exists():
        return baselines

    for out_file in sorted(baselines_dir.glob("*.out")):
        txt = _read_text(out_file)
        for m in FINAL_RE.finditer(txt):
            method = m["method"]
            env = m["env"]
            seed = int(m["seed"])
            ep = float(m["ep"])
            baselines[(method, env, seed)] = ep

    # Keep only expected envs/seeds to avoid accidental mixing.
    envs_set = set(envs)
    seeds_set = set(seeds)
    baselines = {
        k: v
        for k, v in baselines.items()
        if (k[1] in envs_set and k[2] in seeds_set)
    }
    return baselines


def load_vdpo_ep_re_at_step(
    vdpo_dir: Path,
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
    target_global_step: int,
) -> Dict[Tuple[str, int, int], float]:
    """
    Returns dict mapping (env, delay, seed) -> ep_re at target_global_step.

    Mapping from task_id to (seed, env, delay) matches run_all_slurm.sh:
      seed -> env -> delay
    """
    results: Dict[Tuple[str, int, int], float] = {}
    if not vdpo_dir.exists():
        return results

    num_envs = len(envs)
    num_delays = len(delays)
    num_seeds = len(seeds)

    for out_file in sorted(vdpo_dir.glob("*.out")):
        try:
            task_id = int(out_file.stem)
        except ValueError:
            continue

        seed_idx = task_id // (num_envs * num_delays)
        env_idx = (task_id % (num_envs * num_delays)) // num_delays
        delay_idx = task_id % num_delays

        if seed_idx >= num_seeds or env_idx >= num_envs or delay_idx >= num_delays:
            continue

        env = envs[env_idx]
        delay = delays[delay_idx]
        seed = seeds[seed_idx]

        txt = _read_text(out_file)
        ep_at: Optional[float] = None
        for m in VDPO_EVAL_RE.finditer(txt):
            if int(m["gs"]) == target_global_step:
                ep_at = float(m["ep"])

        if ep_at is not None:
            results[(env, delay, seed)] = ep_at

    return results


def compute_ret_nor(
    ret_alg: Dict[Tuple[str, int, int], float],
    baselines: Dict[Tuple[str, str, int], float],
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
) -> Dict[Tuple[str, int], List[float]]:
    """
    Returns dict mapping (env, delay) -> list of Ret_nor values (one per available seed).
    """
    out: Dict[Tuple[str, int], List[float]] = {(e, d): [] for e in envs for d in delays}

    for env in envs:
        for seed in seeds:
            key_rand = ("random", env, seed)
            key_df = ("delay0_sac", env, seed)
            if key_rand not in baselines or key_df not in baselines:
                continue

            ret_rand = baselines[key_rand]
            ret_df = baselines[key_df]
            denom = ret_df - ret_rand
            if abs(denom) < 1e-12:
                continue

            for delay in delays:
                key_alg = (env, delay, seed)
                if key_alg not in ret_alg:
                    continue
                ret_nor = (ret_alg[key_alg] - ret_rand) / denom
                out[(env, delay)].append(float(ret_nor))

    return out


def format_cell(values: List[float], n_total: int) -> str:
    if not values:
        return "—"
    arr = np.asarray(values, dtype=np.float64)  # shape: (n_available,)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    n = len(values)
    return f"{mean:.3f}±{std:.3f} ({n}/{n_total})"


def write_markdown_table(
    ret_nor: Dict[Tuple[str, int], List[float]],
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
) -> str:
    delay_cols = [str(d) for d in delays]
    header = ["Task", "Method", "Action Delay"] + delay_cols
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    n_total = len(seeds)
    method_name = "VDPO (Ret_nor)"

    for env in envs:
        env_short = re.sub(r"-v\d+$", "", env)
        row = [env_short, method_name, ""]
        for d in delays:
            row.append(format_cell(ret_nor[(env, d)], n_total))
        lines.append("| " + " | ".join(row) + " |")

    # Task average across envs: pool available normalized values.
    avg_row = ["Task average", method_name, ""]
    for d in delays:
        pooled: List[float] = []
        for env in envs:
            pooled.extend(ret_nor[(env, d)])
        avg_row.append(format_cell(pooled, n_total * len(envs)))
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vdpo_job_id", type=str, default="411035")
    p.add_argument("--baselines_job_id", type=str, default="411339")
    p.add_argument("--target_global_step", type=int, default=1_000_000)
    p.add_argument("--out_md", type=str, default="results_table_ret_nor.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        vdpo_job_id=args.vdpo_job_id,
        baselines_job_id=args.baselines_job_id,
        vdpo_slurm_dir=Path("output/slurm_logs/g1-pickup-grid-search") / args.vdpo_job_id,
        baselines_slurm_dir=Path("output/slurm_logs/vdpo-baselines") / args.baselines_job_id,
        envs=("HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"),
        delays=(5, 25, 50),
        seeds=(0, 1, 2),
        target_global_step=int(args.target_global_step),
    )

    baselines = load_baselines(cfg.baselines_slurm_dir, cfg.envs, cfg.seeds)
    ret_alg = load_vdpo_ep_re_at_step(
        cfg.vdpo_slurm_dir,
        cfg.envs,
        cfg.delays,
        cfg.seeds,
        cfg.target_global_step,
    )

    ret_nor = compute_ret_nor(ret_alg, baselines, cfg.envs, cfg.delays, cfg.seeds)
    md = write_markdown_table(ret_nor, cfg.envs, cfg.delays, cfg.seeds)

    out_path = Path(args.out_md)
    out_path.write_text(md)

    # Also print a short summary to stdout for quick checks.
    n_cells = len(cfg.envs) * len(cfg.delays)
    n_nonempty = sum(1 for k, v in ret_nor.items() if len(v) > 0)
    print(f"Wrote: {out_path}")
    print(f"VDPO dir: {cfg.vdpo_slurm_dir}")
    print(f"Baselines dir: {cfg.baselines_slurm_dir}")
    print(f"Non-empty cells: {n_nonempty}/{n_cells} (partial completion is expected)")
    print("")
    print(md)


if __name__ == "__main__":
    main()

