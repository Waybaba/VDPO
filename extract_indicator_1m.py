#!/usr/bin/env python3
"""
Extract 1M-step evaluation returns from latest SLURM run.

This script extracts the value printed as:
  global step <step>, trans_decision ep_re <value>
from *.out files under:
  output/slurm_logs/g1-pickup-grid-search/<job_id>/

Then it aggregates per (env, delay) across seeds that contain an entry at:
  target_step = 1_000_000

It also prints the paper's Table 2 VDPO values (Ret_nor) for reference.
Note: paper uses Ret_nor, while this script extracts raw ep_re. They are not directly comparable
unless you additionally run and record Ret_rand and Ret_df.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


RE_EVAL = re.compile(r"global step (\d+), trans_decision ep_re ([\d\.\-eE]+)")


@dataclass(frozen=True)
class RunKey:
    env: str
    delay: int
    seed: int


def find_latest_job(job_root: Path) -> Path:
    job_ids = [p for p in job_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not job_ids:
        raise RuntimeError(f"No job directories found under: {job_root}")
    return max(job_ids, key=lambda p: int(p.name))


def map_task_id(task_id: int, envs: list[str], delays: list[int], seeds: list[int]) -> RunKey:
    # Mirrors run_all_slurm.sh:
    # seed_idx = task_id / (NUM_ENVS * NUM_DELAYS)
    # env_idx  = (task_id % (NUM_ENVS * NUM_DELAYS)) / NUM_DELAYS
    # delay_idx = task_id % NUM_DELAYS
    num_envs = len(envs)
    num_delays = len(delays)
    num_seeds = len(seeds)
    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    seed_idx = task_id // (num_envs * num_delays)
    env_idx = (task_id % (num_envs * num_delays)) // num_delays
    delay_idx = task_id % num_delays
    if seed_idx >= num_seeds or env_idx >= num_envs or delay_idx >= num_delays:
        raise ValueError(f"task_id out of range: {task_id}")
    return RunKey(env=envs[env_idx], delay=delays[delay_idx], seed=seeds[seed_idx])


def extract_task_step_value(out_path: Path) -> tuple[int | None, float | None, int | None]:
    """
    Returns:
      (step_at_target_if_present, value_at_that_step_if_present, max_step_seen)
    """
    target_step = 1_000_000
    found_step = None
    found_value = None
    max_step = None

    with out_path.open("r", errors="ignore") as f:
        for line in f:
            m = RE_EVAL.search(line)
            if not m:
                continue
            step = int(m.group(1))
            val = float(m.group(2))
            if max_step is None or step > max_step:
                max_step = step
            if step == target_step:
                found_step = step
                found_value = val

    return found_step, found_value, max_step


def main() -> None:
    # Must match run_all_slurm.sh
    envs = ["HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"]
    delays = [5, 25, 50]
    seeds = [0, 1, 2]

    job_root = Path("output/slurm_logs/g1-pickup-grid-search")
    latest_job = find_latest_job(job_root)

    # Aggregate per (env, delay) over seeds with an exact 1M evaluation.
    values: dict[tuple[str, int], list[float]] = {(e, d): [] for e in envs for d in delays}
    missing: list[tuple[int, RunKey]] = []

    out_files = sorted([p for p in latest_job.glob("*.out") if p.name.split(".")[0].isdigit()], key=lambda p: int(p.stem))
    if not out_files:
        raise RuntimeError(f"No *.out files found in: {latest_job}")

    for out_path in out_files:
        task_id = int(out_path.stem)
        key = map_task_id(task_id, envs=envs, delays=delays, seeds=seeds)

        step_at_target, value_at_target, max_step_seen = extract_task_step_value(out_path)
        if step_at_target is not None and value_at_target is not None:
            values[(key.env, key.delay)].append(value_at_target)
        else:
            missing.append((task_id, key))

    # Print summary
    print(f"Latest SLURM job: {latest_job}")
    print("Extracted metric: eval/trans_decision ep_re at global_step=1_000_000 (exact match).")
    print("")

    for env in envs:
        row = []
        for delay in delays:
            v = values[(env, delay)]
            if v:
                mean = float(np.mean(v))
                std = float(np.std(v)) if len(v) > 1 else 0.0
                row.append(f"{mean:.1f}±{std:.1f} (n={len(v)})")
            else:
                row.append("—")
        print(f"{env}: " + " | ".join(row))

    print("")
    print("Missing tasks (no exact eval at 1M step).")
    # Print a compact list.
    if missing:
        for _, key in missing[:24]:
            print(f"  {key.env} delay={key.delay} seed={key.seed}")
        if len(missing) > 24:
            print(f"  ... and {len(missing) - 24} more")
    else:
        print("  None")

    # Paper Table 2: Ret_nor (mean±std) for VDPO, delays 5/25/50.
    # Units: normalized indicator in [approx 0..1.3].
    paper_ret_nor = {
        "Ant-v4": {5: (0.72, 0.25), 25: (0.56, 0.06), 50: (0.46, 0.07)},
        "HalfCheetah-v4": {5: (1.03, 0.08), 25: (0.70, 0.17), 50: (0.72, 0.21)},
        "Hopper-v4": {5: (1.22, 0.08), 25: (0.82, 0.40), 50: (0.22, 0.04)},
        "Walker2d-v4": {5: (1.27, 0.04), 25: (0.27, 0.11), 50: (0.11, 0.03)},
    }

    print("")
    print("Paper Table 2 (VDPO) values: Ret_nor = (Ret_alg - Ret_rand) / (Ret_df - Ret_rand).")
    for env in envs:
        row = []
        for delay in delays:
            if env in paper_ret_nor and delay in paper_ret_nor[env]:
                mean, std = paper_ret_nor[env][delay]
                row.append(f"{mean:.2f}±{std:.2f}")
            else:
                row.append("—")
        print(f"{env}: " + " | ".join(row))

    print("")
    print("Note: The extracted ep_re values are raw returns in the environment and cannot be directly compared to Ret_nor")
    print("without also estimating Ret_rand and Ret_df for each (env, delay) using the paper's evaluation protocol.")


if __name__ == "__main__":
    main()

