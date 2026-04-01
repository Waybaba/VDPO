#!/usr/bin/env python3
"""
Extract VDPO eval returns from SLURM stdout logs for the g1PickupVD 5M grid.

Log layout (matches run_all_slurm.sh):
  output/slurm_logs/g1PickupVD/<job_id>/<array_task_id>.out

Grid (loop order: seed -> env -> delay, same indexing as run_all_slurm.sh):
  - env: HalfCheetah-v5, Hopper-v5, Ant-v5, Walker2d-v5
  - delay: 4, 8, 16
  - seeds: 0..9
  - total_timesteps: 5_000_000

Each line: global step <n>, trans_decision ep_re <value>
We take the last (step, value) in the file; a run counts as completed if
last_step >= total_timesteps - 100_000 (same rule as extract_results.py).

Outputs: console table plus .txt / .md (same style as extract_results.py).

Default SLURM array parent job id is fixed to this batch (no need to pass flags).
Use --latest only if you intentionally want the newest directory under the log root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from extract_results import (
    generate_table,
    parse_output_log,
    write_vdpo_results_md,
    write_vdpo_results_txt,
)

# Must match run_all_slurm.sh
ENVS = ["HalfCheetah-v5", "Hopper-v5", "Ant-v5", "Walker2d-v5"]
DELAYS = [4, 8, 16]
SEEDS = list(range(10))
TOTAL_TIMESTEPS = 5_000_000
METHOD = "VDPO"
DEFAULT_LOG_ROOT = Path("output/slurm_logs/g1PickupVD")
# Locked batch: g1PickupVD 5M grid (10 seeds x 4 envs x 3 delays).
LOCKED_SLURM_JOB_ID = "413920"
DEFAULT_RESULTS_PREFIX = Path("output") / "results_g1pickup_vd_5m"


def find_latest_job_dir(job_root: Path) -> Path:
    job_ids = [p for p in job_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not job_ids:
        raise FileNotFoundError(f"No numeric job directories under: {job_root}")
    return max(job_ids, key=lambda p: int(p.name))


def map_task_id(
    task_id: int,
    envs: list[str],
    delays: list[int],
    seeds: list[int],
) -> tuple[int, int, int]:
    """
    Map SLURM array task id -> (env_idx, delay_idx, seed_idx).
    Mirrors run_all_slurm.sh:
      SEED_IDX = TASK_ID // (NUM_ENVS * NUM_DELAYS)
      ENV_IDX = (TASK_ID % (NUM_ENVS * NUM_DELAYS)) // NUM_DELAYS
      DELAY_IDX = TASK_ID % NUM_DELAYS
    """
    num_envs = len(envs)
    num_delays = len(delays)
    num_seeds = len(seeds)
    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    seed_idx = task_id // (num_envs * num_delays)
    env_idx = (task_id % (num_envs * num_delays)) // num_delays
    delay_idx = task_id % num_delays
    if seed_idx >= num_seeds or env_idx >= num_envs or delay_idx >= num_delays:
        raise ValueError(
            f"task_id {task_id} out of range for "
            f"{num_seeds} seeds x {num_envs} envs x {num_delays} delays"
        )
    return env_idx, delay_idx, seed_idx


def extract_from_job_dir(job_dir: Path) -> dict:
    """Build results[(env, delay, seed)] = {value, step, completed}."""
    results: dict = {}
    for log_file in sorted(job_dir.glob("*.out")):
        if not log_file.stem.isdigit():
            continue
        task_id = int(log_file.stem)
        try:
            env_idx, delay_idx, seed_idx = map_task_id(
                task_id, ENVS, DELAYS, SEEDS
            )
        except ValueError as e:
            print(f"  Skip {log_file.name}: {e}", file=sys.stderr)
            continue
        env = ENVS[env_idx]
        delay = DELAYS[delay_idx]
        seed = SEEDS[seed_idx]
        value, step, completed = parse_output_log(log_file, TOTAL_TIMESTEPS)
        if value is None:
            print(f"  Skip {log_file.name}: no trans_decision lines", file=sys.stderr)
            continue
        results[(env, delay, seed)] = {
            "value": value,
            "step": step,
            "completed": completed,
        }
        mark = "ok" if completed else f"incomplete (last {step:,})"
        print(f"  {env} delay={delay} seed={seed}: {value:.2f} [{mark}]")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract VDPO 5M-step grid results from g1PickupVD SLURM logs."
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=LOCKED_SLURM_JOB_ID,
        help=(
            "SLURM array parent job id (log subfolder name). "
            f"Default is locked to this run: {LOCKED_SLURM_JOB_ID}."
        ),
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Ignore --job_id; use the highest numeric job directory under --log_root."
        ),
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Root folder for g1PickupVD logs (default: {DEFAULT_LOG_ROOT}).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=str(DEFAULT_RESULTS_PREFIX),
        help=(
            "Output path prefix (no extension); writes <prefix>_table.txt and "
            "<prefix>_table.md under output/ by default."
        ),
    )
    parser.add_argument(
        "--include_incomplete_stats",
        action="store_true",
        help="If set, mean/std use all parsed values, not only completed runs.",
    )
    args = parser.parse_args()

    log_root = args.log_root
    if not log_root.is_dir():
        print(f"Log root does not exist: {log_root}", file=sys.stderr)
        return 1

    if args.latest:
        job_dir = find_latest_job_dir(log_root)
        print(f"Using latest job dir: {job_dir}\n")
    else:
        job_dir = log_root / str(args.job_id)
        if not job_dir.is_dir():
            print(f"Job log dir not found: {job_dir}", file=sys.stderr)
            return 1
        print(f"Using job dir: {job_dir}\n")

    print("Parsing *.out (5M target, completed if last step >= 4.9M)...")
    results = extract_from_job_dir(job_dir)

    expected = len(ENVS) * len(DELAYS) * len(SEEDS)
    print(f"\nTotal results parsed: {len(results)} / expected {expected}")

    use_completed = not args.include_incomplete_stats
    stats = generate_table(
        results, METHOD, ENVS, DELAYS, SEEDS, use_only_completed=use_completed
    )

    txt_path = Path(f"{args.prefix}_table.txt")
    md_path = Path(f"{args.prefix}_table.md")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    write_vdpo_results_txt(
        results, ENVS, DELAYS, stats, SEEDS, output_path=str(txt_path)
    )
    write_vdpo_results_md(
        results, ENVS, DELAYS, stats, SEEDS, output_path=str(md_path)
    )
    print(f"\nConfig: {job_dir.name} | timesteps={TOTAL_TIMESTEPS} | seeds={len(SEEDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
