#!/usr/bin/env python3
"""
Build a single Ret_nor Markdown table for MuJoCo v4 at 1M global steps,
merging two VDPO grid jobs:
  - delays (4, 8, 16) from one SLURM array job
  - delays (5, 25, 50) from another SLURM array job

Normalization matches extract_ret_nor_md.py / the VDPO paper Sec. 4.2.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_ret_nor_md import (
    compute_ret_nor,
    find_incomplete_vdpo_tasks,
    load_baselines,
    load_vdpo_ep_re_at_step,
    write_markdown_table_raw_returns,
    write_markdown_table_ret_nor_with_raw_baselines,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vdpo_job_id_short_delays",
        type=str,
        default="411547",
        help="Job folder for delays (4, 8, 16).",
    )
    p.add_argument(
        "--vdpo_job_id_mid_delays",
        type=str,
        default="411035",
        help="Job folder for delays (5, 25, 50).",
    )
    p.add_argument("--baselines_job_id", type=str, default="411339")
    p.add_argument("--target_global_step", type=int, default=1_000_000)
    p.add_argument(
        "--out_md",
        type=str,
        default="results_ret_nor_v4_1M_delays_combined.md",
    )
    args = p.parse_args()

    envs = ("HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4")
    seeds = (0, 1, 2)
    delays_short = (4, 8, 16)
    delays_mid = (5, 25, 50)
    delays_all = (4, 8, 16, 5, 25, 50)

    root = Path("output/slurm_logs/g1-pickup-grid-search")
    vdpo_short = root / args.vdpo_job_id_short_delays
    vdpo_mid = root / args.vdpo_job_id_mid_delays
    baselines_dir = Path("output/slurm_logs/vdpo-baselines") / args.baselines_job_id

    baselines = load_baselines(baselines_dir, envs, seeds)
    r_short = load_vdpo_ep_re_at_step(
        vdpo_short, envs, delays_short, seeds, int(args.target_global_step)
    )
    r_mid = load_vdpo_ep_re_at_step(
        vdpo_mid, envs, delays_mid, seeds, int(args.target_global_step)
    )
    ret_alg = {**r_short, **r_mid}
    ret_nor = compute_ret_nor(ret_alg, baselines, envs, delays_all, seeds)
    table_norm_with_baselines = write_markdown_table_ret_nor_with_raw_baselines(
        ret_nor, baselines, envs, delays_all, seeds
    )
    table_raw = write_markdown_table_raw_returns(
        ret_alg, baselines, envs, delays_all, seeds
    )

    inc_s = find_incomplete_vdpo_tasks(
        vdpo_short, int(args.target_global_step), max_task_id=35
    )
    inc_m = find_incomplete_vdpo_tasks(
        vdpo_mid, int(args.target_global_step), max_task_id=35
    )

    tg = int(args.target_global_step)
    lines = []
    lines.append("# VDPO $Ret_{nor}$ — MuJoCo v4, 1M steps (all delay grids combined)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "- **Metric**: $Ret_{nor} = (Ret_{alg} - Ret_{rand}) / (Ret_{df} - Ret_{rand})$ "
        "(same as the paper Table 2 definition; baselines: `random`, `delay0_sac`)."
    )
    lines.append(f"- **Target global step**: {tg:,}.")
    lines.append(
        f"- **Delays 4 / 8 / 16** — VDPO logs: `{vdpo_short}` (job `{args.vdpo_job_id_short_delays}`)."
    )
    lines.append(
        f"- **Delays 5 / 25 / 50** — VDPO logs: `{vdpo_mid}` (job `{args.vdpo_job_id_mid_delays}`)."
    )
    lines.append(f"- **Baselines**: `{baselines_dir}`.")
    lines.append("- **Seeds**: 0, 1, 2 (cells show mean±std and `(n/3)` when partial).")
    lines.append("")
    lines.append("## Incomplete VDPO tasks (no line at target global step)")
    lines.append("")
    lines.append(f"- Short-delay job `{args.vdpo_job_id_short_delays}`: `{inc_s or 'none'}`.")
    lines.append(f"- Mid-delay job `{args.vdpo_job_id_mid_delays}`: `{inc_m or 'none'}`.")
    lines.append("")
    lines.append(
        "## Table 1: $Ret_{nor}$ (action-delay columns) + baseline raw returns (last two columns)"
    )
    lines.append("")
    lines.append(
        "Delay columns **4–50** are **normalized** $Ret_{nor}$. The last two columns are **raw** "
        "evaluation returns (`ep_re` at the target global step) for the `random` policy and "
        "for **delay-free SAC** (`delay0_sac`)."
    )
    lines.append("")
    lines.append("### Experimental conditions (for screenshots / collaborators)")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(
        "| **MuJoCo / task API version** | **Gymnasium MuJoCo v4** "
        "(`HalfCheetah-v4`, `Hopper-v4`, `Ant-v4`, `Walker2d-v4`) |"
    )
    if tg == 1_000_000:
        train_cell = (
            "**1,000,000** global environment steps per run "
            "(**1M steps** = **1 million** steps total per run)"
        )
    else:
        train_cell = f"**{tg:,}** global environment steps per run"
    lines.append(f"| **Training horizon** | {train_cell} |")
    lines.append(
        "| **Metric reporting point** | $Ret_{alg}$, $Ret_{rand}$, $Ret_{df}$ all taken at the "
        f"same **global step = {tg:,}** (aligned with paper Table 2: 1M-step reporting) |"
    )
    lines.append(
        "| **Random seeds** | **3** runs per cell when complete: seeds **0, 1, 2** |"
    )
    lines.append(
        f"| **Action delays in table** | **4, 8, 16** (job `{args.vdpo_job_id_short_delays}`) and "
        f"**5, 25, 50** (job `{args.vdpo_job_id_mid_delays}`) |"
    )
    lines.append("")
    lines.append(
        "*MuJoCo **v4** tasks; each training run uses **one million** environment steps.*"
    )
    lines.append("")
    lines.append(table_norm_with_baselines.rstrip())
    lines.append("")
    lines.append("## Table 2: Raw returns (all columns unnormalized, ep_re at target step)")
    lines.append("")
    lines.append(
        "Same layout as Table 1, but every cell is the **raw** episodic return "
        "(`trans_decision ep_re` for VDPO; `FINAL_RESULT ep_re` for baselines)."
    )
    lines.append("")
    lines.append(table_raw.rstrip())
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 extract_ret_nor_combined_md.py --out_md results_ret_nor_v4_1M_delays_combined.md")
    lines.append("```")
    lines.append("")

    out_path = Path(args.out_md)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
