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


def task_id_to_key(
    task_id: int,
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
) -> Tuple[str, int, int]:
    """Map SLURM array task index to (env, delay, seed); matches run_all_slurm.sh order."""
    num_envs = len(envs)
    num_delays = len(delays)
    num_seeds = len(seeds)
    seed_idx = task_id // (num_envs * num_delays)
    env_idx = (task_id % (num_envs * num_delays)) // num_delays
    delay_idx = task_id % num_delays
    return envs[env_idx], delays[delay_idx], seeds[seed_idx]


def find_incomplete_vdpo_tasks(
    vdpo_dir: Path,
    target_global_step: int,
    max_task_id: int = 35,
) -> List[int]:
    """Task indices whose .out log has no eval line at exactly target_global_step."""
    missing: List[int] = []
    for tid in range(max_task_id + 1):
        out_path = vdpo_dir / f"{tid}.out"
        if not out_path.exists():
            missing.append(tid)
            continue
        txt = _read_text(out_path)
        found = False
        for m in VDPO_EVAL_RE.finditer(txt):
            if int(m["gs"]) == target_global_step:
                found = True
                break
        if not found:
            missing.append(tid)
    return missing


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


def write_markdown_table_ret_nor_with_raw_baselines(
    ret_nor: Dict[Tuple[str, int], List[float]],
    baselines: Dict[Tuple[str, str, int], float],
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
) -> str:
    """
    Same as write_markdown_table, plus two columns: raw ep_re for `random` and `delay0_sac`
    (mean±std over seeds). Delay columns remain Ret_nor.
    """
    extra_cols = ["random (raw ep_re)", "delay0_sac (raw ep_re)"]
    delay_cols = [str(d) for d in delays]
    header = ["Task", "Method", "Action Delay"] + delay_cols + extra_cols
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    n_total = len(seeds)
    method_name = "VDPO (Ret_nor) + baselines"

    def _base_vals(method: str, env: str) -> List[float]:
        out: List[float] = []
        for s in seeds:
            k = (method, env, s)
            if k in baselines:
                out.append(float(baselines[k]))
        return out

    for env in envs:
        env_short = re.sub(r"-v\d+$", "", env)
        row = [env_short, method_name, ""]
        for d in delays:
            row.append(format_cell(ret_nor[(env, d)], n_total))
        row.append(format_cell(_base_vals("random", env), n_total))
        row.append(format_cell(_base_vals("delay0_sac", env), n_total))
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["Task average", method_name, ""]
    for d in delays:
        pooled: List[float] = []
        for env in envs:
            pooled.extend(ret_nor[(env, d)])
        avg_row.append(format_cell(pooled, n_total * len(envs)))
    all_rand: List[float] = []
    all_df: List[float] = []
    for env in envs:
        all_rand.extend(_base_vals("random", env))
        all_df.extend(_base_vals("delay0_sac", env))
    avg_row.append(format_cell(all_rand, n_total * len(envs)))
    avg_row.append(format_cell(all_df, n_total * len(envs)))
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines) + "\n"


def write_markdown_table_raw_returns(
    ret_alg: Dict[Tuple[str, int, int], float],
    baselines: Dict[Tuple[str, str, int], float],
    envs: Tuple[str, ...],
    delays: Tuple[int, ...],
    seeds: Tuple[int, ...],
) -> str:
    """
    Unnormalized evaluation returns (ep_re) at the target global step:
    VDPO per (env, delay) and baselines random / delay0_sac (no action delay).
    """
    extra_cols = ["random", "delay0_sac"]
    delay_cols = [str(d) for d in delays]
    header = ["Task", "Method", "Action Delay"] + delay_cols + extra_cols
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    n_total = len(seeds)
    method_name = "VDPO + baselines (raw ep_re)"

    def _base_vals(method: str, env: str) -> List[float]:
        out: List[float] = []
        for s in seeds:
            k = (method, env, s)
            if k in baselines:
                out.append(float(baselines[k]))
        return out

    for env in envs:
        env_short = re.sub(r"-v\d+$", "", env)
        row = [env_short, method_name, ""]
        for d in delays:
            vals: List[float] = []
            for s in seeds:
                k = (env, d, s)
                if k in ret_alg:
                    vals.append(float(ret_alg[k]))
            row.append(format_cell(vals, n_total))
        row.append(format_cell(_base_vals("random", env), n_total))
        row.append(format_cell(_base_vals("delay0_sac", env), n_total))
        lines.append("| " + " | ".join(row) + " |")

    avg_row = ["Task average", method_name, ""]
    for d in delays:
        pooled: List[float] = []
        for env in envs:
            for s in seeds:
                k = (env, d, s)
                if k in ret_alg:
                    pooled.append(float(ret_alg[k]))
        avg_row.append(format_cell(pooled, n_total * len(envs)))
    all_rand: List[float] = []
    all_df: List[float] = []
    for env in envs:
        all_rand.extend(_base_vals("random", env))
        all_df.extend(_base_vals("delay0_sac", env))
    avg_row.append(format_cell(all_rand, n_total * len(envs)))
    avg_row.append(format_cell(all_df, n_total * len(envs)))
    lines.append("| " + " | ".join(avg_row) + " |")

    return "\n".join(lines) + "\n"


def render_full_report_markdown(
    cfg: Config,
    table_md: str,
    incomplete_task_ids: List[int],
) -> str:
    """Preamble + table for a standalone Markdown document."""
    lines: List[str] = []
    lines.append("# VDPO normalized evaluation returns ($Ret_{nor}$)")
    lines.append("")
    lines.append("## Metric (paper-style normalization)")
    lines.append("")
    lines.append(
        "Define $Ret_{nor} = (Ret_{alg} - Ret_{rand}) / (Ret_{df} - Ret_{rand})$, where:"
    )
    lines.append("")
    lines.append("- $Ret_{alg}$: VDPO return (`trans_decision ep_re`) at the target global step.")
    lines.append("- $Ret_{rand}$: random policy baseline return at the same target step.")
    lines.append("- $Ret_{df}$: delay-free SAC baseline (`delay0_sac`) at the same target step.")
    lines.append("")
    lines.append("Per-cell statistics are mean ± std over available seeds; "
                "`(n/3)` means `n` of `3` seeds contributed after baselines and VDPO logs are both available.")
    lines.append("")
    lines.append("## Data sources")
    lines.append("")
    lines.append(f"- **VDPO grid logs**: `{cfg.vdpo_slurm_dir}` (array job `{cfg.vdpo_job_id}`).")
    lines.append(f"- **Baselines** (`random`, `delay0_sac`): `{cfg.baselines_slurm_dir}`.")
    lines.append(f"- **Target global step**: {cfg.target_global_step:,}.")
    lines.append(f"- **Environments**: {', '.join(cfg.envs)}.")
    lines.append(f"- **Action delays**: {list(cfg.delays)}.")
    lines.append(f"- **Seeds**: {list(cfg.seeds)}.")
    lines.append("")
    lines.append("## Incomplete VDPO runs (excluded)")
    lines.append("")
    if incomplete_task_ids:
        lines.append(
            "The following array task indices had no `global step "
            f"{cfg.target_global_step:,}, trans_decision ep_re ...` line in their `.out` log "
            "and were **skipped** for aggregation:"
        )
        lines.append("")
        for tid in incomplete_task_ids:
            env, delay, seed = task_id_to_key(
                tid, cfg.envs, cfg.delays, cfg.seeds
            )
            lines.append(
                f"- task `{tid}` → `{env}`, delay `{delay}`, seed `{seed}`"
            )
        lines.append("")
    else:
        lines.append("None; all tasks reached the target step.")
        lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append(table_md.rstrip())
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python3 extract_ret_nor_md.py --vdpo_job_id {cfg.vdpo_job_id} "
        f"--baselines_job_id {cfg.baselines_job_id} "
        f"--target_global_step {cfg.target_global_step} "
        f"--out_md <path.md> --full_report"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vdpo_job_id", type=str, default="411035")
    p.add_argument("--baselines_job_id", type=str, default="411339")
    p.add_argument("--target_global_step", type=int, default=1_000_000)
    p.add_argument("--out_md", type=str, default="results_table_ret_nor.md")
    p.add_argument(
        "--full_report",
        action="store_true",
        help="Write a standalone Markdown document with metric definition and data sources.",
    )
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
    table_md = write_markdown_table(ret_nor, cfg.envs, cfg.delays, cfg.seeds)

    incomplete = find_incomplete_vdpo_tasks(
        cfg.vdpo_slurm_dir, cfg.target_global_step, max_task_id=35
    )
    if args.full_report:
        md = render_full_report_markdown(cfg, table_md, incomplete)
    else:
        md = table_md

    out_path = Path(args.out_md)
    out_path.write_text(md)

    # Also print a short summary to stdout for quick checks.
    n_cells = len(cfg.envs) * len(cfg.delays)
    n_nonempty = sum(1 for k, v in ret_nor.items() if len(v) > 0)
    print(f"Wrote: {out_path}")
    print(f"VDPO dir: {cfg.vdpo_slurm_dir}")
    print(f"Baselines dir: {cfg.baselines_slurm_dir}")
    print(f"Non-empty cells: {n_nonempty}/{n_cells} (partial completion is expected)")
    if incomplete:
        print(f"Incomplete VDPO task indices (no target step in log): {incomplete}")
    print("")
    print(md)


if __name__ == "__main__":
    main()

