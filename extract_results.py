#!/usr/bin/env python3
"""
Extract evaluation results from SLURM logs and generate a table.
"""

import os
import re
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Warning: tensorboard not available, will only parse output logs")
    EventAccumulator = None

def parse_output_log(log_file, total_timesteps=5000000):
    """Parse output log to extract final evaluation results."""
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            # Match: global step 50000, trans_decision ep_re 4785.301019354389
            match = re.search(r'global step (\d+), trans_decision ep_re ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                value = float(match.group(2))
                results.append((step, value))
    
    # Return the last evaluation result with completion status
    if results:
        last_step, last_value = results[-1]
        # Consider completed if within 100k steps of total_timesteps
        completed = last_step >= (total_timesteps - 100000)
        return last_value, last_step, completed
    return None, None, False

def parse_tensorboard_log(log_dir):
    """Parse tensorboard log to extract final evaluation result."""
    if EventAccumulator is None:
        return None
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        if 'eval/trans_decision' in ea.Tags()['scalars']:
            scalar_events = ea.Scalars('eval/trans_decision')
            if scalar_events:
                # Return the last (final) evaluation result
                return scalar_events[-1].value
    except Exception as e:
        pass
    
    return None

def extract_results_from_logs():
    """Extract all results from log directories."""
    # Configuration — keep in sync with the batch script you use:
    # run_all_slurm.sh: delays 5,25,50 and seed 0; for run_all.sh use DELAYS=[4,8,16], SEEDS=[0,1,2].
    ENVS = ["HalfCheetah-v4", "Hopper-v4", "Ant-v4", "Walker2d-v4"]
    DELAYS = [5, 25, 50]
    SEEDS = [0]
    METHOD = "VDPO"  # Single method
    
    # Storage: results[(env, delay, seed)] = value
    results = {}
    
    # Method 1: Parse output logs from SLURM logs (FAST - preferred method)
    print("Parsing SLURM output logs (fast method)...")
    TOTAL_TIMESTEPS = 5000000
    slurm_logs_dir = Path("output/slurm_logs/g1-pickup-grid-search")
    if slurm_logs_dir.exists():
        for job_dir in slurm_logs_dir.iterdir():
            if job_dir.is_dir():
                for log_file in job_dir.glob("*.out"):
                    task_id = log_file.stem
                    value, step, completed = parse_output_log(log_file, TOTAL_TIMESTEPS)
                    if value is not None:
                        # Need to reverse engineer (env, delay, seed) from task_id
                        # This matches the calculation in run_all_slurm.sh
                        task_id_int = int(task_id)
                        NUM_DELAYS = len(DELAYS)
                        NUM_SEEDS = len(SEEDS)
                        
                        env_idx = task_id_int // (NUM_DELAYS * NUM_SEEDS)
                        delay_idx = (task_id_int % (NUM_DELAYS * NUM_SEEDS)) // NUM_SEEDS
                        seed_idx = task_id_int % NUM_SEEDS
                        
                        if env_idx < len(ENVS) and delay_idx < len(DELAYS) and seed_idx < len(SEEDS):
                            env = ENVS[env_idx]
                            delay = DELAYS[delay_idx]
                            seed = SEEDS[seed_idx]
                            
                            results[(env, delay, seed)] = {
                                'value': value,
                                'step': step,
                                'completed': completed
                            }
                            status = "✓" if completed else f"✗ ({step:,})"
                            print(f"  Found: {env} delay={delay} seed={seed}: {value:.2f} {status}")
    
    # Method 2: Parse tensorboard logs from logs/VDPO/ (SLOW - only for missing results)
    # Only use tensorboard if output logs are missing
    missing_count = len(ENVS) * len(DELAYS) * len(SEEDS) - len(results)
    if missing_count > 0 and EventAccumulator is not None:
        print(f"\nParsing tensorboard logs for {missing_count} missing results (slow method)...")
        logs_dir = Path("logs/VDPO")
        if logs_dir.exists():
            for env in ENVS:
                for delay in DELAYS:
                    for seed in SEEDS:
                        # Skip if already found
                        if (env, delay, seed) in results:
                            continue
                        
                        log_dir = logs_dir / f"ENV_{env}_DELAYS_{delay}_SEED_{seed}"
                        if log_dir.exists():
                            value = parse_tensorboard_log(str(log_dir))
                            if value is not None:
                                results[(env, delay, seed)] = value
                                print(f"  Found: {env} delay={delay} seed={seed}: {value:.2f}")
    
    return results, METHOD, ENVS, DELAYS, SEEDS

def generate_table(results, method, envs, delays, seeds, use_only_completed=True):
    """Generate the results table."""
    
    # Calculate statistics for each (env, delay) combination
    stats = {}  # stats[(env, delay)] = {'mean': x, 'std': y, 'values': [...], 'completed_count': n}
    
    for env in envs:
        for delay in delays:
            values = []
            completed_values = []
            for seed in seeds:
                key = (env, delay, seed)
                if key in results:
                    result = results[key]
                    if isinstance(result, dict):
                        value = result['value']
                        completed = result.get('completed', False)
                        values.append(value)
                        if completed:
                            completed_values.append(value)
                    else:
                        # Backward compatibility
                        values.append(result)
                        completed_values.append(result)
            
            if use_only_completed:
                # Only use completed runs for statistics
                if completed_values and len(completed_values) >= 1:
                    stats[(env, delay)] = {
                        'mean': np.mean(completed_values),
                        'std': np.std(completed_values),
                        'values': completed_values,
                        'completed_count': len(completed_values),
                        'total_count': len(values)
                    }
            else:
                # Use all available results
                if values:
                    stats[(env, delay)] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values,
                        'completed_count': len(completed_values) if completed_values else 0,
                        'total_count': len(values)
                    }
    
    # Generate table - matching the format from the example
    # Header row
    print("\n" + "="*100)
    print(f"{'Task':<20} {'Method':<25} {'Action Delay':<15}", end="")
    print(f"{'0 (=SAC)':>15}", end="")
    for delay in delays:
        print(f"{delay:>15}", end="")
    print()
    
    # Data rows for each environment
    for env in envs:
        env_short = re.sub(r"-v\d+$", "", env)  # strip Gymnasium version suffix (e.g. -v4, -v5)
        print(f"{env_short}")
        print(f"{'':<20} {method:<25} {'':<15}", end="")
        
        # Delay 0 (not available for VDPO, show —)
        print("—".rjust(15), end="")
        
        for delay in delays:
            key = (env, delay)
            if key in stats:
                mean = stats[key]['mean']
                std = stats[key]['std']
                print(f"{mean:.0f}±{std:.0f}".rjust(15), end="")
            else:
                print("—".rjust(15), end="")
        print()
    
    # Task average row
    print(f"{'Task average'}")
    print(f"{'':<20} {method:<25} {'':<15}", end="")
    
    # Delay 0 (not available)
    print("—".rjust(15), end="")
    
    for delay in delays:
        all_values = []
        for env in envs:
            key = (env, delay)
            if key in stats:
                all_values.extend(stats[key]['values'])
        
        if all_values:
            mean = np.mean(all_values)
            std = np.std(all_values)
            print(f"{mean:.0f}±{std:.0f}".rjust(15), end="")
        else:
            print("—".rjust(15), end="")
    print()
    print("="*100)
    print("\nNote: Only completed runs (≥4.9M steps) are included in statistics. Incomplete runs are excluded.")
    return stats

def write_vdpo_results_txt(results, envs, delays, stats, seeds, output_path="results_table.txt"):
    """Write a brief VDPO-only table to txt. Marks incomplete grids as (mean±std, n/N seeds)."""
    W = 20  # width per number column
    n_expect = len(seeds)
    lines = []
    lines.append("VDPO Results")
    lines.append("")
    hdr = f"{'Task':<18}"
    for d in delays:
        hdr += f" {'Delay ' + str(d):<{W}}"
    lines.append(hdr)
    lines.append("-" * (18 + len(delays) * (W + 1)))

    for env in envs:
        row = [env]
        for delay in delays:
            key = (env, delay)
            if key in stats:
                mean, std = stats[key]['mean'], stats[key]['std']
                n = stats[key]['completed_count']
                s = f"{mean:.0f}±{std:.0f}"
                if n < n_expect:
                    s = f"{s} ({n}/{n_expect} seeds)"
                row.append(s)
            else:
                row.append("—")
        line = f"{row[0]:<18}" + "".join(f" {row[i]:<{W}}" for i in range(1, len(row)))
        lines.append(line)

    lines.append("-" * (18 + len(delays) * (W + 1)))
    row = ["Task average"]
    for delay in delays:
        all_vals = []
        any_incomplete = False
        for env in envs:
            key = (env, delay)
            if key in stats:
                all_vals.extend(stats[key]['values'])
                if stats[key]['completed_count'] < n_expect:
                    any_incomplete = True
        if all_vals:
            s = f"{np.mean(all_vals):.0f}±{np.std(all_vals):.0f}"
            if any_incomplete:
                s = f"{s} (partial)"
            row.append(s)
        else:
            row.append("—")
    line = f"{row[0]:<18}" + "".join(f" {row[i]:<{W}}" for i in range(1, len(row)))
    lines.append(line)

    txt = "\n".join(lines) + "\n"
    Path(output_path).write_text(txt, encoding="utf-8")
    print(f"\nWrote: {output_path}")

def write_vdpo_results_md(results, envs, delays, stats, seeds, output_path="results_table.md"):
    """Write a Markdown table for easier reading."""
    n_expect = len(seeds)
    lines = []
    lines.append("# VDPO Results")
    lines.append("")

    # Header
    header = ["Task"] + [f"Delay {d}" for d in delays]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    # Rows per environment
    for env in envs:
        row = [env]
        for delay in delays:
            key = (env, delay)
            if key in stats:
                mean, std = stats[key]['mean'], stats[key]['std']
                n = stats[key]['completed_count']
                s = f"{mean:.0f}$\\pm${std:.0f}"
                if n < n_expect:
                    s = f"{s} ({n}/{n_expect} seeds)"
                row.append(s)
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Task average row
    avg_row = ["Task average"]
    for delay in delays:
        all_vals = []
        any_incomplete = False
        for env in envs:
            key = (env, delay)
            if key in stats:
                all_vals.extend(stats[key]['values'])
                if stats[key]['completed_count'] < n_expect:
                    any_incomplete = True
        if all_vals:
            s = f"{np.mean(all_vals):.0f}$\\pm${np.std(all_vals):.0f}"
            if any_incomplete:
                s = f"{s} (partial)"
            avg_row.append(s)
        else:
            avg_row.append("—")
    lines.append("| " + " | ".join(avg_row) + " |")
    lines.append("")
    lines.append(f"- Expected runs per cell: {n_expect} seeds")
    lines.append("- Stats include completed runs only (>= 4.9M steps in current script).")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")

def main():
    print("Extracting results from logs...\n")
    
    results, method, envs, delays, seeds = extract_results_from_logs()
    
    print(f"\nTotal results found: {len(results)}")
    print(f"Expected: {len(envs) * len(delays) * len(seeds)} (environments × delays × seeds)")
    
    stats = generate_table(results, method, envs, delays, seeds)
    write_vdpo_results_txt(results, envs, delays, stats, seeds)
    write_vdpo_results_md(results, envs, delays, stats, seeds)

if __name__ == "__main__":
    main()
