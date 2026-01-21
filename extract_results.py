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

def parse_output_log(log_file):
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
    
    # Return the last evaluation result
    if results:
        return results[-1][1]  # Return the value of the last evaluation
    return None

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
    # Configuration
    ENVS = ["HalfCheetah-v5", "Hopper-v5", "Ant-v5", "Walker2d-v5"]
    DELAYS = [4, 8, 16]  # Note: 0 is not in the current setup
    SEEDS = [0, 1, 2]
    METHOD = "VDPO"  # Single method
    
    # Storage: results[(env, delay, seed)] = value
    results = {}
    
    # Method 1: Parse tensorboard logs from logs/VDPO/
    print("Parsing tensorboard logs...")
    logs_dir = Path("logs/VDPO")
    if logs_dir.exists():
        for env in ENVS:
            for delay in DELAYS:
                for seed in SEEDS:
                    log_dir = logs_dir / f"ENV_{env}_DELAYS_{delay}_SEED_{seed}"
                    if log_dir.exists():
                        value = parse_tensorboard_log(str(log_dir))
                        if value is not None:
                            results[(env, delay, seed)] = value
                            print(f"  Found: {env} delay={delay} seed={seed}: {value:.2f}")
    
    # Method 2: Parse output logs from SLURM logs
    print("\nParsing SLURM output logs...")
    slurm_logs_dir = Path("output/slurm_logs/g1-pickup-grid-search")
    if slurm_logs_dir.exists():
        for job_dir in slurm_logs_dir.iterdir():
            if job_dir.is_dir():
                for log_file in job_dir.glob("*.out"):
                    task_id = log_file.stem
                    value = parse_output_log(log_file)
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
                            
                            # Only store if not already found from tensorboard
                            if (env, delay, seed) not in results:
                                results[(env, delay, seed)] = value
                                print(f"  Found: {env} delay={delay} seed={seed}: {value:.2f}")
    
    return results, METHOD, ENVS, DELAYS

def generate_table(results, method, envs, delays):
    """Generate the results table."""
    seeds = [0, 1, 2]
    
    # Calculate statistics for each (env, delay) combination
    stats = {}  # stats[(env, delay)] = {'mean': x, 'std': y, 'values': [...]}
    
    for env in envs:
        for delay in delays:
            values = []
            for seed in seeds:
                key = (env, delay, seed)
                if key in results:
                    values.append(results[key])
            
            if values:
                stats[(env, delay)] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
    
    # Generate table - matching the format from the example
    print("\n" + "="*80)
    print("Results Table")
    print("="*80)
    print(f"{'Task':<20} {'Method':<25} {'Action Delay':<12}", end="")
    for delay in delays:
        print(f"{delay:>15}", end="")
    print()
    print(f"{'0 (=SAC)':<20} {'':<25} {'':<12}", end="")
    for delay in delays:
        print(f"{delay:>15}", end="")
    print()
    
    for env in envs:
        env_short = env.replace("-v5", "")
        print(f"{env_short}")
        print(f"{'':<20} {method:<25} {'':<12}", end="")
        
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
    print(f"{'':<20} {method:<25} {'':<12}", end="")
    
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
    print("="*80)

def main():
    print("Extracting results from logs...\n")
    
    results, method, envs, delays = extract_results_from_logs()
    
    print(f"\nTotal results found: {len(results)}")
    print(f"Expected: {len(envs) * len(delays) * 3} (environments × delays × seeds)")
    
    generate_table(results, method, envs, delays)

if __name__ == "__main__":
    main()
