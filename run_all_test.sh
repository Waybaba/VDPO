#!/bin/bash

# Run VDPO on multiple MuJoCo v4 environments (TEST - reduced timesteps)
# Environments: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4
# Delays: 4, 8, 16
# Seeds: 0, 1, 2 (3 seeds)
# Total timesteps: 1500 per run (for testing)

### Set environment variables
export PYTHONUNBUFFERED=1

### VARIABLES
CONDA_ENV="VDPO"
ENVS=("HalfCheetah-v4" "Hopper-v4" "Ant-v4" "Walker2d-v4")
DELAYS=(4 8 16)
SEEDS=(0 1 2)
TOTAL_TIMESTEPS=1500

### Setup
source ~/.bashrc
conda activate ${CONDA_ENV}

for env in "${ENVS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running ${env} delay=${delay} seed=${seed}"
            python3 VDPO.py --env=${env} --delay=${delay} --total_timesteps=${TOTAL_TIMESTEPS} --seed=${seed}
        done
    done
done
