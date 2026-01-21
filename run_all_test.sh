#!/bin/bash

# Run VDPO on multiple MuJoCo v5 environments (TEST - reduced timesteps)
# Environments: HalfCheetah-v5, Hopper-v5, Ant-v5, Walker2d-v5
# Delays: 0, 4, 8, 12, 16, 20
# Seeds: 0, 1, 2 (3 seeds)
# Total timesteps: 1500 per run (for testing)

### Set environment variables
export PYTHONUNBUFFERED=1

### VARIABLES
CONDA_ENV="VDPO"
ENVS=("HalfCheetah-v5" "Hopper-v5" "Ant-v5" "Walker2d-v5")
DELAYS=(0 4 8 12 16 20)
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
