#!/bin/bash

# Run VDPO on multiple MuJoCo v4 environments
# Environments: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4
# Delays: 4, 8, 16
# Seeds: 0, 1, 2 (3 seeds)
# Total timesteps: 5M per run

ENVS=("HalfCheetah-v4" "Hopper-v4" "Ant-v4" "Walker2d-v4")
DELAYS=(4 8 16)
SEEDS=(0 1 2)
TOTAL_TIMESTEPS=5000000

for env in "${ENVS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running ${env} delay=${delay} seed=${seed}"
            python3 VDPO.py --env=${env} --delay=${delay} --total_timesteps=${TOTAL_TIMESTEPS} --seed=${seed}
        done
    done
done
