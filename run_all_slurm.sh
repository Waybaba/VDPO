#!/bin/bash
# SLURM training script for VDPO - Grid search over all parameters
# Parameters (loop order: env -> delay -> seed):
#   - env: HalfCheetah-v5, Hopper-v5, Ant-v5, Walker2d-v5 (4 values) - outermost loop
#   - delay: 0, 4, 8, 12, 16, 20 (6 values)
#   - seed: 0, 1, 2 (3 values) - innermost loop
# Total combinations: 4 * 6 * 3 = 72

#SBATCH --job-name=g1-pickup-grid-search
#SBATCH --array=0-71%8
#SBATCH --gres=gpu:1
#SBATCH --exclude=al-l40s-0.grasp.maas
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=output/slurm_logs/%x/%A/%a.out
#SBATCH --error=output/slurm_logs/%x/%A/%a.err

ENVS=("HalfCheetah-v5" "Hopper-v5" "Ant-v5" "Walker2d-v5")
DELAYS=(0 4 8 12 16 20)
SEEDS=(0 1 2)
TOTAL_TIMESTEPS=5000000

# Calculate indices from SLURM array task ID
# Loop order: env (outermost) -> delay -> seed (innermost)
TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_DELAYS=${#DELAYS[@]}
NUM_SEEDS=${#SEEDS[@]}

ENV_IDX=$((TASK_ID / (NUM_DELAYS * NUM_SEEDS)))
DELAY_IDX=$(((TASK_ID % (NUM_DELAYS * NUM_SEEDS)) / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

ENV=${ENVS[$ENV_IDX]}
DELAY=${DELAYS[$DELAY_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Running ${ENV} delay=${DELAY} seed=${SEED} (Task ID: ${TASK_ID})"
python3 VDPO.py --env=${ENV} --delay=${DELAY} --total_timesteps=${TOTAL_TIMESTEPS} --seed=${SEED}
