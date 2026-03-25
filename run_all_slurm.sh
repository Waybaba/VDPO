#!/bin/bash
# SLURM training script for VDPO - Grid search over all parameters
# Parameters (loop order: seed -> env -> delay):
#   - seed: 0, 1, 2 (3 values) - outermost loop
#   - env: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4 (4 values)
#   - delay: 5, 25, 50 (3 values) - innermost loop
# Total combinations: 3 * 4 * 3 = 36
# At most 3 array tasks run concurrently (%3)

#SBATCH --job-name=g1-pickup-grid-search
#SBATCH --array=0-35%3
#SBATCH --gres=gpu:1
#SBATCH --exclude=al-l40s-0.grasp.maas
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=output/slurm_logs/%x/%A/%a.out
#SBATCH --error=output/slurm_logs/%x/%A/%a.err

### Set environment variables
export PYTHONUNBUFFERED=1

### VARIABLES
CONDA_ENV="VDPO"
ENVS=("HalfCheetah-v4" "Hopper-v4" "Ant-v4" "Walker2d-v4")
DELAYS=(5 25 50)
SEEDS=(0 1 2)
TOTAL_TIMESTEPS=1000000

# Calculate indices from task ID
# Total combinations: 3 * 4 * 3 = 36
# Loop order: seed (outermost) -> env -> delay (innermost)
# seed_idx = task_id / (4 * 3) = task_id / 12
# env_idx = (task_id % 12) / 3
# delay_idx = task_id % 3

TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_SEEDS=${#SEEDS[@]}
NUM_ENVS=${#ENVS[@]}
NUM_DELAYS=${#DELAYS[@]}

SEED_IDX=$((TASK_ID / (NUM_ENVS * NUM_DELAYS)))
ENV_IDX=$(((TASK_ID % (NUM_ENVS * NUM_DELAYS)) / NUM_DELAYS))
DELAY_IDX=$((TASK_ID % NUM_DELAYS))

# Get parameter values
ENV=${ENVS[$ENV_IDX]}
DELAY=${DELAYS[$DELAY_IDX]}
SEED=${SEEDS[$SEED_IDX]}

### RUN
source ~/.bashrc
conda activate ${CONDA_ENV}

echo "Starting VDPO training - Grid search"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Timesteps: $TOTAL_TIMESTEPS"
echo "Parameters:"
echo "  - env: $ENV (idx: $ENV_IDX)"
echo "  - delay: $DELAY (idx: $DELAY_IDX)"
echo "  - seed: $SEED (idx: $SEED_IDX)"
echo ""

python3 VDPO.py --env=${ENV} --delay=${DELAY} --total_timesteps=${TOTAL_TIMESTEPS} --seed=${SEED}
