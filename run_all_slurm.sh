#!/bin/bash
# SLURM training script for VDPO - Grid search over all parameters
# Parameters (loop order: env -> delay -> seed):
#   - env: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4 (4 values) - outermost loop
#   - delay: 5, 25, 50 (3 values)
#   - seed: 0 (1 value) - innermost loop
# Total combinations: 4 * 3 * 1 = 12
# At most 3 array tasks run concurrently (%3)

#SBATCH --job-name=g1-pickup-grid-search
#SBATCH --array=0-11
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
SEEDS=(0)
TOTAL_TIMESTEPS=5000000

# Calculate indices from task ID
# Total combinations: 4 * 3 * 1 = 12
# Loop order: env (outermost) -> delay -> seed (innermost)
# env_idx = task_id / (3 * 1) = task_id / 3
# delay_idx = (task_id % 3) / 1
# seed_idx = task_id % 1

TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_DELAYS=${#DELAYS[@]}
NUM_SEEDS=${#SEEDS[@]}

ENV_IDX=$((TASK_ID / (NUM_DELAYS * NUM_SEEDS)))
DELAY_IDX=$(((TASK_ID % (NUM_DELAYS * NUM_SEEDS)) / NUM_SEEDS))
SEED_IDX=$((TASK_ID % NUM_SEEDS))

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
