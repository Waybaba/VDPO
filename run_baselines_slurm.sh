#!/bin/bash
# SLURM script for baseline runs used by Ret_nor normalization.
# Methods:
#   - delay0_sac: delay-free SAC reference policy (Ret_df)
#   - random: random policy baseline (Ret_rand)
#
# Loop order: method -> seed -> env
#   - method: delay0_sac, random (2 values) - outermost loop
#   - seed: 0, 1, 2 (3 values)
#   - env: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4 (4 values) - innermost loop
# Total combinations: 2 * 3 * 4 = 24

#SBATCH --job-name=vdpo-baselines
#SBATCH --array=0-23%8
#SBATCH --gres=gpu:1
#SBATCH --exclude=al-l40s-0.grasp.maas
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=output/slurm_logs/%x/%A/%a.out
#SBATCH --error=output/slurm_logs/%x/%A/%a.err

export PYTHONUNBUFFERED=1

CONDA_ENV="VDPO"
METHODS=("delay0_sac" "random")
SEEDS=(0 1 2)
ENVS=("HalfCheetah-v4" "Hopper-v4" "Ant-v4" "Walker2d-v4")
TOTAL_TIMESTEPS=1000000

TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_METHODS=${#METHODS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_ENVS=${#ENVS[@]}

METHOD_IDX=$((TASK_ID / (NUM_SEEDS * NUM_ENVS)))
SEED_IDX=$(((TASK_ID % (NUM_SEEDS * NUM_ENVS)) / NUM_ENVS))
ENV_IDX=$((TASK_ID % NUM_ENVS))

METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}
ENV=${ENVS[$ENV_IDX]}

source ~/.bashrc
conda activate ${CONDA_ENV}

echo "Starting baseline run"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Method: $METHOD (idx: $METHOD_IDX)"
echo "Seed: $SEED (idx: $SEED_IDX)"
echo "Env: $ENV (idx: $ENV_IDX)"
echo "Total Timesteps: $TOTAL_TIMESTEPS"
echo ""

python3 run_baselines.py \
  --method=${METHOD} \
  --env=${ENV} \
  --seed=${SEED} \
  --total_timesteps=${TOTAL_TIMESTEPS}

