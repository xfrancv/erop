#!/bin/bash
#SBATCH --job-name=resnet_blr
#SBATCH --array=0-214%50
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/resnet_blr_%a.log
#SBATCH --time=00:30:00

# ---------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------
source ~/.bashrc
# conda activate my_env
ml Python

mkdir -p logs

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BACKBONE="resnet"
ESTIMATOR="gp_linear"
CKPT_PATH="src/age_resnet50_regression.pth"

# Fractions (43 total)
FRACTIONS=(0.001 0.0012 0.0015 0.0018 0.002 0.0025 0.003 0.0035 0.004 0.005 0.006 0.0075 0.008 0.009 0.01 0.012 0.015 0.018 0.02 0.025 0.03 0.035 0.04 0.05 0.06 0.075 0.08 0.09 0.1 0.12 0.15 0.18 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.75 0.8 0.9 1.0)

# ---------------------------------------------------------------------
# INDEX CALCULATION
# ---------------------------------------------------------------------
NUM_FRACS=${#FRACTIONS[@]} # 43

# The ID runs from 0 to 214
ID=$SLURM_ARRAY_TASK_ID

# Calculate Indices
# Inner loop: Fraction (changes fastest)
# Outer loop: Fold
FRAC_IDX=$((ID % NUM_FRACS))
FOLD=$((ID / NUM_FRACS))

FRAC=${FRACTIONS[$FRAC_IDX]}

echo "--- Job Info ---"
echo "Backbone: $BACKBONE"
echo "Estimator: $ESTIMATOR"
echo "Fold: $FOLD"
echo "Fraction: $FRAC"
echo "----------------"

# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
python -m src.train \
    --data_dir "data" \
    --ckpt_path "$CKPT_PATH" \
    --backbone_name "$BACKBONE" \
    --estimator "$ESTIMATOR" \
    --fold "$FOLD" \
    --fraction "$FRAC" \
    --output_dir "results" \
    --use_oracle_noise