#!/bin/bash
#SBATCH --job-name=farl_gp
#SBATCH --array=0-214%50
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/farl_gp_%a.log
#SBATCH --time=00:45:00

source ~/.bashrc
ml Python
mkdir -p logs

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
BACKBONE="farl"
ESTIMATOR="gp_rbf"  # Options: gp_linear, gp_rbf, blr
CKPT_PATH="/mnt/data/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"

FRACTIONS=(0.001 0.0012 0.0015 0.0018 0.002 0.0025 0.003 0.0035 0.004 0.005 0.006 0.0075 0.008 0.009 0.01 0.012 0.015 0.018 0.02 0.025 0.03 0.035 0.04 0.05 0.06 0.075 0.08 0.09 0.1 0.12 0.15)
# 0.18 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.75 0.8 0.9 1.0)

# ---------------------------------------------------------------------
# INDEX CALCULATION
# ---------------------------------------------------------------------
NUM_FRACS=${#FRACTIONS[@]}
ID=$SLURM_ARRAY_TASK_ID

FRAC_IDX=$((ID % NUM_FRACS))
FOLD=$((ID / NUM_FRACS))
FRAC=${FRACTIONS[$FRAC_IDX]}

echo "Running FaRL | Fold $FOLD | Frac $FRAC"

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
    #--use_oracle_noise