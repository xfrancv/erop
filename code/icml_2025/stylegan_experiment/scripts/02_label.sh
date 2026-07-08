#!/usr/bin/env bash
set -e

# --------------------
# User config
# --------------------
ROOT_DIR=/mnt/data/experiments/ICML_2025
STYLEGAN_ENV=stylegan3

# --------------------
# Paths
# --------------------
DATA_DIR="$ROOT_DIR/data"

# --------------------
# Environment
# --------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STYLEGAN_ENV"

ml Python

cd "$ROOT_DIR"

python src/create_oracle_predictions.py
python src/create_benchmark.py