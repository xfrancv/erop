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
STYLEGAN_DIR="$ROOT_DIR/modules/stylegan3"
DATA_DIR="$ROOT_DIR/data"
NETWORK="$STYLEGAN_DIR/stylegan3-t-ffhq-1024x1024.pkl"

if [[ ! -f "$NETWORK" ]]; then
  echo "StyleGAN3 network not found. Downloading..."

  wget \
    --content-disposition \
    'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-ffhq-1024x1024.pkl' \
    --output-document "$NETWORK"
fi

# --------------------
# Environment
# --------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STYLEGAN_ENV"

ml CUDA/12.4.0
ml GCC/10

# Copy stylegan image generation script that processes the sampled latents in batches
cp "$ROOT_DIR/modules/stylegan3_extra/gen_images_batched.py" "$ROOT_DIR/modules/stylegan3/"

cd "$STYLEGAN_DIR"

# --------------------
# Generate images
# --------------------
python gen_images_batched.py \
  --outdir="$DATA_DIR/train" \
  --trunc=1 \
  --seeds=381572-500000 \
  --network="$NETWORK"

python gen_images_batched.py \
  --outdir="$DATA_DIR/test" \
  --trunc=1 \
  --seeds=500001-600000 \
  --network="$NETWORK"
