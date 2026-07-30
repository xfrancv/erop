#!/bin/bash
#SBATCH --job-name=erop   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xfrancv@fel.cvut.cz   # Where to send mail
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # request one GPU (the gpu partition rejects jobs with none)

# Base-predictor training for ONE dataset.
#
#   ./run_base_pred_training.sh <dataset> [noadapt]
#   sbatch -J erop-cifar100 run_base_pred_training.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to train every
# dataset in turn.
#
# [noadapt] is optional and selects the misspecified-model-prior variant, i.e.
# training *without* the BCTS calibration:
#
#   (none)    output: runs/<dataset>/           --calibration bcts
#   noadapt   output: runs/<dataset>_noadapt/   (no --calibration flag)

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100 organamnist organsmnist tissuemnist)

EPOCHS=30
DEVICE=cuda

usage() {
    echo "usage: $0 <dataset> [noadapt]" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    echo "  [noadapt]: omit for the default (BCTS-calibrated) run" >&2
    exit 1
}

{ [ $# -eq 1 ] || [ $# -eq 2 ]; } || usage

DIR_SUFFIX=""
EXTRA_ARGS=(--calibration bcts)
case "${2:-}" in
    "")       ;;                                   # default run
    noadapt)  DIR_SUFFIX="_noadapt"; EXTRA_ARGS=() ;;
    *)        echo "error: unknown mode '$2'" >&2; usage ;;
esac

source .venv/bin/activate

# run <dataset> -- train one base predictor.
run() {
    local ds=$1
    local dir="runs/${ds}${DIR_SUFFIX}"
    echo "=== ${ds}${DIR_SUFFIX}${EXTRA_ARGS:+ (${EXTRA_ARGS[*]})} ==="
    # "${EXTRA_ARGS[@]+...}" guards against the empty-array-under-set-u error on
    # bash < 4.4 (older cluster nodes).
    python run_base_predictor_exp.py "$ds" "$dir" \
        --epochs "$EPOCHS" --device "$DEVICE" \
        "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
}

case "$1" in
    bloodmnist|cifar10|dermamnist|fashion_mnist|cifar100|tissuemnist|organamnist|organsmnist)
        run "$1"
        ;;
    all)
        for ds in "${DATASETS[@]}"; do
            run "$ds"
        done
        ;;
    *)
        echo "error: unknown dataset '$1'" >&2
        usage
        ;;
esac
