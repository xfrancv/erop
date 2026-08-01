#!/bin/bash
#SBATCH --job-name=erop   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=your@email   # Where to send mail
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # request one GPU (the gpu partition rejects jobs with none)


# Base-predictor training run_base_pred_training.shfor ONE dataset.
#
#   ./run_base_pred_training.sh <dataset> [nocalib]
#   sbatch -J erop-cifar100 run_base_pred_training.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to train every
# dataset in turn.
#
# [nocalib] is optional and selects the uncalibrated variant, i.e. training
# *without* the BCTS calibration:
#
#   (none)    output: runs/<dataset>/           --calibration bcts
#   nocalib   output: runs/<dataset>_nocalib/   (no --calibration flag)
#
# run_rejopt_eval.sh's "nocalib" mode reads runs/<dataset>_nocalib/ for its
# no-calibration ablation.

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100 organamnist organsmnist tissuemnist)

EPOCHS=30
DEVICE=cuda

usage() {
    echo "usage: $0 <dataset> [nocalib]" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    echo "  [nocalib]: omit for the default (BCTS-calibrated) run" >&2
    exit 1
}

{ [ $# -eq 1 ] || [ $# -eq 2 ]; } || usage

DIR_SUFFIX=""
EXTRA_ARGS=(--calibration bcts)
case "${2:-}" in
    "")       ;;                                   # default run
    nocalib)  DIR_SUFFIX="_nocalib"; EXTRA_ARGS=() ;;
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
    python base_predictor_training.py "$ds" "$dir" \
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
