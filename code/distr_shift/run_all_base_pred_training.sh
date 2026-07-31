#!/bin/bash
# Run all real-data experiments.
#
#   ./run_all_base_pred_training.sh           # run sequentially in this shell
#   ./run_all_base_pred_training.sh sbatch    # submit each run as a Slurm job

set -e

USE_SBATCH=0
case "${1:-}" in
    "") ;;
    sbatch) USE_SBATCH=1 ;;
    *) echo "usage: $0 [sbatch]" >&2; exit 1 ;;
esac

# run <job-name> <dataset> [mode]
run() {
    local job="$1"; shift
    if [ "$USE_SBATCH" -eq 1 ]; then
        sbatch -J "$job" ./run_base_pred_training.sh "$@"
    else
        ./run_base_pred_training.sh "$@"
    fi
}

run erop-blood   bloodmnist
run erop-blood   bloodmnist nocalib

run erop-cif10   cifar10
run erop-cif10   cifar10 nocalib

run erop-cif100  cifar100
run erop-cif100  cifar100 nocalib

run erop-derma   dermamnist
run erop-derma   dermamnist nocalib

run erop-fashion fashion_mnist
run erop-fashion fashion_mnist nocalib

run erop-tissue  tissuemnist
run erop-tissue  tissuemnist nocalib

run erop-organa  organamnist
run erop-organa  organamnist nocalib

run erop-organa  organsmnist
run erop-organa  organsmnist nocalib
