#!/bin/bash
# Run all real-data experiments.
#
#   ./run_all_real_exp.sh           # run sequentially in this shell
#   ./run_all_real_exp.sh sbatch    # submit each run as a Slurm job

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
        sbatch -J "$job" ./run_real_exp.sh "$@"
    else
        ./run_real_exp.sh "$@"
    fi
}

run erop-blood   bloodmnist
run erop-blood   bloodmnist noadapt
run erop-blood   bloodmnist beta

run erop-cif10   cifar10
run erop-cif10   cifar10 noadapt
run erop-cif10   cifar10 beta

run erop-cif100  cifar100
run erop-cif100  cifar100 noadapt
run erop-cif100  cifar100 beta

run erop-derma   dermamnist
run erop-derma   dermamnist noadapt
run erop-derma   dermamnist beta

run erop-fashion fashion_mnist
run erop-fashion fashion_mnist noadapt
run erop-fashion fashion_mnist beta

run erop-tissue  tissuemnist
run erop-tissue  tissuemnist noadapt
run erop-tissue  tissuemnist beta

run erop-organa  organamnist
run erop-organa  organamnist noadapt
run erop-organa  organamnist beta

run erop-organa  organsmnist
run erop-organa  organsmnist noadapt
run erop-organa  organsmnist beta
