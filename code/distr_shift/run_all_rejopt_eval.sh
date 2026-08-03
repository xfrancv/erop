#!/bin/bash
# Run all real-data experiments.
#
#   ./run_rejopt_eval.sh           # run sequentially in this shell
#   ./run_rejopt_eval.sh sbatch    # submit each run as a Slurm job

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
        sbatch -J "$job" ./run_rejopt_eval.sh "$@"
    else
        ./run_rejopt_eval.sh "$@"
    fi
}

run erop-blood   bloodmnist
run erop-blood   bloodmnist nocalib
run erop-blood   bloodmnist beta
run erop-blood   bloodmnist transductive

run erop-cif10   cifar10
run erop-cif10   cifar10 nocalib
run erop-cif10   cifar10 beta
run erop-cif10   cifar10 transductive

run erop-cif100  cifar100
run erop-cif100  cifar100 nocalib
run erop-cif100  cifar100 beta
run erop-cif100  cifar100 transductive

run erop-derma   dermamnist
run erop-derma   dermamnist nocalib
run erop-derma   dermamnist beta
run erop-derma   dermamnist transductive

run erop-fashion fashion_mnist
run erop-fashion fashion_mnist nocalib
run erop-fashion fashion_mnist beta
run erop-fashion fashion_mnist transductive

run erop-tissue  tissuemnist
run erop-tissue  tissuemnist nocalib
run erop-tissue  tissuemnist beta
run erop-tissue  tissuemnist transductive

run erop-organa  organamnist
run erop-organa  organamnist nocalib
run erop-organa  organamnist beta
run erop-organa  organamnist transductive

run erop-organa  organsmnist
run erop-organa  organsmnist nocalib
run erop-organa  organsmnist beta
run erop-organa  organsmnist transductive

