#!/bin/bash
#SBATCH --job-name=erop   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xfrancv@fel.cvut.cz   # Where to send mail
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --partition=cpu

# Real-data reject-option experiments for ONE dataset.
#
#   ./run_all_real_exp.sh <dataset>
#   sbatch -J erop-cifar100 run_all_real_exp.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to run every
# dataset in turn (the script's previous behaviour). Each dataset needs its
# base predictor at runs/<dataset>/model.pt (run_base_predictor_exp.py).

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100)

usage() {
    echo "usage: $0 <dataset>" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    exit 1
}

[ $# -eq 1 ] || usage

source .venv/bin/activate

# Shared sweep arguments. CIFAR-100 has only 100 pool examples per class, so
# its adaptation sizes stop where the eval set would be starved.
REGRET_TARGETS="0.0005 0.001 0.005 0.01 0.05"
SIZES_DEFAULT="1 2 5 10 50 100 200 500"
SIZES_CIFAR100="1 2 5 10 50 100"

# run <dataset> <extra args...> -- one sweep of the reject-option experiment.
run() {
    local ds=$1; shift
    local model="runs/$ds/model.pt"
    if [ ! -f "$model" ]; then
        echo "error: $model not found; train it first with" >&2
        echo "       python run_base_predictor_exp.py $ds runs/$ds" >&2
        return 1
    fi
    echo "=== $ds: $* ==="
    python run_real_reject_option_exp.py "$model" "runs/$ds/" --sweep \
        --sizes $SIZES --regret-target $REGRET_TARGETS "$@"
}


run_bloodmnist() {
    SIZES="$SIZES_DEFAULT"

    run bloodmnist --pair-ratio 1 1 --pair-rest-ratio 1 1

    run bloodmnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 10

    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 

    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 10
}


run_cifar10() {
    SIZES="$SIZES_DEFAULT"

    run cifar10 --pair-ratio 1 1 --pair-rest-ratio 1 1

    run cifar10 --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 10

    run cifar10 --test-prior 0.43 0.01 0.01 0.25 0.01 0.25 0.01 0.01 0.01 0.01

    run cifar10 --test-prior 0.43 0.01 0.01 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 10
}


run_dermamnist() {
    SIZES="$SIZES_DEFAULT"

    run dermamnist --pair-ratio 1 1 --pair-rest-ratio 1 1

    run dermamnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 10

    run dermamnist --test-prior 0.01 0.01 0.46 0.01 0.25 0.25 0.01

    run dermamnist --test-prior 0.01 0.01 0.46 0.01 0.25 0.25 0.01 --dirichlet 10
}

run_fashion_mnist() {
    SIZES="$SIZES_DEFAULT"

    run fashion_mnist --pair-ratio 1 1 --pair-rest-ratio 1 1

    run fashion_mnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 10

    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01

    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 10
}


run_cifar100() {
    SIZES="$SIZES_CIFAR100"

    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 3

    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 3 --dirichlet 10
    
}


case "$1" in
    bloodmnist|cifar10|dermamnist|fashion_mnist|cifar100)
        "run_$1"
        ;;
    all)
        for ds in "${DATASETS[@]}"; do
            "run_$ds"
        done
        ;;
    *)
        echo "error: unknown dataset '$1'" >&2
        usage
        ;;
esac
