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
#   ./run_all_real_exp.sh <dataset> [mode]
#   sbatch -J erop-cifar100 run_all_real_exp.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to run every
# dataset in turn (the script's previous behaviour).
#
# [mode] is optional and selects which base predictor / output directory to use
# (both are the same directory) and whether the misspecified model prior is
# used:
#
#   (none)    model & output: runs/<dataset>/           (default)
#   noadapt   model & output: runs/<dataset>_noadapt/   (e.g. uncalibrated base)
#   beta      model & output: runs/<dataset>/           + --beta $BETA_SUM
#
# Each chosen directory needs its base predictor at <dir>/model.pt
# (run_base_predictor_exp.py).

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100 octmnist organamnist tissuemnist)

# Symmetric-Dirichlet model-prior concentration for the "beta" mode.
BETA_SUM=20

usage() {
    echo "usage: $0 <dataset> [mode]" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    echo "  [mode]   : noadapt | beta   (omit for the default run)" >&2
    exit 1
}

{ [ $# -eq 1 ] || [ $# -eq 2 ]; } || usage

# The second argument is a mode keyword; it maps to a directory suffix (used for
# both the model input and the output) and to extra args for the Python script.
DIR_SUFFIX=""
EXTRA_ARGS=()
case "${2:-}" in
    "")       ;;                                   # default run
    noadapt)  DIR_SUFFIX="_noadapt" ;;
    beta)     EXTRA_ARGS=(--beta "$BETA_SUM") ;;
    *)        echo "error: unknown mode '$2'" >&2; usage ;;
esac

source .venv/bin/activate

#
REGRET_TARGETS="0.0001 0.001 0.01"

# run <dataset> <extra args...> -- one sweep of the reject-option experiment.
run() {
    local ds=$1; shift
    local dir="runs/${ds}${DIR_SUFFIX}"
    local model="$dir/model.pt"
    if [ ! -f "$model" ]; then
        echo "error: $model not found; train it first with" >&2
        echo "       python run_base_predictor_exp.py $ds $dir" >&2
        return 1
    fi
    echo "=== ${ds}${DIR_SUFFIX}${EXTRA_ARGS:+ (${EXTRA_ARGS[*]})}: $* ==="
    mkdir -p "$dir"
    # "${EXTRA_ARGS[@]+...}" guards against the empty-array-under-set-u error on
    # bash < 4.4 (older cluster nodes).
    python run_real_reject_option_exp.py "$model" "$dir/" --sweep \
        --sizes $SIZES --regret-target $REGRET_TARGETS \
        "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}" "$@"
}


run_bloodmnist() {
    SIZES="1 2 5 10 50 100 200 500"

     #
#     run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 20 --trials-prior 10
     run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 20 --trials-prior 10 --percentile-band 50

}


run_cifar10() {
    SIZES="1 2 5 10 50 100 200 500"

#      run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 20  --trials-prior 10 
      run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 20  --trials-prior 10 --percentile-band 50

}


run_dermamnist() {
    SIZES="1 2 5 10 50 100 200"

#    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 20  --n-eval 200
    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 20  --n-eval 200 --percentile-band 50

}

run_fashion_mnist() {
    SIZES="1 2 5 10 50 100 200 500"

#    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20  --trials-prior 10
    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20  --trials-prior 10  --percentile-band 50
}

run_octmnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run octmnist --test-prior 0.25 0.25 0.25 0.25   --dirichlet 20 --percentile-band 50
}

run_organamnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run organamnist --test-prior 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.09 0.1    --dirichlet 20 --percentile-band 50
}

run_tissuemnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run tissuemnist --test-prior 0.2 0.1 0.1 0.1 0.1 0.1 0.2 0.1   --dirichlet 20 --percentile-band 50
}


run_cifar100() {
    SIZES="1 2 5 10 50 100 200 500"

     run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 50   --trials-prior 10  --percentile-band 50
    
}


case "$1" in
    bloodmnist|cifar10|dermamnist|fashion_mnist|cifar100|tissuemnist|organamnist|octmnist)
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
