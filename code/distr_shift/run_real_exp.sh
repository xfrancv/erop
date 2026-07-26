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
#   ./run_all_real_exp.sh <dataset> [suffix]
#   sbatch -J erop-cifar100 run_all_real_exp.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to run every
# dataset in turn (the script's previous behaviour). Each dataset needs its
# base predictor at runs/<dataset>/model.pt (run_base_predictor_exp.py).
#
# [suffix] is optional: when given, it is appended (with a leading underscore)
# to the dataset name for the OUTPUT directory only, so results land in
# runs/<dataset>_<suffix>/ instead of runs/<dataset>/ while still reusing the
# base predictor at runs/<dataset>/model.pt. E.g.
#
#   ./run_all_real_exp.sh fashion_mnist no_adapt   -> runs/fashion_mnist_no_adapt/

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100)

usage() {
    echo "usage: $0 <dataset> [suffix]" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    echo "  [suffix] : optional output-dir suffix (runs/<dataset>_<suffix>/)" >&2
    exit 1
}

{ [ $# -eq 1 ] || [ $# -eq 2 ]; } || usage

# Optional output-dir suffix from the second argument (empty if not given).
OUT_SUFFIX=""
[ $# -eq 2 ] && OUT_SUFFIX="_$2"

source .venv/bin/activate

# 
REGRET_TARGETS="0.0001 0.001 0.01"

# run <dataset> <extra args...> -- one sweep of the reject-option experiment.
run() {
    local ds=$1; shift
    local model="runs/$ds/model.pt"
    local outdir="runs/${ds}${OUT_SUFFIX}/"
    if [ ! -f "$model" ]; then
        echo "error: $model not found; train it first with" >&2
        echo "       python run_base_predictor_exp.py $ds runs/$ds" >&2
        return 1
    fi
    echo "=== $ds${OUT_SUFFIX:+ [$OUT_SUFFIX]}: $* ==="
    mkdir -p "$outdir"
    python run_real_reject_option_exp.py "$model" "$outdir" --sweep \
        --sizes $SIZES --regret-target $REGRET_TARGETS "$@"
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
