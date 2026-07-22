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


    # base-acc: v   aurc: v  cov@reg: x
#    run bloodmnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 20
    # base-acc: v  aurc: (v)  cov@reg: (v)
#    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 20
    # base-acc: x  aurc: x  cov@reg: x
#    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 100
    # base-acc: (v)  aurc: v  cov@reg: x
#    run bloodmnist --prior-classes 3 6 0 --prior-weights 1 1 1 --prior-rest-weight 1 --dirichlet 100
    # base-acc: x  aurc: v  cov@reg:  x
#    run bloodmnist --prior-classes 3 6 0 --prior-weights 2 2 3 --prior-rest-weight 1 --dirichlet 100
    # base-acc: x  aurc: v  cov@reg: x
#    run bloodmnist --prior-classes 3 6 0 --prior-weights 2 2 4 --prior-rest-weight 1 --dirichlet 100
    # base-acc: x  aurc: x  cov@reg: x
#    run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 100
    # base-acc: x  aurc: x  cov@reg: x
#    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 100
    # base-acc: (v)  aurc: v  cov@reg: x
#    run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 50
    # base-acc: (v) aurc: v  cov@reg: x
#    run bloodmnist --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 50 


     # base-acc: (v)  aurc: (v)  cov@reg: (v)
    run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 50 --trials-prior 10

     # base-acc: v  aurc: v  cov@reg: (v)
    run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 20 --trials-prior 10

     # base-acc: v  aurc: v  cov@reg: (v)
    run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 10 --trials-prior 10

}


run_cifar10() {
    SIZES="$SIZES_DEFAULT"


     # base-acc:  (v)  aurc: v  cov@reg: v
#     run cifar10 --prior-classes 3 5 2 --prior-weights 1 1 1 --prior-rest-weight 1 --dirichlet 100

    # base-acc: (v) aurc: (v)  cov@reg: v
#     run cifar10 --prior-classes 3 5 2 --prior-weights 2 2 3 --prior-rest-weight 1 --dirichlet 100

    # base-acc: (v) aurc: (v)  cov@reg: v
#     run cifar10 --test-prior 0.01 0.43 0.01 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 100

    # base-acc: v aurc: v  cov@reg: v
#     run cifar10 --test-prior 0.01 0.43 0.01 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 50



     # base-acc: v  aurc: v  cov@reg: v
     run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 50  --trials-prior 10

     # base-acc: v  aurc: v  cov@reg: v
     run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 20  --trials-prior 10

     # base-acc: v  aurc: v  cov@reg: v
     run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 10  --trials-prior 10
}


run_dermamnist() {
    SIZES="1 2 5 10 50 100"

#    run dermamnist --pair-ratio 1 1 --pair-rest-ratio 1 1
#    run dermamnist --test-prior 0.01 0.01 0.46 0.01 0.25 0.25 0.01
#    run dermamnist --prior-classes 4 5 2 --prior-weights 1 1 1 --prior-rest-weight 1 --dirichlet 100
    # base-acc: x  aurc:   cov@reg: 
#    run dermamnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 100
    # base-acc: (v)  aurc: (v)  cov@reg: v
#    run dermamnist --prior-classes 4 5 2 --prior-weights 2 2 3 --prior-rest-weight 1 --dirichlet 100

    # base-acc: x  aurc: x  cov@reg: v
#    run dermamnist --test-prior 0.01 0.01 0.46 0.01 0.25 0.25 0.01 --dirichlet 100

#    run dermamnist --prior-classes 4 5 2 --prior-weights 2 2 3 --prior-rest-weight 1 --dirichlet 100 --n-eval 200
#    run dermamnist --test-prior 0.01 0.24 0.23 0.01 0.25 0.25 0.01 --dirichlet 50  --n-eval 200
#    run dermamnist --test-prior 0.01 0.24 0.23 0.01 0.25 0.25 0.01 --dirichlet 20  --n-eval 200
#    run dermamnist --test-prior 0.01 0.24 0.23 0.01 0.25 0.25 0.01 --dirichlet 10  --n-eval 200

#
    # base-acc: v  aurc: v  cov@reg: v
    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 50  --n-eval 200

    # base-acc: v  aurc: (v)  cov@reg: v
    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 10  --n-eval 200

    # base-acc: v  aurc: v  cov@reg: v
    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 20  --n-eval 200


}

run_fashion_mnist() {
    SIZES="$SIZES_DEFAULT"

#    run fashion_mnist --pair-ratio 1 1 --pair-rest-ratio 1 1
#    run fashion_mnist --pair-ratio 1 1 --pair-rest-ratio 1 1 --dirichlet 20
#    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01
#    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20
#    run fashion_mnist --prior-classes 0 6 2 --prior-weights 1 1 1 --prior-rest-weight 1 --dirichlet 100
#    run fashion_mnist --prior-classes 0 6 2 --prior-weights 2 2 3 --prior-rest-weight 1 --dirichlet 100
    # base-acc: v  aurc:  v  cov@reg: v
#    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 50
    # base-acc: x  aurc: v  cov@reg: v
#    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 100


    # base-acc: v  aurc: v  cov@reg: v
    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 50   --trials-prior 10
    
    # base-acc: v  aurc: v  cov@reg: v
    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20   --trials-prior 10
    
    # base-acc: v  aurc: v  cov@reg: v
    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 10   --trials-prior 10

}

run_retinamnist() {
    SIZES="1 2 5 10 50"

    run retinamnist --prior-classes 1 2 0 --prior-weights 1 1 1 --prior-rest-weight 1 --dirichlet 100 --n-eval 100

}


run_cifar100() {
    SIZES="$SIZES_DEFAULT"

#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 3
#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 3 --dirichlet 100
#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 3 --dirichlet 500
#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 5
#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 5 --dirichlet 100
#    run cifar100 --pair-ratio 1 1 --pair-rest-ratio 1 5 --dirichlet 500
#    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 1 --dirichlet 100
    # base-acc: v  aurc: v  cov@reg: v
#    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 100
    # base-acc: v  aurc: v  cov@reg: v
#    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 50


     # base-acc: v  aurc: v  cov@reg: v
#    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 50   --trials-prior 10

     # base-acc:   aurc:   cov@reg: 
    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 20   --trials-prior 10

     # base-acc:   aurc:   cov@reg: 
    run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 10   --trials-prior 10
    
}


case "$1" in
    bloodmnist|cifar10|dermamnist|fashion_mnist|cifar100|retinamnist)
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
