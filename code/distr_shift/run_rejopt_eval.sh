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
#   ./run_rejopt_eval.sh <dataset> [mode]
#   sbatch -J erop-cifar100 run_rejopt_eval.sh cifar100
#
# <dataset> is one of the keys listed in DATASETS below, or "all" to run every
# dataset in turn.
#
# [mode] is optional and selects which base predictor / output directory to use
# (both are the same directory) and whether the misspecified model prior is
# used:
#
#   (none)    model: runs/<dataset>/model.pt          output: runs/<dataset>/
#   nocalib   model: runs/<dataset>_nocalib/model.pt  output: runs/<dataset>_nocalib/
#             (the uncalibrated base predictor, i.e. the "nocalib" mode of
#              run_base_pred_training.sh)
#   beta      model: runs/<dataset>/model.pt          output: runs/<dataset>/beta/
#             (and appends --beta $BETA_SUM to rejopt_eval.py)
#   transductive
#             model: runs/<dataset>/model.pt          output: runs/<dataset>/transductive/
#             (and appends --eval-on-adapt: adaptation and evaluation on the
#              same examples; the disjoint-split run is the default mode, so
#              the two are directly comparable)
#
# Every mode also carries its own sampling effort -- the number of sampled
# target priors and the number of trials at each -- as the <MODE>_TRIALS_PRIOR
# and <MODE>_TRIALS constants below, so one mode can be run cheaper or deeper
# than the rest. They all start at the paper's 20 x 20.
#
# The model directory needs its base predictor at <dir>/model.pt
# (base_predictor_training.py).

set -u

DATASETS=(bloodmnist cifar10 dermamnist fashion_mnist cifar100 organamnist organsmnist tissuemnist)

# Symmetric-Dirichlet model-prior concentration for the "beta" mode.
BETA_SUM=20

# Sampling effort, per mode: <MODE>_TRIALS_PRIOR is the number of sampled target
# priors (--trials-prior) and <MODE>_TRIALS the number of trials run at each of
# them (--trials), so one mode costs TRIALS_PRIOR x TRIALS sweeps per dataset.
# Each mode carries its own pair, so an expensive mode can be run at reduced
# effort without touching the others. All four are set to the 20 x 20 the paper
# runs use; the mode dispatch below copies the selected pair into
# TRIALS_PRIOR / TRIALS.
DEFAULT_TRIALS_PRIOR=20
DEFAULT_TRIALS=100

NOCALIB_TRIALS_PRIOR=20
NOCALIB_TRIALS=20

BETA_TRIALS_PRIOR=20
BETA_TRIALS=20

TRANSDUCTIVE_TRIALS_PRIOR=20
TRANSDUCTIVE_TRIALS=100

usage() {
    echo "usage: $0 <dataset> [mode]" >&2
    echo "  <dataset>: ${DATASETS[*]} | all" >&2
    echo "  [mode]   : nocalib | beta | transductive   (omit for the default run)" >&2
    exit 1
}

{ [ $# -eq 1 ] || [ $# -eq 2 ]; } || usage

# The second argument is a mode keyword. DIR_SUFFIX selects the model directory
# (runs/<dataset><DIR_SUFFIX>/); OUT_SUBDIR is an extra output-only subdirectory;
# EXTRA_ARGS are appended to the Python script; TRIALS_PRIOR / TRIALS take the
# mode's pair from the constants above.
DIR_SUFFIX=""
OUT_SUBDIR=""
EXTRA_ARGS=()
# dermamnist is the one dataset that pins a small disjoint evaluation set;
# --eval-on-adapt has no separate evaluation set and rejects --n-eval.
DERMA_NEVAL=(--n-eval 150)
case "${2:-}" in
    "")       TRIALS_PRIOR=$DEFAULT_TRIALS_PRIOR
              TRIALS=$DEFAULT_TRIALS ;;           # default run
    nocalib)  DIR_SUFFIX="_nocalib"
              TRIALS_PRIOR=$NOCALIB_TRIALS_PRIOR
              TRIALS=$NOCALIB_TRIALS ;;
    beta)     OUT_SUBDIR="beta"; EXTRA_ARGS=(--beta "$BETA_SUM")
              TRIALS_PRIOR=$BETA_TRIALS_PRIOR
              TRIALS=$BETA_TRIALS ;;
    transductive)
              OUT_SUBDIR="transductive"; EXTRA_ARGS=(--eval-on-adapt)
              DERMA_NEVAL=()
              TRIALS_PRIOR=$TRANSDUCTIVE_TRIALS_PRIOR
              TRIALS=$TRANSDUCTIVE_TRIALS ;;
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
    local outdir="$dir${OUT_SUBDIR:+/$OUT_SUBDIR}/"
    if [ ! -f "$model" ]; then
        echo "error: $model not found; train it first with" >&2
        echo "       python base_predictor_training.py $ds $dir" >&2
        return 1
    fi
    echo "=== ${ds}${DIR_SUFFIX}${OUT_SUBDIR:+/$OUT_SUBDIR}${EXTRA_ARGS:+ (${EXTRA_ARGS[*]})}" \
         "[${TRIALS_PRIOR} priors x ${TRIALS} trials]: $* ==="
    mkdir -p "$outdir"
    # "${EXTRA_ARGS[@]+...}" guards against the empty-array-under-set-u error on
    # bash < 4.4 (older cluster nodes).
    python rejopt_eval.py "$model" "$outdir" \
        --sizes $SIZES --regret-target $REGRET_TARGETS \
        "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}" "$@" \
        --trials-prior "$TRIALS_PRIOR" --trials "$TRIALS"
}


run_bloodmnist() {
    SIZES="1 2 5 10 50 100 200 500"

     #
     run bloodmnist --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 20 --percentile-band 50

}


run_cifar10() {
    SIZES="1 2 5 10 50 100 200 500"

      run cifar10 --test-prior 0.01 0.01 0.43 0.25 0.01 0.25 0.01 0.01 0.01 0.01 --dirichlet 20 --percentile-band 50 

}


run_dermamnist() {
    SIZES="1 2 5 10 50 100 200"

    run dermamnist --test-prior 0.1 0.1 0.1 0.1 0.25 0.25 0.1 --dirichlet 20 \
        "${DERMA_NEVAL[@]+${DERMA_NEVAL[@]}}" --percentile-band 50

}

run_fashion_mnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run fashion_mnist --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20  --percentile-band 50 
}

run_organamnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run organamnist --test-prior 0.07 0.07 0.07 0.07 0.2 0.2 0.07 0.04 0.07 0.07 0.07    --dirichlet 20 --percentile-band 50 
}


run_tissuemnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run tissuemnist --test-prior 0.2 0.1 0.1 0.1 0.1 0.1 0.2 0.1   --dirichlet 20 --percentile-band 50
}


run_cifar100() {
    SIZES="1 2 5 10 50 100 200 500"

     run cifar100 --prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5 --dirichlet 50 --percentile-band 50
}


run_organsmnist() {
    SIZES="1 2 5 10 50 100 200 500"

    run organsmnist --test-prior 0.07 0.07 0.07 0.07 0.2 0.2 0.07 0.04 0.07 0.07 0.07    --dirichlet 20 --percentile-band 50 
}


case "$1" in
    bloodmnist|cifar10|dermamnist|fashion_mnist|cifar100|tissuemnist|organamnist|organsmnist)
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
