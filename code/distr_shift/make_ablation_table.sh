#!/bin/bash
# Ablation summary table: for every dataset the LATEST run of each of the three
# configurations -- default, misspecified model prior (beta), and uncalibrated
# base predictor (nocalib).
#
#   ./summary_table_ablation.sh                # writes summary_table_ablation.txt
#   ./summary_table_ablation.sh --sizes 1 10   # extra args go to summary_table.py
#
# The three configurations live in
#   runs/<dataset>/<timestamp>/           (default)
#   runs/<dataset>/beta/<timestamp>/      (--beta, misspecified model prior)
#   runs/<dataset>_nocalib/<timestamp>/   (base predictor without BCTS)
# each holding one timestamped subdirectory per run; the most recent one that
# actually contains a sweep report is used.

set -u

DATASETS=(bloodmnist cifar10 cifar100 dermamnist fashion_mnist organamnist organsmnist tissuemnist)
REPORT=real_reject_option_sweep_report.txt

# latest_report <run dir> -- path of the newest <dir>/<timestamp>/$REPORT, if any.
# The timestamps are zero-padded and fixed-width, so the lexicographic order of
# the glob is chronological. Directories that nest their own timestamps one
# level deeper (runs/<dataset>/beta/) hold no report themselves and are skipped.
latest_report() {
    local dir=$1 d newest=""
    for d in "$dir"/*/; do
        [ -f "$d$REPORT" ] && newest=$d
    done
    [ -n "$newest" ] && printf '%s\n' "$newest$REPORT"
}

reports=()
missing=()
for ds in "${DATASETS[@]}"; do
    for dir in "runs/$ds" "runs/$ds/beta" "runs/${ds}_nocalib"; do
        r=$(latest_report "$dir")
        if [ -n "$r" ]; then
            reports+=("$r")
            echo "$dir: $r" >&2
        else
            missing+=("$dir")
        fi
    done
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "error: no $REPORT found under: ${missing[*]}" >&2
    echo "       run ./run_rejopt_eval.sh <dataset> [beta | nocalib] first" >&2
    exit 1
fi

python summary_table.py --reports "${reports[@]}" \
    --size 100 \
    --precision 3 \
    --output summary_table_ablation.txt \
    "$@"
