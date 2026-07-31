#!/bin/bash
# Summary table across datasets, built from the LATEST run of each.
#
#   ./summary_tables.sh                 # writes summary_table.txt
#   ./summary_tables.sh --sizes 1 10    # extra args are passed to summary_table.py
#
# Each dataset's output directory runs/<dataset>/ holds one timestamped
# subdirectory per run (YYYYmmdd_HHMMSS); this script picks the most recent one
# that actually contains a sweep report. Subdirectories without a report
# directly inside them (e.g. runs/<dataset>/beta/, which nests its own
# timestamps) are skipped -- see summary_table_ablation.sh for those.

set -u

DATASETS=(bloodmnist cifar10 cifar100 dermamnist fashion_mnist organamnist organsmnist tissuemnist)
REPORT=real_reject_option_sweep_report.txt

# latest_report <run dir> -- path of the newest <dir>/<timestamp>/$REPORT, if any.
# The timestamps are zero-padded and fixed-width, so the lexicographic order of
# the glob is chronological.
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
    r=$(latest_report "runs/$ds")
    if [ -n "$r" ]; then
        reports+=("$r")
        echo "$ds: $r" >&2
    else
        missing+=("$ds")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "error: no $REPORT found for: ${missing[*]}" >&2
    echo "       run ./run_rejopt_eval.sh <dataset> first" >&2
    exit 1
fi

python summary_table.py --reports "${reports[@]}" \
    --size 1 10 100 \
    --precision 3 \
    --output summary_table.txt \
    "$@"
