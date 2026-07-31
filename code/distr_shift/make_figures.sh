#!/bin/bash
# Refresh the paper's RC-curve figures from the LATEST run of each dataset, then
# re-render them as styled PDFs.
#
#   ./make_figures.sh
#
# Two phases:
#
#   1. Copy the figure data of the newest run,
#        runs/<dataset>/<timestamp>/{aurc,aurec}_vs_n_test.{figspec.npz,png}
#      into
#        figures/rc-curves/<dataset>/
#      The *.figspec.json files are deliberately NOT copied: those carry the
#      hand-tuned presentation (labels, titles, colours, legend placement) and
#      the ones already in figures/ are kept.
#
#   2. Re-render each retained *.figspec.json with the paper style sheet, which
#      picks up the freshly copied *.figspec.npz data.

set -u

source .venv/bin/activate

DATASETS=(bloodmnist cifar10 cifar100 dermamnist fashion_mnist organamnist organsmnist tissuemnist)
FIGURES=(aurc_vs_n_test aurec_vs_n_test)
OUT_ROOT=figures/rc-curves
STYLE=./styles/paper.mplstyle

# latest_run <run dir> -- newest <dir>/<timestamp>/ holding the wanted figures.
# The timestamps are zero-padded and fixed-width, so the lexicographic order of
# the glob is chronological.
latest_run() {
    local dir=$1 d newest=""
    for d in "$dir"/*/; do
        [ -f "$d${FIGURES[0]}.figspec.npz" ] && newest=$d
    done
    [ -n "$newest" ] && printf '%s\n' "$newest"
}

# --- phase 1: copy the figure data of the latest run ----------------------
missing=()
for ds in "${DATASETS[@]}"; do
    src=$(latest_run "runs/$ds")
    if [ -z "$src" ]; then
        missing+=("runs/$ds")
        continue
    fi
    dst="$OUT_ROOT/$ds"
    mkdir -p "$dst"
    echo "$ds: $src -> $dst/"
    for fig in "${FIGURES[@]}"; do
        for ext in figspec.npz png; do
            cp "$src$fig.$ext" "$dst/$fig.$ext"
        done
        if [ ! -f "$dst/$fig.figspec.json" ]; then
            missing+=("$dst/$fig.figspec.json")
        fi
    done
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "error: missing (no run with figures, or no retained figspec json):" >&2
    printf '       %s\n' "${missing[@]}" >&2
    exit 1
fi

# --- phase 2: re-render the retained figspecs -----------------------------
for ds in "${DATASETS[@]}"; do
    for fig in "${FIGURES[@]}"; do
        python render_figspecs.py "$OUT_ROOT/$ds/$fig.figspec.json" \
            --style "$STYLE" --format pdf
    done
done
