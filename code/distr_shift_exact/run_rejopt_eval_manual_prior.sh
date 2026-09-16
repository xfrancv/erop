#!/usr/bin/env bash
# Run the reject-option evaluation against the hand-built prior set for one
# dataset (or "all"). Theta comes from runs/<ds>/priors_manual.txt, which
# ./create_prior.sh writes; nothing in model.pt or eval_log_post.npz depends
# on it, so no retraining is involved.
#
#   ./run_rejopt_eval_manual_prior.sh                 # every dataset
#   ./run_rejopt_eval_manual_prior.sh organsmnist
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(fashion_mnist cifar10 cifar100 dermamnist bloodmnist tissuemnist
          organamnist organsmnist)

COVERAGE=0.8

usage() { echo "usage: $0 [dataset|all]" >&2; exit 2; }
[[ $# -le 1 ]] || usage

target=${1:-all}

# create_prior.py emits C = Y + 1 priors, and the S6.4 breakdown runs one
# stratum per element, so the sweep costs C + 2 strata each of which is itself
# O(C * Y) per triplet. On cifar100 that is 103 strata at ~80 s apiece -- ~2.4 h,
# against 2 min for the Y = 10 datasets. The breakdown is also unreadable there
# (101 near-identical panels, a 101-entry legend), so drop it and keep the main
# 'shifted' stratum, which is the one S6.5 actually plots.
extra_args() {
  case "$1" in
    cifar100) echo "--no-supplementary" ;;
    *)        echo "" ;;
  esac
}

run_one() {
  local ds=$1
  local extra
  read -r -a extra <<< "$(extra_args "$ds")"
  echo "=== $ds (coverage = $COVERAGE) ${extra[*]:-}"
  python rejopt_eval.py "runs/$ds/" \
      --priors "runs/$ds/priors_manual.txt" \
      --out-dir "runs/$ds/rejopt_manual" \
      --coverage "$COVERAGE" \
      ${extra[@]+"${extra[@]}"}
}

if [[ "$target" == all ]]; then
  for ds in "${DATASETS[@]}"; do run_one "$ds"; done
else
  run_one "$target"
fi
