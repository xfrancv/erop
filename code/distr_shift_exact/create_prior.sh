#!/usr/bin/env bash
# Build the sliding-pair prior set Theta for one dataset (or "all").
#
# Writes runs/<ds>/priors_manual.txt from runs/<ds>/priors.txt, which is what
# `rejopt_eval.py --priors` reads; see create_prior.py for the construction.
#
#   ./create_prior.sh                 # every dataset, tau = 0.35
#   ./create_prior.sh organsmnist
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(fashion_mnist cifar10 cifar100 dermamnist bloodmnist tissuemnist
          organamnist organsmnist)

TAU=0.35

usage() { echo "usage: $0 [dataset|all]" >&2; exit 2; }
[[ $# -le 1 ]] || usage

target=${1:-all}

run_one() {
  local ds=$1
  echo "=== $ds (tau = $TAU)"
  python create_prior.py "runs/$ds/priors.txt" "$TAU" \
      "runs/$ds/priors_manual.txt"
}

if [[ "$target" == all ]]; then
  for ds in "${DATASETS[@]}"; do run_one "$ds"; done
else
  run_one "$target"
fi
