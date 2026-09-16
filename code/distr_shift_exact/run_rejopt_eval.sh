#!/usr/bin/env bash
# Run the reject-option evaluation for one dataset (or "all").
#
#   ./run_rejopt_eval.sh fashion_mnist        # main + supplementary strata
#   ./run_rejopt_eval.sh all
#   ./run_rejopt_eval.sh cifar10 nocalib      # against the uncalibrated model
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(fashion_mnist cifar10 cifar100 dermamnist bloodmnist tissuemnist
          organamnist organsmnist)

usage() { echo "usage: $0 <dataset|all> [nocalib]" >&2; exit 2; }
[[ $# -ge 1 ]] || usage

target=$1
mode=${2:-main}

run_one() {
  local ds=$1
  case "$mode" in
    main)    python rejopt_eval.py "runs/$ds" ;;
    nocalib) python rejopt_eval.py "runs/${ds}_nocalib" ;;
    *)       usage ;;
  esac
}

if [[ "$target" == all ]]; then
  for ds in "${DATASETS[@]}"; do run_one "$ds"; done
else
  run_one "$target"
fi
