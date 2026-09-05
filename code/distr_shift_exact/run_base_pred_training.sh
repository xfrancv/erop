#!/usr/bin/env bash
# Train the base predictor for one dataset (or "all"), in the paper configuration.
#
#   ./run_base_pred_training.sh fashion_mnist          # default: BCTS
#   ./run_base_pred_training.sh all
#   ./run_base_pred_training.sh cifar10 nocalib        # the miscalibration ablation
#   ./run_base_pred_training.sh cifar10 ts             # scalar-temperature ablation
#
# The ablations write runs/<ds>_nocalib/ and runs/<ds>_ts/ so they never
# overwrite the run of record. README S6.1 makes BCTS the default for a reason:
# the whole re-weighting theta_y / p_tr(y) inherits the base model's
# calibration error, and a per-class error there does not cancel.
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(fashion_mnist cifar10 cifar100 dermamnist bloodmnist tissuemnist
          organamnist organsmnist)
EPOCHS=${EPOCHS:-30}

usage() { echo "usage: $0 <dataset|all> [nocalib|ts]" >&2; exit 2; }
[[ $# -ge 1 ]] || usage

target=$1
mode=${2:-calib}

run_one() {
  local ds=$1
  case "$mode" in
    calib)   python base_predictor_training.py "$ds" "runs/$ds" \
               --epochs "$EPOCHS" --calibration bcts ;;
    nocalib) python base_predictor_training.py "$ds" "runs/${ds}_nocalib" \
               --epochs "$EPOCHS" --calibration none ;;
    ts)      python base_predictor_training.py "$ds" "runs/${ds}_ts" \
               --epochs "$EPOCHS" --calibration temperature ;;
    *)       usage ;;
  esac
}

if [[ "$target" == all ]]; then
  for ds in "${DATASETS[@]}"; do run_one "$ds"; done
else
  run_one "$target"
fi
