#!/usr/bin/env bash
# Compute every C7 baseline and score it -- the pre-launch check that decides
# whether the competition has discoverable structure at all.
#
# The number that matters is bayes_total vs bayes_epistemic: the same base
# predictor, a different ranking. compare_baselines.py is what answers it,
# because it bootstraps both on the *same* resampled batches; two independently
# bootstrapped marginal intervals are the wrong test and a conservative one.
#
#   ./run_baselines.sh                        # out/kaggle
#   ./run_baselines.sh out/smoke_kaggle
set -euo pipefail
cd "$(dirname "$0")"

KAGGLE=${1:-out/kaggle}
SUBS=${2:-out/submissions}
PY=${PYTHON:-python}

$PY baseline_solutions.py "$KAGGLE" "$SUBS"

for usage in Public Private; do
  echo
  echo "################################################################"
  echo "# $usage -- paired comparison (this is the C8 criterion)"
  echo "################################################################"
  $PY compare_baselines.py "$KAGGLE/solution.csv" "$SUBS"/*.csv \
      --usage "$usage" --vs bayes_total
done

echo
echo "################################################################"
echo "# per-submission detail and figures (Private)"
echo "################################################################"
for f in "$SUBS"/*.csv; do
  $PY evaluate.py "$f" "$KAGGLE/solution.csv" --usage Private \
      --out-dir "$SUBS/Private"
done

cat <<'NOTE'

Read the output in this order:
  1. true_plugin must be exactly +0.000000. Anything else means generation and
     scoring have diverged; stop and fix that before reading anything else.
  2. The paired "diff vs ref" interval for bayes_epistemic must exclude 0.
     That is the C8 separation criterion.
  3. The last row of the per-size table is the gap between the top two
     contenders at each batch size. Sizes whose gap is ~0 are dead weight in
     the average -- that is the evidence for the open questions C9.8 (is m = 1
     worth its ninth?) and C9.9 (should the nine sizes be weighted equally?).
NOTE
