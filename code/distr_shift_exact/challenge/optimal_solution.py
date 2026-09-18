#!/usr/bin/env python3
"""The intended optimum: Bayesian base predictor, epistemic rejector (C1).

Under 0/1 loss the posterior expected **regret** of answering ``y_hat`` is
exactly the epistemic uncertainty

    E(x, D, y_hat) = T(x, D, y_hat) - A(x, D),

because the aleatoric term ``A(x, D)`` is the expected loss of the per-``theta``
plugin predictor -- which is what the metric's reference predictor is. So the
predictor minimising ``Reg@c`` is the Bayesian learned-prior rule ``H(x, D)``
ranked by ``E(x, D)``, and *not* the same rule ranked by total uncertainty
``T(x, D)``: ``T`` also charges for aleatoric noise that the reference
predictor pays too, and which rejection therefore cannot recover.

This script exists so the organisers can verify the intended optimum really is
the optimum before launch. **It is not shipped to students**, and neither the
competition rules nor the starter notebook may name the argument above.

    python optimal_solution.py out/kaggle out/submissions/optimal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chal.predictors import OPTIMAL, PREDICTORS, Problem, predict, write_submission


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kaggle_dir", type=Path, help="output of prepare_kaggle_data.py")
    p.add_argument("out", type=Path, help="submission file to write")
    p.add_argument("--batches-file", default="test_batches.csv")
    args = p.parse_args()

    prob = Problem(args.kaggle_dir, args.batches_file)
    pred, conf = predict(prob, OPTIMAL)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_submission(args.out, prob.row_id, pred, conf)

    inf = prob.inference
    print(f"{OPTIMAL}: {PREDICTORS[OPTIMAL]}")
    print(f"  {prob.n:,} rows, mean T = {inf.total.mean():.4f}, "
          f"mean A = {inf.aleatoric.mean():.4f}, mean E = {inf.epistemic.mean():.4f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
