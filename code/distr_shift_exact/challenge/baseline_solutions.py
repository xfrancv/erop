#!/usr/bin/env python3
"""The baselines of C7, run on the test data and written as submission files.

Every baseline is built from the **public** competition files only, exactly as a
competitor would build it. The one exception is the true-prior oracle, which
needs ``theta_*`` itself and so reads the local-only ``batch_meta.csv``; it is
the metric's own reference, not a competitor.

+---+---------------------+----------------------------+------------------------+
| # | name                | base predictor             | uncertainty score      |
+---+---------------------+----------------------------+------------------------+
| 1 | ``base``            | argmax_y p_tr(y | x)       | 1 - its posterior      |
| 2 | ``map_plugin``      | h(x, theta_map)            | 1 - its posterior      |
| 3 | ``bayes_total``     | H(x, D)                    | T(x, D)                |
| 4 | ``bayes_epistemic`` | H(x, D)                    | E(x, D), ties by T     |
| 5 | ``true_plugin``     | h(x, theta_*)  [oracle]    | 1 - its posterior      |
| 6 | ``bayes_aleatoric`` | H(x, D)                    | A(x, D), ties by T     |
+---+---------------------+----------------------------+------------------------+

(1) is also the **train-prior plugin** of S3 row 4: ``h(x, theta_tr)`` maximises
``(theta_tr,y / p_tr(y)) p_tr(y | x)``, and that factor is constant in ``y``, so
it is the raw base predictor. Since ``theta_tr`` is no longer a member of
``Theta`` (C3.3) it is not an admissible hypothesis at all, which is what makes
it beatable on every batch.

(5) has **identically zero regret** by construction -- it *is* what the metric
measures against. It is the pipeline's sanity check, not a competitor: any
number other than 0.000000 from ``evaluate.py`` means solution.csv and the
oracle disagree, i.e. the generation and the scoring have diverged.

(6) is the optional sixth row of S3. It is free once ``T`` and ``E`` exist and
it says which half of ``T = A + E`` is doing the ranking work.

**The comparison the competition rests on is (3) against (4).** They share the
base predictor ``H(x, D)`` and differ *only* in the ranking, so their gap
isolates the value of scoring epistemic rather than total uncertainty. If they
are not separated by more than their bootstrap intervals, the competition has no
discoverable structure and ``tau``, the grid or the coverage need retuning (C7).

Run with::

    python baseline_solutions.py out/kaggle out/submissions
    python baseline_solutions.py out/kaggle out/submissions --only bayes_total bayes_epistemic

Then score them:

    python evaluate.py out/submissions/bayes_epistemic.csv out/kaggle/solution.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chal.predictors import PREDICTORS, Problem, predict, write_submission


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kaggle_dir", type=Path,
                   help="output of prepare_kaggle_data.py (or make_dev_benchmark.py)")
    p.add_argument("out_dir", type=Path, help="directory receiving the submissions")
    p.add_argument("--only", nargs="+", choices=list(PREDICTORS),
                   default=list(PREDICTORS),
                   help="subset of baselines to run (default: all six)")
    p.add_argument("--batches-file", default="test_batches.csv",
                   help="name of the batch listing inside kaggle_dir")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob = Problem(args.kaggle_dir, args.batches_file)
    print(f"{prob.n:,} rows in {len(prob.starts):,} batches, C = {prob.C} priors")

    for name in args.only:
        pred, conf = predict(prob, name)
        path = out_dir / f"{name}.csv"
        write_submission(path, prob.row_id, pred, conf)
        print(f"  {name:<17} {PREDICTORS[name]}")
        print(f"  {'':<17} -> {path}")

    if prob._inf is not None:
        inf = prob.inference
        zero = float((inf.epistemic <= 1e-12).mean())
        print(f"\nuncertainty decomposition over all rows:")
        print(f"  mean T = {inf.total.mean():.4f}"
              f"   mean A = {inf.aleatoric.mean():.4f}"
              f"   mean E = {inf.epistemic.mean():.4f}")
        print(f"  E is exactly zero on {zero:.1%} of rows -- there every prior in "
              f"Theta votes for\n  the same label, so the ranking is decided by the "
              f"tie-break on T. That is why\n  the tie-break is part of the "
              f"rejector, not a detail (S3).")
        # How often adaptation actually changes the answer, i.e. the headroom
        # the whole competition is played inside.
        base_pred = prob.log_post.argmax(axis=1)
        print(f"  H(x, D) differs from the non-adapted base predictor on "
              f"{float((inf.bayes_pred != base_pred).mean()):.1%} of rows")

    print(f"\nscore them with:")
    for name in args.only:
        print(f"  python evaluate.py {out_dir}/{name}.csv "
              f"{args.kaggle_dir}/solution.csv")


if __name__ == "__main__":
    main()
