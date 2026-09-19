#!/usr/bin/env python3
"""The baselines of C7, run on an organiser directory, written as submissions.

The directory is ``organiser/test`` or ``organiser/dev`` of ``hard_package.py``.
Its ``predictions.csv`` is the TRUE label model, so every baseline here is what
a competitor with a perfect model of ``q(y | x)`` would achieve; what separates
them is only how they adapt to the location and how they rank.

+---+---------------------+----------------------------+------------------------+
| # | name                | base predictor             | uncertainty score      |
+---+---------------------+----------------------------+------------------------+
| 1 | ``base``            | argmax_y q(y | x)          | 1 - its posterior      |
| 2 | ``map_plugin``      | h(x, theta_map)            | 1 - its posterior      |
| 3 | ``bayes_total``     | H(x, D)                    | T(x, D)                |
| 4 | ``bayes_epistemic`` | H(x, D)                    | E(x, D), ties by T     |
| 5 | ``true_plugin``     | h(x, theta_*)  [oracle]    | 1 - its posterior      |
| 6 | ``bayes_aleatoric`` | H(x, D)                    | A(x, D), ties by T     |
+---+---------------------+----------------------------+------------------------+

(5) has **identically zero regret** by construction -- it *is* the reference
predictor. Any other number means the generation and the scoring have diverged.

**The location prior** ``p(theta)`` of (2), (3), (4) and (6) is chosen with
``--location-prior``: ``uniform`` (what a competitor who ignores the hint
uses), ``true`` (the secret ``w``, from ``location_prior.csv``) or a CSV file,
e.g. the EM estimate of ``estimate_location_prior.py``. ``--suffix`` keeps the
three runs apart in one directory. The spec's question -- does estimating ``w``
by EM pay? -- is ``bayes_epistemic_em`` against ``bayes_epistemic_uniform``.

    python baseline_solutions.py out/v3/kaggle/organiser/test out/v3/submissions \
        --location-prior uniform --suffix _uniform
    python evaluate.py out/v3/submissions/bayes_epistemic_uniform.csv \
        out/v3/kaggle/organiser/test/solution.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chal.locprior import read_location_prior
from chal.predictors import PREDICTORS, Problem, predict, write_submission


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kaggle_dir", type=Path,
                   help="organiser/test or organiser/dev of hard_package.py")
    p.add_argument("out_dir", type=Path, help="directory receiving the submissions")
    p.add_argument("--only", nargs="+", choices=list(PREDICTORS),
                   default=list(PREDICTORS),
                   help="subset of baselines to run (default: all six)")
    p.add_argument("--batches-file", default="test_batches.csv",
                   help="name of the batch listing inside kaggle_dir")
    p.add_argument("--location-prior", default="uniform",
                   help="'uniform', 'true' (kaggle_dir/location_prior.csv) or "
                        "a location-prior CSV file (default uniform)")
    p.add_argument("--suffix", default="",
                   help="appended to every submission name, e.g. _em")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.location_prior == "uniform":
        w = None
    elif args.location_prior == "true":
        w = read_location_prior(Path(args.kaggle_dir) / "location_prior.csv")
    else:
        w = read_location_prior(Path(args.location_prior))
    prob = Problem(args.kaggle_dir, args.batches_file, location_prior=w)
    print(f"{prob.n:,} rows in {len(prob.starts):,} batches, C = {prob.C} "
          f"locations, location prior: {args.location_prior}")

    for name in args.only:
        pred, conf = predict(prob, name)
        path = out_dir / f"{name}{args.suffix}.csv"
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
        print(f"  python evaluate.py {out_dir}/{name}{args.suffix}.csv "
              f"{args.kaggle_dir}/solution.csv")


if __name__ == "__main__":
    main()
