#!/usr/bin/env python3
"""Build the offline benchmark students score themselves on (C7).

Runs the **same** batch-generation protocol as ``prepare_kaggle_data.py`` --
literally the same function, ``chal.generate.draw_rows`` -- but over the student
development split, whose labels students already have. Together with
``evaluate.py`` this lets them measure a predictor before spending one of their
five daily submissions.

Outputs, all shipped to students:

``dev_test.csv``              one row per development batch: ``id_test``, ``m``

All development ``row_id`` and ``id_test`` values are offset by
``DEV_ID_OFFSET`` so they cannot be confused with the competition's.
``dev_test_batches.csv``      one row per image in those batches (no image key)
``dev_predictions.csv``       the base posterior for each of those rows
``dev_solution.csv``          ground truth + reference predictor, **no** ``Usage``
``dev_sample_submission.csv`` the non-adapted base predictor
``dev_batch_meta.csv``        ``theta_*`` per batch

``dev_solution.csv`` is given away, unlike the competition's ``solution.csv``:
the development labels are public anyway (they are in ``dev.csv``), so nothing
is protected by withholding it, and having it is what makes the benchmark
useful. ``dev_batch_meta.csv`` is likewise public here -- knowing ``theta_*`` on
the development batches is exactly what lets a student check their adaptation
against the oracle while they work.

The development split is much smaller than an evaluation pool, so the grid is
capped: C4/S6.2 requires ``m_max <= |pool| / 10``, and the batch counts are
scaled down to keep the benchmark quick to regenerate.

    python make_dev_benchmark.py out/model priors_tissuemnist_challenge.txt \\
        out/kaggle out/kaggle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chal import data as chdata
from chal.generate import draw_rows, reference_and_base
from chal.priors import read_prior_set
from chal.protocol import POOL_SIZE_RATIO, SIZE_GRID

# Development identifiers live above this, the competition's below it, so the
# two never collide. Any value past the competition's 126 000 rows would do.
DEV_ID_OFFSET = 1_000_000


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="output of train_base_model.py")
    p.add_argument("priors", type=Path, help="output of make_priors.py")
    p.add_argument("kaggle_dir", type=Path,
                   help="output of prepare_kaggle_data.py; its dev.csv supplies "
                        "the image ids, so the benchmark refers to posteriors "
                        "students already have")
    p.add_argument("out_dir", type=Path, help="directory receiving the benchmark")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--grid", type=int, nargs="+", default=list(SIZE_GRID))
    p.add_argument("--n-min", type=int, default=50,
                   help="minimum batches per size (default 50; the competition "
                        "uses 200, but the benchmark is meant to be cheap)")
    p.add_argument("--batch-scale", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.model_dir / "log_post.npz", allow_pickle=True)
    log_post_dev = npz["log_post_dev"].astype(np.float64)
    y_dev = npz["y_dev"]
    train_prior = npz["train_prior"]
    log_train_prior = np.log(train_prior)
    Y = chdata.NUM_CLASSES

    ps = read_prior_set(args.priors, train_prior)
    dev = pd.read_csv(args.kaggle_dir / "dev.csv")
    assert len(dev) == len(y_dev), (
        f"dev.csv has {len(dev):,} rows but the model dir has {len(y_dev):,} "
        f"development examples -- they must come from the same run")
    assert np.array_equal(dev["label"].to_numpy(), y_dev), \
        "dev.csv labels disagree with the model directory"

    grid = tuple(m for m in args.grid if m * POOL_SIZE_RATIO <= len(y_dev))
    dropped = [m for m in args.grid if m not in grid]
    if dropped:
        print(f"grid truncated to {list(grid)}: the development split holds "
              f"{len(y_dev):,}, and C4 requires m_max <= |pool| / "
              f"{POOL_SIZE_RATIO}; dropped {dropped}")

    rng = np.random.default_rng(args.seed)
    drawn = draw_rows(y_dev, ps.theta, rng, Y, grid=grid, n_min=args.n_min,
                      scale=args.batch_scale)

    # Shuffle the public identifiers, as C4 does for the competition files, and
    # offset them out of the competition's range. The two id spaces would
    # otherwise both start at 0 and overlap, so one loader reading "batches +
    # predictions" and caching by row_id would have development rows silently
    # overwrite test rows. The metric and evaluate.py both catch that, but late;
    # an offset prevents it. Nothing depends on the ids being 0-based -- they
    # exist only to align files with each other.
    batch_public_id = DEV_ID_OFFSET + rng.permutation(drawn.n_batches)
    id_test = batch_public_id[drawn.gen_batch]
    row_id = DEV_ID_OFFSET + rng.permutation(drawn.n_rows)

    lp_rows = log_post_dev[drawn.pool_row]
    theta_of_row = ps.theta[drawn.batch_theta[drawn.gen_batch]]
    pred_ref, _unc, base_pred, base_conf = reference_and_base(
        lp_rows, log_train_prior, theta_of_row)
    label = y_dev[drawn.pool_row]


    order = np.lexsort((drawn.slot, id_test))
    batch_order = np.argsort(batch_public_id)

    pd.DataFrame({"id_test": batch_public_id[batch_order],
                  "m": drawn.batch_m[batch_order]}).to_csv(
        out_dir / "dev_test.csv", index=False)
    # No image key, exactly as test_batches.csv (C9.7): the benchmark must not
    # teach a habit the competition forbids.
    pd.DataFrame({"row_id": row_id[order], "id_test": id_test[order],
                  "slot": drawn.slot[order]}).to_csv(
        out_dir / "dev_test_batches.csv", index=False)
    dpost = np.exp(lp_rows[order])
    pd.DataFrame({"row_id": row_id[order],
                  **{f"p{y}": dpost[:, y] for y in range(Y)}}).to_csv(
        out_dir / "dev_predictions.csv", index=False)
    pd.DataFrame({"row_id": row_id[order], "pred": base_pred[order],
                  "confidence": base_conf[order]}).to_csv(
        out_dir / "dev_sample_submission.csv", index=False)
    # No Usage column: the development benchmark has nothing to hide.
    pd.DataFrame({"row_id": row_id[order], "id_test": id_test[order],
                  "slot": drawn.slot[order],
                  "m": drawn.m_of_row[order], "label": label[order],
                  "pred_ref": pred_ref[order]}).to_csv(
        out_dir / "dev_solution.csv", index=False)
    pd.DataFrame({"id_test": batch_public_id[batch_order],
                  "m": drawn.batch_m[batch_order],
                  "theta_star_index": drawn.batch_theta[batch_order],
                  "dup_fraction": drawn.batch_dup[batch_order]}).to_csv(
        out_dir / "dev_batch_meta.csv", index=False)

    print(f"{drawn.n_batches:,} batches, {drawn.n_rows:,} rows over sizes "
          f"{list(grid)}")
    print(f"base predictor accuracy      : {float((base_pred == label).mean()):.4f}")
    print(f"reference predictor accuracy : {float((pred_ref == label).mean()):.4f}")
    print(f"\nfiles in {out_dir}/: dev_test.csv dev_test_batches.csv "
          f"dev_predictions.csv\n                     dev_solution.csv "
          f"dev_sample_submission.csv dev_batch_meta.csv")
    print("\nstudents run:  python evaluate.py my_submission.csv dev_solution.csv")


if __name__ == "__main__":
    main()
