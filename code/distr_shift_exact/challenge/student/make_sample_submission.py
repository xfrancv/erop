#!/usr/bin/env python3
"""Build a submission from the base predictor -- the baseline to beat.

For every image in a batch listing it emits the label the **non-adapted** base
predictor thinks is most likely, and that label's probability as the confidence.
It reads those probabilities straight out of ``predictions.csv``.

This is the baseline: it ignores the batch entirely and answers each image on
its own, under the training label prior. Since the training prior is not one of
the eight admissible test priors, it is wrong on every batch by construction.

It also shows the submission format, which is three columns:

    row_id      copied from the batch listing; one row per test image
    pred        the predicted label, an integer 0..7
    confidence  any real number; higher means "keep this prediction"

Only the *ranking* induced by ``confidence`` matters -- the organiser sorts by
it, keeps the most confident 80 %, and scores those. Ties are broken by
ascending ``row_id``, so a constant confidence is scored deterministically (and
poorly). Note the ranking is pooled **across batches** of the same size, so your
confidences have to be comparable between batches, not just within one.

    # the offline development benchmark
    python make_sample_submission.py --data-dir ../competition-data \
        --batches dev_test_batches.csv --out dev_sample_submission.csv

    # the real thing, for uploading to Kaggle
    python make_sample_submission.py --data-dir ../competition-data \
        --batches test_batches.csv --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 8


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="directory holding the competition CSV files")
    p.add_argument("--batches", default="dev_test_batches.csv",
                   help="batch listing to predict: 'dev_test_batches.csv' for "
                        "the offline benchmark (default) or 'test_batches.csv' "
                        "for the Kaggle submission")
    p.add_argument("--predictions", default=None,
                   help="predictions file (default: predictions.csv, or "
                        "dev_predictions.csv when --batches is a dev file)")
    p.add_argument("--out", type=Path, default=Path("dev_sample_submission.csv"))
    args = p.parse_args()

    batches = pd.read_csv(args.data_dir / args.batches)
    pred_file = args.predictions or ("dev_predictions.csv"
                                     if args.batches.startswith("dev_")
                                     else "predictions.csv")
    preds = pd.read_csv(args.data_dir / pred_file)

    cols = [f"p{y}" for y in range(NUM_CLASSES)]
    post = preds[cols].to_numpy(dtype=np.float64)

    # Line up every row of the batch listing with its image's posterior.
    pos = pd.Index(preds["row_id"]).get_indexer(batches["row_id"])
    missing = int((pos < 0).sum())
    if missing:
        raise SystemExit(
            f"{missing} rows of {args.batches} have no entry in {pred_file}")
    rows = post[pos]

    pred = rows.argmax(axis=1)
    confidence = rows[np.arange(len(pred)), pred]

    out = pd.DataFrame({"row_id": batches["row_id"].to_numpy(),
                        "pred": pred.astype(np.int64),
                        "confidence": confidence})
    out.sort_values("row_id", kind="stable").to_csv(args.out, index=False)

    print(f"wrote {args.out}  ({len(out):,} rows, from {args.batches} "
          f"+ {pred_file})")
    print(f"  predicted label counts: {np.bincount(pred, minlength=NUM_CLASSES).tolist()}")
    print(f"  confidence in [{confidence.min():.4f}, {confidence.max():.4f}]")
    print("\nThis is the baseline. It never looks at the other images in a "
          "batch,\nso it cannot know the batch's label prior.")


if __name__ == "__main__":
    main()
