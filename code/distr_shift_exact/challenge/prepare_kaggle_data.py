#!/usr/bin/env python3
"""Generate every file the Kaggle competition needs (C4, C5).

Reads the outputs of ``train_base_model.py`` and the priors file, draws the test
batches of C4 for each of the three ``Usage`` pools, and writes:

``test_priors.csv``       ``Theta``, the admissible test priors
``train_prior.csv``       ``theta_tr``
``predictions.csv``       calibrated base posterior on every released image
``test.csv``              one row per batch: ``id_test``, ``m``
``test_batches.csv``      one row per test image (no image key: C9.7)
``sample_submission.csv`` the non-adapted base predictor, as the baseline
``dev.csv``               labeled student development images
``solution.csv``          ground truth + reference predictor + ``Usage``  [HIDDEN]
``batch_meta.csv``        ``theta_*`` per batch                           [LOCAL ONLY]
``row_image.csv``         which eval image each row drew                 [LOCAL ONLY]
``manifest.json``         seed, counts, duplicate rates, checks

The last two are **never uploaded**. ``solution.csv`` goes to Kaggle but not to
students; ``batch_meta.csv`` goes nowhere -- it exists so ``baseline_solutions.py``
can build the true-prior oracle, which needs ``theta_*`` itself rather than just
the reference predictor's output.

``id_test`` and ``row_id`` are **random permutations**, not generation order:
batches are generated usage by usage and size by size, so sequential ids would
leak the ``Usage`` partition outright.

Run with::

    python prepare_kaggle_data.py out/model priors_tissuemnist_challenge.txt out/kaggle
    python prepare_kaggle_data.py out/model priors.txt out/smoke --n-min 5 --batch-scale 20
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from chal import data as chdata
from chal.ids import random_ids
from chal.generate import draw_rows, reference_and_base
from chal.priors import read_prior_set
from chal.protocol import BATCH_SCALE, N_MIN, SIZE_GRID
from chal.splits import USAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="output of train_base_model.py")
    p.add_argument("priors", type=Path, help="output of make_priors.py")
    p.add_argument("out_dir", type=Path, help="directory receiving the upload")
    p.add_argument("--seed", type=int, default=20260912,
                   help="master seed; per-usage streams are spawned from it so "
                        "regenerating one usage does not perturb the others")
    p.add_argument("--grid", type=int, nargs="+", default=list(SIZE_GRID),
                   help=f"batch sizes (default {list(SIZE_GRID)})")
    p.add_argument("--n-min", type=int, default=N_MIN,
                   help=f"minimum batches per size (default {N_MIN}); this is "
                        f"the constant that sets the metric's precision (C4)")
    p.add_argument("--batch-scale", type=int, default=BATCH_SCALE,
                   help=f"N(m) = max(n_min, ceil(scale / m)) (default {BATCH_SCALE})")
    p.add_argument("--balance-pools", action="store_true",
                   help="subsample each usage pool to equal class counts. A "
                        "batch draws a class from theta_* and then an image "
                        "uniformly from that class, so an image is drawn with "
                        "probability theta_y / n_y; averaged over theta the "
                        "theta_y washes out but the 1/n_y does not, and on an "
                        "imbalanced pool the number of times an img_id appears "
                        "is a strong signal of its class. Equal n_y removes "
                        "that. See audit_leakage.py.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    grid = tuple(int(m) for m in args.grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs ------------------------------------------------------------
    npz = np.load(args.model_dir / "log_post.npz", allow_pickle=True)
    log_post_eval = npz["log_post_eval"].astype(np.float64)
    y_eval = npz["y_eval"]
    pool_of_eval = npz["pool_of_eval"]
    log_post_dev = npz["log_post_dev"].astype(np.float64)
    y_dev = npz["y_dev"]
    train_prior = npz["train_prior"]
    log_train_prior = np.log(train_prior)
    Y = chdata.NUM_CLASSES

    ps = read_prior_set(args.priors, train_prior)
    assert ps.Y == Y, f"priors file has Y = {ps.Y}, expected {Y}"

    master = np.random.default_rng(args.seed)
    usage_seeds = master.spawn(len(USAGES))

    # --- draw the batches, usage by usage ---------------------------------
    # Same code path as make_dev_benchmark.py (chal.generate), so the benchmark
    # students score themselves on cannot drift from the real thing.
    rows_batch, rows_slot, rows_eval_idx, rows_usage = [], [], [], []
    batch_usage, batch_m, batch_theta, batch_dup = [], [], [], []
    next_batch = 0
    for u_i, usage in enumerate(USAGES):
        pool_idx = np.flatnonzero(pool_of_eval == u_i)
        rng = np.random.default_rng(usage_seeds[u_i])
        if args.balance_pools:
            pool_idx = _balance(pool_idx, y_eval[pool_idx], Y, rng)
        drawn = draw_rows(y_eval[pool_idx], ps.theta, rng, Y, grid=grid,
                          n_min=args.n_min, scale=args.batch_scale)
        rows_batch.append(drawn.gen_batch + next_batch)
        rows_slot.append(drawn.slot)
        rows_eval_idx.append(pool_idx[drawn.pool_row])
        rows_usage.append(np.full(drawn.n_rows, u_i, dtype=np.int8))
        batch_usage.append(np.full(drawn.n_batches, u_i, dtype=np.int8))
        batch_m.append(drawn.batch_m)
        batch_theta.append(drawn.batch_theta)
        batch_dup.append(drawn.batch_dup)
        next_batch += drawn.n_batches
        print(f"{usage:<8} {drawn.n_batches:>6,} batches, "
              f"{drawn.n_rows:>8,} rows")

    gen_batch = np.concatenate(rows_batch)
    slot = np.concatenate(rows_slot)
    eval_idx = np.concatenate(rows_eval_idx)
    usage_of_row = np.concatenate(rows_usage)
    batch_usage = np.concatenate(batch_usage)
    batch_m = np.concatenate(batch_m)
    batch_theta = np.concatenate(batch_theta)
    batch_dup = np.concatenate(batch_dup)
    m_of_row = batch_m[gen_batch]
    n_rows, n_batches = len(slot), next_batch

    # --- shuffle the public identifiers -----------------------------------
    # Generation order is usage-major then size-major; sequential ids would hand
    # the Usage partition to anyone who sorted the file.
    batch_public_id = master.permutation(n_batches)   # generation index -> id_test
    id_test = batch_public_id[gen_batch]
    row_id = master.permutation(n_rows)

    # --- identifiers ------------------------------------------------------
    # Test rows carry NO image key (C9.7). An image recurs across batches
    # because the pool is finite and drawn from with replacement, and publishing
    # a per-image key hands competitors a channel the intended solution does not
    # use: an image that keeps appearing in batches enriched on classes {i, i+1}
    # is probably class i or i+1. Posteriors are therefore published per *row*.
    # The true identity is written to row_image.csv, which is never uploaded and
    # exists so audit_leakage.py can measure the residual channel.
    used_eval = np.unique(eval_idx)
    name_of_eval = np.empty(len(y_eval), dtype=object)
    name_of_eval[used_eval] = random_ids(len(used_eval), master)
    n_named = len(used_eval)

    # --- the reference predictor (C5) -------------------------------------
    # h(x, theta_*) with the batch's true prior, on the same calibrated
    # posterior students get. Shipping its *output* keeps theta_* out of
    # solution.csv entirely.
    lp_rows = log_post_eval[eval_idx]
    theta_of_row = ps.theta[batch_theta[gen_batch]]
    pred_ref, unc_ref, base_pred, base_conf = reference_and_base(
        lp_rows, log_train_prior, theta_of_row)
    label = y_eval[eval_idx]

    # --- check before writing ----------------------------------------------
    # The C8 assertions run on the arrays, not the files, so a failure leaves
    # no half-valid upload on disk.
    checks = _checks(id_test, slot, m_of_row, row_id, usage_of_row, eval_idx,
                     pool_of_eval, grid)

    # --- write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((slot, id_test))       # tidy: by batch, then slot

    pd.DataFrame({
        "id": np.arange(ps.C),
        **{f"p{y}": ps.theta[:, y] for y in range(Y)},
    }).to_csv(out_dir / "test_priors.csv", index=False)

    pd.DataFrame({f"p{y}": [train_prior[y]] for y in range(Y)}).to_csv(
        out_dir / "train_prior.csv", index=False)

    batch_order = np.argsort(batch_public_id)
    pd.DataFrame({
        "id_test": batch_public_id[batch_order],
        "m": batch_m[batch_order],
    }).to_csv(out_dir / "test.csv", index=False)

    pd.DataFrame({
        "row_id": row_id[order],
        "id_test": id_test[order],
        "slot": slot[order],
    }).to_csv(out_dir / "test_batches.csv", index=False)

    pd.DataFrame({
        "row_id": row_id[order],
        "pred": base_pred[order],
        "confidence": base_conf[order],
    }).to_csv(out_dir / "sample_submission.csv", index=False)

    pd.DataFrame({
        "row_id": row_id[order],
        "id_test": id_test[order],
        "slot": slot[order],
        "m": m_of_row[order],
        "label": label[order],
        "pred_ref": pred_ref[order],
        "Usage": np.array(USAGES, dtype=object)[usage_of_row[order]],
    }).to_csv(out_dir / "solution.csv", index=False)

    # dev.csv is a plain labeled sample: a label and its posterior, nothing to
    # join on. It deliberately carries no identifier -- there is nothing for one
    # to point at, and giving development images a stable id that test rows lack
    # would invite the question of why, which points straight at the linkage
    # rule 2 forbids.
    dev_post = np.exp(log_post_dev)
    pd.DataFrame({"label": y_dev,
                  **{f"p{y}": dev_post[:, y] for y in range(Y)}}).to_csv(
        out_dir / "dev.csv", index=False)

    # predictions.csv is keyed by row_id, one row per test row. A row of an
    # image drawn twice therefore appears twice, with identical probabilities --
    # detectable if a competitor goes looking, which the rules forbid (C9.7),
    # but not handed over.
    pred_post = np.exp(lp_rows[order])
    # Written at full precision, with no float_format: pandas emits the
    # shortest string that round-trips to the same float64, so a competitor
    # reading predictions.csv gets *exactly* the posterior pred_ref was
    # computed from. Rounding here would let a knife-edge argmax flip, and the
    # true-prior oracle would then no longer score exactly 0.000000 -- losing
    # the one check that says generation and scoring agree.
    pd.DataFrame({"row_id": row_id[order],
                  **{f"p{y}": pred_post[:, y] for y in range(Y)}}).to_csv(
        out_dir / "predictions.csv", index=False)

    # Local only: theta_* per batch, for the oracle baseline and diagnostics.
    pd.DataFrame({
        "id_test": batch_public_id[batch_order],
        "m": batch_m[batch_order],
        "usage": np.array(USAGES, dtype=object)[batch_usage[batch_order]],
        "theta_star_index": batch_theta[batch_order],
        "dup_fraction": batch_dup[batch_order],
    }).to_csv(out_dir / "batch_meta.csv", index=False)

    # Local only: the identity the public files withhold, for audit_leakage.py.
    pd.DataFrame({"row_id": row_id[order],
                  "img_id": name_of_eval[eval_idx][order]}).to_csv(
        out_dir / "row_image.csv", index=False)

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "grid": list(grid),
        "n_min": args.n_min,
        "batch_scale": args.batch_scale,
        "num_classes": Y,
        "C": ps.C,
        "usages": list(USAGES),
        "n_batches": int(n_batches),
        "n_rows": int(n_rows),
        "n_distinct_images": int(n_named),
        "balanced_pools": bool(args.balance_pools),
        "images_published": False,
        "rows_per_usage": {u: int((usage_of_row == i).sum())
                           for i, u in enumerate(USAGES)},
        "batches_per_size": {int(m): int((batch_m == m).sum()) for m in grid},
        "rows_per_size_public": {
            int(m): int(((m_of_row == m) & (usage_of_row == 0)).sum())
            for m in grid},
        "duplicate_fraction_per_size": {
            int(m): float(np.mean(batch_dup[batch_m == m]))
            for m in grid},
        "mean_reference_uncertainty": float(unc_ref.mean()),
        "base_predictor_accuracy": float((base_pred == label).mean()),
        "reference_predictor_accuracy": float((pred_ref == label).mean()),
        "checks": checks,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{n_batches:,} batches, {n_rows:,} rows, {n_named:,} distinct images "
          f"(posteriors only; no pixels are published)")
    print(f"base predictor accuracy      : {manifest['base_predictor_accuracy']:.4f}")
    print(f"reference predictor accuracy : "
          f"{manifest['reference_predictor_accuracy']:.4f}")
    print(f"mean duplicate fraction at m = {max(grid)}: "
          f"{manifest['duplicate_fraction_per_size'][max(grid)]:.4f}")
    print(f"\nupload to Kaggle : test_priors.csv train_prior.csv predictions.csv "
          f"test.csv\n                   test_batches.csv sample_submission.csv "
          f"dev.csv")
    print(f"give Kaggle only : solution.csv")
    print(f"keep local       : batch_meta.csv row_image.csv manifest.json")


def _balance(pool_idx: np.ndarray, y_pool: np.ndarray, num_classes: int,
             rng: np.random.Generator) -> np.ndarray:
    """Subsample a pool to equal class counts, keeping the rarest class whole."""
    counts = np.bincount(y_pool, minlength=num_classes)
    k = int(counts.min())
    keep = np.concatenate([rng.permutation(np.flatnonzero(y_pool == c))[:k]
                           for c in range(num_classes)])
    return pool_idx[np.sort(keep)]


def _checks(id_test, slot, m_of_row, row_id, usage_of_row, eval_idx,
            pool_of_eval, grid) -> dict:
    """The C8 pre-launch assertions, run at generation time."""
    # max(slot) + 1 == m for every batch, and no slot missing.
    order = np.lexsort((slot, id_test))
    s, b, m = slot[order], id_test[order], m_of_row[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(b)) + 1])
    ends = np.concatenate([starts[1:], [len(b)]])
    assert np.all(s[ends - 1] + 1 == m[starts]), "max(slot) + 1 != m somewhere"
    assert np.all(ends - starts == m[starts]), "a batch is missing rows"
    assert np.array_equal(s, np.concatenate(
        [np.arange(k) for k in m[starts]])), "slots are not 0..m-1"

    assert len(np.unique(row_id)) == len(row_id), "row_id is not unique"

    # The three usage pools are disjoint at the image level.
    pools = [set(np.unique(eval_idx[usage_of_row == i]).tolist())
             for i in range(len(USAGES))]
    for i in range(len(USAGES)):
        for j in range(i + 1, len(USAGES)):
            assert not (pools[i] & pools[j]), (
                f"pools {USAGES[i]} and {USAGES[j]} share images")
    assert all(pool_of_eval[list(p)].min() == pool_of_eval[list(p)].max()
               for p in pools if p)

    # Development and evaluation images are disjoint structurally, not by
    # coincidence: development comes from the MedMNIST train split and
    # evaluation from val + test (C3.1), so no assertion on identifiers is
    # needed -- and comparing two independent sets of random ids would have
    # asserted nothing anyway.

    usages_seen = sorted(set(np.array(USAGES, dtype=object)[
        np.unique(usage_of_row)].tolist()))
    assert usages_seen == sorted(USAGES), f"usages present: {usages_seen}"
    assert "Ignored" in USAGES, "Kaggle's literal is 'Ignored', not 'Ignore'"

    sizes = sorted(set(int(v) for v in np.unique(m_of_row)))
    assert sizes == sorted(grid), f"sizes present {sizes} != grid {sorted(grid)}"

    return {
        "slots_are_0_to_m_minus_1": True,
        "row_id_unique": True,
        "usage_pools_disjoint": True,
        "dev_disjoint_from_test": "structural: different MedMNIST splits",
        "all_grid_sizes_present": True,
        "usage_literals": list(USAGES),
    }


if __name__ == "__main__":
    main()
