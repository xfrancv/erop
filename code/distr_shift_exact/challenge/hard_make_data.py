#!/usr/bin/env python3
"""Build every array of the hard variant except the model's outputs.

The hard variant (``tasks/hard_variant.md``) releases **images**, not a trained
network's posteriors, and describes the shift only as "locations". This script
does everything that does not need the reference model, and needs no torch:

1. **Split.** The original TissueMNIST training split is divided, stratified by
   class, into the *training data* (released, labeled, with a location per
   image) and a *development pool* (``--dev-fraction``, default 10 %) that the
   development batches are drawn from. The test batches are drawn from the
   original validation + test splits, partitioned into the three Kaggle
   ``Usage`` pools exactly as in the easy variant.
2. **Locations.** ``chal.locations`` solves the linear program that moves the
   easy variant's 8 priors as little as possible so that every training image
   can be placed; location 8 takes the remainder. The 9 resulting priors are
   written to ``--priors-out``.
3. **Batches.** Development and test batches are drawn by the easy variant's
   own sampler (``chal.generate.draw_rows``) under the 9 location priors, each
   batch from one location. Every location gets exactly the same number of
   batches of every size in every usage (``N(m)`` is rounded up to a multiple
   of 9 for that).
4. **Transformation.** Every released image -- each training image once, each
   development and test *row* separately -- is rotated by 90, 180 or 270
   degrees and given light Gaussian noise (``chal.transform``).

Outputs in ``out_dir`` (all organiser-only; ``hard_package.py`` decides what is
published):

``hard_data.npz``  images, labels, locations, batches, ids, sources
``priors.txt``     the 9 location priors (also written to ``--priors-out``)
``report.txt``     the split, the location plan, the batch counts

Then train the reference model on the released training images::

    python hard_make_data.py out/hard/data --priors-out priors_tissuemnist_hard.txt
    python train_base_model.py out/hard/model --hard-data out/hard/data
    python hard_package.py out/hard/data out/hard/model out/hard/kaggle

A capped smoke run, for the plumbing only::

    python hard_make_data.py out/hard_smoke/data --n-min 5 --batch-scale 20 \\
        --dev-n-min 5 --dev-batch-scale 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from chal import data as chdata
from chal.generate import draw_rows
from chal.locations import EPS_DEFAULT, N_MIN_DEFAULT, assign_locations, \
    plan_locations
from chal.priors import TV_TOL, PriorSet, read_prior_set, write_prior_set
from chal.protocol import BATCH_SCALE, N_MIN, POOL_SIZE_RATIO, SIZE_GRID
from chal.splits import USAGES, _assign_pools
from chal.transform import NOISE_STD_DEFAULT, transform
from make_dev_benchmark import DEV_ID_OFFSET
from prepare_kaggle_data import _balance, _checks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--data-root", type=Path, default=chdata.DATA_ROOT)
    p.add_argument("--base-priors", type=Path,
                   default=Path("priors_tissuemnist_challenge.txt"),
                   help="the 8 priors locations 0..7 start from")
    p.add_argument("--priors-out", type=Path, default=None,
                   help="also write the 9 location priors here (they always "
                        "go to out_dir/priors.txt)")
    p.add_argument("--dev-fraction", type=float, default=0.10,
                   help="share of the original training split held out as the "
                        "development pool (default 0.10)")
    p.add_argument("--eps", type=float, default=EPS_DEFAULT,
                   help=f"floor on every location prior (default {EPS_DEFAULT})")
    p.add_argument("--n-min-location", type=int, default=N_MIN_DEFAULT,
                   help=f"minimum size of locations 0..7 (default {N_MIN_DEFAULT})")
    p.add_argument("--noise-std", type=float, default=NOISE_STD_DEFAULT,
                   help=f"Gaussian pixel noise, grey levels (default "
                        f"{NOISE_STD_DEFAULT})")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--grid", type=int, nargs="+", default=list(SIZE_GRID))
    p.add_argument("--n-min", type=int, default=N_MIN,
                   help=f"test: minimum batches per size (default {N_MIN})")
    p.add_argument("--batch-scale", type=int, default=BATCH_SCALE,
                   help=f"test: N(m) = max(n_min, ceil(scale / m)) "
                        f"(default {BATCH_SCALE})")
    p.add_argument("--dev-n-min", type=int, default=50,
                   help="development: minimum batches per size (default 50)")
    p.add_argument("--dev-batch-scale", type=int, default=500)
    p.add_argument("--balance-pools", action="store_true",
                   help="subsample each test pool to equal class counts; see "
                        "prepare_kaggle_data.py")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The data directory always holds the priors, so hard_package.py finds
    # them there; --priors-out writes a second copy, e.g. the committed file.
    priors_files = [out_dir / "priors.txt"] + (
        [args.priors_out] if args.priors_out else [])
    Y = chdata.NUM_CLASSES
    grid = tuple(int(m) for m in args.grid)

    master = np.random.default_rng(args.seed)
    (rng_loc, rng_train_tf, rng_dev, rng_dev_tf, rng_test_tf,
     rng_ids) = master.spawn(6)
    usage_seeds = master.spawn(len(USAGES))

    # --- 1. split -----------------------------------------------------------
    ds = chdata.load(args.data_root)
    X_orig, y_orig = ds.splits["train"]
    idx_train, idx_devpool = train_test_split(
        np.arange(len(y_orig)), test_size=args.dev_fraction, stratify=y_orig,
        random_state=args.seed % (2 ** 32))
    idx_train, idx_devpool = np.sort(idx_train), np.sort(idx_devpool)
    y_train, y_devpool = y_orig[idx_train], y_orig[idx_devpool]

    X_eval = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
    y_eval = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
    pool_of_eval = _assign_pools(y_eval, seed=args.seed + 2)

    # --- 2. locations -------------------------------------------------------
    D = np.bincount(y_train, minlength=Y)
    pooled = D / D.sum()
    base = read_prior_set(args.base_priors, pooled).theta
    plan = plan_locations(D, base, eps=args.eps, n_min=args.n_min_location)
    loc_train = assign_locations(y_train, plan, rng_loc)
    theta = plan.priors
    K = len(base)
    labels = [f"location {l} (from prior {l} of {args.base_priors.name}, "
              f"TV moved {plan.tv_change[l]:.4f})" for l in range(K)]
    labels.append(f"location {K} (the remainder of the training data)")
    ps = PriorSet(theta, labels, pooled)
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    assert off.min() >= TV_TOL, f"two locations are only TV {off.min():.4f} apart"
    assert ps.tv_to_train.min() >= TV_TOL, \
        "a location's prior coincides with the pooled training prior"
    # The competitor's route to the priors is the class frequencies of each
    # location in the released training data. They must be the priors exactly.
    for l in range(plan.L):
        freq = np.bincount(y_train[loc_train == l], minlength=Y)
        assert np.allclose(freq / freq.sum(), theta[l], atol=1e-12)
    header = [
        "Theta of the HARD variant: one label prior per location "
        "(tasks/hard_variant.md)",
        f"L = {ps.C} locations over Y = {ps.Y} classes; each prior is the "
        f"exact class frequency of its location's training images",
        f"locations 0..{K - 1} are the priors of {args.base_priors.name} "
        f"moved by TV <= {plan.tv_change.max():.4f}; location {K} is the "
        f"remainder",
        f"eps = {args.eps:g}, n_min = {args.n_min_location}, seed = {args.seed}",
        "p(theta) = 1/L uniform: every location contributes the same number "
        "of test batches",
        "TV to theta_tr below is against the POOLED training class frequency",
    ]
    for path in priors_files:
        write_prior_set(ps, path, header=header)

    # --- 3a. development batches ------------------------------------------
    dev_grid = tuple(m for m in grid
                     if m * POOL_SIZE_RATIO <= len(y_devpool))
    if dev_grid != grid:
        print(f"development grid truncated to {list(dev_grid)} (pool of "
              f"{len(y_devpool):,})")
    dev = draw_rows(y_devpool, theta, rng_dev, Y, grid=dev_grid,
                    n_min=args.dev_n_min, scale=args.dev_batch_scale,
                    balanced=True)
    dev_batch_id = DEV_ID_OFFSET + rng_ids.permutation(dev.n_batches)
    dev_id_test = dev_batch_id[dev.gen_batch]
    dev_row_id = DEV_ID_OFFSET + rng_ids.permutation(dev.n_rows)
    dev_order = np.lexsort((dev.slot, dev_id_test))
    dev_border = np.argsort(dev_batch_id)
    dev_source = idx_devpool[dev.pool_row[dev_order]]

    # --- 3b. test batches, usage by usage (as prepare_kaggle_data.py) ------
    rows_batch, rows_slot, rows_eval_idx, rows_usage = [], [], [], []
    batch_usage, batch_m, batch_theta, batch_dup = [], [], [], []
    next_batch = 0
    for u_i, usage in enumerate(USAGES):
        pool_idx = np.flatnonzero(pool_of_eval == u_i)
        rng = np.random.default_rng(usage_seeds[u_i])
        if args.balance_pools:
            pool_idx = _balance(pool_idx, y_eval[pool_idx], Y, rng)
        drawn = draw_rows(y_eval[pool_idx], theta, rng, Y, grid=grid,
                          n_min=args.n_min, scale=args.batch_scale,
                          balanced=True)
        rows_batch.append(drawn.gen_batch + next_batch)
        rows_slot.append(drawn.slot)
        rows_eval_idx.append(pool_idx[drawn.pool_row])
        rows_usage.append(np.full(drawn.n_rows, u_i, dtype=np.int8))
        batch_usage.append(np.full(drawn.n_batches, u_i, dtype=np.int8))
        batch_m.append(drawn.batch_m)
        batch_theta.append(drawn.batch_theta)
        batch_dup.append(drawn.batch_dup)
        next_batch += drawn.n_batches

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

    # Sequential ids would hand out the Usage partition; see prepare_kaggle_data.
    batch_public_id = rng_ids.permutation(n_batches)
    id_test = batch_public_id[gen_batch]
    row_id = rng_ids.permutation(n_rows)
    checks = _checks(id_test, slot, m_of_row, row_id, usage_of_row, eval_idx,
                     pool_of_eval, grid)
    order = np.lexsort((slot, id_test))
    border = np.argsort(batch_public_id)

    # --- 4. transformation --------------------------------------------------
    # Per row, not per source image: a test image drawn into two batches is
    # released as two different arrays (chal/transform.py).
    X_train_t, rot_train = transform(X_orig[idx_train], rng_train_tf,
                                     args.noise_std)
    X_dev_t, rot_dev = transform(X_orig[dev_source], rng_dev_tf, args.noise_std)
    X_test_t, rot_test = transform(X_eval[eval_idx[order]], rng_test_tf,
                                   args.noise_std)
    # A random token tying the model's outputs to this exact dataset.
    run_id = "".join(f"{v:08x}" for v in master.integers(0, 1 << 32, 4))

    np.savez(
        out_dir / "hard_data.npz",
        run_id=run_id, seed=args.seed, noise_std=args.noise_std,
        grid=np.array(grid), dev_grid=np.array(dev_grid),
        usages=np.array(USAGES), priors=theta, location_sizes=plan.sizes,
        pooled_prior=pooled,
        # training data, in original order; location is the released column
        train_images=X_train_t, train_label=y_train, train_location=loc_train,
        train_source=idx_train, train_rot=rot_train,
        # development rows, in public order: by id_test, then slot
        dev_images=X_dev_t, dev_row_id=dev_row_id[dev_order],
        dev_id_test=dev_id_test[dev_order], dev_slot=dev.slot[dev_order],
        dev_m=dev.m_of_row[dev_order], dev_label=y_orig[dev_source],
        dev_location=dev.batch_theta[dev.gen_batch][dev_order],
        dev_source=dev_source, dev_rot=rot_dev,
        dev_batch_id=dev_batch_id[dev_border],
        dev_batch_m=dev.batch_m[dev_border],
        dev_batch_location=dev.batch_theta[dev_border],
        dev_batch_dup=dev.batch_dup[dev_border],
        # test rows, in public order
        test_images=X_test_t, test_row_id=row_id[order],
        test_id_test=id_test[order], test_slot=slot[order],
        test_m=m_of_row[order], test_label=y_eval[eval_idx[order]],
        test_location=batch_theta[gen_batch][order],
        test_usage=usage_of_row[order], test_source=eval_idx[order],
        test_rot=rot_test,
        test_batch_id=batch_public_id[border], test_batch_m=batch_m[border],
        test_batch_location=batch_theta[border],
        test_batch_usage=batch_usage[border], test_batch_dup=batch_dup[border],
    )

    # --- report -------------------------------------------------------------
    cls = "  ".join(f"{c:>6}" for c in range(Y))
    lines = [
        "Hard variant: split, locations, batches (tasks/hard_variant.md)",
        "=" * 78,
        f"timestamp : {datetime.now().isoformat(timespec='seconds')}",
        f"command   : {' '.join(sys.argv)}",
        f"run_id    : {run_id}",
        "-" * 78,
        "split",
        f"  training data (released, 9 locations) : {len(y_train):,}",
        f"  development pool                      : {len(y_devpool):,}",
        f"  evaluation (val + test)               : {len(y_eval):,}",
        "-" * 78,
        f"locations: eps = {args.eps:g}, n_min = {args.n_min_location}, "
        f"LP delta = {plan.delta:.4f}",
        f"  {'loc':>3} {'size':>7} {'share':>6} {'TV moved':>9}  "
        f"class frequency {cls}",
    ] + [
        f"  {l:>3} {plan.sizes[l]:>7,} {plan.sizes[l] / len(y_train):>6.1%} "
        f"{(plan.tv_change[l] if l < K else float('nan')):>9.4f}  "
        f"{'':>15} " + "  ".join(f"{v:.4f}" for v in theta[l])
        for l in range(plan.L)
    ] + [
        "  pooled training class frequency           "
        + "  ".join(f"{v:.4f}" for v in pooled),
        f"  min pairwise TV between locations : {off.min():.4f}",
        f"  TV to pooled prior in [{ps.tv_to_train.min():.4f}, "
        f"{ps.tv_to_train.max():.4f}]",
        "-" * 78,
        "batches (each from one location; every location the same number)",
        f"  development : {dev.n_batches:,} batches, {dev.n_rows:,} rows, "
        f"sizes {list(dev_grid)}",
        f"  test        : {n_batches:,} batches, {n_rows:,} rows, "
        f"sizes {list(grid)}"
        + ("  (balanced pools)" if args.balance_pools else ""),
    ] + [
        f"    {u:<8}  {int((batch_usage == i).sum()):,} batches, "
        f"{int((usage_of_row == i).sum()):,} rows"
        for i, u in enumerate(USAGES)
    ] + [
        f"  test batches per location: "
        f"{np.bincount(batch_theta, minlength=plan.L).tolist()}",
        "-" * 78,
        f"transformation: rotation by 90/180/270 degrees, Gaussian noise "
        f"std {args.noise_std:g} grey levels, per released row",
        f"  checks: {', '.join(k for k, v in checks.items() if v is True)}",
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)
    print(f"wrote {out_dir}/hard_data.npz, {out_dir}/report.txt, "
          + ", ".join(str(p) for p in priors_files))


if __name__ == "__main__":
    main()
