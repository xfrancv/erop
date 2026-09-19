#!/usr/bin/env python3
"""How well can the released test images be matched back to TissueMNIST?

The hard variant publishes pixels, and TissueMNIST's labels are public. Every
released image is rotated by 90/180/270 degrees and given light Gaussian noise
(``chal/transform.py``); the rules forbid trying to recover the labels. This
script measures what the transformation alone buys, from the attacker's side:

1. **exact** -- undo each of the 4 rotations and look the bytes up in a hash of
   the public validation + test images. This is the attack that recovered every
   test label of an unprotected release in seconds.
2. **nearest neighbour** -- undo each rotation and take the public image at the
   smallest Euclidean distance. The match is counted as found when that image is
   the true source; the label is recovered when its label is right.

``--sweep`` re-applies the transformation at other noise levels to the same
sample of source images, so the trade-off can be read off before regenerating
the data: the noise has to stay small enough not to hurt a classifier (check
with ``train_base_model.py --hard-data``), and nearest-neighbour matching
survives far more noise than that.

    python hard_audit_matching.py out/hard/data
    python hard_audit_matching.py out/hard/data --n 2000 --sweep 0 2 8 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chal import data as chdata
from chal.transform import transform


def nearest(Q: np.ndarray, R: np.ndarray, R_sq: np.ndarray,
            chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Index of and squared distance to the nearest row of ``R`` per query."""
    idx = np.empty(len(Q), dtype=np.int64)
    dist = np.empty(len(Q))
    for a in range(0, len(Q), chunk):
        q = Q[a:a + chunk]
        d2 = (q * q).sum(1)[:, None] + R_sq[None, :] - 2.0 * q @ R.T
        idx[a:a + chunk] = d2.argmin(axis=1)
        dist[a:a + chunk] = d2[np.arange(len(q)), idx[a:a + chunk]]
    return idx, dist


def attack(X_rows: np.ndarray, X_pub: np.ndarray, pub_hash: dict,
           R: np.ndarray, R_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(exact_hit_index or -1, nearest_index)`` for each released row."""
    n = len(X_rows)
    exact = np.full(n, -1, dtype=np.int64)
    best = np.full(n, np.inf)
    nn = np.zeros(n, dtype=np.int64)
    for r in range(4):
        U = np.rot90(X_rows, r, axes=(1, 2))
        for i in range(n):
            j = pub_hash.get(U[i].tobytes())
            if j is not None and exact[i] < 0:
                exact[i] = j
        idx, d = nearest(U.reshape(n, -1).astype(np.float32), R, R_sq)
        better = d < best
        best[better], nn[better] = d[better], idx[better]
    return exact, nn


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="output of hard_make_data.py")
    p.add_argument("--data-root", type=Path, default=chdata.DATA_ROOT)
    p.add_argument("--n", type=int, default=1000,
                   help="test rows to attack (default 1000)")
    p.add_argument("--sweep", type=float, nargs="*", default=[],
                   help="also re-transform the same sources at these noise "
                        "standard deviations (grey levels)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    hd = np.load(args.data_dir / "hard_data.npz")
    ds = chdata.load(args.data_root)
    X_pub = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
    y_pub = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
    pub_hash = {x.tobytes(): i for i, x in enumerate(X_pub)}
    R = X_pub.reshape(len(X_pub), -1).astype(np.float32)
    R_sq = (R * R).sum(1)

    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(hd["test_label"]), size=min(args.n, len(hd["test_label"])),
                      replace=False)
    src = hd["test_source"][take]
    label = hd["test_label"][take]

    print(f"{len(take):,} released test rows attacked against the "
          f"{len(X_pub):,} public val + test images")
    print(f"{'noise std':>10} {'exact':>8} {'NN found':>9} {'NN label':>9}   "
          f"(label accuracy by chance of the majority class: "
          f"{np.bincount(y_pub).max() / len(y_pub):.3f})")

    def report(tag: str, X_rows: np.ndarray) -> None:
        exact, nn = attack(X_rows, X_pub, pub_hash, R, R_sq)
        print(f"{tag:>10} {float((exact >= 0).mean()):>8.3f} "
              f"{float((nn == src).mean()):>9.3f} "
              f"{float((y_pub[nn] == label).mean()):>9.3f}")

    report(f"{float(hd['noise_std']):g} (rel.)", hd["test_images"][take])
    for std in args.sweep:
        X_rows, _k = transform(X_pub[src], np.random.default_rng(args.seed + 1),
                               std)
        report(f"{std:g}", X_rows)
    print("\n'exact'    : undoing a rotation reproduces a public image byte for "
          "byte\n'NN found' : the nearest public image is the true source\n"
          "'NN label' : the nearest public image has the right label -- the "
          "leak itself")


if __name__ == "__main__":
    main()
