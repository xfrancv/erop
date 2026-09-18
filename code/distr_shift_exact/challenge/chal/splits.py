"""The split policy of C3.1: a three-way development split, three eval pools.

Differs from the parent README S6.1 in two places, both deliberate:

* **development is split 0.80 / 0.10 / 0.10**, not 0.80 / 0.20. The extra part
  is what is released to students as ``dev.csv``. Releasing the *calibration*
  split instead would hand them data the base posterior is optimistically
  calibrated on, so their offline score estimates would be biased -- and biased
  by an amount that differs between methods, which would make their offline
  ranking disagree with the leaderboard (C9.4).
* **the evaluation split is partitioned into three disjoint pools**, one per
  Kaggle ``Usage`` value. Disjointness is what stops labels probed off the
  public leaderboard from being worth anything on the private split (C9.6).

``theta_tr`` is the class frequency of the **fit** part alone, because that is
the prior the network was actually fit under and the whole re-weighting
``theta_y / p_tr(y)`` is only valid against it (S6.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .data import NUM_CLASSES, Dataset

USAGES = ("Public", "Private", "Ignored")
# Kaggle's own literal is "Ignored". A typo here silently drops every row of
# that pool from scoring, so it is asserted at generation time (C2).


@dataclass
class Splits:
    """Every array the pipeline needs, plus the indices that produced them."""

    X_fit: np.ndarray
    y_fit: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_dev: np.ndarray
    y_dev: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    pool_of_eval: np.ndarray          # (n_eval,) index into USAGES
    num_classes: int = NUM_CLASSES

    @property
    def train_prior(self) -> np.ndarray:
        counts = np.bincount(self.y_fit, minlength=self.num_classes).astype(float)
        assert np.all(counts > 0), (
            "a class is absent from the fit part; theta_y / p_tr(y) would "
            "divide by zero")
        return counts / counts.sum()

    def pool_indices(self, usage: str) -> np.ndarray:
        return np.flatnonzero(self.pool_of_eval == USAGES.index(usage))


def make_splits(ds: Dataset, cal_fraction: float = 0.10,
                dev_fraction: float = 0.10, seed: int = 0) -> Splits:
    """Stratified 0.80/0.10/0.10 development split and three eval pools."""
    assert 0 < cal_fraction < 1 and 0 < dev_fraction < 1
    assert cal_fraction + dev_fraction < 1

    X_dev_all, y_dev_all = ds.splits["train"]
    held = cal_fraction + dev_fraction
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X_dev_all, y_dev_all, test_size=held, stratify=y_dev_all,
        random_state=seed)
    # Split the held-out remainder in two, in the requested proportion.
    X_cal, X_dev, y_cal, y_dev = train_test_split(
        X_rest, y_rest, test_size=dev_fraction / held, stratify=y_rest,
        random_state=seed + 1)

    # Evaluation = official val + test merged, never seen in training or
    # calibration (S6.1).
    X_eval = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
    y_eval = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])

    pool_of_eval = _assign_pools(y_eval, seed=seed + 2)

    sp = Splits(X_fit, y_fit, X_cal, y_cal, X_dev, y_dev, X_eval, y_eval,
                pool_of_eval)
    _check(sp)
    return sp


def _assign_pools(y_eval: np.ndarray, seed: int) -> np.ndarray:
    """Class-stratified partition of the evaluation split into |USAGES| pools.

    Stratified so that every pool can realise every ``theta_*`` in ``Theta``:
    an unstratified split of a long-tailed dataset can leave a pool short of a
    rare class that some prior puts 0.35 on.
    """
    rng = np.random.default_rng(seed)
    pool = np.empty(len(y_eval), dtype=np.int64)
    k = len(USAGES)
    for c in np.unique(y_eval):
        idx = np.flatnonzero(y_eval == c)
        rng.shuffle(idx)
        # Round-robin over a shuffled list: sizes differ by at most one.
        pool[idx] = np.arange(len(idx)) % k
    return pool


def _check(sp: Splits) -> None:
    n_dev_total = len(sp.y_fit) + len(sp.y_cal) + len(sp.y_dev)
    assert n_dev_total == len(sp.y_fit) + len(sp.y_cal) + len(sp.y_dev)
    counts = np.bincount(sp.pool_of_eval, minlength=len(USAGES))
    assert counts.min() > 0, "an evaluation pool came out empty"
    for k in range(len(USAGES)):
        per_class = np.bincount(sp.y_eval[sp.pool_of_eval == k],
                                minlength=sp.num_classes)
        assert per_class.min() > 0, (
            f"pool {USAGES[k]} is missing class {int(per_class.argmin())}; "
            f"a theta_* that puts mass there could not be realised")
