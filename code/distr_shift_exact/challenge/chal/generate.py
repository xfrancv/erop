"""The one batch-generation path, shared by the Kaggle data and the dev benchmark.

``prepare_kaggle_data.py`` and ``make_dev_benchmark.py`` both call
:func:`draw_rows` and :func:`reference_and_base`. A divergence between the two
is the single most likely source of a silent scoring mismatch -- a student's
offline number not matching the leaderboard -- so they share the code rather
than agreeing by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import plugin_for_prior
from .protocol import BATCH_SCALE, N_MIN, SIZE_GRID, BatchSampler, check_grid, \
    generate_batches


@dataclass
class DrawnRows:
    """Row-level and batch-level arrays for one pool's worth of batches."""

    gen_batch: np.ndarray      # (n_rows,) generation index of the row's batch
    slot: np.ndarray           # (n_rows,) 0..m-1
    pool_row: np.ndarray       # (n_rows,) index into the pool that was passed in
    batch_m: np.ndarray        # (n_batches,)
    batch_theta: np.ndarray    # (n_batches,) index into Theta
    batch_dup: np.ndarray      # (n_batches,) fraction of repeated images

    @property
    def n_rows(self) -> int:
        return len(self.slot)

    @property
    def n_batches(self) -> int:
        return len(self.batch_m)

    @property
    def m_of_row(self) -> np.ndarray:
        return self.batch_m[self.gen_batch]


def draw_rows(y_pool: np.ndarray, theta: np.ndarray, rng: np.random.Generator,
              num_classes: int, grid=SIZE_GRID, n_min: int = N_MIN,
              scale: int = BATCH_SCALE) -> DrawnRows:
    """Draw every batch of one pool and flatten them into rows (C4).

    Slots are ``0..m-1`` in draw order, which is uniformly random because the
    ``m`` examples were drawn i.i.d. -- so any prefix of slots is a uniform
    subsample of the batch, which is what the optional constant-budget pooling
    of ``evaluate.py --budget`` relies on.
    """
    check_grid(len(y_pool), grid)
    sampler = BatchSampler(y_pool, num_classes)
    batches = generate_batches(sampler, theta, rng, grid=grid, n_min=n_min,
                               scale=scale)

    gen_batch = np.concatenate(
        [np.full(b.m, i, dtype=np.int64) for i, b in enumerate(batches)])
    slot = np.concatenate([np.arange(b.m, dtype=np.int64) for b in batches])
    pool_row = np.concatenate([b.idx for b in batches])
    return DrawnRows(
        gen_batch=gen_batch, slot=slot, pool_row=pool_row,
        batch_m=np.array([b.m for b in batches]),
        batch_theta=np.array([b.theta_star_index for b in batches]),
        batch_dup=np.array([b.dup_fraction for b in batches]))


def reference_and_base(log_post_rows: np.ndarray, log_train_prior: np.ndarray,
                       theta_of_row: np.ndarray):
    """The reference predictor and the non-adapted baseline, per row (C5).

    ``pred_ref = h(x, theta_*)`` is the plugin Bayes rule given the batch's
    **true** prior, evaluated on the same calibrated posterior students get.
    Shipping its output rather than ``theta_*`` keeps ``theta_*`` out of
    ``solution.csv`` entirely.

    The baseline is ``argmax_y p_tr(y | x)``, which is also the train-prior
    plugin: the factor ``theta_tr,y / p_tr(y)`` is constant in ``y``.
    """
    pred_ref, unc_ref = plugin_for_prior(log_post_rows, log_train_prior,
                                         theta_of_row)
    base_pred = log_post_rows.argmax(axis=1)
    base_conf = np.exp(log_post_rows[np.arange(len(base_pred)), base_pred])
    return pred_ref, unc_ref, base_pred, base_conf
