"""Generating the test batches: the grid, the batch counts, the sampler (C4).

One *batch* is ``m`` unlabeled images drawn i.i.d. from ``p_te(. | theta_*)``
for a single ``theta_*`` drawn uniformly from ``Theta``. Every image of every
batch is a submission row and every one is scored -- the parent README's
constant curve budget is **not** applied (C9.2), so the pooled ranking at size
``m`` has ``B_m = N(m) * m`` rows.

``m_max`` is 100, not the 500 of the parent S6.2 grid. On the trained base
model the two best rejectors are already separated by *exactly* zero at
``m >= 100`` -- the posterior over ``Theta`` has concentrated on ``theta_*`` and
both answer identically -- so ``m = 200`` and ``m = 500`` cost 77 % of all rows
and return no ranking information about the top of the leaderboard. ``m = 100``
is kept rather than cutting at 50 because weaker entries than the reference
rejectors are still separable there: a non-adapting predictor scores ~0.099 at
``m = 100`` against ~0.000 for anything that adapts.

``N_MIN`` is the load-bearing constant, not ``2000``. Rows inside one batch
share ``theta_*`` and the same adaptation evidence, so they are strongly
correlated: with intra-batch correlation ``rho`` the effective sample size is
``N(m) m / (1 + (m - 1) rho)``, capped at ``N(m) / rho`` however large ``m``
gets. Precision therefore tracks the number of *batches*. Dropping ``N_MIN``
to the unclipped ``ceil(2000 / 500) = 4`` at ``m = 500`` would make one ninth
of the final score almost pure noise.

Sampling **with replacement** is what makes a batch exactly i.i.d. from
``p_te(. | theta_*)`` and what stops a prior spiked on a rare class from
exhausting the pool. A query image may therefore recur inside its own batch;
that is not a leak here, because the challenge conditions on the whole batch
including the query anyway (C3.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE_GRID = (1, 2, 5, 10, 20, 50, 100)
N_MIN = 200
BATCH_SCALE = 2000          # N(m) = max(N_MIN, ceil(BATCH_SCALE / m))
# C4 / S6.2: m_max must not exceed a tenth of the pool it is drawn from, or
# duplicate contamination inside a batch stops being negligible.
POOL_SIZE_RATIO = 10


def n_batches(m: int, n_min: int = N_MIN, scale: int = BATCH_SCALE) -> int:
    """``N(m) = max(N_min, ceil(scale / m))`` (C4)."""
    assert m >= 1
    return int(max(n_min, -(-scale // m)))


def check_grid(pool_size: int, grid=SIZE_GRID) -> None:
    m_max = max(grid)
    assert m_max * POOL_SIZE_RATIO <= pool_size, (
        f"m_max = {m_max} exceeds |pool| / {POOL_SIZE_RATIO} = "
        f"{pool_size // POOL_SIZE_RATIO}; shrink the grid or enlarge the pool")


@dataclass
class Batch:
    """One test batch: which pool rows it drew, under which prior."""

    idx: np.ndarray            # (m,) indices into the *pool*
    theta_star_index: int
    m: int

    @property
    def dup_fraction(self) -> float:
        """Fraction of slots whose image occurs more than once in the batch."""
        _, counts = np.unique(self.idx, return_counts=True)
        return float((counts > 1).sum() / len(self.idx)) if len(self.idx) else 0.0


class BatchSampler:
    """Draws batches from one evaluation pool (C4, step 2).

    A class is drawn i.i.d. from ``theta_*`` and then an example of that class
    uniformly with replacement, which makes the batch exactly an i.i.d. sample
    from ``p_te(. | theta_*)`` under the pool's empirical class conditionals.
    """

    def __init__(self, y_pool: np.ndarray, num_classes: int):
        self.y_pool = np.asarray(y_pool)
        self.Y = num_classes
        order = np.argsort(self.y_pool, kind="stable")
        self.by_class = order
        self.counts = np.bincount(self.y_pool, minlength=num_classes)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])

    def sample(self, m: int, theta_star: np.ndarray, theta_star_index: int,
               rng: np.random.Generator) -> Batch:
        support = theta_star > 0
        assert np.all(self.counts[support] > 0), (
            "theta_* puts mass on a class with no examples in this pool")
        cls = rng.choice(self.Y, size=m, p=theta_star)
        pos = (rng.random(m) * self.counts[cls]).astype(np.int64)
        idx = self.by_class[self.offsets[cls] + pos]
        return Batch(idx=idx, theta_star_index=theta_star_index, m=m)


def generate_batches(sampler: BatchSampler, theta: np.ndarray,
                     rng: np.random.Generator, grid=SIZE_GRID,
                     n_min: int = N_MIN, scale: int = BATCH_SCALE
                     ) -> list[Batch]:
    """Every batch of one usage, in grid order.

    ``theta_*`` is drawn uniformly over ``Theta`` -- the same ``p(theta)`` the
    model is given, so the setting is well-specified by construction.
    """
    C = len(theta)
    out = []
    for m in grid:
        for _ in range(n_batches(m, n_min, scale)):
            c = int(rng.integers(C))
            out.append(sampler.sample(m, theta[c], c, rng))
    return out
