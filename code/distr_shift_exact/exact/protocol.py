"""The evaluation protocol of README S6: grid, trials, pooling, budgets.

One *trial* is one labeled test set ``D'`` of ``m + 1`` examples drawn i.i.d.
from ``p_te(. | theta_*)``; it yields ``m + 1`` triplets ``(x, D, y)`` by letting
each example in turn be the query and the other ``m`` be the adaptation set
(S6.3). Because the ``m + 1`` examples are drawn **with replacement** from the
evaluation split, the test set is exactly i.i.d. -- which is what the model in
S2 assumes and what keeps spiked priors on rare classes from exhausting the
pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# S6.2. m_max is capped per dataset by the evaluation-split size; see
# ``resolve_grid``.
SIZE_GRID = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
# S6.2 asserts m_max <= |eval| / EVAL_SIZE_RATIO.
EVAL_SIZE_RATIO = 10

# S6.4 pooled ranking budget and the trial-count clip.
BUDGET_B = 2000
N_MIN = 50
N_MAX = 2000


def n_trials(m: int, budget: int = BUDGET_B,
             n_min: int = N_MIN, n_max: int = N_MAX) -> int:
    """``N(m) = clip(ceil(B / (m+1)), N_min, N_max)`` (S6.4)."""
    return int(np.clip(int(np.ceil(budget / (m + 1))), n_min, n_max))


def subsample_layout(m: int, n_trials_: int, budget: int = BUDGET_B
                     ) -> tuple[int, int]:
    """How many triplets each trial contributes to the pooled ranking.

    S6.4 asks for the ``B`` pooled triplets to be split evenly across trials,
    which is only exactly possible when ``N(m)`` divides ``B``. Returns
    ``(per_trial, remainder)``: ``remainder`` trials contribute one extra
    triplet each, and the bootstrap re-draws *which* trials those are on every
    replicate so the choice cannot bias a curve.

    Every trial's ``m + 1`` triplets are already in a uniformly random order
    (the ``m + 1`` examples were drawn i.i.d.), so "take the first ``k`` slots"
    is itself a uniform subsample -- no extra randomisation is needed.
    """
    n = m + 1
    pooled = n_trials_ * n
    if pooled <= budget:
        return n, 0
    per_trial, remainder = divmod(budget, n_trials_)
    assert per_trial + (1 if remainder else 0) <= n
    return per_trial, remainder


def resolve_grid(eval_size: int, grid=SIZE_GRID, strict: bool = False
                 ) -> tuple[tuple[int, ...], str | None]:
    """Apply the S6.2 assert ``m_max <= |eval| / 10``.

    The assert is a bound on *duplicate contamination* (S6.3), and on the
    smaller evaluation splits it genuinely bites: DermaMNIST's val+test is about
    3000 examples, which caps ``m_max`` at 300 and drops the last grid point.
    Rather than refuse to run, the grid is truncated and the caller reports it;
    ``strict=True`` turns it back into a hard failure.
    """
    m_allowed = eval_size // EVAL_SIZE_RATIO
    kept = tuple(m for m in grid if m <= m_allowed)
    if len(kept) == len(grid):
        return kept, None
    note = (f"adaptation grid truncated to m <= {m_allowed} "
            f"(|eval| = {eval_size}, S6.2 requires m_max <= |eval|/"
            f"{EVAL_SIZE_RATIO}); dropped "
            f"{[m for m in grid if m > m_allowed]}")
    if strict:
        raise AssertionError(note)
    assert kept, f"evaluation split of {eval_size} is too small for any m"
    return kept, note


@dataclass
class Trial:
    """One test set ``D'``: which eval examples it drew, and under which prior."""

    idx: np.ndarray            # (m+1,) indices into the evaluation split
    y: np.ndarray              # (m+1,) their labels
    theta_star: np.ndarray     # (Y,) the true test prior of this trial
    theta_star_index: int      # index in Theta, or -1 when theta_* is off-grid
    dup_fraction: float        # fraction of triplets whose query recurs in D^(i)
    dup_mean: float            # mean number of such recurrences per triplet


class TrialSampler:
    """Draws test sets from the evaluation split (S6.3, step 2).

    Classes are drawn i.i.d. from ``theta_*`` and then an example of that class
    is drawn uniformly with replacement, which makes ``D'`` exactly an i.i.d.
    sample from ``p_te(. | theta_*)`` where ``p_te(x | y)`` is the empirical
    class-conditional of the evaluation split.
    """

    def __init__(self, y_eval: np.ndarray, num_classes: int):
        self.y_eval = np.asarray(y_eval)
        self.Y = num_classes
        order = np.argsort(self.y_eval, kind="stable")
        counts = np.bincount(self.y_eval, minlength=num_classes)
        self.by_class = order
        self.counts = counts
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def sample(self, m: int, theta_star: np.ndarray, theta_star_index: int,
               rng: np.random.Generator) -> Trial:
        support = theta_star > 0
        assert np.all(self.counts[support] > 0), (
            "theta_* puts mass on a class with no evaluation examples")
        n = m + 1
        cls = rng.choice(self.Y, size=n, p=theta_star)
        pos = (rng.random(n) * self.counts[cls]).astype(np.int64)
        idx = self.by_class[self.offsets[cls] + pos]

        # S6.3's "known residual leak": with replacement the query can recur
        # inside its own D^(i), since leave-one-out removes only index i.
        if n > 1:
            _, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
            extra = cnt[inv] - 1
            dup_fraction = float((extra > 0).mean())
            dup_mean = float(extra.mean())
        else:
            dup_fraction = dup_mean = 0.0

        return Trial(idx=idx, y=self.y_eval[idx], theta_star=theta_star,
                     theta_star_index=theta_star_index,
                     dup_fraction=dup_fraction, dup_mean=dup_mean)


class ThetaStarDrawer:
    """How ``theta_*`` is drawn for a stratum (S6.4, Appendix A.1).

    ``mode``:

    ``"shifted"``
        uniform over ``Theta \\ {theta_1}`` -- the main panels' convention.
        S6.4 states it as *conditioning* the pooled trials on
        ``theta_* != theta_1``; drawing conditionally is the same distribution
        but keeps ``N(m)`` and the budget ``B`` exactly at their nominal values,
        which post-hoc filtering would not.
    ``"marginal"``
        uniform over all of ``Theta`` -- the supplementary unstratified panel,
        and the only mode in which ``theta_*`` and the model's ``p(theta)``
        agree (Appendix A.5).
    ``"fixed"``
        always ``Theta[index]`` -- the per-``theta_*`` supplementary breakdown.
    ``"dirichlet"``
        ``theta_* ~ Dir(s * theta_tr)``, so ``theta_* not in Theta`` almost
        surely while the model still uses ``Theta``: the misspecified second arm
        of Appendix A.1, without which the epistemic-calibration claim is not
        falsifiable.
    """

    def __init__(self, theta: np.ndarray, mode: str = "shifted",
                 index: int = 0, train_prior: np.ndarray | None = None,
                 concentration: float = 20.0):
        self.theta = theta
        self.mode = mode
        self.index = index
        self.train_prior = train_prior
        self.concentration = concentration
        if mode == "shifted":
            assert len(theta) >= 2, (
                "mode 'shifted' needs at least two priors in Theta")
            self.choices = np.arange(1, len(theta))
        elif mode == "marginal":
            self.choices = np.arange(len(theta))
        elif mode == "fixed":
            self.choices = np.array([index])
        elif mode == "dirichlet":
            assert train_prior is not None
            self.choices = None
        else:
            raise ValueError(f"unknown theta_* mode: {mode}")

    @property
    def label(self) -> str:
        if self.mode == "fixed":
            return f"theta_*=theta[{self.index}]"
        if self.mode == "dirichlet":
            return f"dirichlet(s={self.concentration:g})"
        return self.mode

    def draw(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        if self.mode == "dirichlet":
            return rng.dirichlet(self.concentration * self.train_prior), -1
        c = int(rng.choice(self.choices))
        return self.theta[c], c
