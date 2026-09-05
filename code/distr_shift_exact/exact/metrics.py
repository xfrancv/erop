"""Selective risk / regret curves, their summary scalars, and the bootstrap.

README S4 defines the metrics on a **single pooled ranking** of ``B`` triplets
per adaptation size: rank by ascending uncertainty, accept the ``k`` least
uncertain, and read off

    coverage(k) = k / B
    risk(k)     = mean loss of the accepted triplets
    regret(k)   = the same, measured against the true-prior plugin

with ``AuRC``/``AuRegC`` their averages over ``k`` and ``Reg@c`` the regret at a
fixed coverage. S6.4 fixes the resampling unit for the confidence intervals:
the **trial**, not the triplet -- triplets from one test set share ``m - 1``
adaptation points and are strongly dependent, so bootstrapping them directly
would badly understate the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_COVERAGE = 0.8
BOOTSTRAP_REPS = 1000
CI_ALPHA = 0.05
# Replicates per vectorised chunk; caps peak memory of the bootstrap.
_CHUNK = 250
# Number of coverage points retained for the stored risk-coverage curve.
CURVE_POINTS = 200


# S3's table, in order. ``optional`` marks the sixth row S3 lists as optional
# (score = A(x, D)), which completes the T = A + E decomposition; it is always
# computed and reported, but kept out of the default S6.5 panels.
@dataclass(frozen=True)
class Rejector:
    key: str
    label: str
    optional: bool = False


REJECTORS: tuple[Rejector, ...] = (
    Rejector("bayes_total", "Bayesian, total"),
    Rejector("bayes_epistemic", "Bayesian, epistemic"),
    Rejector("map_plugin", "MAP plugin"),
    Rejector("train_plugin", "train-prior plugin"),
    Rejector("true_plugin", "true-prior plugin (oracle)"),
    Rejector("bayes_aleatoric", "Bayesian, aleatoric", optional=True),
)
REJECTOR_KEYS = tuple(r.key for r in REJECTORS)
ORACLE_KEY = "true_plugin"


@dataclass
class Pool:
    """The pooled triplets of one (stratum, m) cell, laid out per trial.

    ``loss``/``score``/``tiebreak`` are ``(n_rejectors, N, n)``; ``oracle_loss``
    is ``(N, n)``. Keeping the trial axis separate is what lets the bootstrap
    resample trials rather than triplets.
    """

    loss: np.ndarray
    score: np.ndarray
    tiebreak: np.ndarray
    oracle_loss: np.ndarray
    m: int
    per_trial: int
    remainder: int
    keys: tuple[str, ...] = REJECTOR_KEYS

    @property
    def n_trials(self) -> int:
        return self.oracle_loss.shape[0]

    @property
    def budget(self) -> int:
        return self.per_trial * self.n_trials + self.remainder


def _pick_indices(trial_ids: np.ndarray, n: int, per_trial: int,
                  remainder: int) -> np.ndarray:
    """Flat indices of the pooled subsample, one row per replicate.

    ``trial_ids`` is ``(R, N)``. Each selected trial contributes its first
    ``per_trial`` slots; the first ``remainder`` columns -- themselves a random
    selection, since ``trial_ids`` is drawn at random -- contribute one more.
    Slot order inside a trial is already uniformly random (S6.3 draws the
    ``m + 1`` examples i.i.d.), so a prefix is a uniform subsample.
    """
    base = (trial_ids[:, :, None] * n + np.arange(per_trial)[None, None, :])
    base = base.reshape(len(trial_ids), -1)
    if remainder:
        extra = trial_ids[:, :remainder] * n + per_trial
        base = np.concatenate([base, extra], axis=1)
    return base


def _curves(loss: np.ndarray, oracle: np.ndarray, score: np.ndarray,
            tiebreak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk and regret curves for ``(R, B)`` inputs.

    Ties in the score are broken by ``tiebreak`` -- which for the epistemic
    rejector is the total uncertainty, as S3 requires: ``E`` is exactly zero
    whenever every prior in ``Theta`` votes for the same label, which is common
    enough that the tie-break governs a large part of the ranking.
    """
    order = np.lexsort((tiebreak, score), axis=-1)
    k = np.arange(1, loss.shape[1] + 1)[None, :]
    risk = np.cumsum(np.take_along_axis(loss, order, axis=1), axis=1) / k
    diff = np.take_along_axis(loss - oracle, order, axis=1)
    regret = np.cumsum(diff, axis=1) / k
    return risk, regret


@dataclass
class CellResult:
    """Point estimates and bootstrap intervals for one (stratum, m) cell."""

    m: int
    n_trials: int
    budget: int
    scalars: dict[str, dict[str, float]] = field(default_factory=dict)
    ci: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    coverage_grid: list[float] = field(default_factory=list)


def _scalars_from_curves(risk: np.ndarray, regret: np.ndarray,
                         coverage: float) -> dict[str, np.ndarray]:
    B = risk.shape[1]
    k_at_c = max(1, int(np.ceil(coverage * B)))
    return {
        "aurc": risk.mean(axis=1),
        "auregc": regret.mean(axis=1),
        "regret_at_c": regret[:, k_at_c - 1],
        "risk_at_c": risk[:, k_at_c - 1],
        "accuracy_full": 1.0 - risk[:, -1],
        "regret_full": regret[:, -1],
    }


def evaluate_pool(pool: Pool, coverage: float = DEFAULT_COVERAGE,
                  reps: int = BOOTSTRAP_REPS, alpha: float = CI_ALPHA,
                  rng: np.random.Generator | None = None,
                  keep_curves: bool = True) -> CellResult:
    """Point estimate plus percentile bootstrap CIs for every rejector.

    The replicates are **paired**: every rejector is evaluated on the same
    resampled trials, so two rejectors' bands are directly comparable and the
    per-replicate difference between them is meaningful. Drawing a fresh
    resample per rejector would leave each interval valid on its own but make
    any comparison between them look noisier than it is.
    """
    rng = rng or np.random.default_rng(0)
    N, n = pool.oracle_loss.shape
    per_trial, remainder = pool.per_trial, pool.remainder
    B = pool.budget

    res = CellResult(m=pool.m, n_trials=N, budget=B)
    grid_k = np.unique(np.linspace(1, B, min(CURVE_POINTS, B)).astype(int)) - 1
    res.coverage_grid = [float((i + 1) / B) for i in grid_k]

    oracle_flat = pool.oracle_loss.reshape(-1)
    flat = [(pool.loss[r].reshape(-1), pool.score[r].reshape(-1),
             pool.tiebreak[r].reshape(-1)) for r in range(len(pool.keys))]

    def scalars_for(idx):
        """All rejectors' scalars on one (R, B) index matrix."""
        out = []
        for loss_f, score_f, tb_f in flat:
            risk, regret = _curves(loss_f[idx], oracle_flat[idx],
                                   score_f[idx], tb_f[idx])
            out.append((risk, regret, _scalars_from_curves(risk, regret, coverage)))
        return out

    point_idx = _pick_indices(np.arange(N)[None, :], n, per_trial, remainder)
    for key, (risk, regret, sc) in zip(pool.keys, scalars_for(point_idx)):
        res.scalars[key] = {k: float(v[0]) for k, v in sc.items()}
        if keep_curves:
            res.curves[key] = {"risk": [float(v) for v in risk[0, grid_k]],
                               "regret": [float(v) for v in regret[0, grid_k]]}

    if reps <= 0:
        return res

    draws = {key: {k: [] for k in res.scalars[key]} for key in pool.keys}
    done = 0
    while done < reps:
        R = min(_CHUNK, reps - done)
        # One set of resampled trials, shared by every rejector in this chunk.
        trial_ids = rng.integers(0, N, size=(R, N))
        idx = _pick_indices(trial_ids, n, per_trial, remainder)
        for key, (_, _, sc) in zip(pool.keys, scalars_for(idx)):
            for k, v in sc.items():
                draws[key][k].append(v)
        done += R

    res.ci = {
        key: {k: (float(np.quantile(np.concatenate(v), alpha / 2)),
                  float(np.quantile(np.concatenate(v), 1 - alpha / 2)))
              for k, v in per_key.items()}
        for key, per_key in draws.items()}
    return res
