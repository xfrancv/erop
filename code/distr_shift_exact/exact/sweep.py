"""The experiment's inner loop: trials -> pooled triplets -> metrics.

One call to :func:`run_cell` covers every trial at one adaptation size ``m`` for
one ``theta_*`` stratum and returns the pooled arrays S4 ranks. The base model
is never re-run here: the calibrated ``log p_tr(y | x)`` of the whole evaluation
split is computed once by the caller, and a trial is just a row-gather.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inference import infer_trial, plugin_for_prior
from .metrics import REJECTOR_KEYS, Pool
from .protocol import ThetaStarDrawer, TrialSampler, n_trials, subsample_layout


@dataclass
class CellDiagnostics:
    """Per-cell sanity numbers that S6.3 and S7 ask to be logged, not hidden."""

    dup_fraction: float = 0.0      # triplets whose query recurs inside its D
    dup_mean: float = 0.0          # mean number of such recurrences
    map_correct: float = float("nan")   # P(theta_map == theta_*)
    post_true: float = float("nan")     # mean p(theta_* | D)
    mean_total: float = 0.0
    mean_aleatoric: float = 0.0
    mean_epistemic: float = 0.0
    theta_star_counts: dict[int, int] = field(default_factory=dict)


def run_cell(m: int, drawer: ThetaStarDrawer, sampler: TrialSampler,
             log_post_eval: np.ndarray, log_train_prior: np.ndarray,
             train_prior: np.ndarray, log_theta: np.ndarray,
             log_p_theta: np.ndarray, rng: np.random.Generator,
             budget: int, n_min: int, n_max: int
             ) -> tuple[Pool, CellDiagnostics]:
    """Run every trial of one (stratum, m) cell and pool their triplets."""
    N = n_trials(m, budget, n_min, n_max)
    n = m + 1
    R = len(REJECTOR_KEYS)

    loss = np.empty((R, N, n))
    score = np.empty((R, N, n))
    tiebreak = np.empty((R, N, n))
    oracle_loss = np.empty((N, n))

    diag = CellDiagnostics()
    dup_f, dup_m, map_ok, post_true = [], [], [], []
    tot, ale, epi = [], [], []

    for j in range(N):
        theta_star, star_idx = drawer.draw(rng)
        trial = sampler.sample(m, theta_star, star_idx, rng)
        lp = log_post_eval[trial.idx]                       # (n, Y)
        y = trial.y

        inf = infer_trial(lp, log_train_prior, log_theta, log_p_theta)
        tr_pred, tr_unc = plugin_for_prior(lp, log_train_prior, train_prior)
        st_pred, st_unc = plugin_for_prior(lp, log_train_prior, theta_star)

        ell_star = (st_pred != y).astype(float)
        oracle_loss[j] = ell_star

        rows = [
            # (prediction, score, tie-break) for each row of the S3 table.
            (inf.bayes_pred, inf.total, inf.total),
            (inf.bayes_pred, inf.epistemic, inf.total),
            (inf.map_pred, inf.map_unc, inf.map_unc),
            (tr_pred, tr_unc, tr_unc),
            (st_pred, st_unc, st_unc),
            (inf.bayes_pred, inf.aleatoric, inf.total),
        ]
        for r, (pred, sc, tb) in enumerate(rows):
            loss[r, j] = (pred != y).astype(float)
            score[r, j] = sc
            tiebreak[r, j] = tb

        dup_f.append(trial.dup_fraction)
        dup_m.append(trial.dup_mean)
        tot.append(inf.total.mean())
        ale.append(inf.aleatoric.mean())
        epi.append(inf.epistemic.mean())
        if star_idx >= 0:
            map_ok.append(float((inf.map_index == star_idx).mean()))
            post_true.append(float(inf.pth_adapt[:, star_idx].mean()))
            diag.theta_star_counts[star_idx] = \
                diag.theta_star_counts.get(star_idx, 0) + 1

    diag.dup_fraction = float(np.mean(dup_f))
    diag.dup_mean = float(np.mean(dup_m))
    diag.mean_total = float(np.mean(tot))
    diag.mean_aleatoric = float(np.mean(ale))
    diag.mean_epistemic = float(np.mean(epi))
    if map_ok:
        diag.map_correct = float(np.mean(map_ok))
        diag.post_true = float(np.mean(post_true))

    per_trial, remainder = subsample_layout(m, N, budget)
    pool = Pool(loss=loss, score=score, tiebreak=tiebreak,
                oracle_loss=oracle_loss, m=m,
                per_trial=per_trial, remainder=remainder)
    return pool, diag
