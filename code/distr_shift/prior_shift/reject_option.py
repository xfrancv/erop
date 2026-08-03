"""Reject-option (selective prediction) metrics for label-prior adaptation.

The computation layer shared by the reject-option experiment: the predictor
registry, the :class:`Aggregation` value that keeps the tables and figures
agreeing on one central statistic, the uncertainty decomposition, and the
selective risk/regret curves with their areas (AuRC, AuRC50, AuGRC).

Pure numpy and strings -- no plotting. The figure builders that consume these
arrays live in the top-level ``reject_figures`` module, the experiment driver
in ``rejopt_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# The reject-option predictors (order fixed for tables and figures). Both share
# one base predictor -- the Bayesian learned-prior rule -- and differ only in
# the uncertainty score they rank by.
REJECT_LABELS = {
    "bayes_total": "Bayesian, total uncertainty",
    "bayes_epistemic": "Bayesian, epistemic uncertainty",
}
REJECT_COLORS = {
    "bayes_total": "C1",
    "bayes_epistemic": "C2",
}


def accuracy(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(pred == y_true))


# Replicate-axis aggregation, shared by the report tables and every figure
# builder: how the solid line and the shaded band are computed from the
# replicate axis, and how titles describe them.
#
# This is an explicit value, passed to whoever formats a table or builds a
# figure -- not process-global state. The tables and the figures of one run must
# report the *same* central statistic, and threading one object through is what
# guarantees that; a module-level setting would make the result depend on the
# order in which the driver happened to configure and render.
@dataclass(frozen=True)
class Aggregation:
    """How to collapse the replicate axis into a line, a band and a title.

    ``band`` is the spread of the default (mean-centred) mode: ``"sem"`` gives
    ``std / sqrt(reps)``, ``"std"`` the plain std. ``desc`` is the title
    fragment, with a ``{reps}`` placeholder receiving the replicate count the
    figure was given, and ``noun`` the word for one replicate (used by the
    percentile-band title).

    ``pct`` switches to percentile mode (the ``--percentile-band`` flag): the
    line becomes the pointwise median and the band the central ``pct``%
    interval. It overrides ``band``/``desc``, but not the replicate axis.

    The default reproduces the fixed-prior convention (mean line, s.e.m. band
    over trials); the dirichlet mode builds one with ``band="std"`` over
    sampled priors via :meth:`over_priors`.
    """

    band: str = "sem"
    desc: str = "mean ± s.e.m., {reps} trials"
    noun: str = "trials"
    pct: float | None = None

    def __post_init__(self):
        if self.band not in ("sem", "std"):
            raise ValueError(f"band must be 'sem' or 'std', got {self.band!r}")
        if self.pct is not None and not 0 <= self.pct <= 100:
            raise ValueError(
                f"percentile band must be in [0, 100], got {self.pct}")

    @classmethod
    def over_priors(cls, n_priors: int, trials: int,
                    pct: float | None = None) -> "Aggregation":
        """The dirichlet-mode convention: the replicate axis is the sampled
        priors (the arrays hold per-prior means), so the band is ±1 std over
        those, and the title names both counts."""
        return cls(band="std",
                   desc=f"{n_priors}x{trials} runs, ±1 std over {{reps}} priors",
                   noun="priors", pct=pct)

    def with_percentile(self, pct: float | None) -> "Aggregation":
        """A copy in percentile mode (or back out of it, with ``None``)."""
        return replace(self, pct=pct)

    def series(self, arr: np.ndarray, axis: int, reps: int):
        """``(center, lower, upper)`` of the plotted curve along the replicate
        ``axis``. Default: the mean and mean ± (std or std/sqrt(reps)) per
        ``band``. Percentile mode: the pointwise median and the central
        ``[lo, hi]`` percentile interval -- in general asymmetric about the
        median, but always containing it."""
        if self.pct is None:
            center = arr.mean(axis=axis)
            s = arr.std(axis=axis)
            half = s / np.sqrt(reps) if self.band == "sem" else s
            return center, center - half, center + half
        lo, hi = (100 - self.pct) / 2, (100 + self.pct) / 2
        return (np.median(arr, axis=axis),
                np.percentile(arr, lo, axis=axis),
                np.percentile(arr, hi, axis=axis))

    def center(self, arr, axis=None):
        """Center statistic for the report tables, matching the figures' solid
        line: the mean over the replicate axis, or the pointwise median in
        percentile mode. Keeps the tables and figures reporting the same
        central value."""
        return (np.median if self.pct is not None else np.mean)(arr, axis=axis)

    def describe(self, reps: int) -> str:
        """Title fragment describing the aggregation, e.g. 'mean ± s.e.m., 10
        trials' or, in percentile mode, 'median, central 80% band, 10
        trials'."""
        if self.pct is not None:
            return f"median, central {self.pct:g}% band, {reps} {self.noun}"
        return self.desc.format(reps=reps)


def bayesian_posterior_and_aleatoric(
    train_posterior: np.ndarray,   # (n, Y) = p_tr(y | x)
    train_prior: np.ndarray,       # (Y,)
    alpha_samples: np.ndarray,     # (S, Y) posterior draws of the test prior
    loss: np.ndarray,              # (Y, Y) with loss[yhat, y]
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture label posterior and aleatoric uncertainty in one pass.

    Returns ``(bayes_post, aleatoric)`` where ``bayes_post`` is the (n, Y)
    posterior-averaged label distribution ``hat p(y | x, D)`` (identical to
    ``posterior_label_probabilities``) and ``aleatoric`` is the (n,) average
    over draws of the per-draw minimal conditional risk,

        A_hat(x, D) = (1/S) sum_s min_yhat sum_y p(y | x, alpha_s) loss[yhat, y].
    """
    R = train_posterior / train_prior[None, :]
    n, Y = R.shape
    acc_post = np.zeros((n, Y))
    acc_alea = np.zeros(n)
    for alpha in alpha_samples:
        w = R * alpha[None, :]
        w /= w.sum(axis=1, keepdims=True)
        acc_post += w
        acc_alea += (w @ loss.T).min(axis=1)
    S = len(alpha_samples)
    return acc_post / S, acc_alea / S


def epistemic_metrics(
    epi_u: np.ndarray,         # (n,) epistemic uncertainty per eval example
    losses_bayes: np.ndarray,  # (n,) losses of the Bayesian base predictor
    losses_ref: np.ndarray,    # (n,) losses of the true-prior reference
    threshold: float,
) -> tuple[float, float, float]:
    """(avg epistemic uncertainty, avg regret, portion with negligible epi).

    All three characterize the Bayesian learned-prior predictor ``h(x, D)``:
    they are identical for the total- and epistemic-uncertainty reject-option
    predictors (which differ only in their ranking score) and involve no
    selection, so they are reported once, not per predictor.
    """
    return (
        float(epi_u.mean()),
        float((losses_bayes - losses_ref).mean()),
        float((epi_u < threshold).mean()),
    )


def coverage_at_target(curve: np.ndarray, target: float, k_min: int = 10) -> float:
    """Largest coverage ``k/n`` such that ``curve(j) <= target`` for all
    ``j`` from ``k_min`` up to ``k`` (the dual of AuRC: how many predictions
    can be accepted while the selective metric stays within budget).

    The first ``k_min - 1`` ranks are a grace region excluded from the prefix
    condition: selective metrics at tiny ``k`` are 0/1-grained (one unlucky
    example makes ``curve(1) = 1``), which would zero the statistic in
    otherwise-clean trials.
    """
    n = len(curve)
    bad = np.flatnonzero(curve[k_min - 1:] > target)
    return 1.0 if bad.size == 0 else float(bad[0] + k_min - 1) / n


def selective_curves(
    losses_pred: np.ndarray,   # (n,) per-example loss of the evaluated predictor
    losses_ref: np.ndarray,    # (n,) per-example loss of the true-prior reference
    u: np.ndarray,             # (n,) uncertainty score used for ranking
) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk and regret at every rank k = 1..n.

    Examples are sorted by ascending ``u`` (stable, so ties keep input order);
    ``risk(k)`` / ``regret(k)`` average the first k losses / loss differences.
    """
    order = np.argsort(u, kind="stable")
    k = np.arange(1, len(u) + 1)
    risk = np.cumsum(losses_pred[order]) / k
    regret = np.cumsum(losses_pred[order] - losses_ref[order]) / k
    return risk, regret


def generalize_curve(curve: np.ndarray) -> np.ndarray:
    """A selective curve rescaled to its generalized counterpart.

    The selective metrics divide the prefix sum by the accepted count ``k``, the
    generalized ones by the evaluation-set size ``n``, so ``genrisk(k) =
    coverage(k) * risk(k)`` and likewise for the regret. Ranks run along the
    last axis, so a single curve or any stack of them (per-trial, per-size)
    works.

    The area under the generalized curve (AuGRC) weights the rank-i example by
    ``(n - i + 1) / n^2``, against ``sum_{k>=i} 1/k ~ ln(n/i)`` per ``n`` for the
    AuRC -- the latter inflates the top-ranked example's fair ``1/n`` share by a
    factor of ``ln n``, which is what makes the AuRC sensitive to the noisy
    low-coverage tail. Note the AuGRC weights sum to ``(n + 1) / 2n ~ 1/2``, so
    the two areas are on different scales and must not share an axis or a table
    column.
    """
    n = curve.shape[-1]
    return curve * (np.arange(1, n + 1) / n)


# Coverage floor of the truncated areas (AuRC50). A module constant, not a CLI
# flag: the figure filename derives from it (see ``make_trunc_sweep_figure``).
MIN_COVERAGE = 0.5

# Shared header note and caveat for the AuRC50 report blocks. The caveat goes
# under the numbers, where they are read.
AURC50_NOTE = (f"areas over coverage >= {MIN_COVERAGE:g} only; same scale as "
               f"the AuRC above")
AURC50_CAVEAT = (
    f"  note: invariant to the ranking within the accepted top "
    f"{round(100 * MIN_COVERAGE)}%, so gaps between\n"
    f"  predictors compress -- a smaller gap here does not mean the rankings "
    f"are more similar")


def truncated_area(curve: np.ndarray,
                   min_coverage: float = MIN_COVERAGE) -> np.ndarray:
    """Mean of a selective curve over the ranks with ``coverage >= min_coverage``.

    Ranks ``k = 1..n`` run along the last axis (as in ``generalize_curve``), so
    a single curve, a per-trial stack and a per-size stack all work; the rank
    axis is reduced. Retained are ``k0 = ceil(min_coverage * n)`` .. ``n``, so
    with ``min_coverage = 0`` this is the plain AuRC -- the areas stay on the
    AuRC scale ([0, 1] under 0/1 loss) and may share a table with it. Averaging
    over the window rather than integrating over it is what keeps that scale:
    the integral over a window of width ``1 - min_coverage`` would be smaller by
    exactly that factor.

    Dropping the low-coverage ranks drops where the selective metrics are
    noisiest (``risk(1)`` is one example, hence 0/1-grained) and where no
    deployment operates. As a rank statistic this weights example ``i`` by
    ``(sum_{k>=max(i, k0)} 1/k) / (n - k0 + 1)``, i.e. ``ln(1/min_coverage)``
    over the retained count -- a *constant* for every ``i <= k0`` -- decaying as
    ``ln(n/i)`` below that. The full AuRC instead weights rank 1 by ``ln n``
    times its fair ``1/n`` share: the sensitivity to the noisy tail that AuGRC
    attacks with linear weights and this attacks by truncation.

    The flat top-half weights are exactly an invariance: **the result does not
    change under any re-ordering within the first k0 ranks** (every retained
    ``risk(k)`` is a prefix mean containing all of them). Since the entries of
    ``REJECT_LABELS`` share one base predictor and differ only in their ranking
    score, their curves meet at coverage 1 and this truncation compresses the
    gaps between them -- it discards signal along with noise, so a smaller gap
    here is not evidence that the rankings agree more.
    """
    n = curve.shape[-1]
    k0 = min(n, max(1, int(np.ceil(min_coverage * n))))
    return curve[..., k0 - 1:].mean(axis=-1)


# ----------------------------------------------------------------------------
# Single-number summary of each sweep curve
# ----------------------------------------------------------------------------
# Each sweep table shows a measure as a curve over the adaptation-set size
# (one row per size in ``--sizes``). These helpers append one ``avg`` row that
# collapses each column to a single scalar: the mean of the per-size cells.
# Because every size carries the same replicate count, that equals the plain
# mean over the whole (size x replicate) array.
# See ``tasks/single_number_summary_polished.md`` -- and its caveats:
# the mean is over the *sampled* sizes, so it is only comparable across runs
# with the same ``--sizes`` and must not be read as a substitute for the curve.
SWEEP_AVG_LABEL = "avg"


def sweep_avg_row(data: dict, names, decimals: int, agg: Aggregation,
                  warn=None, scale: float = 1.0) -> str:
    """Format the ``avg`` row for a per-predictor sweep table (AuRC, AuRC50,
    AuGRC, coverage-at-target): ``agg``'s center statistic over sizes and
    replicates of each predictor column, in the table's own field width (24)
    and ``decimals``. ``warn`` (the AuRC tables' per-size warn array) adds the
    matching mean-warn cell when given. ``scale`` multiplies each centered
    value before formatting (the area tables report on a ``x1000`` scale); it
    must match the per-size rows' scaling."""
    row = f"{SWEEP_AVG_LABEL:>8}"
    if warn is not None:
        row += f"{float(np.mean(warn)):>8.2f}"
    row += "".join(f"{float(agg.center(data[name])) * scale:>24.{decimals}f}"
                   for name in names)
    return row


def sweep_epi_avg_row(epi_metrics: np.ndarray, agg: Aggregation) -> str:
    """Format the ``avg`` row for the epistemic-uncertainty metrics table, whose
    columns are the three fixed metrics (not per predictor). ``epi_metrics`` is
    (sizes, replicates, 3); each column is collapsed with ``agg``'s center
    statistic over sizes and replicates."""
    return (f"{SWEEP_AVG_LABEL:>8}{agg.center(epi_metrics[:, :, 0]):>14.4f}"
            f"{agg.center(epi_metrics[:, :, 1]):>14.4f}"
            f"{agg.center(epi_metrics[:, :, 2]):>14.3f}")
