"""Figure builders for the reject-option experiment.

Every builder turns the arrays produced by ``prior_shift.reject_option`` into a
declarative ``figspec.FigureSpec`` and writes it (PNG plus a ``.figspec.json``
sidecar that ``render_figspecs.py`` can re-render). Kept out of the
``prior_shift`` package so the library stays free of the plotting layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import figspec
from prior_shift.reject_option import (
    MIN_COVERAGE,
    REJECT_COLORS,
    REJECT_LABELS,
    _agg_desc,
    _center,
    _series,
)


def make_curves_at_n_figure(
    risk_curves: dict, regret_curves: dict, n_test: int, out_dir: str,
    metrics: tuple[str, str] = ("selective risk", "selective regret"),
    area_label: str = "AuRC",
    fname_prefix: str = "coverage_curves",
    suptitle_prefix: str = "Coverage curves",
) -> str:
    """Side-by-side risk- and regret-coverage curves for one adaptation-set
    size of a sweep. ``risk_curves`` / ``regret_curves`` map predictor name to
    a (trials, n_eval) array of per-trial curves at that size. The size
    appears in the panel titles and in the file name; the written path is
    returned.

    ``metrics``, ``area_label``, ``fname_prefix`` and ``suptitle_prefix`` select
    the flavour: the defaults draw the selective curves;
    ``make_gen_curves_at_n_figure`` passes the generalized ones.
    """
    trials, n_eval = next(iter(risk_curves.values())).shape
    coverage = np.arange(1, n_eval + 1) / n_eval

    panels = []
    for curves, metric, is_regret in (
        (risk_curves, metrics[0], False),
        (regret_curves, metrics[1], True),
    ):
        series = []
        for name in REJECT_LABELS:
            center, lo, hi = _series(curves[name], 0, trials)
            area = curves[name].mean(axis=1)     # per-trial area
            series.append(figspec.Series(
                x=coverage, center=center, lower=lo, upper=hi,
                color=REJECT_COLORS[name],
                label=f"{REJECT_LABELS[name]}  "
                      f"({area_label} {_center(area):.4f} ± {area.std():.4f})"))
        panels.append(figspec.Panel(
            series=series,
            hlines=[figspec.HLine(0.0)] if is_regret else [],
            xlabel="coverage", ylabel=metric,
            title=f"{metric.capitalize()}-coverage curve, "
                  f"$n_\\mathrm{{test}}$ = {n_test}",
            legend=True, legend_loc="upper left"))

    spec = figspec.FigureSpec(
        panels=panels, nrows=1, ncols=2, figsize=[13.0, 5.0],
        suptitle=f"{suptitle_prefix} at n_test = {n_test} ({_agg_desc(trials)})",
        tight_rect=[0.0, 0.0, 1.0, 0.94])
    sub = Path(out_dir) / "coverage_curves"
    sub.mkdir(parents=True, exist_ok=True)
    return figspec.write(spec, str(sub / f"{fname_prefix}_n{n_test}.png"))


def make_gen_curves_at_n_figure(
    gen_risk_curves: dict, gen_regret_curves: dict, n_test: int, out_dir: str,
) -> str:
    """Generalized counterpart of ``make_curves_at_n_figure``."""
    return make_curves_at_n_figure(
        gen_risk_curves, gen_regret_curves, n_test, out_dir,
        metrics=("generalized risk", "generalized regret"),
        area_label="AuGRC",
        fname_prefix="gen_coverage_curves",
        suptitle_prefix="Generalized coverage curves")


def sweep_panels(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict, trials: int,
    metrics: tuple[str, str] = ("AuRC (selective risk)",
                                "AuRC (selective regret)"),
    ylabels: tuple[str, str] | None = None,
    xlabel: str = "number of unlabeled adaptation examples $n$",
    titles: tuple[str, str] | None = None,
) -> list:
    """The two AuRC-vs-size panels (risk, regret) as a list of ``figspec.Panel``.

    Split out of ``make_sweep_figure`` so the same panels can be embedded in a
    multi-figure layout (e.g. the real-data accuracy + AuRC overview) instead of
    being written to their own file. See ``make_sweep_figure`` for the argument
    meanings."""
    ylabels = ylabels or metrics
    titles = titles or tuple(
        f"{m} vs. test-set size ({_agg_desc(trials)})" for m in metrics)
    x = np.asarray(sizes, dtype=float)

    panels = []
    for aurc, title, ylabel, is_regret in (
        (aurc_risk, titles[0], ylabels[0], False),
        (aurc_regret, titles[1], ylabels[1], True),
    ):
        series = [
            figspec.Series(
                x=x, center=c, lower=lo, upper=hi, marker="o",
                color=REJECT_COLORS[name], label=REJECT_LABELS[name])
            for name in REJECT_LABELS
            for c, lo, hi in [_series(aurc[name], 1, trials)]]
        panels.append(figspec.Panel(
            series=series,
            hlines=[figspec.HLine(0.0)] if is_regret else [],
            xscale="log", xticks=list(sizes),
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            legend=True, grid_which="both"))
    return panels


def make_sweep_figure(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict,
    trials: int, out_dir: str,
    metrics: tuple[str, str] = ("AuRC (selective risk)",
                                "AuRC (selective regret)"),
    fname: str = "aurc_vs_n_test",
    ylabels: tuple[str, str] | None = None,
    xlabel: str = "number of unlabeled adaptation examples $n$",
    titles: tuple[str, str] | None = None,
    figsize: tuple[float, float] | None = (13.0, 5.0),
) -> None:
    """Area under the coverage curves vs. the adaptation-set size.

    ``metrics`` and ``fname`` select the flavour: the defaults plot the AuRC of
    the selective curves; ``make_gen_sweep_figure`` plots the AuGRC of the
    generalized ones. The two areas are on different scales (the AuGRC weights
    sum to ~1/2), so they get separate figures rather than shared axes.

    ``ylabels`` defaults to ``metrics`` and overrides just the y-axis text, for
    flavours whose full name is too long for the title (the truncated one spells
    its coverage window out there, and keeps a short name for the title).

    ``xlabel``, ``titles`` (a (risk, regret) pair) and ``figsize`` let a caller
    restyle the figure -- the real-data driver passes a dataset-named title and
    ``figsize=None`` so the size defers to the active style sheet. The defaults
    reproduce the historical synthetic look.
    """
    panels = sweep_panels(sizes, aurc_risk, aurc_regret, trials,
                          metrics=metrics, ylabels=ylabels, xlabel=xlabel,
                          titles=titles)
    spec = figspec.FigureSpec(panels=panels, nrows=1, ncols=2,
                              figsize=list(figsize) if figsize else None)
    figspec.write(spec, f"{out_dir}/{fname}.png")


def make_gen_sweep_figure(
    sizes: list[int], augrc_risk: dict, augrc_regret: dict,
    trials: int, out_dir: str,
) -> None:
    """Generalized counterpart of ``make_sweep_figure``."""
    make_sweep_figure(
        sizes, augrc_risk, augrc_regret, trials, out_dir,
        metrics=("AuGRC (generalized risk)", "AuGRC (generalized regret)"),
        fname="gen_aurc_vs_n_test")


def make_trunc_sweep_figure(
    sizes: list[int], aurc50_risk: dict, aurc50_regret: dict,
    trials: int, out_dir: str,
) -> None:
    """Truncated counterpart of ``make_sweep_figure`` (see ``truncated_area``).

    Unlike the AuGRC these areas are on the AuRC scale, but they still get their
    own figure: the point is to read them against each other, not against areas
    taken over a different coverage window.
    """
    pct = round(100 * MIN_COVERAGE)
    make_sweep_figure(
        sizes, aurc50_risk, aurc50_regret, trials, out_dir,
        metrics=(f"AuRC{pct} (selective risk)",
                 f"AuRC{pct} (selective regret)"),
        ylabels=(f"AuRC (selective risk, coverage >= {MIN_COVERAGE:g})",
                 f"AuRC (selective regret, coverage >= {MIN_COVERAGE:g})"),
        fname=trunc_sweep_fname())


def trunc_sweep_fname() -> str:
    """Filename stem of the truncated-area sweep figure, derived from
    ``MIN_COVERAGE`` so it cannot go stale (``aurc50_vs_n_test`` at 0.5)."""
    return f"aurc{round(100 * MIN_COVERAGE)}_vs_n_test"


def make_epistemic_metrics_figure(
    sizes: list[int], epi_metrics: np.ndarray, threshold: float, out_dir: str,
) -> None:
    """Two panels vs. n. Panel 1 overlays the average regret and average
    epistemic uncertainty (both 0/1-loss units) as two lines on one axes;
    panel 2 shows the portion with negligible epistemic uncertainty.

    ``epi_metrics`` is (len(sizes), trials, 3) with columns (avg epistemic
    uncertainty, avg regret, portion below ``threshold``). All three are
    properties of the Bayesian learned-prior predictor.
    """
    x = np.asarray(sizes, dtype=float)
    trials = epi_metrics.shape[1]
    xlabel = "number of unlabeled adaptation examples $n$"

    # ---- Panel 1: regret and epistemic uncertainty overlaid ----
    p1_series = []
    for col, label, color in (
        (1, "average regret (full coverage)", "C1"),
        (0, "average epistemic uncertainty", "C2"),
    ):
        center, lo, hi = _series(epi_metrics[:, :, col], 1, trials)
        p1_series.append(figspec.Series(
            x=x, center=center, lower=lo, upper=hi, marker="o",
            color=color, label=label))
    panel1 = figspec.Panel(
        series=p1_series, hlines=[figspec.HLine(0.0)],
        xscale="log", xticks=list(sizes), xlabel=xlabel,
        ylabel="0/1-loss units", legend=True, grid_which="both")

    # ---- Panel 2: portion with negligible epistemic uncertainty ----
    center, lo, hi = _series(epi_metrics[:, :, 2], 1, trials)
    panel2 = figspec.Panel(
        series=[figspec.Series(x=x, center=center, lower=lo, upper=hi,
                               marker="o", color="C0")],
        xscale="log", xticks=list(sizes), xlabel=xlabel, ylim=[-0.02, 1.02],
        ylabel=f"portion with epistemic uncertainty < {threshold:g}",
        grid_which="both")

    spec = figspec.FigureSpec(
        panels=[panel1, panel2], nrows=1, ncols=2, figsize=[12.0, 4.6],
        suptitle="Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"({_agg_desc(trials)})")
    figspec.write(spec, f"{out_dir}/epistemic_metrics_vs_n_test.png")


def make_cov_target_figure(
    sizes: list[int], cov_regret: list[dict], trials: int,
    regret_descs: list[str], out_dir: str,
) -> None:
    """Coverage-at-target vs. n: one panel per regret budget.

    ``cov_regret`` is a list (one entry per budget) of dicts mapping predictor
    name to a (len(sizes), trials) coverage array. Coverage at a *risk* budget
    was dropped with ``--risk-target``; the file name is unchanged so the
    figspec sidecars keep their paths.
    """
    x = np.asarray(sizes, dtype=float)
    ncols = len(cov_regret)

    panels = []
    for c, covs in enumerate(cov_regret):
        series = [
            figspec.Series(
                x=x, center=center, lower=lo, upper=hi, marker="o",
                color=REJECT_COLORS[name], label=REJECT_LABELS[name])
            for name in REJECT_LABELS
            for center, lo, hi in [_series(covs[name], 1, trials)]]
        panels.append(figspec.Panel(
            series=series, ylim=[-0.02, 1.05], xscale="log",
            xticks=list(sizes),
            xlabel="number of unlabeled adaptation examples $n$",
            ylabel="coverage at target",
            title=f"coverage @ regret <= {regret_descs[c]}",
            legend=True, legend_loc="lower right", grid_which="both"))

    spec = figspec.FigureSpec(
        panels=panels, nrows=1, ncols=ncols,
        figsize=[5.8 * ncols, 4.8],
        suptitle=f"Coverage at target vs. test-set size ({_agg_desc(trials)})",
        tight_rect=[0.0, 0.0, 1.0, 0.94])
    figspec.write(spec, f"{out_dir}/cov_at_target_vs_n_test.png")
