"""Figure builders for the reject-option experiment.

Every builder turns the arrays produced by ``prior_shift.reject_option`` into a
declarative ``figspec.FigureSpec`` and writes it (PNG plus a ``.figspec.json``
sidecar that ``render_figspecs.py`` can re-render). Kept out of the
``prior_shift`` package so the library stays free of the plotting layer.

Each builder takes the run's :class:`~prior_shift.reject_option.Aggregation` as
an explicit ``agg`` argument -- the same object the driver hands to the report
tables, so a figure's solid line and its table cell are the same statistic by
construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import figspec
from prior_shift.reject_option import (
    MIN_COVERAGE,
    REJECT_COLORS,
    REJECT_LABELS,
    Aggregation,
    generalize_curve,
)


def make_curves_at_n_figure(
    risk_curves: dict, regret_curves: dict, n_test: int, out_dir: str,
    agg: Aggregation,
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
            center, lo, hi = agg.series(curves[name], 0, trials)
            area = curves[name].mean(axis=1)     # per-trial area
            series.append(figspec.Series(
                x=coverage, center=center, lower=lo, upper=hi,
                color=REJECT_COLORS[name],
                label=f"{REJECT_LABELS[name]}  "
                      f"({area_label} {agg.center(area):.4f} ± {area.std():.4f})"))
        panels.append(figspec.Panel(
            series=series,
            hlines=[figspec.HLine(0.0)] if is_regret else [],
            xlabel="coverage", ylabel=metric,
            title=f"{metric.capitalize()}-coverage curve, "
                  f"$n_\\mathrm{{test}}$ = {n_test}",
            legend=True, legend_loc="upper left"))

    spec = figspec.FigureSpec(
        panels=panels, nrows=1, ncols=2, figsize=[13.0, 5.0],
        suptitle=f"{suptitle_prefix} at n_test = {n_test} ({agg.describe(trials)})",
        tight_rect=[0.0, 0.0, 1.0, 0.94])
    sub = Path(out_dir) / "coverage_curves"
    sub.mkdir(parents=True, exist_ok=True)
    return figspec.write(spec, str(sub / f"{fname_prefix}_n{n_test}.png"))


def make_gen_curves_at_n_figure(
    gen_risk_curves: dict, gen_regret_curves: dict, n_test: int, out_dir: str,
    agg: Aggregation,
) -> str:
    """Generalized counterpart of ``make_curves_at_n_figure``."""
    return make_curves_at_n_figure(
        gen_risk_curves, gen_regret_curves, n_test, out_dir, agg,
        metrics=("generalized risk", "generalized regret"),
        area_label="AuGRC",
        fname_prefix="gen_coverage_curves",
        suptitle_prefix="Generalized coverage curves")


def sweep_panels(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict, trials: int,
    agg: Aggregation,
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
        f"{m} vs. test-set size ({agg.describe(trials)})" for m in metrics)
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
            for c, lo, hi in [agg.series(aurc[name], 1, trials)]]
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
    trials: int, out_dir: str, agg: Aggregation,
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
    panels = sweep_panels(sizes, aurc_risk, aurc_regret, trials, agg,
                          metrics=metrics, ylabels=ylabels, xlabel=xlabel,
                          titles=titles)
    spec = figspec.FigureSpec(panels=panels, nrows=1, ncols=2,
                              figsize=list(figsize) if figsize else None)
    figspec.write(spec, f"{out_dir}/{fname}.png")


def make_gen_sweep_figure(
    sizes: list[int], augrc_risk: dict, augrc_regret: dict,
    trials: int, out_dir: str, agg: Aggregation,
) -> None:
    """Generalized counterpart of ``make_sweep_figure``."""
    make_sweep_figure(
        sizes, augrc_risk, augrc_regret, trials, out_dir, agg,
        metrics=("AuGRC (generalized risk)", "AuGRC (generalized regret)"),
        fname="gen_aurc_vs_n_test")


def make_trunc_sweep_figure(
    sizes: list[int], aurc50_risk: dict, aurc50_regret: dict,
    trials: int, out_dir: str, agg: Aggregation,
) -> None:
    """Truncated counterpart of ``make_sweep_figure`` (see ``truncated_area``).

    Unlike the AuGRC these areas are on the AuRC scale, but they still get their
    own figure: the point is to read them against each other, not against areas
    taken over a different coverage window.
    """
    pct = round(100 * MIN_COVERAGE)
    make_sweep_figure(
        sizes, aurc50_risk, aurc50_regret, trials, out_dir, agg,
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
    agg: Aggregation,
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
        center, lo, hi = agg.series(epi_metrics[:, :, col], 1, trials)
        p1_series.append(figspec.Series(
            x=x, center=center, lower=lo, upper=hi, marker="o",
            color=color, label=label))
    panel1 = figspec.Panel(
        series=p1_series, hlines=[figspec.HLine(0.0)],
        xscale="log", xticks=list(sizes), xlabel=xlabel,
        ylabel="0/1-loss units", legend=True, grid_which="both")

    # ---- Panel 2: portion with negligible epistemic uncertainty ----
    center, lo, hi = agg.series(epi_metrics[:, :, 2], 1, trials)
    panel2 = figspec.Panel(
        series=[figspec.Series(x=x, center=center, lower=lo, upper=hi,
                               marker="o", color="C0")],
        xscale="log", xticks=list(sizes), xlabel=xlabel, ylim=[-0.02, 1.02],
        ylabel=f"portion with epistemic uncertainty < {threshold:g}",
        grid_which="both")

    spec = figspec.FigureSpec(
        panels=[panel1, panel2], nrows=1, ncols=2, figsize=[12.0, 4.6],
        suptitle="Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"({agg.describe(trials)})")
    figspec.write(spec, f"{out_dir}/epistemic_metrics_vs_n_test.png")


def make_cov_target_figure(
    sizes: list[int], cov_regret: list[dict], trials: int,
    regret_descs: list[str], out_dir: str, agg: Aggregation,
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
            for center, lo, hi in [agg.series(covs[name], 1, trials)]]
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
        suptitle=f"Coverage at target vs. test-set size ({agg.describe(trials)})",
        tight_rect=[0.0, 0.0, 1.0, 0.94])
    figspec.write(spec, f"{out_dir}/cov_at_target_vs_n_test.png")


def base_accuracy_panel(
    sizes: list[int], base_acc: dict, trials: int, agg: Aggregation,
    xlabel: str = "number of unlabeled adaptation examples $n$",
    title: str | None = None,
) -> figspec.Panel:
    """The base-predictor-accuracy-vs-size panel as a ``figspec.Panel``.

    Three predictors: Bayesian learned prior, plugin with the true (target)
    prior, and plugin with the unadapted training prior. Each ``base_acc``
    entry is a (len(sizes), trials) array. The two plugins ignore the
    adaptation examples, so with a fixed evaluation set they are flat in n;
    under ``--eval-on-adapt`` they vary with n through the evaluation sample
    itself. Split out as its own
    ``figspec.Panel`` so ``make_base_accuracy_figure`` can write it as an
    independent single-panel figure."""
    x = np.asarray(sizes, dtype=float)
    style = (
        ("bayes_learned", "Bayesian, learned prior", "C1", "o", "-"),
        ("plugin_train", "Plugin, training prior (no adaptation)", "C4", "s", "-"),
        ("plugin_true", "Plugin, true test prior (oracle)", "C0", None, "--"),
    )
    series = []
    for key, label, color, marker, ls in style:
        center, lo, hi = agg.series(base_acc[key], 1, trials)
        series.append(figspec.Series(
            x=x, center=center, lower=lo, upper=hi, marker=marker,
            linestyle=ls, color=color, label=label))

    if title is None:
        title = f"Base-predictor accuracy vs. test-set size ({agg.describe(trials)})"
    return figspec.Panel(
        series=series, xscale="log", xticks=list(sizes),
        xlabel=xlabel, ylabel="test accuracy", title=title,
        legend=True, legend_loc="lower right", grid_which="both")


def make_base_accuracy_figure(
    sizes: list[int], base_acc: dict, trials: int, out_dir: str,
    agg: Aggregation,
) -> None:
    """Base-predictor accuracy vs. the adaptation-set size (one-panel figure)."""
    panel = base_accuracy_panel(sizes, base_acc, trials, agg)
    spec = figspec.FigureSpec(panels=[panel], figsize=[8.5, 5.0])
    figspec.write(spec, f"{out_dir}/base_accuracy_vs_n_test.png")


def make_sweep_area_figures(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict,
    reps: int, out_dir: str, short_name: str, agg: Aggregation,
) -> None:
    """The AuRC (risk) and AuReC (regret) sweep panels as two *independent*
    single-panel figures, ``aurc_vs_n_test.png`` and ``aurec_vs_n_test.png``.

    Written separately (rather than combined, or bundled with the accuracy panel
    into one overview figure) so each curve drops into a paper on its own. The
    accuracy panel is its own figure via ``make_base_accuracy_figure``.
    ``figsize=None`` so the size follows the render style sheet."""
    xlabel = "number of unlabeled adaptation examples $m$"
    risk_panel, regret_panel = sweep_panels(
        sizes, aurc_risk, aurc_regret, reps, agg, xlabel=xlabel,
        titles=(f"{short_name}: AuRC vs. adaptation set size $m$",
                f"{short_name}: AuReC vs. adaptation set size $m$"))
    for panel, fname in ((risk_panel, "aurc_vs_n_test"),
                         (regret_panel, "aurec_vs_n_test")):
        figspec.write(figspec.FigureSpec(panels=[panel], figsize=None),
                      f"{out_dir}/{fname}.png")


def make_epi_regret_calibration_figure(sizes, epi_by_size, out_dir) -> None:
    """Average epistemic uncertainty vs. average realized regret of the
    Bayesian predictor: one point per (sampled prior, adaptation size), each
    the pair of means over that draw's trials, with the identity line. Under a
    well-specified prior the points concentrate on the diagonal; a
    misspecified model prior (``--beta``) or heavy truncation pushes them off
    it. ``epi_by_size`` is (len(sizes), N, 3) per-prior means with columns
    (avg epi, avg regret, portion negligible)."""
    import matplotlib.pyplot as plt

    epi_by_size = np.asarray(epi_by_size)
    S, N, _ = epi_by_size.shape
    cmap = plt.get_cmap("viridis")
    series = []
    for i, n in enumerate(sizes):
        series.append(figspec.Series(
            kind="scatter", x=epi_by_size[i, :, 0], center=epi_by_size[i, :, 1],
            size=36, color=list(cmap(i / max(S - 1, 1))), alpha=0.85,
            edgecolors="none", zorder=3, label=f"n = {n}"))
    vals = epi_by_size[:, :, :2]
    lo = min(0.0, float(vals.min()))
    hi = float(vals.max()) * 1.05 + 1e-9
    series.append(figspec.Series(
        x=np.array([lo, hi]), center=np.array([lo, hi]), color="0.4",
        linestyle="--", linewidth=1.0, label="y = x (calibrated)"))

    panel = figspec.Panel(
        series=series,
        xlabel="average epistemic uncertainty (per sampled prior)",
        ylabel="average realized regret at full coverage (per sampled prior)",
        title="Epistemic-uncertainty calibration of the Bayesian "
              f"predictor\none point per (sampled prior, n); {N} priors",
        legend=True)
    spec = figspec.FigureSpec(panels=[panel], figsize=[6.6, 6.0])
    figspec.write(spec, f"{out_dir}/epi_vs_regret_calibration.png")


def write_sweep_figures(sizes, res, reps: int, out_dir: str,
                        agg: Aggregation, regret_targets, epi_threshold: float,
                        display_name: str = "") -> None:
    """Write every figure of one sweep.

    The fan-out the driver calls once: the two area sweeps, the generalized and
    truncated sweeps, the epistemic-metrics and coverage-at-target figures, the
    base-accuracy figure, and the per-size coverage-curve pair. ``res`` is the
    :class:`~prior_shift.sweep.SweepResult`; ``reps`` the replicate count of its
    second axis, matching ``agg``.

    The generalized curves are derived here rather than carried in ``res``: they
    are a rescaling of the selective curves by the coverage (no re-ranking), and
    nothing but these figures needs them.
    """
    names = list(REJECT_LABELS.keys())
    gen_risk = {n: [generalize_curve(c) for c in res.risk_curves[n]]
                for n in names}
    gen_regret = {n: [generalize_curve(c) for c in res.regret_curves[n]]
                  for n in names}

    # aurc_vs_n_test + aurec_vs_n_test: the AuRC (risk) and AuReC (regret)
    # sweeps as two independent single-panel figures (dataset-named titles,
    # "$m$" for the adaptation-set size, figsize=None so the size follows the
    # style sheet).
    short_name = display_name.split(" (")[0] if display_name else "real data"
    make_sweep_area_figures(sizes, res.aurc_risk, res.aurc_regret, reps,
                            out_dir, short_name, agg)
    make_gen_sweep_figure(sizes, res.augrc_risk, res.augrc_regret, reps,
                          out_dir, agg)
    make_trunc_sweep_figure(sizes, res.aurc50_risk, res.aurc50_regret, reps,
                            out_dir, agg)
    make_epistemic_metrics_figure(sizes, res.epi_metrics, epi_threshold,
                                  out_dir, agg)
    make_cov_target_figure(sizes, res.cov_regret, reps,
                           [f"{e:g}" for e in regret_targets], out_dir, agg)
    make_base_accuracy_figure(sizes, res.base_acc, reps, out_dir, agg)
    for i, n in enumerate(sizes):
        make_curves_at_n_figure(
            {name: res.risk_curves[name][i] for name in names},
            {name: res.regret_curves[name][i] for name in names},
            n, out_dir, agg)
        make_gen_curves_at_n_figure(
            {name: gen_risk[name][i] for name in names},
            {name: gen_regret[name][i] for name in names},
            n, out_dir, agg)
