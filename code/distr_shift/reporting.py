"""Everything the reject-option experiment writes as text or JSON.

Two outputs, from the same numbers:

* the **human-facing report** ``real_reject_option_sweep_report.txt`` -- the
  header lines describing how the run was configured, then the fixed-width
  metric tables built by :func:`sweep_tables`;
* the **machine-readable** ``results.json`` built by :func:`build_results`,
  which ``summary_table.py`` reads. Keeping the two apart is what lets the
  report be reworded freely: nothing downstream parses it.

Both take the run's :class:`~prior_shift.reject_option.Aggregation`, so a table
cell and the matching figure's solid line are the same statistic by
construction. Kept out of the ``prior_shift`` package: this is presentation, not
method.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from prior_shift.reject_option import (
    AURC50_CAVEAT,
    AURC50_NOTE,
    REJECT_LABELS,
    Aggregation,
    sweep_avg_row,
    sweep_epi_avg_row,
)
from prior_shift.sweep import SweepResult

REPORT_FILENAME = "real_reject_option_sweep_report.txt"


def save_run_args(args, filename: str, extra: dict | None = None,
                  ignored: set[str] | None = None) -> Path:
    """Write the run's argument setting to ``args.out_dir / filename``.

    ``extra`` holds values resolved after parsing (the target prior and how it
    was obtained, the resolved evaluation size) that are not argparse arguments.
    ``ignored`` names the arguments the run does not read; they are omitted
    rather than recorded at their (unused) defaults, which would invite a reader
    to reconstruct the run wrongly.
    """
    fields = {k: v for k, v in vars(args).items() if k not in (ignored or ())}
    width = max(len(k) for k in (*fields, *(extra or ()))) + 1

    path = Path(args.out_dir) / filename
    lines = [
        f"timestamp : {datetime.now().isoformat(timespec='seconds')}",
        f"command   : {' '.join(sys.argv)}",
        "",
        "[arguments]",
    ]
    lines += [f"{k:<{width}}: {v}" for k, v in sorted(fields.items())]
    if extra:
        lines += ["", "[resolved]"]
        lines += [f"{k:<{width}}: {v}" for k, v in extra.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


# Same threshold as base_predictor_training.CALIB_RATIO_THRESHOLD.
CALIB_RATIO_THRESHOLD = 1.5


def calibration_lines(bundle, class_names) -> list[str]:
    """Report lines describing the base model's calibration and its
    marginal-consistency check (computed at training time on held-out
    train-distribution data — it cannot be recomputed here, where the pool is
    shifted by design)."""
    T = bundle["temperature"]
    bias = bundle.get("calib_bias")
    mode = bundle.get("calibration",
                      "temperature" if bias is None else "bcts")
    line = f"calibration  : {mode}, T={T:.3f}"
    if bias is not None and np.any(bias):
        line += f", bias in [{np.min(bias):+.2f}, {np.max(bias):+.2f}]"
    lines = [line]
    ratio = bundle.get("marginal_ratio")
    if ratio is None:
        lines.append("calib check  : unavailable (bundle predates the "
                     "consistency check; retrain with base_predictor_training.py)")
        return lines
    ratio = np.asarray(ratio, dtype=float)
    lines.append(f"calib check  : mean posterior / class frequency = "
                 f"{np.array2string(ratio, precision=2)}")
    bad = [class_names[i] for i in np.flatnonzero(
        (ratio > CALIB_RATIO_THRESHOLD) | (ratio < 1 / CALIB_RATIO_THRESHOLD))]
    if bad:
        lines.append("!!! CALIBRATION WARNING: ratio outside "
                     f"[1/{CALIB_RATIO_THRESHOLD:g}, {CALIB_RATIO_THRESHOLD:g}]"
                     " for: " + ", ".join(bad))
        lines.append("    the label-shift MCMC reads this bias as prior "
                     "shift; the learned prior cannot be trusted.")
    return lines


def confusable_report_line(pair_idx, class_names, source: str) -> str:
    """Report line naming the dataset's confusable pair.

    Reporting only: the pair is a property of the dataset (the registry's
    designated hard pair, echoed by the per-draw lines in dirichlet mode) and
    plays no part in building the target prior -- ``target_prior_report_line``
    says how that was obtained.
    """
    if pair_idx is None:
        return "confusable   : none (the dataset registry names no pair)"
    i, j = pair_idx
    return (f"confusable   : {class_names[i]} / {class_names[j]}  "
            f"(classes {i}, {j}; {source})")


# How the target prior was obtained, spelled out because the default is
# degenerate on purpose: with the training prior as target there is no label
# shift, so the adaptive methods and the no-adaptation baseline coincide up to
# posterior noise. That must be readable off the report rather than inferred
# from the numbers.
TRAIN_PRIOR_DEFAULT_HOW = ("training prior (DEFAULT: no label shift -- "
                           "degenerate control)")


TRAIN_PRIOR_DEFAULT_WARNING = (
    "warning: the target prior defaults to the TRAINING prior, i.e. no label "
    "shift;\n         pass --test-prior or "
    "--prior-classes/--prior-weights/--prior-rest-weight\n         for a "
    "shifted target.")


def target_prior_report_line(how: str, dirichlet_mode: bool) -> str:
    """Report line saying how the target prior was obtained. In dirichlet mode
    the configured prior is the *centre* of the sampling distribution, not the
    target of any single run, so the line says so."""
    role = "central prior for Dir(s * p), " if dirichlet_mode else ""
    return f"target prior : {role}{how}"


def class_weights_how(classes, weights, rest_weight, class_names) -> str:
    """The ``how`` fragment of ``target_prior_report_line`` for a target prior
    built from class weights."""
    named = ", ".join(f"{class_names[c]}={w:g}"
                      for c, w in zip(classes, weights))
    if rest_weight is None or len(classes) == len(class_names):
        return f"class weights  {named} (all classes named)"
    return f"class weights  {named}, rest={rest_weight:g} (per class)"


def eval_set_report_field(args) -> str:
    """The report header's evaluation-set field: a size for the disjoint split,
    a statement of identity under ``--eval-on-adapt`` (where the evaluation size
    is the swept size, so no single number describes it)."""
    return ("eval set = adaptation set" if args.eval_on_adapt
            else f"n_eval {args.n_eval}")


# The area-under-curve report tables (AuRC / AuRC50 / AuGRC, risk and regret)
# print on a x1000 scale so more significant digits survive the fixed decimal
# width. Applies to the text tables only -- figures and the raw arrays (hence
# the win-rate comparisons, which are scale-invariant) are untouched.
AREA_SCALE = 1000


AREA_SCALE_TAG = "x1000"


WIN_RATE_REF = "bayes_total"


def win_rates(aurc: dict, sizes: list[int], names: list[str],
              ref: str = WIN_RATE_REF) -> dict:
    """Paired head-to-head win rates of ``ref`` against each competitor.

    Dirichlet-mode only. ``aurc[name]`` is the ``(len(sizes), N)`` array of
    per-prior mean areas of one measure (AuRC / AuRC50 / AuGRC, risk or regret
    -- the same quantity the corresponding table centres). A cell is the
    percentage of the ``N`` sampled priors on which ``ref`` has the strictly
    lower (better) area, at one adaptation size or, under the ``"all"`` key,
    pooled over every (size, prior) pair.

    Returned as ``{row label: {competitor: percentage}}`` with row labels
    ``str(size)`` and ``"all"``; ``ref`` never appears as a competitor (no
    self-comparison). The single source of these numbers: ``win_rate_block``
    formats this for the text report, and the results file records it verbatim.
    """
    def pct(win_arr) -> float:
        return 100.0 * float(np.mean(win_arr))

    rows = {str(n): {name: pct(aurc[ref][i] < aurc[name][i])
                     for name in names if name != ref}
            for i, n in enumerate(sizes)}
    rows["all"] = {name: pct(aurc[ref] < aurc[name])
                   for name in names if name != ref}
    return rows


def win_rate_block(metric: str, aurc: dict, sizes: list[int], names: list[str],
                   ref: str = WIN_RATE_REF, measure: str = "AuRC") -> list[str]:
    """Report lines rendering :func:`win_rates` as a fixed-width table.

    The ``ref`` column is left blank (no self-comparison); the ``all`` row pools
    the comparison over every (size, prior) pair."""
    n_priors = aurc[ref].shape[1]
    ref_label = REJECT_LABELS[ref]
    rows = win_rates(aurc, sizes, names, ref)
    out = [
        "-" * 76,
        f"win% {measure} ({metric}): sampled priors (of {n_priors}) where "
        f"'{ref_label}' has the lower area than each competitor",
        f"{'n_test':>8}{'':>8}"
        + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names),
    ]
    for label in [str(n) for n in sizes] + ["all"]:
        out.append(f"{label:>8}{'':>8}" + "".join(
            f"{'':>24}" if name == ref else f"{rows[label][name]:>24.1f}"
            for name in names))
    return out


# Version tag of the results file. Bump on any change that a reader could not
# absorb silently (a renamed/removed key, a changed unit or scale); adding a new
# key is backwards compatible and does not need a bump.
RESULTS_SCHEMA = "erop.rejopt_eval.results/1"


RESULTS_FILENAME = "results.json"


def _rows_by_size(data: dict, sizes: list[int], names: list[str],
                  agg: Aggregation) -> dict:
    """One metric's table as ``{row label: {predictor: value}}``.

    ``data[name]`` is the ``(len(sizes), reps)`` array; each cell is collapsed
    with ``agg``'s center statistic, exactly as the text table's cell is. The
    ``"avg"`` row collapses sizes and replicates together, matching
    ``sweep_avg_row``. Values are unscaled (the text tables' ``x1000`` is a
    display choice, recorded as ``area_scale`` in the metadata)."""
    rows = {str(n): {name: float(agg.center(data[name][i])) for name in names}
            for i, n in enumerate(sizes)}
    rows["avg"] = {name: float(agg.center(data[name])) for name in names}
    return rows


def build_results(sizes, args, res: SweepResult, reps: int, agg: Aggregation,
                  meta: dict, report_win_rate: bool) -> dict:
    """The run's numbers as a plain JSON-able dict.

    The machine-readable counterpart of the text report: the same central
    statistics, through the same ``agg``, but keyed by name rather than laid out
    in fixed-width columns. ``summary_table.py`` reads this, so the text report
    is free to change its wording and column widths without breaking the
    downstream tables (see ``tasks/summary_table.md``).

    Row labels are ``str(size)`` plus ``"avg"`` (``"all"`` for the win rates,
    matching the text report's spelling in each table).
    """
    names = list(REJECT_LABELS.keys())
    areas = {
        "aurc": {"risk": res.aurc_risk, "regret": res.aurc_regret},
        "aurc50": {"risk": res.aurc50_risk, "regret": res.aurc50_regret},
        "augrc": {"risk": res.augrc_risk, "regret": res.augrc_regret},
    }
    results = {
        "schema": RESULTS_SCHEMA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        **meta,
        "predictors": {name: REJECT_LABELS[name] for name in names},
        "sizes": [int(n) for n in sizes],
        "reps": int(reps),
        "aggregation": {
            "band": agg.band, "percentile": agg.pct, "noun": agg.noun,
            "center": "median" if agg.pct is not None else "mean",
        },
        # The text tables multiply the areas by this; the values below do not.
        "area_scale": AREA_SCALE,
        "areas": {
            measure: {metric: _rows_by_size(data, sizes, names, agg)
                      for metric, data in per_metric.items()}
            for measure, per_metric in areas.items()
        },
        "coverage_at_regret": [
            {"target": float(eps),
             "rows": _rows_by_size(cov, sizes, names, agg)}
            for eps, cov in zip(args.regret_target, res.cov_regret)
        ],
        "epistemic": {
            "threshold": float(args.epi_threshold),
            "rows": {
                str(n): {
                    "avg_epistemic": float(agg.center(res.epi_metrics[i, :, 0])),
                    "avg_regret": float(agg.center(res.epi_metrics[i, :, 1])),
                    "portion_negligible": float(
                        agg.center(res.epi_metrics[i, :, 2])),
                }
                for i, n in enumerate(sizes)
            },
        },
        "transductive_floor": {
            str(n): {
                "regret_vs_oracle": float(agg.center(res.floor["regret"][i])),
                "accuracy": float(agg.center(res.floor["accuracy"][i])),
                "classes_seen": float(agg.center(res.floor["classes"][i])),
                "eval_size": int(res.curve_len[i]),
            }
            for i, n in enumerate(sizes)
        },
        "identifiability_warn_fraction": {
            str(n): float(res.warned[i].mean()) for i, n in enumerate(sizes)
        },
    }
    if report_win_rate:
        results["win_rate"] = {
            "reference": WIN_RATE_REF,
            "areas": {
                measure: {metric: win_rates(data, sizes, names)
                          for metric, data in per_metric.items()}
                for measure, per_metric in areas.items()
            },
        }
    return results


def dirichlet_header_lines(args, misspec_line) -> list[str]:
    """Report lines describing the prior-sampling setup and the aggregation."""
    lines = [
        f"prior model  : alpha ~ Dir(s * p), s = {args.dirichlet:g}, "
        f"p = central target prior below, {args.trials_prior} draws",
        ("model prior  : matched, beta = s * p (well specified)"
         if misspec_line is None else
         f"model prior  : symmetric, beta = {args.beta:g} per class"),
    ]
    if misspec_line:
        lines.append(misspec_line)
    lines.append(
        f"aggregation  : means pool the {args.trials_prior * args.trials} "
        f"runs; ± bands/std are over the {args.trials_prior} per-prior means")
    return lines


def sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names,
                         max_lines=20) -> list[str]:
    """Per-draw report summary of the sampled target priors (the full vectors
    go to sampled_priors.txt)."""
    lines = []
    for j, a in enumerate(alphas[:max_lines]):
        top = np.argsort(-a)[:3]
        desc = "  top: " + " ".join(f"{class_names[c]}={a[c]:.3f}"
                                    for c in top)
        if pair_idx is not None:
            i_, j_ = pair_idx
            desc = (f"  pair: {class_names[i_]}={a[i_]:.3f} "
                    f"{class_names[j_]}={a[j_]:.3f}" + desc)
        lines.append(f"prior draw {j + 1:>3} (seed {prior_seeds[j]}):{desc}")
    if len(alphas) > max_lines:
        lines.append(f"  ... {len(alphas) - max_lines} more draws in "
                     f"sampled_priors.txt")
    return lines


def write_sampled_priors(out_dir: Path, alphas, prior_seeds,
                          beta_gen) -> None:
    """Write every sampled target prior (audit trail: an unlucky draw must be
    distinguishable from a bug)."""
    lines = ["# target priors sampled from Dir(s * p)",
             "# generator concentration beta = "
             + np.array2string(beta_gen, precision=6, threshold=10 ** 6)]
    for j, (a, s) in enumerate(zip(alphas, prior_seeds)):
        lines.append(f"draw {j + 1}  seed {s}")
        lines.append(np.array2string(a, precision=8, threshold=10 ** 6,
                                     max_line_width=100))
    (out_dir / "sampled_priors.txt").write_text("\n".join(lines) + "\n")


def sweep_tables(sizes, args, res: SweepResult, agg: Aggregation,
                 report_win_rate: bool = False) -> list[str]:
    """The report's metric tables, as report lines appended after the header.

    Shared by the fixed-prior sweep (replicate axis = trials) and the dirichlet
    sweep (replicate axis = sampled priors, arrays holding per-prior means); the
    caller passes the matching ``agg``, so what each cell centres describes the
    right thing. ``report_win_rate`` (dirichlet mode only) appends a
    per-competitor win-rate block after each area table.

    Every area comes from ``res``, computed per replicate from that replicate's
    full selective curve, so no number here depends on the ragged-curve
    truncation the figures need.
    """
    names = list(REJECT_LABELS.keys())
    warned, epi_metrics = res.warned, res.epi_metrics
    lines: list[str] = []

    for metric, aurc in (("risk", res.aurc_risk), ("regret", res.aurc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC ({metric})  [{AREA_SCALE_TAG}]")
        lines.append(f"{'n_test':>8}{'warn':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            row = f"{n:>8}{warned[i].mean():>8.2f}"
            row += "".join(f"{agg.center(aurc[name][i]) * AREA_SCALE:>24.4f}"
                           for name in names)
            lines.append(row)
        lines.append(sweep_avg_row(aurc, names, decimals=4, agg=agg,
                                   warn=warned, scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, aurc, sizes, names,
                                        measure="AuRC"))
    for metric, aurc50 in (("risk", res.aurc50_risk),
                           ("regret", res.aurc50_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC50 ({metric})  [{AREA_SCALE_TAG}]  ({AURC50_NOTE})")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{agg.center(aurc50[name][i]) * AREA_SCALE:>24.4f}"
                                   for name in names))
        lines.append(sweep_avg_row(aurc50, names, decimals=4, agg=agg,
                                   scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, aurc50, sizes, names,
                                        measure="AuRC50"))
    lines.append(AURC50_CAVEAT)
    for metric, augrc in (("risk", res.augrc_risk),
                          ("regret", res.augrc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuGRC ({metric})  [{AREA_SCALE_TAG}]  (normalized by "
                     f"n_eval; not on the AuRC scale)")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{agg.center(augrc[name][i]) * AREA_SCALE:>24.4f}"
                                   for name in names))
        lines.append(sweep_avg_row(augrc, names, decimals=4, agg=agg,
                                   scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, augrc, sizes, names,
                                        measure="AuGRC"))
    for eps, cov in zip(args.regret_target, res.cov_regret):
        lines.append("-" * 76)
        lines.append(f"coverage @ regret <= {eps:g}")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{agg.center(cov[name][i]):>24.3f}"
                                   for name in names))
        lines.append(sweep_avg_row(cov, names, decimals=3, agg=agg))
    lines.append("-" * 76)
    lines.append("Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"(threshold={args.epi_threshold:g})")
    lines.append(f"{'n_test':>8}{'avg epi':>14}{'avg regret':>14}{'portion negl':>14}")
    for i, n in enumerate(sizes):
        lines.append(f"{n:>8}{agg.center(epi_metrics[i, :, 0]):>14.4f}"
                     f"{agg.center(epi_metrics[i, :, 1]):>14.4f}"
                     f"{agg.center(epi_metrics[i, :, 2]):>14.3f}")
    lines.append(sweep_epi_avg_row(epi_metrics, agg))
    # Transductive floor: the best a prior-adaptation method could do on the
    # evaluated batch, measured against the same population-oracle reference as
    # every regret above. Under --eval-on-adapt it is the honest lower bound on
    # regret, and a negative value says the population oracle is beatable on a
    # batch that size -- the reason regret may go negative.
    lines.append("-" * 76)
    lines.append("Transductive floor: plugin at the evaluation batch's "
                 f"empirical prior  [{AREA_SCALE_TAG}]")
    lines.append(f"{'n_test':>8}{'regret vs oracle':>18}{'accuracy':>12}"
                 f"{'classes seen':>14}{'eval size':>12}")
    floor = res.floor
    for i, n in enumerate(sizes):
        lines.append(f"{n:>8}{agg.center(floor['regret'][i]) * AREA_SCALE:>18.4f}"
                     f"{agg.center(floor['accuracy'][i]):>12.4f}"
                     f"{agg.center(floor['classes'][i]):>14.1f}"
                     f"{res.curve_len[i]:>12}")
    if float(np.min(floor["regret"])) < 0:
        lines.append("  note: a negative floor means the plugin at the batch's "
                     "empirical prior beats the")
        lines.append("        population oracle on that batch, so the regret "
                     "of every method may go negative;")
        lines.append("        'coverage @ regret <= eps' is then a budget "
                     "against the population oracle,")
        lines.append("        not against an attainable optimum.")
    lines.append("=" * 76)
    if warned.any():
        lines.append("!!! IDENTIFIABILITY WARNING: 'warn' = fraction of trials "
                     "where the learned prior was only weakly identifiable "
                     "(see README).")
    return lines


def write_report(out_dir: Path, lines: list[str]) -> str:
    """Write the text report, echo it to stdout, and return it."""
    report = "\n".join(lines)
    (out_dir / REPORT_FILENAME).write_text(report + "\n")
    print(report)
    return report


def write_results(out_dir: Path, sizes, args, res: SweepResult, reps: int,
                  agg: Aggregation, meta: dict, report_win_rate: bool) -> None:
    """Write the machine-readable ``results.json`` (see :func:`build_results`)."""
    results = build_results(sizes, args, res, reps, agg, meta, report_win_rate)
    (out_dir / RESULTS_FILENAME).write_text(
        json.dumps(results, indent=2, sort_keys=False) + "\n")
