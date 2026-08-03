"""Reject-option / test-prior adaptation on real datasets.

Consumes a **trained, temperature-calibrated neural-network base predictor**
produced by ``base_predictor_training.py`` (its ``model.pt`` bundle); everything
downstream -- the MCMC prior learning, the plugin corrections, the
reject-option predictors and their risk/regret-coverage curves -- comes from
``prior_shift`` (the metrics in ``prior_shift.reject_option``) and
``reject_figures``.

Label shift is *simulated*: the real test pool (the base script's val+test
merge) is labeled, so it can be resampled to a chosen target prior
``p_te(y)``. The script always **sweeps** the adaptation-set size: every
measure is reported as a curve over ``--sizes`` (pass a single size for a
one-point run). Per trial an adaptation pool of ``max(--sizes)`` examples is
drawn at the target prior from the whole pool -- per class, so it is stratified
-- and the swept sizes are nested prefixes of that pool.

Two evaluation designs, selected by ``--eval-on-adapt``
(``merge_eval_adapt_sets_polished.md``):

- **disjoint** (default): the remainder of the pool feeds a separate evaluation
  set, whose size ``--n-eval`` defaults to the largest all-distinct set that
  remainder supports -- maximising the eval size and so minimising the variance
  of the reported metrics.
- **transductive** (``--eval-on-adapt``): the adaptation set *is* the
  evaluation set, the deployment setting (the prior is learned from the very
  batch that has to be classified) and the one in which the epistemic term is
  exactly the posterior-expected regret on the scored points. ``--n-eval`` is
  then meaningless and rejected: the evaluation size is ``--sizes``.

There is no optimal-Bayes upper bound here (the true class conditionals are
unknown for real data); the regret reference is the plugin given the true
(target) test prior -- in transductive mode that population oracle is *not*
optimal on the finite batch, so regret may be negative; the reported
"transductive floor" (the plugin at the batch's empirical prior) says how much
of that is finite-batch prior mismatch.

Predictors / baselines:

- Plugin, training prior (no adaptation)
- Plugin, true test prior (oracle target prior)
- Bayesian, learned prior (MCMC from the unlabeled adaptation inputs)

The supervised-prior baseline (prior counted from the adaptation-set labels --
``baseline_learned_prior_subervised_data.md``) is not computed here; under
``--eval-on-adapt`` it would coincide exactly with the empirical-prior plugin
reported as the transductive floor.

and the two reject-option predictors: the Bayesian predictor ranked by total
and by epistemic uncertainty.

Run with::

    python base_predictor_training.py bloodmnist runs/blood      # train base model
    python rejopt_eval.py runs/blood/model.pt runs/blood
    python rejopt_eval.py runs/blood/model.pt runs/blood \\
        --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 --dirichlet 20

Two interfaces set the target prior: ``--test-prior`` (an explicit vector) and
``--prior-classes`` / ``--prior-weights`` / ``--prior-rest-weight`` (relative
per-class weights, normalised). **By default the target prior is the training
prior**, i.e. no label shift at all -- the degenerate control that shows what
the methods cost when no adaptation is needed. In a fixed-prior run that makes
every predictor coincide up to posterior noise; under ``--dirichlet`` the draws
around it are genuinely shifted.

With ``--dirichlet SUM_PARAMS`` the experiment repeats over target priors
sampled from ``Dir(s * p)`` centered on the configured prior
(``--trials-prior`` draws x ``--trials`` trials each). The Bayesian methods
and the supervised baseline use the matching Dirichlet (well specified;
``--beta`` deliberately misspecifies the model prior instead), the regret
reference uses each sampled prior, adaptation draws truncate to pool
availability while evaluation draws use replacement, and all figures average
the runs with ±1-std-over-priors bands. Extra outputs:
``epi_vs_regret_calibration.png`` (per-draw epistemic uncertainty vs.
realized regret against the y = x line) and ``sampled_priors.txt``. See
``tasks/multiple_priors_polished.md``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import reporting
from base_predictor_training import make_model, to_tensor
from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS
from prior_shift import zero_one_loss_matrix
from prior_shift.reject_option import REJECT_LABELS, Aggregation
from prior_shift.sampling import (
    max_distinct_eval,
    sample_target_prior,
    target_counts,
    target_prior_from_weights,
)
from prior_shift.sweep import SweepResult, run_sweep, stack_curves
from reject_figures import (
    make_epi_regret_calibration_figure,
    trunc_sweep_fname,
    write_sweep_figures,
)
from reporting import (
    TRAIN_PRIOR_DEFAULT_HOW,
    TRAIN_PRIOR_DEFAULT_WARNING,
    calibration_lines,
    class_weights_how,
    confusable_report_line,
    dirichlet_header_lines,
    eval_set_report_field,
    sampled_prior_lines,
    save_run_args,
    target_prior_report_line,
    write_sampled_priors,
)

# Mode-dependent defaults, named rather than buried in the resolution code.
# In dirichlet mode the largest all-distinct evaluation size is undefined (it
# varies per draw), so the evaluation set is a fixed size drawn with
# replacement where a class falls short.
DEFAULT_TRIALS_PRIOR = 5
DIRICHLET_DEFAULT_N_EVAL = 1000


def load_base_predictor(bundle_path: Path, device: torch.device):
    """Reconstruct the trained network and return ``(model, bundle)``."""
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    model = make_model(bundle["arch"], bundle["in_channels"],
                       bundle["num_classes"])
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    return model, bundle


@torch.no_grad()
def calibrated_posterior(model, X, bundle, device, batch_size=512):
    """Calibrated ``p_tr(y | x)`` for a uint8 image array.

    Applies the bundle's calibration: ``softmax(logits / T + bias)``. Bundles
    from before the BCTS change carry no ``calib_bias``; zero bias reproduces
    plain temperature scaling.
    """
    Xt = to_tensor(X, bundle["norm_mean"], bundle["norm_std"])
    T = bundle["temperature"]
    bias = bundle.get("calib_bias")
    bias_t = (torch.zeros(bundle["num_classes"]) if bias is None
              else torch.from_numpy(np.asarray(bias)).float())
    out = []
    for i in range(0, len(Xt), batch_size):
        logits = model(Xt[i:i + batch_size].to(device))
        out.append(torch.softmax(logits / T + bias_t, dim=1).cpu().numpy())
    return np.concatenate(out)





















































def _sweep_outputs(sizes, args, out_dir: Path, lines: list[str],
                   res: SweepResult, reps: int, agg: Aggregation,
                   meta: dict, display_name: str = "",
                   report_win_rate: bool = False) -> None:
    """Write everything one sweep produces: the report, the results file and
    every figure.

    Shared by the fixed-prior sweep (replicate axis = trials) and the dirichlet
    sweep (replicate axis = sampled priors, arrays holding per-prior means).
    ``lines`` is the caller's report header, which the metric tables are
    appended to; ``reps`` is the replicate count of the arrays' second axis,
    matching ``agg``.
    """
    lines = lines + reporting.sweep_tables(sizes, args, res, agg,
                                           report_win_rate)
    reporting.write_report(out_dir, lines)
    reporting.write_results(out_dir, sizes, args, res, reps, agg, meta,
                            report_win_rate)
    write_sweep_figures(sizes, res, reps, args.out_dir, agg,
                        args.regret_target, args.epi_threshold,
                        display_name=display_name)
    print(f"\nreport and figures written to {out_dir}/: "
          f"{reporting.REPORT_FILENAME}, {reporting.RESULTS_FILENAME}, "
          f"aurc_vs_n_test.png, "
          f"aurec_vs_n_test.png, gen_aurc_vs_n_test.png, "
          f"{trunc_sweep_fname()}.png, "
          f"epistemic_metrics_vs_n_test.png, cov_at_target_vs_n_test.png, "
          f"base_accuracy_vs_n_test.png, "
          f"coverage_curves/[gen_]coverage_curves_n<n_test>.png "
          f"(two per size)")


def run_sweep_report(P, y_pool, train_prior, target_prior, bundle, spec,
                     class_names, loss, args, out_dir: Path,
                     header_lines: list[str]) -> None:
    """Drive the fixed-prior sweep, print/save the report, write the figures.

    The replicate axis is the trial, so the default aggregation applies (mean
    line, s.e.m. band), with ``--percentile-band`` layered on when asked for.
    """
    sizes = sorted(args.sizes)
    agg = Aggregation().with_percentile(args.percentile_band)
    master_rng = np.random.default_rng(args.seed)
    res = run_sweep(
        P, y_pool, train_prior, target_prior, sizes, args.trials,
        args.n_eval, loss, master_rng, args.epi_threshold,
        args.regret_target, sampler=args.sampler, beta=args.beta,
        eval_on_adapt=args.eval_on_adapt)
    shortfalls = res.short_adapt | res.short_eval

    lines = [
        "=" * 76,
        "AuRC vs. number of unlabeled adaptation examples (real data)",
        "=" * 76,
        f"timestamp    : {datetime.now().isoformat(timespec='seconds')}",
        f"command      : {' '.join(sys.argv)}",
        f"base model   : {spec.display_name} ({bundle['dataset']}), "
        f"arch {bundle['arch']}",
        *calibration_lines(bundle, class_names),
        f"pool size    : {len(y_pool)}   trials {args.trials}   "
        f"{eval_set_report_field(args)}   sizes {sizes}",
        f"prior beta   : {args.beta:g} per class (symmetric Dirichlet)",
        *header_lines,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"target vector: {np.array2string(target_prior, precision=3)}",
    ]
    if shortfalls:
        pretty = ", ".join(class_names[c] for c in sorted(shortfalls))
        lines.append(f"note         : classes resampled WITH replacement "
                     f"(pool too small at this target prior): {pretty}")
        if args.eval_on_adapt:
            lines.append("               under --eval-on-adapt those "
                         "duplicates are evaluation examples too, which "
                         "inflates the apparent coverage")
    meta = {
        "dataset": bundle["dataset"],
        "display_name": spec.display_name,
        "arch": bundle["arch"],
        "mode": "fixed",
        "trials": args.trials,
        "eval_on_adapt": bool(args.eval_on_adapt),
        "n_eval": None if args.eval_on_adapt else int(args.n_eval),
        "train_prior": train_prior.tolist(),
        "target_prior": target_prior.tolist(),
    }
    _sweep_outputs(sizes, args, out_dir, lines, res, args.trials, agg, meta,
                   display_name=spec.display_name)



def run_dirichlet_sweep_report(P, y_pool, train_prior, central_prior, bundle,
                               spec, class_names, pair_idx, loss, args,
                               out_dir: Path, header_lines: list[str],
                               model_beta, misspec_line) -> None:
    """Sweep repeated over N target priors sampled from Dir(s * central).

    Per draw the full fixed-prior sweep runs (``run_sweep``) with the sampled
    prior in the true-prior role and the trial axis is collapsed to per-prior
    means, so every downstream table/figure aggregates N per-prior means (the
    dirichlet-mode error-bar convention)."""
    sizes = sorted(args.sizes)
    names = list(REJECT_LABELS.keys())
    N, T, S = args.trials_prior, args.trials, len(sorted(args.sizes))
    master = np.random.default_rng(args.seed)
    prior_seeds = master.integers(1 << 32, size=N)
    beta_gen = args.dirichlet * central_prior

    def per_predictor():
        return {n: np.zeros((S, N)) for n in names}

    # Per-prior means, on the same layout run_sweep uses for per-trial values.
    # Every area collapses by averaging, and AuRC / AuRC50 / AuGRC are all
    # rank-means of a curve, so averaging the per-trial areas is exactly the
    # area of the averaged curve -- the numbers match the pre-refactor path.
    d = SweepResult(
        aurc_risk=per_predictor(), aurc_regret=per_predictor(),
        aurc50_risk=per_predictor(), aurc50_regret=per_predictor(),
        augrc_risk=per_predictor(), augrc_regret=per_predictor(),
        cov_regret=[per_predictor() for _ in args.regret_target],
        risk_curves={n: [[] for _ in sizes] for n in names},
        regret_curves={n: [[] for _ in sizes] for n in names},
        curve_len=np.zeros(S, dtype=int), warned=np.zeros((S, N)),
        epi_metrics=np.zeros((S, N, 3)),
        base_acc={k: np.zeros((S, N))
                  for k in ("bayes_learned", "plugin_true", "plugin_train")},
        floor={k: np.zeros((S, N)) for k in ("regret", "accuracy", "classes")},
        realized_n=np.zeros((S, N)), short_adapt=set(), short_eval=set())
    alphas = np.zeros((N, len(central_prior)))

    for j in range(N):
        prng = np.random.default_rng(prior_seeds[j])
        alpha = sample_target_prior(prng, beta_gen)
        alphas[j] = alpha
        r = run_sweep(
            P, y_pool, train_prior, alpha, sizes, T, args.n_eval, loss,
            prng, args.epi_threshold, args.regret_target,
            sampler=args.sampler, beta=model_beta, adapt_replace=False,
            eval_on_adapt=args.eval_on_adapt,
            progress_desc=f"prior {j + 1}/{N}")
        d.short_adapt |= r.short_adapt
        d.short_eval |= r.short_eval
        for area_d, area in ((d.aurc_risk, r.aurc_risk),
                             (d.aurc_regret, r.aurc_regret),
                             (d.aurc50_risk, r.aurc50_risk),
                             (d.aurc50_regret, r.aurc50_regret),
                             (d.augrc_risk, r.augrc_risk),
                             (d.augrc_regret, r.augrc_regret)):
            for n in names:
                area_d[n][:, j] = area[n].mean(axis=1)
        for n in names:
            for i in range(S):
                d.risk_curves[n][i].append(r.risk_curves[n][i].mean(axis=0))
                d.regret_curves[n][i].append(r.regret_curves[n][i].mean(axis=0))
        d.warned[:, j] = r.warned.mean(axis=1)
        d.epi_metrics[:, j] = r.epi_metrics.mean(axis=1)
        for ei in range(len(args.regret_target)):
            for n in names:
                d.cov_regret[ei][n][:, j] = r.cov_regret[ei][n].mean(axis=1)
        for k in d.base_acc:
            d.base_acc[k][:, j] = r.base_acc[k].mean(axis=1)
        for k in d.floor:
            d.floor[k][:, j] = r.floor[k].mean(axis=1)
        d.realized_n[:, j] = r.realized_n.mean(axis=1)

    # Sampled priors differ, so per-prior mean curves can differ in length too
    # (truncation of the adaptation pool); cut each size to its shortest draw.
    for n in names:
        d.risk_curves[n], d.curve_len = stack_curves(d.risk_curves[n])
        d.regret_curves[n], _ = stack_curves(d.regret_curves[n])
    realized_d = d.realized_n

    write_sampled_priors(out_dir, alphas, prior_seeds, beta_gen)
    # The replicate axis here is the sampled prior, not the trial: the arrays
    # above hold per-prior means, so the band is ±1 std over those.
    agg = Aggregation.over_priors(N, T, pct=args.percentile_band)

    lines = [
        "=" * 76,
        "AuRC vs. number of unlabeled adaptation examples "
        "(real data, sampled target priors)",
        "=" * 76,
        f"timestamp    : {datetime.now().isoformat(timespec='seconds')}",
        f"command      : {' '.join(sys.argv)}",
        f"base model   : {spec.display_name} ({bundle['dataset']}), "
        f"arch {bundle['arch']}",
        *calibration_lines(bundle, class_names),
        f"pool size    : {len(y_pool)}   priors {N} x trials {T}   "
        f"{eval_set_report_field(args)}   sizes {sizes}",
        *dirichlet_header_lines(args, misspec_line),
        *header_lines,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"central prior: {np.array2string(central_prior, precision=3)}",
        *sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names),
    ]
    if np.any(realized_d < np.asarray(sizes)[:, None]):
        per_size = ", ".join(
            f"{n}->{realized_d[i].mean():.1f}" for i, n in enumerate(sizes)
            if realized_d[i].mean() < n)
        what = ("adaptation/evaluation sets" if args.eval_on_adapt
                else "adaptation sets")
        lines.append(f"note         : {what} truncated to pool "
                     f"availability (no replacement); mean realized n: "
                     f"{per_size}")
    if d.short_eval:
        lines.append("note         : evaluation sets drawn WITH replacement "
                     "where a class's pool fell short (expected in dirichlet "
                     "mode)")

    meta = {
        "dataset": bundle["dataset"],
        "display_name": spec.display_name,
        "arch": bundle["arch"],
        "mode": "dirichlet",
        "trials": T,
        "trials_prior": N,
        "dirichlet": float(args.dirichlet),
        "model_prior_misspecified": misspec_line is not None,
        "eval_on_adapt": bool(args.eval_on_adapt),
        "n_eval": None if args.eval_on_adapt else int(args.n_eval),
        "train_prior": train_prior.tolist(),
        "central_prior": central_prior.tolist(),
        "sampled_priors": alphas.tolist(),
    }
    _sweep_outputs(sizes, args, out_dir, lines, d, N, agg, meta,
                   display_name=spec.display_name, report_win_rate=True)

    make_epi_regret_calibration_figure(sizes, d.epi_metrics, args.out_dir)
    print(f"calibration figure and sampled priors written to {out_dir}/: "
          f"epi_vs_regret_calibration.png, sampled_priors.txt")




def build_parser() -> argparse.ArgumentParser:
    """The command-line interface. Split out of ``main`` so the flags can be
    read (and tested) without running an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=str,
                        help="Path to a model.pt bundle from base_predictor_training.py.")
    parser.add_argument("out_dir", type=str,
                        help="Directory receiving the report and figures.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Labeled evaluation-set size, disjoint from the "
                             "adaptation set. Default: the maximum "
                             "all-distinct size at the target prior (the "
                             "adaptation set is drawn first, the rest go to "
                             "evaluation); in dirichlet mode, where that "
                             "maximum is undefined, a fixed 1000 drawn with "
                             "replacement where needed. Pass an integer to "
                             "pin it. Rejected with --eval-on-adapt, which "
                             "has no separate evaluation set.")
    parser.add_argument(
        "--eval-on-adapt", action="store_true",
        help="Transductive evaluation: score every predictor on the very "
             "examples the test prior was learned from, i.e. the evaluation "
             "set IS the adaptation set and its size is --sizes. The "
             "deployment setting (a batch arrives, its prior is learned from "
             "it, it has to be classified), and the setting in which the "
             "epistemic term is exactly the posterior-expected regret on the "
             "scored points. Regret is still measured against the plugin at "
             "the true target prior, which on a finite batch is not optimal, "
             "so regret may go negative -- see the transductive-floor table. "
             "Default: the disjoint adaptation/evaluation split.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[50, 100, 200, 500, 1000, 2000],
        help="Adaptation-set sizes: the swept variable every measure is "
             "reported against. A single value gives a one-point run.")
    parser.add_argument(
        "--epi-threshold", type=float, default=0.001,
        help="Epistemic uncertainty below this value counts as negligible "
             "in the reported portion metric.")
    parser.add_argument(
        "--regret-target", type=float, nargs="+", default=[0.002],
        help="Regret budget(s) for the coverage-at-target metric; the metric "
             "is computed for each value given.")
    parser.add_argument("--test-prior", type=float, nargs="+", default=None,
                        help="Explicit target test prior (Y floats, summing to "
                             "1). Default: the training prior itself, i.e. no "
                             "label shift.")
    parser.add_argument("--prior-classes", type=int, nargs="+", default=None,
                        metavar="Y",
                        help="0-based class indices whose target-prior weights "
                             "are given by --prior-weights; every class not "
                             "listed gets --prior-rest-weight. Distinct, in "
                             "[0, Y). Conflicts with --test-prior.")
    parser.add_argument("--prior-weights", type=float, nargs="+", default=None,
                        metavar="W",
                        help="Non-negative weights aligned positionally with "
                             "--prior-classes. Relative, not probabilities: "
                             "the full weight vector is normalised to sum 1.")
    parser.add_argument("--prior-rest-weight", type=float, default=None,
                        metavar="R",
                        help="Per-class weight of every class not named by "
                             "--prior-classes (so they stay equiprobable among "
                             "themselves). Required unless --prior-classes "
                             "names all Y classes.")
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Per-class concentration of the symmetric Dirichlet MODEL prior "
             "on the test prior (default 1). With many classes the default "
             "carries Y pseudo-counts and overwhelms small unlabeled samples, "
             "so the posterior hugs the near-uniform prior and the epistemic "
             "uncertainty underestimates the true regret; values < 1 spread "
             "the prior over skewed priors instead, but their spiky draws "
             "degrade the Bayesian point decision at small n. In dirichlet "
             "mode the model prior defaults to the matched generator "
             "Dir(s * p) (well specified); passing --beta there replaces it "
             "with the symmetric prior, i.e. deliberately misspecifies the "
             "model while the data keep being generated from Dir(s * p).")
    parser.add_argument(
        "--dirichlet", type=float, default=None, metavar="SUM_PARAMS",
        help="Enable dirichlet mode: repeat the experiment over target priors "
             "sampled from Dir(s * p), where s = SUM_PARAMS (> 0) is the "
             "total concentration and p the configured target prior "
             "(--test-prior, the --prior-classes weights, or the training "
             "prior by default) in its central role. Larger s concentrates "
             "the draws around p; s -> inf recovers the fixed-prior "
             "experiment.")
    parser.add_argument(
        "--trials-prior", type=int, default=None, metavar="N",
        help="Number of sampled target priors in dirichlet mode (default 5); "
             "each runs the full --trials loop, so N * trials runs total. "
             "Requires --dirichlet.")
    parser.add_argument(
        "--sampler", choices=("mh", "gibbs"), default="mh",
        help="Posterior sampler for the test prior: random-walk "
             "Metropolis-Hastings (mh, default) or the latent-variable "
             "Gibbs sampler (gibbs).")
    parser.add_argument(
        "--percentile-band", type=float, default=None, metavar="X",
        help="Draw the figures' uncertainty bands as the central X%% "
             "percentile interval (X in [0, 100]; e.g. 80 -> 10th-90th "
             "percentile) with the solid line at the pointwise median, instead "
             "of the default mean +/- s.e.m. (or std) band.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def validate_args(parser: argparse.ArgumentParser,
                  args) -> tuple[list[str], bool]:
    """Check the flag combinations and fill in the mode-dependent defaults.

    Returns ``(weight_flags, dirichlet_mode)``: the target-prior weight flags
    the user actually passed (empty unless that strategy was chosen) and whether
    dirichlet mode is on. Runs before the model is loaded, so a flag clash fails
    immediately rather than a minute in.
    """
    if args.percentile_band is not None and not 0 <= args.percentile_band <= 100:
        parser.error("--percentile-band must be in [0, 100]")
    if args.eval_on_adapt and args.n_eval is not None:
        parser.error("--n-eval is meaningless with --eval-on-adapt: the "
                     "evaluation set is the adaptation set, so its size is "
                     "--sizes")

    # The two target-prior strategies are mutually exclusive; with neither the
    # target is the training prior (no shift).
    weight_flags = [f for f, v in (("--prior-classes", args.prior_classes),
                                   ("--prior-weights", args.prior_weights),
                                   ("--prior-rest-weight",
                                    args.prior_rest_weight))
                    if v is not None]
    if weight_flags and args.test_prior is not None:
        clash = ", ".join(weight_flags)
        sys.exit(f"error: --test-prior and {clash} are different ways to build "
                 "the target prior; pass only one strategy")

    dirichlet_mode = args.dirichlet is not None
    if args.trials_prior is not None and not dirichlet_mode:
        sys.exit("error: --trials-prior requires --dirichlet")
    if dirichlet_mode and args.dirichlet <= 0:
        sys.exit("error: --dirichlet needs a positive total concentration")
    if dirichlet_mode and args.trials_prior is None:
        args.trials_prior = DEFAULT_TRIALS_PRIOR
    if dirichlet_mode and args.trials_prior <= 0:
        sys.exit("error: --trials-prior must be positive")
    if args.beta is not None and args.beta <= 0:
        sys.exit("error: --beta must be positive")
    if not dirichlet_mode and args.beta is None:
        args.beta = 1.0
    return weight_flags, dirichlet_mode


def resolve_target_prior(args, Y: int, class_names, train_prior,
                         weight_flags: list[str]):
    """The run's target prior and a phrase saying how it was obtained.

    Three strategies: the explicit ``--test-prior`` vector, the
    ``--prior-classes``/``--prior-weights`` relative weights, or -- with
    neither -- the training prior itself, i.e. no label shift. That last one is
    the deliberate degenerate control, so it is named in the report and warned
    about on stdout by the caller.
    """
    if args.test_prior is not None:
        target_prior = np.asarray(args.test_prior, dtype=float)
        if len(target_prior) != Y or not np.isclose(target_prior.sum(), 1.0):
            sys.exit(f"error: --test-prior must be {Y} floats summing to 1")
        return target_prior, "explicit (--test-prior)"

    if weight_flags:
        if args.prior_classes is None or args.prior_weights is None:
            sys.exit("error: --prior-classes and --prior-weights must be given "
                     "together")
        k = len(args.prior_classes)
        if len(args.prior_weights) != k:
            sys.exit(f"error: --prior-classes has {k} entries but "
                     f"--prior-weights has {len(args.prior_weights)}; they are "
                     "aligned positionally and must match")
        bad = [c for c in args.prior_classes if not 0 <= c < Y]
        if bad:
            sys.exit(f"error: --prior-classes entries must be in [0, {Y}), got "
                     f"{bad}")
        if len(set(args.prior_classes)) != k:
            sys.exit("error: --prior-classes must not repeat a class index")
        if any(w < 0 for w in args.prior_weights):
            sys.exit("error: --prior-weights must be non-negative")
        if args.prior_rest_weight is not None and args.prior_rest_weight < 0:
            sys.exit("error: --prior-rest-weight must be non-negative")
        if k < Y and args.prior_rest_weight is None:
            sys.exit(f"error: --prior-classes names {k} of {Y} classes, so "
                     "--prior-rest-weight is required for the remaining "
                     f"{Y - k}")
        if k == Y and args.prior_rest_weight is not None:
            print("warning: --prior-classes names every class, so "
                  "--prior-rest-weight is ignored")
        total = sum(args.prior_weights) + (Y - k) * (args.prior_rest_weight
                                                     if k < Y else 0.0)
        if total <= 0:
            sys.exit("error: the --prior-weights / --prior-rest-weight weights "
                     "are all zero, so the target prior is undefined")
        rest = args.prior_rest_weight if k < Y else None
        target_prior = target_prior_from_weights(
            Y, args.prior_classes, args.prior_weights, rest)
        zero = [class_names[c] for c in np.flatnonzero(target_prior <= 0)]
        if zero:
            print(f"warning: zero weight on {', '.join(zero)}; "
                  "absent from the adaptation and evaluation sets")
        return target_prior, class_weights_how(
            args.prior_classes, args.prior_weights, rest, class_names)

    # No target-prior flag: the target IS the training prior, i.e. no label
    # shift. Deliberate (the degenerate control), but invisible in the numbers
    # -- every predictor then coincides up to posterior noise -- so it is named
    # in the report and warned about on stdout.
    return train_prior / train_prior.sum(), TRAIN_PRIOR_DEFAULT_HOW


def resolve_model_prior(args, Y: int, target_prior, class_names,
                        dirichlet_mode: bool):
    """The Dirichlet concentration the *methods* use, and the misspecification
    banner when it deliberately disagrees with the generator.

    The data-generating Dirichlet is always ``Dir(s * p)``. The model prior
    matches it (well specified) unless ``--beta`` overrides it with a symmetric
    prior -- the deliberate-misspecification control. Outside dirichlet mode the
    model prior is just the scalar ``--beta`` (``sample_prior_posterior``
    broadcasts it).
    """
    if not dirichlet_mode:
        return args.beta, None
    if np.any(target_prior <= 0):
        zero = ", ".join(class_names[c]
                         for c in np.flatnonzero(target_prior <= 0))
        sys.exit("error: --dirichlet needs positive central mass on every "
                 f"class, but these have none: {zero}. Adjust "
                 "--test-prior / --prior-weights / --prior-rest-weight.")
    if args.beta is None:
        return args.dirichlet * target_prior, None
    misspec_line = ("!!! MODEL PRIOR MISSPECIFIED via --beta: methods "
                    f"use symmetric Dirichlet({args.beta:g}) while "
                    "target priors are drawn from Dir(s * p) !!!")
    print(misspec_line)
    return np.full(Y, args.beta), misspec_line


def resolve_eval_size(args, Y: int, y_pool, target_prior, class_names,
                      dirichlet_mode: bool) -> bool:
    """Check the pool can serve the requested draws and settle ``args.n_eval``.

    The adaptation set (``max(--sizes)`` examples at the target prior) is drawn
    first from the whole pool; the remainder feeds evaluation, and ``--n-eval``
    defaults to the largest all-distinct evaluation set that remainder supports.
    Returns whether that default was used (the report says so).
    """
    n_adapt = max(args.sizes)
    pool_counts = np.bincount(y_pool, minlength=Y)

    if args.eval_on_adapt:
        # No separate evaluation draw, so no evaluation feasibility to check:
        # the only pool requirement is that the target prior's classes are
        # present at all (a shortfall is handled by replacement/truncation, as
        # for any adaptation draw).
        wanted = [c for c in range(Y) if dirichlet_mode or target_prior[c] > 0]
        missing = [c for c in wanted if pool_counts[c] == 0]
        if missing:
            names = ", ".join(f"{class_names[c]} (class {c})" for c in missing)
            sys.exit(f"error: the pool has no examples of: {names}; "
                     + ("dirichlet mode needs every class present in the pool."
                        if dirichlet_mode else
                        "adjust --test-prior / --prior-weights."))
        print(f"adapt size   : {n_adapt}   eval size : same set "
              f"(--eval-on-adapt; evaluation size = each swept size)")
        return False

    if dirichlet_mode:
        # Per-draw feasibility is handled by truncation (adaptation) and
        # replacement (evaluation); only a class with no pool examples at all
        # is fatal, since every class has positive mass under every draw.
        missing = [c for c in range(Y) if pool_counts[c] == 0]
        if missing:
            names = ", ".join(f"{class_names[c]} (class {c})" for c in missing)
            sys.exit(f"error: the pool has no examples of: {names}; dirichlet "
                     f"mode needs every class present in the pool.")
        n_eval_auto = args.n_eval is None
        if n_eval_auto:
            args.n_eval = DIRICHLET_DEFAULT_N_EVAL
        n_eval_note = "  (dirichlet-mode default)" if n_eval_auto else ""
        print(f"adapt size   : {n_adapt} (truncated per draw to pool "
              f"availability)   eval size : {args.n_eval}{n_eval_note}")
        return n_eval_auto

    adapt_counts = target_counts(n_adapt, target_prior)
    eval_avail = np.maximum(0, pool_counts - adapt_counts)

    wanted = [c for c in range(Y) if target_prior[c] > 0]
    missing = [c for c in wanted if pool_counts[c] == 0]
    if missing:
        names = ", ".join(f"{class_names[c]} (class {c})" for c in missing)
        sys.exit(f"error: target prior wants class(es) absent from the pool: "
                 f"{names}. Adjust --test-prior / --prior-weights.")
    exhausted = [c for c in wanted if eval_avail[c] == 0]
    if exhausted:
        names = ", ".join(f"{class_names[c]} (class {c})" for c in exhausted)
        sys.exit(f"error: the adaptation set of {n_adapt} examples exhausts "
                 f"class(es) {names}, leaving no evaluation examples for them. "
                 f"Reduce the adaptation size or the target mass on them.")

    n_eval_auto = args.n_eval is None
    if n_eval_auto:
        args.n_eval = max_distinct_eval(eval_avail, target_prior)
        if args.n_eval <= 0:
            sys.exit("error: no evaluation examples available at this target "
                     "prior; reduce the adaptation size.")
    n_eval_note = "  (auto max at target prior)" if n_eval_auto else ""
    print(f"adapt size   : {n_adapt}   eval size : {args.n_eval}{n_eval_note}")
    return n_eval_auto


def load_test_pool(bundle, model, device):
    """The labeled pool and its calibrated posteriors.

    The pool is the same val+test merge ``base_predictor_training.py`` scored
    on, so the downstream resampling sees every example the base model was
    evaluated against but none it was trained or calibrated on.
    """
    ds = load_dataset(bundle["dataset"])
    if "val" in ds.splits:
        X_pool = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
        y_pool = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
    else:
        X_pool, y_pool = ds.splits["test"]
    return calibrated_posterior(model, X_pool, bundle, device), y_pool


def run_args_extra(args, bundle, train_prior, target_prior, class_names,
                   pair_idx, pair_source: str, prior_how: str,
                   dirichlet_mode: bool, misspec_line, n_eval_auto: bool):
    """The ``[resolved]`` block of ``rejopt_eval_args.txt``: the values settled
    after parsing, which a reader would otherwise have to reconstruct."""
    extra = {
        "dataset": bundle["dataset"],
        "arch": bundle["arch"],
        "train_prior": np.array2string(train_prior, precision=4),
        ("central_target_prior" if dirichlet_mode else "target_test_prior"):
            np.array2string(target_prior, precision=4),
        "confusable_pair": (None if pair_idx is None else
                            f"{class_names[pair_idx[0]]} / "
                            f"{class_names[pair_idx[1]]} "
                            f"(classes {pair_idx[0]}, {pair_idx[1]}; "
                            f"{pair_source}; reporting only)"),
        "target_prior_from": prior_how,
        "eval_set": ("same as adaptation set (--eval-on-adapt; size = --sizes)"
                     if args.eval_on_adapt else
                     f"disjoint, n_eval {args.n_eval}"
                     + ((" (dirichlet-mode default)" if dirichlet_mode
                         else " (auto max at target prior)")
                        if n_eval_auto else " (explicit)")),
    }
    if dirichlet_mode:
        extra["model_prior"] = (
            "matched: beta = s * central prior (well specified)"
            if misspec_line is None else
            f"symmetric beta = {args.beta:g} per class (MISSPECIFIED)")
    return extra


def run_args_ignored(args, dirichlet_mode: bool) -> set[str]:
    """Arguments this run never reads, omitted from the args file rather than
    recorded at their unused defaults."""
    ignored = set() if dirichlet_mode else {"dirichlet", "trials_prior"}
    if args.eval_on_adapt:
        ignored.add("n_eval")   # rejected at parse time; never read
    return ignored


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    weight_flags, dirichlet_mode = validate_args(parser, args)

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    # All outputs go to a timestamped subdirectory of the requested out_dir,
    # so repeated runs never overwrite each other's report/figures.
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    # --- base predictor + calibrated posteriors on the whole test pool ------
    model, bundle = load_base_predictor(Path(args.model), device)
    Y = bundle["num_classes"]
    train_prior = np.asarray(bundle["train_prior"], dtype=float)
    loss = zero_one_loss_matrix(Y)
    class_names = bundle["class_names"]
    spec = DATASETS[bundle["dataset"]]

    for line in calibration_lines(bundle, class_names):
        print(line)

    P, y_pool = load_test_pool(bundle, model, device)

    # --- target prior -------------------------------------------------------
    # The dataset registry's designated confusable pair. Reporting only (the
    # report line and, in dirichlet mode, the per-draw pair marginals); it takes
    # no part in building the target prior.
    if spec.confusable_pair is not None:
        pair_idx = tuple(class_names.index(c) for c in spec.confusable_pair)
        pair_source = "registry default"
    else:
        pair_idx, pair_source = None, "none"

    target_prior, prior_how = resolve_target_prior(
        args, Y, class_names, train_prior, weight_flags)

    # Two independent report lines: the confusable pair is a dataset property,
    # the target prior is how this run was configured. Conflating them (as the
    # old single line did) would claim a pair-targeted shift on default runs.
    header_lines = [
        confusable_report_line(pair_idx, class_names, pair_source),
        target_prior_report_line(prior_how, dirichlet_mode),
    ]
    for line in header_lines:
        print(line)
    if prior_how is TRAIN_PRIOR_DEFAULT_HOW and not dirichlet_mode:
        print(TRAIN_PRIOR_DEFAULT_WARNING)

    model_beta, misspec_line = resolve_model_prior(
        args, Y, target_prior, class_names, dirichlet_mode)
    n_eval_auto = resolve_eval_size(
        args, Y, y_pool, target_prior, class_names, dirichlet_mode)

    save_run_args(args, "rejopt_eval_args.txt",
                  extra=run_args_extra(args, bundle, train_prior, target_prior,
                                       class_names, pair_idx, pair_source,
                                       prior_how, dirichlet_mode, misspec_line,
                                       n_eval_auto),
                  ignored=run_args_ignored(args, dirichlet_mode))

    if dirichlet_mode:
        run_dirichlet_sweep_report(P, y_pool, train_prior, target_prior,
                                   bundle, spec, class_names, pair_idx,
                                   loss, args, out_dir, header_lines,
                                   model_beta, misspec_line)
    else:
        run_sweep_report(P, y_pool, train_prior, target_prior, bundle,
                         spec, class_names, loss, args, out_dir, header_lines)


if __name__ == "__main__":
    main()
