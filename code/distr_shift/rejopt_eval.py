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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import figspec
from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS
from prior_shift import (
    bayes_decision,
    corrected_posterior,
    sample_prior_posterior,
    zero_one_loss_matrix,
)
from prior_shift.reject_option import (
    AURC50_CAVEAT,
    AURC50_NOTE,
    REJECT_LABELS,
    _agg_desc,
    _center,
    _series,
    accuracy,
    bayesian_posterior_and_aleatoric,
    configure_aggregation,
    configure_percentile_band,
    coverage_at_target,
    epistemic_metrics,
    generalize_curve,
    selective_curves,
    sweep_avg_row,
    sweep_epi_avg_row,
    truncated_area,
)
from reject_figures import (
    make_cov_target_figure,
    make_curves_at_n_figure,
    make_epistemic_metrics_figure,
    make_gen_curves_at_n_figure,
    make_gen_sweep_figure,
    make_trunc_sweep_figure,
    sweep_panels,
    trunc_sweep_fname,
)
from base_predictor_training import make_model, to_tensor

try:  # progress bars are optional -- the script runs without tqdm installed.
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def _progress(total: int, desc: str):
    """A tqdm progress bar, or a no-op stand-in if tqdm is unavailable.

    The returned object supports ``update(k)`` and ``close()`` and, as a context
    manager, closes itself on exit. Bars render to stderr so the printed result
    tables on stdout stay clean.
    """
    if tqdm is not None:
        return tqdm(total=total, desc=desc)

    class _NoBar:
        def update(self, n: int = 1):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _NoBar()


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

# Accuracy-table predictors (no optimal-Bayes bound: no true conditionals).
PREDICTOR_LABELS = {
    "plugin_true_test_prior": "Plugin, true test prior (oracle)",
    "plugin_supervised_prior": "Plugin, supervised prior estimate",
    "bayes_learned_prior": "Bayesian, learned prior (proposed)",
    "plugin_train_prior": "Plugin, training prior (no adaptation)",
}


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


def target_prior_from_weights(Y, classes, weights, rest_weight):
    """Target prior from a subset of classes, their weights, and a fallback.

    ``classes`` and ``weights`` are positionally aligned sequences of length
    ``k <= Y``; every class *not* named in ``classes`` gets ``rest_weight``
    individually (a per-class weight, not a total mass to share, so the
    unnamed classes are always equiprobable among themselves). The result is
    the normalised weight vector

        p(c) = w(c) / sum_c' w(c'),

    with ``w(c) = weights[i]`` where ``c == classes[i]``, else ``rest_weight``.
    With ``k == Y`` no class is unnamed and ``rest_weight`` is unused.

    Validation of the arguments happens at the call site (argument parsing),
    which owns the error messages; this only assumes them well formed.
    """
    w = np.full(Y, float(rest_weight if rest_weight is not None else 0.0))
    w[np.asarray(classes, dtype=int)] = np.asarray(weights, dtype=float)
    return w / w.sum()


def class_weights_how(classes, weights, rest_weight, class_names) -> str:
    """The ``how`` fragment of ``target_prior_report_line`` for a target prior
    built from class weights."""
    named = ", ".join(f"{class_names[c]}={w:g}"
                      for c, w in zip(classes, weights))
    if rest_weight is None or len(classes) == len(class_names):
        return f"class weights  {named} (all classes named)"
    return f"class weights  {named}, rest={rest_weight:g} (per class)"


def target_counts(m, target_prior):
    """Per-class integer counts for an ``m``-sample draw at ``target_prior``.

    Floor of ``m * target_prior``, with the rounding remainder handed to the
    largest fractional parts so the counts sum to exactly ``m``. Deterministic
    in ``(m, target_prior)`` -- used both to draw and to size the pools.
    """
    tp = np.asarray(target_prior, dtype=float)
    counts = np.floor(m * tp).astype(int)
    remainder = int(m - counts.sum())
    if remainder > 0:
        frac = m * tp - counts
        counts[np.argsort(-frac)[:remainder]] += 1
    return counts


def resample_to_prior(source_idx, labels, target_prior, m, rng,
                      replace_short=True):
    """Draw ``m`` indices from ``source_idx`` so labels follow ``target_prior``.

    Sampling is without replacement per class where the pool is large enough.
    A class whose request exceeds its pool is flagged in the returned ``short``
    set and either drawn WITH replacement (``replace_short=True``, the default)
    or truncated to the whole class pool (``replace_short=False`` -- the
    dirichlet-mode adaptation rule, where duplicated inputs would double-count
    evidence in the MCMC likelihood), in which case the returned index set is
    shorter than ``m``. A class the target wants but that has *no* examples in
    ``source_idx`` is skipped and named in the returned ``absent`` set instead
    of raising -- callers guard against this upstream, so in normal operation
    ``absent`` is empty.
    """
    Y = len(target_prior)
    counts = target_counts(m, target_prior)

    chosen, short, absent = [], set(), set()
    for c in range(Y):
        if counts[c] == 0:
            continue
        pool_c = source_idx[labels[source_idx] == c]
        if len(pool_c) == 0:
            absent.add(c)
            continue
        take = counts[c]
        replace = take > len(pool_c)
        if replace:
            short.add(c)
            if not replace_short:
                take, replace = len(pool_c), False
        chosen.append(rng.choice(pool_c, size=take, replace=replace))
    idx = (np.concatenate(chosen) if chosen
           else np.empty(0, dtype=np.asarray(source_idx).dtype))
    rng.shuffle(idx)
    return idx, short, absent


def eval_set_report_field(args) -> str:
    """The report header's evaluation-set field: a size for the disjoint split,
    a statement of identity under ``--eval-on-adapt`` (where the evaluation size
    is the swept size, so no single number describes it)."""
    return ("eval set = adaptation set" if args.eval_on_adapt
            else f"n_eval {args.n_eval}")


def max_distinct_eval(eval_avail, target_prior):
    """Largest all-distinct evaluation size at ``target_prior`` given per-class
    availability ``eval_avail``: ``floor(min_c eval_avail[c] / target[c])`` over
    classes with positive target mass (0 if there are none)."""
    tp = np.asarray(target_prior, dtype=float)
    caps = [eval_avail[c] / tp[c] for c in range(len(tp)) if tp[c] > 0]
    return int(np.floor(min(caps))) if caps else 0


def split_adapt_eval(all_idx, y, target_prior, n_adapt, n_eval, rng,
                     adapt_replace=True):
    """Adaptation-first stratified split of the whole pool (the default,
    disjoint evaluation design -- ``--eval-on-adapt`` draws no evaluation set of
    its own and does not call this).

    Draw ``n_adapt`` adaptation indices at the target prior from the whole pool
    (``resample_to_prior`` draws per class, so this is stratified by
    construction), give the disjoint remainder to evaluation, and draw
    ``n_eval`` evaluation indices from that remainder. With
    ``adapt_replace=False`` the adaptation draw is truncated to pool
    availability instead of resampling with replacement (dirichlet mode); the
    evaluation draw always allows replacement. Returns ``(adapt_idx, eval_idx,
    short_adapt, short_eval, absent)`` where the short sets name the classes
    that fell short on each side.
    """
    adapt_idx, short_a, absent_a = resample_to_prior(
        all_idx, y, target_prior, n_adapt, rng, replace_short=adapt_replace)
    eval_source = np.setdiff1d(all_idx, adapt_idx)
    eval_idx, short_e, absent_e = resample_to_prior(
        eval_source, y, target_prior, n_eval, rng)
    return adapt_idx, eval_idx, short_a, short_e, absent_a | absent_e


@dataclass
class SweepResult:
    """Everything one sweep reports, on a common ``(len(sizes), reps)`` layout.

    ``run_sweep`` fills it with ``reps = trials``; the dirichlet driver builds
    the same shape with ``reps = trials_prior`` from per-prior means, so
    ``_sweep_outputs`` serves both. Every scalar area (AuRC, AuRC50, AuGRC) is
    computed per replicate from that replicate's *full* selective curve, so the
    numbers never depend on the ragged-curve truncation below.

    ``risk_curves`` / ``regret_curves`` map a predictor name to a **list over
    sizes** of ``(reps, curve_len[i])`` arrays -- a list, not one array,
    because under ``--eval-on-adapt`` the curve length is the evaluation size,
    which is the swept size itself. They exist only for the per-size
    coverage-curve figures.
    """
    aurc_risk: dict
    aurc_regret: dict
    aurc50_risk: dict
    aurc50_regret: dict
    augrc_risk: dict
    augrc_regret: dict
    cov_regret: list
    risk_curves: dict
    regret_curves: dict
    curve_len: np.ndarray
    warned: np.ndarray
    epi_metrics: np.ndarray
    base_acc: dict
    floor: dict
    realized_n: np.ndarray
    short_adapt: set
    short_eval: set


def stack_curves(per_size_lists):
    """Stack ragged per-replicate selective curves into one array per size.

    ``per_size_lists[i]`` is the list of that size's per-replicate curves.
    Replicates can differ in length (dirichlet-mode truncation of the
    adaptation pool, which under ``--eval-on-adapt`` is also the evaluation
    set), and the figures need a rectangular array, so each size's curves are
    cut to the shortest replicate. Returns ``(stacked, lengths)`` with
    ``stacked[i]`` of shape ``(reps, lengths[i])``; the lengths are reported so
    a cut is visible rather than silent.
    """
    stacked, lengths = [], []
    for curves in per_size_lists:
        m = min(len(c) for c in curves)
        stacked.append(np.stack([np.asarray(c)[:m] for c in curves]))
        lengths.append(m)
    return stacked, np.asarray(lengths, dtype=int)


def run_sweep(P, y, train_prior, target_prior, sizes, trials, n_eval, loss,
              master_rng, epi_threshold, regret_targets=(0.002,),
              sampler="mh", beta=None, adapt_replace=True,
              eval_on_adapt=False, progress_desc="sweep"):
    """AuRC and epistemic metrics as a function of the adaptation-set size.

    Per trial the adaptation pool of size ``max(sizes)`` is drawn at the target
    prior from the whole pool (``resample_to_prior`` draws per class, so it is
    stratified) and the ``n`` adaptation examples are nested prefixes of it, so
    neighbouring sizes share draws and the curves reflect ``n`` rather than
    re-sampling noise. With ``eval_on_adapt`` that prefix is *also* the
    evaluation set (``merge_eval_adapt_sets_polished.md``); otherwise the
    disjoint remainder feeds a fixed ``n_eval`` evaluation set
    (``split_adapt_eval``).

    ``adapt_replace`` is a dirichlet-mode knob; with ``adapt_replace=False``
    the truncated pool can be shorter than a requested size, so the realized
    per-size counts are reported in ``SweepResult.realized_n``.
    """
    Y = len(train_prior)
    N = len(y)
    all_idx = np.arange(N)
    sizes = sorted(sizes)
    n_max = sizes[-1]
    names = list(REJECT_LABELS.keys())
    S = len(sizes)

    def per_predictor():
        return {n: np.zeros((S, trials)) for n in names}

    aurc_risk, aurc_regret = per_predictor(), per_predictor()
    aurc50_risk, aurc50_regret = per_predictor(), per_predictor()
    augrc_risk, augrc_regret = per_predictor(), per_predictor()
    cov_regret = [per_predictor() for _ in regret_targets]
    warned = np.zeros((S, trials), dtype=bool)
    epi_metrics = np.zeros((S, trials, 3))
    # Full per-size curves, kept for the per-n coverage-curve figures. Ragged
    # until stack_curves cuts each size to its shortest replicate.
    risk_curves = {n: [[] for _ in sizes] for n in names}
    regret_curves = {n: [[] for _ in sizes] for n in names}
    # Base-predictor accuracy vs. n: Bayesian learned prior, plugin with the
    # true (target) prior, plugin with the unadapted training prior. Only the
    # Bayesian predictor uses the adaptation examples, but under
    # ``eval_on_adapt`` the evaluation set changes with n, so the two plugins
    # vary with n too (through the evaluation sample, not through adaptation).
    base_acc = {k: np.zeros((S, trials))
                for k in ("bayes_learned", "plugin_true", "plugin_train")}
    # Transductive floor: the plugin at the evaluation batch's own empirical
    # prior -- what any prior-adaptation method could at best reach on that
    # batch, and hence how much of the (possibly negative) regret against the
    # population oracle is plain finite-batch prior mismatch.
    floor = {k: np.zeros((S, trials))
             for k in ("regret", "accuracy", "classes")}
    short_adapt: set[int] = set()
    short_eval: set[int] = set()
    realized_n = np.zeros((S, trials), dtype=int)

    with _progress(total=trials * len(sizes), desc=progress_desc) as bar:
        for t in range(trials):
            rng = np.random.default_rng(master_rng.integers(1 << 32))
            if eval_on_adapt:
                pool_idx, short_a, _absent = resample_to_prior(
                    all_idx, y, target_prior, n_max, rng,
                    replace_short=adapt_replace)
                eval_idx = None
            else:
                pool_idx, eval_idx, short_a, short_e, _absent = (
                    split_adapt_eval(all_idx, y, target_prior, n_max, n_eval,
                                     rng, adapt_replace=adapt_replace))
                short_eval |= short_e
            short_adapt |= short_a

            for i, n in enumerate(sizes):
                adapt_idx = pool_idx[:n]
                realized_n[i, t] = len(adapt_idx)
                idx_ev = adapt_idx if eval_on_adapt else eval_idx
                post_ev = P[idx_ev]
                y_ev = y[idx_ev]
                # The regret reference and the two non-adaptive plugins. They
                # are loop-invariant only with a fixed evaluation set; under
                # eval_on_adapt the evaluation set is the size-n prefix, so
                # they are recomputed per size.
                h_true = bayes_decision(
                    corrected_posterior(post_ev, train_prior, target_prior),
                    loss)
                losses_ref = loss[h_true, y_ev]
                # Plugin with the unadapted training prior: alpha = train_prior
                # leaves p_tr(y|x) unchanged, so this is the base classifier's
                # own Bayes decision.
                h_train = bayes_decision(post_ev, loss)

                emp_prior = np.bincount(y_ev, minlength=Y) / len(y_ev)
                h_emp = bayes_decision(
                    corrected_posterior(post_ev, train_prior, emp_prior), loss)
                floor["regret"][i, t] = float(
                    np.mean(loss[h_emp, y_ev] - losses_ref))
                floor["accuracy"][i, t] = accuracy(h_emp, y_ev)
                floor["classes"][i, t] = int(np.count_nonzero(emp_prior))

                mcmc = sample_prior_posterior(
                    P[adapt_idx], train_prior, rng=rng, sampler=sampler,
                    beta=beta)
                warned[i, t] = mcmc.identifiability_warning() is not None

                bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
                    post_ev, train_prior, mcmc.samples, loss)
                cond_risk_bayes = bayes_post @ loss.T
                h_bayes = cond_risk_bayes.argmin(axis=1)
                total = cond_risk_bayes.min(axis=1)

                base_acc["bayes_learned"][i, t] = accuracy(h_bayes, y_ev)
                base_acc["plugin_train"][i, t] = accuracy(h_train, y_ev)
                base_acc["plugin_true"][i, t] = accuracy(h_true, y_ev)

                predictors = {
                    "bayes_total": (h_bayes, total),
                    "bayes_epistemic": (h_bayes, total - aleatoric),
                }
                curve_set = {
                    name: selective_curves(loss[h, y_ev], losses_ref, u)
                    for name, (h, u) in predictors.items()
                }
                for name, (risk, regret) in curve_set.items():
                    risk_curves[name][i].append(risk)
                    regret_curves[name][i].append(regret)
                    # Every area from the full realized curve, so ragged
                    # lengths never reach the reported numbers.
                    aurc_risk[name][i, t] = risk.mean()
                    aurc_regret[name][i, t] = regret.mean()
                    aurc50_risk[name][i, t] = truncated_area(risk)
                    aurc50_regret[name][i, t] = truncated_area(regret)
                    augrc_risk[name][i, t] = generalize_curve(risk).mean()
                    augrc_regret[name][i, t] = generalize_curve(regret).mean()
                    for ei, eps in enumerate(regret_targets):
                        cov_regret[ei][name][i, t] = coverage_at_target(regret, eps)
                epi_metrics[i, t] = epistemic_metrics(
                    total - aleatoric, loss[h_bayes, y_ev], losses_ref,
                    epi_threshold)
                bar.update(1)

    stacked_risk, curve_len = {}, None
    stacked_regret = {}
    for name in names:
        stacked_risk[name], curve_len = stack_curves(risk_curves[name])
        stacked_regret[name], _ = stack_curves(regret_curves[name])

    return SweepResult(
        aurc_risk=aurc_risk, aurc_regret=aurc_regret,
        aurc50_risk=aurc50_risk, aurc50_regret=aurc50_regret,
        augrc_risk=augrc_risk, augrc_regret=augrc_regret,
        cov_regret=cov_regret, risk_curves=stacked_risk,
        regret_curves=stacked_regret, curve_len=curve_len, warned=warned,
        epi_metrics=epi_metrics, base_acc=base_acc, floor=floor,
        realized_n=realized_n, short_adapt=short_adapt, short_eval=short_eval)


def base_accuracy_panel(
    sizes: list[int], base_acc: dict, trials: int,
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
        center, lo, hi = _series(base_acc[key], 1, trials)
        series.append(figspec.Series(
            x=x, center=center, lower=lo, upper=hi, marker=marker,
            linestyle=ls, color=color, label=label))

    if title is None:
        title = f"Base-predictor accuracy vs. test-set size ({_agg_desc(trials)})"
    return figspec.Panel(
        series=series, xscale="log", xticks=list(sizes),
        xlabel=xlabel, ylabel="test accuracy", title=title,
        legend=True, legend_loc="lower right", grid_which="both")


def make_base_accuracy_figure(
    sizes: list[int], base_acc: dict, trials: int, out_dir: str,
) -> None:
    """Base-predictor accuracy vs. the adaptation-set size (one-panel figure)."""
    panel = base_accuracy_panel(sizes, base_acc, trials)
    spec = figspec.FigureSpec(panels=[panel], figsize=[8.5, 5.0])
    figspec.write(spec, f"{out_dir}/base_accuracy_vs_n_test.png")


def make_sweep_area_figures(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict,
    reps: int, out_dir: str, short_name: str,
) -> None:
    """The AuRC (risk) and AuReC (regret) sweep panels as two *independent*
    single-panel figures, ``aurc_vs_n_test.png`` and ``aurec_vs_n_test.png``.

    Written separately (rather than combined, or bundled with the accuracy panel
    into one overview figure) so each curve drops into a paper on its own. The
    accuracy panel is its own figure via ``make_base_accuracy_figure``.
    ``figsize=None`` so the size follows the render style sheet."""
    xlabel = "number of unlabeled adaptation examples $m$"
    risk_panel, regret_panel = sweep_panels(
        sizes, aurc_risk, aurc_regret, reps, xlabel=xlabel,
        titles=(f"{short_name}: AuRC vs. adaptation set size $m$",
                f"{short_name}: AuReC vs. adaptation set size $m$"))
    for panel, fname in ((risk_panel, "aurc_vs_n_test"),
                         (regret_panel, "aurec_vs_n_test")):
        figspec.write(figspec.FigureSpec(panels=[panel], figsize=None),
                      f"{out_dir}/{fname}.png")


# The area-under-curve report tables (AuRC / AuRC50 / AuGRC, risk and regret)
# print on a x1000 scale so more significant digits survive the fixed decimal
# width. Applies to the text tables only -- figures and the raw arrays (hence
# the win-rate comparisons, which are scale-invariant) are untouched.
AREA_SCALE = 1000
AREA_SCALE_TAG = "x1000"


def win_rate_block(metric: str, aurc: dict, sizes: list[int], names: list[str],
                   ref: str = "bayes_total", measure: str = "AuRC") -> list[str]:
    """Report lines: how often ``ref`` beats each competitor over sampled priors.

    Dirichlet-mode only. ``aurc[name]`` is the ``(len(sizes), N)`` array of
    per-prior mean areas of the ``measure`` in question (AuRC / AuRC50 / AuGRC,
    risk or regret -- the same quantity the corresponding table centres). For
    each competitor column the cell is the percentage of the ``N`` sampled
    priors on which ``ref`` has the strictly lower (better) area at that
    adaptation size -- a paired, per-prior head-to-head win rate. The ``ref``
    column itself is left blank (no self-comparison); the ``all`` row pools the
    comparison over every (size, prior) pair."""
    n_priors = aurc[ref].shape[1]
    ref_label = REJECT_LABELS[ref]
    out = [
        "-" * 76,
        f"win% {measure} ({metric}): sampled priors (of {n_priors}) where "
        f"'{ref_label}' has the lower area than each competitor",
        f"{'n_test':>8}{'':>8}"
        + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names),
    ]

    def cell(win_arr) -> str:
        return f"{100.0 * float(np.mean(win_arr)):>24.1f}"

    for i, n in enumerate(sizes):
        row = f"{n:>8}{'':>8}"
        row += "".join(f"{'':>24}" if name == ref
                       else cell(aurc[ref][i] < aurc[name][i]) for name in names)
        out.append(row)
    row = f"{'all':>8}{'':>8}"
    row += "".join(f"{'':>24}" if name == ref
                   else cell(aurc[ref] < aurc[name]) for name in names)
    out.append(row)
    return out


def _sweep_outputs(sizes, args, out_dir: Path, lines: list[str],
                   res: SweepResult, reps: int, display_name: str = "",
                   report_win_rate: bool = False) -> None:
    """Append the sweep metric tables to ``lines``, write/print the report and
    build every sweep figure. Shared by the fixed-prior sweep (replicate axis =
    trials) and the dirichlet sweep (replicate axis = sampled priors, arrays
    hold per-prior means; ``configure_aggregation`` is set by the caller so the
    figure bands/titles describe the right thing). ``reps`` is the replicate
    count of the arrays' second axis. ``report_win_rate`` (dirichlet mode only)
    appends a per-competitor win-rate block after each AuRC table.

    Every area comes from ``res`` (computed per replicate from its full curve);
    only the *generalized curves* are derived here, and only for the per-size
    figures -- a rescaling of the selective curves by the coverage, so no
    re-ranking is needed."""
    names = list(REJECT_LABELS.keys())
    aurc_risk, aurc_regret = res.aurc_risk, res.aurc_regret
    aurc50_risk, aurc50_regret = res.aurc50_risk, res.aurc50_regret
    augrc_risk, augrc_regret = res.augrc_risk, res.augrc_regret
    warned, epi_metrics, cov_regret = res.warned, res.epi_metrics, res.cov_regret
    risk_curves, regret_curves, base_acc = (res.risk_curves, res.regret_curves,
                                            res.base_acc)
    gen_risk_curves = {n: [generalize_curve(c) for c in risk_curves[n]]
                       for n in names}
    gen_regret_curves = {n: [generalize_curve(c) for c in regret_curves[n]]
                         for n in names}

    for metric, aurc in (("risk", aurc_risk), ("regret", aurc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC ({metric})  [{AREA_SCALE_TAG}]")
        lines.append(f"{'n_test':>8}{'warn':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            row = f"{n:>8}{warned[i].mean():>8.2f}"
            row += "".join(f"{_center(aurc[name][i]) * AREA_SCALE:>24.4f}"
                           for name in names)
            lines.append(row)
        lines.append(sweep_avg_row(aurc, names, decimals=4, warn=warned,
                                   scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, aurc, sizes, names,
                                        measure="AuRC"))
    for metric, aurc50 in (("risk", aurc50_risk), ("regret", aurc50_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC50 ({metric})  [{AREA_SCALE_TAG}]  ({AURC50_NOTE})")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{_center(aurc50[name][i]) * AREA_SCALE:>24.4f}"
                                   for name in names))
        lines.append(sweep_avg_row(aurc50, names, decimals=4, scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, aurc50, sizes, names,
                                        measure="AuRC50"))
    lines.append(AURC50_CAVEAT)
    for metric, augrc in (("risk", augrc_risk), ("regret", augrc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuGRC ({metric})  [{AREA_SCALE_TAG}]  (normalized by "
                     f"n_eval; not on the AuRC scale)")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{_center(augrc[name][i]) * AREA_SCALE:>24.4f}"
                                   for name in names))
        lines.append(sweep_avg_row(augrc, names, decimals=4, scale=AREA_SCALE))
        if report_win_rate:
            lines.extend(win_rate_block(metric, augrc, sizes, names,
                                        measure="AuGRC"))
    regret_fig_descs = [f"{e:g}" for e in args.regret_target]
    blocks = [(f"coverage @ regret <= {e:g}", cov)
              for e, cov in zip(args.regret_target, cov_regret)]
    for label, cov in blocks:
        lines.append("-" * 76)
        lines.append(label)
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{_center(cov[name][i]):>24.3f}" for name in names))
        lines.append(sweep_avg_row(cov, names, decimals=3))
    lines.append("-" * 76)
    lines.append("Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"(threshold={args.epi_threshold:g})")
    lines.append(f"{'n_test':>8}{'avg epi':>14}{'avg regret':>14}{'portion negl':>14}")
    for i, n in enumerate(sizes):
        lines.append(f"{n:>8}{_center(epi_metrics[i, :, 0]):>14.4f}"
                     f"{_center(epi_metrics[i, :, 1]):>14.4f}"
                     f"{_center(epi_metrics[i, :, 2]):>14.3f}")
    lines.append(sweep_epi_avg_row(epi_metrics))
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
        lines.append(f"{n:>8}{_center(floor['regret'][i]) * AREA_SCALE:>18.4f}"
                     f"{_center(floor['accuracy'][i]):>12.4f}"
                     f"{_center(floor['classes'][i]):>14.1f}"
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
    report = "\n".join(lines)
    (out_dir / "real_reject_option_sweep_report.txt").write_text(report + "\n")
    print(report)

    # aurc_vs_n_test + aurec_vs_n_test: the AuRC (risk) and AuReC (regret) sweeps
    # as two independent single-panel figures (dataset-named titles, "$m$" for
    # the adaptation-set size, figsize=None so the size follows the style sheet).
    short_name = display_name.split(" (")[0] if display_name else "real data"
    make_sweep_area_figures(sizes, aurc_risk, aurc_regret, reps, args.out_dir,
                            short_name)
    make_gen_sweep_figure(sizes, augrc_risk, augrc_regret, reps, args.out_dir)
    make_trunc_sweep_figure(sizes, aurc50_risk, aurc50_regret, reps,
                            args.out_dir)
    make_epistemic_metrics_figure(sizes, epi_metrics, args.epi_threshold,
                                  args.out_dir)
    make_cov_target_figure(sizes, cov_regret, reps, regret_fig_descs,
                           args.out_dir)
    make_base_accuracy_figure(sizes, base_acc, reps, args.out_dir)
    for i, n in enumerate(sizes):
        make_curves_at_n_figure(
            {name: risk_curves[name][i] for name in names},
            {name: regret_curves[name][i] for name in names},
            n, args.out_dir)
        make_gen_curves_at_n_figure(
            {name: gen_risk_curves[name][i] for name in names},
            {name: gen_regret_curves[name][i] for name in names},
            n, args.out_dir)
    print(f"\nreport and figures written to {out_dir}/: "
          f"real_reject_option_sweep_report.txt, aurc_vs_n_test.png, "
          f"aurec_vs_n_test.png, gen_aurc_vs_n_test.png, "
          f"{trunc_sweep_fname()}.png, "
          f"epistemic_metrics_vs_n_test.png, cov_at_target_vs_n_test.png, "
          f"base_accuracy_vs_n_test.png, "
          f"coverage_curves/[gen_]coverage_curves_n<n_test>.png "
          f"(two per size)")


def run_sweep_report(P, y_pool, train_prior, target_prior, bundle, spec,
                     class_names, loss, args, out_dir: Path,
                     header_lines: list[str]) -> None:
    """Drive the fixed-prior sweep, print/save the report, write the figures."""
    sizes = sorted(args.sizes)
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
    _sweep_outputs(sizes, args, out_dir, lines, res, args.trials,
                   display_name=spec.display_name)


def sample_target_prior(rng, beta_gen, max_tries=100):
    """One target prior drawn from ``Dir(beta_gen)``, guarded against the
    numerical underflow of tiny concentrations (numpy returns NaN when every
    gamma draw underflows to zero); such draws are rejected and redrawn."""
    for _ in range(max_tries):
        alpha = rng.dirichlet(beta_gen)
        if np.all(np.isfinite(alpha)) and abs(alpha.sum() - 1.0) < 1e-6:
            return alpha
    sys.exit("error: could not draw a finite target prior from Dir(s * p); "
             "--dirichlet is likely too small")


def _dirichlet_header_lines(args, misspec_line) -> list[str]:
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


def _sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names,
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


def _write_sampled_priors(out_dir: Path, alphas, prior_seeds,
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

    _write_sampled_priors(out_dir, alphas, prior_seeds, beta_gen)
    configure_aggregation(
        "std", f"{N}x{T} runs, ±1 std over {{reps}} priors",
        noun="priors")

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
        *_dirichlet_header_lines(args, misspec_line),
        *header_lines,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"central prior: {np.array2string(central_prior, precision=3)}",
        *_sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names),
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

    _sweep_outputs(sizes, args, out_dir, lines, d, N,
                   display_name=spec.display_name, report_win_rate=True)

    make_epi_regret_calibration_figure(sizes, d.epi_metrics, args.out_dir)
    print(f"calibration figure and sampled priors written to {out_dir}/: "
          f"epi_vs_regret_calibration.png, sampled_priors.txt")


def main() -> None:
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
    args = parser.parse_args()
    if args.percentile_band is not None and not 0 <= args.percentile_band <= 100:
        parser.error("--percentile-band must be in [0, 100]")
    if args.eval_on_adapt and args.n_eval is not None:
        parser.error("--n-eval is meaningless with --eval-on-adapt: the "
                     "evaluation set is the adaptation set, so its size is "
                     "--sizes")

    configure_percentile_band(args.percentile_band)

    # The two target-prior strategies are mutually exclusive; with neither the
    # target is the training prior (no shift). Checked here, before the model is
    # loaded, so a flag clash fails immediately rather than a minute in.
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
        args.trials_prior = 5
    if dirichlet_mode and args.trials_prior <= 0:
        sys.exit("error: --trials-prior must be positive")
    if args.beta is not None and args.beta <= 0:
        sys.exit("error: --beta must be positive")
    if not dirichlet_mode and args.beta is None:
        args.beta = 1.0

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

    for line in calibration_lines(bundle, bundle["class_names"]):
        print(line)

    ds = load_dataset(bundle["dataset"])
    # Test pool = the same val+test merge the base script scored on.
    if "val" in ds.splits:
        X_pool = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
        y_pool = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
    else:
        X_pool, y_pool = ds.splits["test"]
    P = calibrated_posterior(model, X_pool, bundle, device)

    # --- target prior -------------------------------------------------------
    spec = DATASETS[bundle["dataset"]]

    # The dataset registry's designated confusable pair. Reporting only (the
    # report line and, in dirichlet mode, the per-draw pair marginals); it takes
    # no part in building the target prior.
    if spec.confusable_pair is not None:
        pair_idx = tuple(class_names.index(c) for c in spec.confusable_pair)
        pair_source = "registry default"
    else:
        pair_idx, pair_source = None, "none"

    if args.test_prior is not None:
        target_prior = np.asarray(args.test_prior, dtype=float)
        if len(target_prior) != Y or not np.isclose(target_prior.sum(), 1.0):
            sys.exit(f"error: --test-prior must be {Y} floats summing to 1")
        prior_how = "explicit (--test-prior)"
    elif weight_flags:
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
        target_prior = target_prior_from_weights(
            Y, args.prior_classes, args.prior_weights,
            args.prior_rest_weight if k < Y else None)
        zero = [class_names[c] for c in np.flatnonzero(target_prior <= 0)]
        if zero:
            print(f"warning: zero weight on {', '.join(zero)}; "
                  "absent from the adaptation and evaluation sets")
        prior_how = class_weights_how(
            args.prior_classes, args.prior_weights,
            args.prior_rest_weight if len(args.prior_classes) < Y else None,
            class_names)
    else:
        # No target-prior flag: the target IS the training prior, i.e. no label
        # shift. Deliberate (the degenerate control), but invisible in the
        # numbers -- every predictor then coincides up to posterior noise -- so
        # it is named in the report and warned about on stdout.
        target_prior = train_prior / train_prior.sum()
        prior_how = TRAIN_PRIOR_DEFAULT_HOW

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

    # --- model prior (dirichlet mode) ---------------------------------------
    # The data-generating Dirichlet is always Dir(s * p). The model prior the
    # methods use matches it (well specified) unless --beta overrides it with
    # a symmetric prior -- the deliberate-misspecification control.
    misspec_line = None
    if dirichlet_mode:
        if np.any(target_prior <= 0):
            zero = ", ".join(class_names[c]
                             for c in np.flatnonzero(target_prior <= 0))
            sys.exit("error: --dirichlet needs positive central mass on every "
                     f"class, but these have none: {zero}. Adjust "
                     "--test-prior / --prior-weights / --prior-rest-weight.")
        if args.beta is None:
            model_beta = args.dirichlet * target_prior
        else:
            model_beta = np.full(Y, args.beta)
            misspec_line = ("!!! MODEL PRIOR MISSPECIFIED via --beta: methods "
                            f"use symmetric Dirichlet({args.beta:g}) while "
                            "target priors are drawn from Dir(s * p) !!!")
            print(misspec_line)
    else:
        model_beta = args.beta   # scalar; sample_prior_posterior broadcasts

    # --- resolve the evaluation-set size ------------------------------------
    # The adaptation set (n_adapt examples at the target prior) is drawn first
    # from the whole pool; the remainder feeds evaluation. n_eval defaults to
    # the largest all-distinct evaluation set that remainder supports.
    n_adapt = max(args.sizes)
    pool_counts = np.bincount(y_pool, minlength=Y)
    if args.eval_on_adapt:
        # No separate evaluation draw, so no evaluation feasibility to check:
        # the only pool requirement is that the target prior's classes are
        # present at all (a shortfall is handled by replacement/truncation, as
        # for any adaptation draw).
        wanted = [c for c in range(Y)
                  if dirichlet_mode or target_prior[c] > 0]
        missing = [c for c in wanted if pool_counts[c] == 0]
        if missing:
            names = ", ".join(f"{class_names[c]} (class {c})" for c in missing)
            sys.exit(f"error: the pool has no examples of: {names}; "
                     + ("dirichlet mode needs every class present in the pool."
                        if dirichlet_mode else
                        "adjust --test-prior / --prior-weights."))
        n_eval_auto = False
        print(f"adapt size   : {n_adapt}   eval size : same set "
              f"(--eval-on-adapt; evaluation size = each swept size)")
    elif dirichlet_mode:
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
            args.n_eval = 1000
        n_eval_note = "  (dirichlet-mode default)" if n_eval_auto else ""
        print(f"adapt size   : {n_adapt} (truncated per draw to pool "
              f"availability)   eval size : {args.n_eval}{n_eval_note}")
    else:
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
    ignored = set() if dirichlet_mode else {"dirichlet", "trials_prior"}
    if args.eval_on_adapt:
        ignored.add("n_eval")   # rejected at parse time; never read
    save_run_args(args, "rejopt_eval_args.txt",
                  extra=extra, ignored=ignored)

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
