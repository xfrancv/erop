"""Reject-option / test-prior adaptation on real datasets.

The real-data counterpart of ``run_synth_reject_option_exp.py``. Where the
synthetic script draws from a Gaussian generator and fits logistic regression,
this one consumes a **trained, temperature-calibrated neural-network base
predictor** produced by ``run_base_predictor_exp.py`` (its ``model.pt`` bundle)
and everything downstream — the MCMC prior learning, the plugin corrections,
the reject-option predictors and their risk/regret-coverage curves — is reused
unchanged from the synthetic pipeline.

Label shift is *simulated*: the real test pool (the base script's val+test
merge) is labeled, so it can be resampled to a chosen target prior
``p_te(y)``. Per trial the split is **adaptation-first**: the adaptation set
(``--n-test`` examples, or ``max(--sizes)`` in a sweep) is drawn at the target
prior from the whole pool -- per class, so it is stratified -- and the disjoint
remainder feeds the evaluation set. ``--n-eval`` defaults to the largest
all-distinct evaluation set that remainder supports, which maximises the eval
size and so minimises the variance of the reported metrics. The adaptation
inputs feed the MCMC (and their labels the supervised-prior baseline); every
predictor is scored on the evaluation set. There is no optimal-Bayes upper
bound here (the true class conditionals are unknown for real data); the regret
reference is the plugin given the true (target) test prior.

Predictors / baselines (same as the synthetic experiment):

- Plugin, training prior (no adaptation)
- Plugin, true test prior (oracle target prior)
- Plugin, supervised prior estimate (prior counted from the adaptation-set
  labels -- ``baseline_learned_prior_subervised_data.md``)
- Bayesian, learned prior (MCMC from the unlabeled adaptation inputs)

and the three reject-option predictors (Bayesian total / epistemic uncertainty,
supervised-prior plugin).

Run with::

    python run_base_predictor_exp.py bloodmnist runs/blood      # train base model
    python run_real_reject_option_exp.py runs/blood/model.pt runs/blood
    python run_real_reject_option_exp.py runs/blood/model.pt runs/blood --sweep

The target prior defaults to the training prior with the dataset's confusable
pair skewed to an asymmetric split (so a genuine, pair-targeted shift always
exists). Three knobs shape it: ``--confusable-pair I J`` chooses which two
classes are the pair (default: the dataset registry's), ``--pair-rest-ratio
A B`` sets the pair's total mass vs. the remaining classes (default: keep the
pair's training mass), and ``--pair-ratio`` splits the pair's mass between its
two classes. ``--test-prior`` overrides all three with an explicit vector.

With ``--dirichlet SUM_PARAMS`` the experiment instead repeats over target
priors sampled from ``Dir(s * p)`` centered on that configured prior
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

from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS
from prior_shift import (
    bayes_decision,
    corrected_posterior,
    sample_prior_posterior,
    zero_one_loss_matrix,
)
from run_base_predictor_exp import make_model, to_tensor
from run_synth_bayesian_learning_exp import _progress, accuracy, save_run_args
from run_synth_reject_option_exp import (
    AURC50_CAVEAT,
    AURC50_NOTE,
    REJECT_LABELS,
    _agg_desc,
    _band,
    _resolve_risk_targets,
    bayesian_posterior_and_aleatoric,
    configure_aggregation,
    configure_oracle,
    coverage_at_target,
    epistemic_metrics,
    generalize_curve,
    make_cov_target_figure,
    make_curve_figures,
    make_curves_at_n_figure,
    make_epistemic_metrics_figure,
    make_gen_curve_figures,
    make_gen_curves_at_n_figure,
    make_gen_sweep_figure,
    make_sweep_figure,
    make_trunc_sweep_figure,
    oracle_curves,
    selective_curves,
    trunc_sweep_fname,
    truncated_area,
)

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


# Same threshold as run_base_predictor_exp.CALIB_RATIO_THRESHOLD.
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
                     "consistency check; retrain with run_base_predictor_exp.py)")
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
    """Report line naming the actually-used confusable pair (or 'none')."""
    if pair_idx is None:
        return "confusable   : none (target = training prior, no shift)"
    i, j = pair_idx
    return (f"confusable   : {class_names[i]} / {class_names[j]}  "
            f"(classes {i}, {j}; {source})")


def default_target_prior(train_prior, pair_idx, pair_ratio=(1.0, 7.0),
                         pair_rest_ratio=None):
    """Target prior obtained by re-weighting the confusable pair.

    The confusable pair ``(i, j)`` is skewed within itself by ``pair_ratio``
    and, optionally, rescaled against the remaining classes by
    ``pair_rest_ratio``:

    - ``pair_rest_ratio=(a, b)`` sets the pair's total mass to ``a/(a+b)`` and
      the remaining classes' total to ``b/(a+b)``; the latter is distributed
      *proportionally to the training prior* over the non-pair classes.
    - ``pair_rest_ratio=None`` keeps the pair's training combined mass
      ``p_tr[i] + p_tr[j]``, which leaves every non-pair class at its training
      probability exactly -- reproducing the original behaviour.

    With ``pair_idx=None`` there is no pair to skew, so the target is the
    (normalised) training prior itself -- no shift.
    """
    q = np.array(train_prior, dtype=float)
    if pair_idx is None:
        return q / q.sum()
    i, j = pair_idx

    rest_mask = np.ones(len(q), dtype=bool)
    rest_mask[[i, j]] = False
    Q_rest = q[rest_mask].sum()

    if pair_rest_ratio is None:
        pair_total = q[i] + q[j]
    else:
        a, b = pair_rest_ratio
        pair_total = a / (a + b)
    # With no non-pair classes (pair covers everything) the rest has nowhere to
    # go, so the pair must hold all the mass.
    if Q_rest == 0:
        pair_total = 1.0
    rest_total = 1.0 - pair_total

    p = np.zeros_like(q)
    if rest_total > 0 and Q_rest > 0:
        p[rest_mask] = rest_total * q[rest_mask] / Q_rest
    r = np.array(pair_ratio, dtype=float)
    r = r / r.sum()
    p[i], p[j] = pair_total * r[0], pair_total * r[1]
    return p


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


def max_distinct_eval(eval_avail, target_prior):
    """Largest all-distinct evaluation size at ``target_prior`` given per-class
    availability ``eval_avail``: ``floor(min_c eval_avail[c] / target[c])`` over
    classes with positive target mass (0 if there are none)."""
    tp = np.asarray(target_prior, dtype=float)
    caps = [eval_avail[c] / tp[c] for c in range(len(tp)) if tp[c] > 0]
    return int(np.floor(min(caps))) if caps else 0


def split_adapt_eval(all_idx, y, target_prior, n_adapt, n_eval, rng,
                     adapt_replace=True):
    """Adaptation-first stratified split of the whole pool.

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


def run_real_trial(P, y, train_prior, target_prior, n_test, n_eval, loss, rng,
                   epi_threshold=1e-3, sampler="mh", beta=None, sup_beta=None,
                   adapt_replace=True):
    """One trial: resample a shifted adaptation pool + eval set, run predictors.

    ``P`` is the (N, Y) calibrated posterior of the whole labeled pool. The
    adaptation set (``n_test`` examples at the target prior) is drawn first from
    the whole pool; the disjoint remainder feeds the ``n_eval`` evaluation set
    (see ``split_adapt_eval``). ``sup_beta`` (a (Y,) Dirichlet concentration)
    switches the supervised prior estimate from raw label counts to the
    Dirichlet posterior mean; ``adapt_replace=False`` truncates the adaptation
    draw instead of resampling with replacement (both are dirichlet-mode
    settings).
    """
    Y = len(train_prior)
    N = len(y)
    all_idx = np.arange(N)
    adapt_idx, eval_idx, short_a, short_e, absent = split_adapt_eval(
        all_idx, y, target_prior, n_test, n_eval, rng,
        adapt_replace=adapt_replace)

    post_adapt = P[adapt_idx]
    post_ev = P[eval_idx]
    y_ev = y[eval_idx]

    # Bayesian prior learning from the unlabeled adaptation inputs.
    mcmc = sample_prior_posterior(post_adapt, train_prior, rng=rng,
                                  sampler=sampler, beta=beta)
    counts = np.bincount(y[adapt_idx], minlength=Y).astype(float)
    if sup_beta is None:
        supervised_prior = counts / counts.sum()
    else:
        supervised_prior = (counts + sup_beta) / (counts.sum() + sup_beta.sum())

    bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
        post_ev, train_prior, mcmc.samples, loss)
    cond_risk_bayes = bayes_post @ loss.T
    h_bayes = cond_risk_bayes.argmin(axis=1)
    total = cond_risk_bayes.min(axis=1)

    post_sup = corrected_posterior(post_ev, train_prior, supervised_prior)
    cond_risk_sup = post_sup @ loss.T
    h_sup = cond_risk_sup.argmin(axis=1)

    # Regret reference: plugin given the true (target) test prior.
    h_true = bayes_decision(
        corrected_posterior(post_ev, train_prior, target_prior), loss)
    losses_ref = loss[h_true, y_ev]

    predictors = {
        "bayes_total": (h_bayes, total),
        "bayes_epistemic": (h_bayes, total - aleatoric),
    }
    # Oracle envelope: risk ranked by actual loss, regret by actual regret.
    oracle = (oracle_curves(loss[h_bayes, y_ev], losses_ref)
              if "oracle" in REJECT_LABELS else None)

    # Accuracy of the four plugin/Bayesian predictors on the eval set. The
    # supervised-prior plugin is kept here (accuracy reference) though it is no
    # longer a reject-option predictor.
    acc = {
        "plugin_train_prior": accuracy(
            bayes_decision(
                corrected_posterior(post_ev, train_prior, train_prior), loss), y_ev),
        "plugin_true_test_prior": accuracy(h_true, y_ev),
        "plugin_supervised_prior": accuracy(h_sup, y_ev),
        "bayes_learned_prior": accuracy(h_bayes, y_ev),
    }

    epi = epistemic_metrics(
        total - aleatoric, loss[h_bayes, y_ev], losses_ref, epi_threshold)

    return {
        "predictors": predictors,
        "oracle": oracle,
        "loss": loss,
        "y_ev": y_ev,
        "losses_ref": losses_ref,
        "accuracy": acc,
        "learned_prior": mcmc.posterior_mean,
        "epi": epi,
        "short": short_a | short_e,
        "short_eval": short_e,
        "n_realized": len(adapt_idx),
        "absent": absent,
        "ident_warn": mcmc.identifiability_warning(),
    }


def run_sweep(P, y, train_prior, target_prior, sizes, trials, n_eval, loss,
              master_rng, epi_threshold, risk_targets=None,
              regret_targets=(0.002,), sampler="mh", beta=None, sup_beta=None,
              adapt_replace=True, progress_desc="sweep"):
    """AuRC and epistemic metrics as a function of the adaptation-set size.

    Per trial the adaptation pool of size ``max(sizes)`` is drawn first at the
    target prior from the whole pool, and the disjoint remainder feeds the
    fixed ``n_eval`` evaluation set (``split_adapt_eval``). The ``n`` adaptation
    examples are nested prefixes of that pool, so neighbouring sizes share
    draws and the curves reflect ``n`` rather than re-sampling noise.

    ``sup_beta`` and ``adapt_replace`` are the dirichlet-mode knobs of
    ``run_real_trial``; with ``adapt_replace=False`` the truncated pool can be
    shorter than a requested size, so the realized per-size adaptation counts
    are returned alongside the metrics.
    """
    Y = len(train_prior)
    N = len(y)
    all_idx = np.arange(N)
    sizes = sorted(sizes)
    n_max = sizes[-1]
    names = list(REJECT_LABELS.keys())
    aurc_risk = {n: np.zeros((len(sizes), trials)) for n in names}
    aurc_regret = {n: np.zeros((len(sizes), trials)) for n in names}
    warned = np.zeros((len(sizes), trials), dtype=bool)
    epi_metrics = np.zeros((len(sizes), trials, 3))
    rts, _ = _resolve_risk_targets(risk_targets)
    cov_risk = [{n: np.zeros((len(sizes), trials)) for n in names}
                for _ in rts]
    cov_regret = [{n: np.zeros((len(sizes), trials)) for n in names}
                  for _ in regret_targets]
    # Full per-size curves, kept for the per-n coverage-curve figures.
    risk_curves = {n: np.zeros((len(sizes), trials, n_eval)) for n in names}
    regret_curves = {n: np.zeros((len(sizes), trials, n_eval)) for n in names}
    # Base-predictor accuracy vs. n: Bayesian learned prior, plugin with the
    # true (target) prior, plugin with the supervised prior estimate. The
    # true-prior plugin does not use the adaptation examples, so it is constant
    # in n (flat curve); the other two adapt from the n examples.
    base_acc = {k: np.zeros((len(sizes), trials))
                for k in ("bayes_learned", "plugin_true", "plugin_supervised")}
    short_adapt: set[int] = set()
    short_eval: set[int] = set()
    realized_n = np.zeros((len(sizes), trials), dtype=int)

    with _progress(total=trials * len(sizes), desc=progress_desc) as bar:
        for t in range(trials):
            rng = np.random.default_rng(master_rng.integers(1 << 32))
            pool_idx, eval_idx, short_a, short_e, _absent = split_adapt_eval(
                all_idx, y, target_prior, n_max, n_eval, rng,
                adapt_replace=adapt_replace)
            short_adapt |= short_a
            short_eval |= short_e

            post_ev = P[eval_idx]
            y_ev = y[eval_idx]
            h_true = bayes_decision(
                corrected_posterior(post_ev, train_prior, target_prior), loss)
            losses_ref = loss[h_true, y_ev]
            resolved_rts = [float(losses_ref.mean()) if rt is None else rt
                            for rt in rts]
            acc_true = accuracy(h_true, y_ev)   # constant in n (no adaptation)

            for i, n in enumerate(sizes):
                adapt_idx = pool_idx[:n]
                realized_n[i, t] = len(adapt_idx)
                mcmc = sample_prior_posterior(
                    P[adapt_idx], train_prior, rng=rng, sampler=sampler,
                    beta=beta)
                warned[i, t] = mcmc.identifiability_warning() is not None

                bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
                    post_ev, train_prior, mcmc.samples, loss)
                cond_risk_bayes = bayes_post @ loss.T
                h_bayes = cond_risk_bayes.argmin(axis=1)
                total = cond_risk_bayes.min(axis=1)

                counts = np.bincount(y[adapt_idx], minlength=Y).astype(float)
                if sup_beta is None:
                    supervised_prior = counts / counts.sum()
                else:
                    supervised_prior = ((counts + sup_beta)
                                        / (counts.sum() + sup_beta.sum()))
                post_sup = corrected_posterior(
                    post_ev, train_prior, supervised_prior)
                cond_risk_sup = post_sup @ loss.T
                h_sup = cond_risk_sup.argmin(axis=1)

                base_acc["bayes_learned"][i, t] = accuracy(h_bayes, y_ev)
                base_acc["plugin_supervised"][i, t] = accuracy(h_sup, y_ev)
                base_acc["plugin_true"][i, t] = acc_true

                predictors = {
                    "bayes_total": (h_bayes, total),
                    "bayes_epistemic": (h_bayes, total - aleatoric),
                }
                curve_set = {
                    name: selective_curves(loss[h, y_ev], losses_ref, u)
                    for name, (h, u) in predictors.items()
                }
                if "oracle" in REJECT_LABELS:
                    curve_set["oracle"] = oracle_curves(
                        loss[h_bayes, y_ev], losses_ref)
                for name, (risk, regret) in curve_set.items():
                    risk_curves[name][i, t] = risk
                    regret_curves[name][i, t] = regret
                    aurc_risk[name][i, t] = risk.mean()
                    aurc_regret[name][i, t] = regret.mean()
                    for ti, rt_val in enumerate(resolved_rts):
                        cov_risk[ti][name][i, t] = coverage_at_target(risk, rt_val)
                    for ei, eps in enumerate(regret_targets):
                        cov_regret[ei][name][i, t] = coverage_at_target(regret, eps)
                epi_metrics[i, t] = epistemic_metrics(
                    total - aleatoric, loss[h_bayes, y_ev], losses_ref,
                    epi_threshold)
                bar.update(1)

    return (aurc_risk, aurc_regret, warned, epi_metrics, short_adapt,
            short_eval, cov_risk, cov_regret, risk_curves, regret_curves,
            base_acc, realized_n)


def make_base_accuracy_figure(
    sizes: list[int], base_acc: dict, trials: int, out_dir: str,
) -> None:
    """Base-predictor accuracy vs. the adaptation-set size, for the three
    predictors: Bayesian learned prior, plugin with the true (target) prior,
    and plugin with the supervised prior estimate. Each is a (len(sizes),
    trials) array; the true-prior plugin is flat in n (drawn as a curve for
    direct comparison)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(sizes, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    style = (
        ("bayes_learned", "Bayesian, learned prior", "C1", "o", "-"),
        ("plugin_supervised", "Plugin, supervised prior estimate", "C4", "s", "-"),
        ("plugin_true", "Plugin, true test prior (oracle)", "C0", None, "--"),
    )
    for key, label, color, marker, ls in style:
        mean = base_acc[key].mean(axis=1)
        sem = _band(base_acc[key], 1, trials)
        ax.plot(x, mean, lw=1.8, marker=marker, ls=ls, color=color, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of unlabeled adaptation examples $n$")
    ax.set_ylabel("test accuracy")
    ax.set_title("Base-predictor accuracy vs. test-set size "
                 f"({_agg_desc(trials)})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/base_accuracy_vs_n_test.png", dpi=130)
    plt.close(fig)


def _sweep_outputs(sizes, args, out_dir: Path, lines: list[str], aurc_risk,
                   aurc_regret, warned, epi_metrics, cov_risk, cov_regret,
                   risk_curves, regret_curves, base_acc, reps: int) -> None:
    """Append the sweep metric tables to ``lines``, write/print the report and
    build every sweep figure. Shared by the fixed-prior sweep (replicate axis =
    trials) and the dirichlet sweep (replicate axis = sampled priors, arrays
    hold per-prior means; ``configure_aggregation`` is set by the caller so the
    figure bands/titles describe the right thing). ``reps`` is the replicate
    count of the arrays' second axis."""
    names = list(REJECT_LABELS.keys())
    # Generalized curves and their areas: a rescaling of the selective curves by
    # the coverage, so no re-ranking is needed.
    gen_risk_curves = {n: generalize_curve(risk_curves[n]) for n in names}
    gen_regret_curves = {n: generalize_curve(regret_curves[n]) for n in names}
    augrc_risk = {n: gen_risk_curves[n].mean(axis=-1) for n in names}
    augrc_regret = {n: gen_regret_curves[n].mean(axis=-1) for n in names}
    # Areas over the high-coverage window only: a slice of the same curves.
    aurc50_risk = {n: truncated_area(risk_curves[n]) for n in names}
    aurc50_regret = {n: truncated_area(regret_curves[n]) for n in names}

    for metric, aurc in (("risk", aurc_risk), ("regret", aurc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC ({metric})")
        lines.append(f"{'n_test':>8}{'warn':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            row = f"{n:>8}{warned[i].mean():>8.2f}"
            row += "".join(f"{aurc[name][i].mean():>24.4f}" for name in names)
            lines.append(row)
    for metric, aurc50 in (("risk", aurc50_risk), ("regret", aurc50_regret)):
        lines.append("-" * 76)
        lines.append(f"AuRC50 ({metric})  ({AURC50_NOTE})")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{aurc50[name][i].mean():>24.4f}"
                                   for name in names))
    lines.append(AURC50_CAVEAT)
    for metric, augrc in (("risk", augrc_risk), ("regret", augrc_regret)):
        lines.append("-" * 76)
        lines.append(f"AuGRC ({metric})  (normalized by n_eval; not on the "
                     f"AuRC scale)")
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{augrc[name][i].mean():>24.4f}"
                                   for name in names))
    rts, rt_descs = _resolve_risk_targets(args.risk_target)
    risk_fig_descs = ["reference risk" if rt is None else d
                      for rt, d in zip(rts, rt_descs)]
    regret_fig_descs = [f"{e:g}" for e in args.regret_target]
    blocks = [(f"coverage @ risk <= {d}"
               + (" (per-trial reference full-coverage risk)" if rt is None else ""),
               cov) for rt, d, cov in zip(rts, rt_descs, cov_risk)]
    blocks += [(f"coverage @ regret <= {e:g}", cov)
               for e, cov in zip(args.regret_target, cov_regret)]
    for label, cov in blocks:
        lines.append("-" * 76)
        lines.append(label)
        lines.append(f"{'n_test':>8}"
                     + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            lines.append(f"{n:>8}"
                         + "".join(f"{cov[name][i].mean():>24.3f}" for name in names))
    lines.append("-" * 76)
    lines.append("Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"(threshold={args.epi_threshold:g})")
    lines.append(f"{'n_test':>8}{'avg epi':>14}{'avg regret':>14}{'portion negl':>14}")
    for i, n in enumerate(sizes):
        lines.append(f"{n:>8}{epi_metrics[i, :, 0].mean():>14.4f}"
                     f"{epi_metrics[i, :, 1].mean():>14.4f}"
                     f"{epi_metrics[i, :, 2].mean():>14.3f}")
    lines.append("=" * 76)
    if warned.any():
        lines.append("!!! IDENTIFIABILITY WARNING: 'warn' = fraction of trials "
                     "where the learned prior was only weakly identifiable "
                     "(see README).")
    report = "\n".join(lines)
    (out_dir / "real_reject_option_sweep_report.txt").write_text(report + "\n")
    print(report)

    make_sweep_figure(sizes, aurc_risk, aurc_regret, reps, args.out_dir)
    make_gen_sweep_figure(sizes, augrc_risk, augrc_regret, reps, args.out_dir)
    make_trunc_sweep_figure(sizes, aurc50_risk, aurc50_regret, reps,
                            args.out_dir)
    make_epistemic_metrics_figure(sizes, epi_metrics, args.epi_threshold,
                                  args.out_dir)
    make_cov_target_figure(sizes, cov_risk, cov_regret, reps,
                           risk_fig_descs, regret_fig_descs, args.out_dir)
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
          f"gen_aurc_vs_n_test.png, {trunc_sweep_fname()}.png, "
          f"epistemic_metrics_vs_n_test.png, cov_at_target_vs_n_test.png, "
          f"base_accuracy_vs_n_test.png, "
          f"coverage_curves/[gen_]coverage_curves_n<n_test>.png "
          f"(two per size)")


def run_sweep_report(P, y_pool, train_prior, target_prior, bundle, spec,
                     class_names, loss, args, out_dir: Path,
                     conf_line: str) -> None:
    """Drive the fixed-prior sweep, print/save the report, write the figures."""
    sizes = sorted(args.sizes)
    master_rng = np.random.default_rng(args.seed)
    (aurc_risk, aurc_regret, warned, epi_metrics, short_adapt, short_eval,
     cov_risk, cov_regret, risk_curves, regret_curves, base_acc,
     _realized_n) = run_sweep(
        P, y_pool, train_prior, target_prior, sizes, args.trials,
        args.n_eval, loss, master_rng, args.epi_threshold,
        args.risk_target, args.regret_target, sampler=args.sampler,
        beta=args.beta)
    shortfalls = short_adapt | short_eval

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
        f"n_eval {args.n_eval}   sizes {sizes}",
        f"prior beta   : {args.beta:g} per class (symmetric Dirichlet)",
        conf_line,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"target prior : {np.array2string(target_prior, precision=3)}",
    ]
    if shortfalls:
        pretty = ", ".join(class_names[c] for c in sorted(shortfalls))
        lines.append(f"note         : classes resampled WITH replacement "
                     f"(pool too small at this target prior): {pretty}")
    _sweep_outputs(sizes, args, out_dir, lines, aurc_risk, aurc_regret,
                   warned, epi_metrics, cov_risk, cov_regret, risk_curves,
                   regret_curves, base_acc, args.trials)


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epi_by_size = np.asarray(epi_by_size)
    S, N, _ = epi_by_size.shape
    fig, ax = plt.subplots(figsize=(6.6, 6))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(sizes):
        ax.scatter(epi_by_size[i, :, 0], epi_by_size[i, :, 1], s=36,
                   color=cmap(i / max(S - 1, 1)), alpha=0.85,
                   edgecolors="none", zorder=3, label=f"n = {n}")
    vals = epi_by_size[:, :, :2]
    lo = min(0.0, float(vals.min()))
    hi = float(vals.max()) * 1.05 + 1e-9
    ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1,
            label="y = x (calibrated)")
    ax.set_xlabel("average epistemic uncertainty (per sampled prior)")
    ax.set_ylabel("average realized regret at full coverage "
                  "(per sampled prior)")
    ax.set_title("Epistemic-uncertainty calibration of the Bayesian "
                 f"predictor\none point per (sampled prior, n); {N} priors")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/epi_vs_regret_calibration.png", dpi=130)
    plt.close(fig)


def run_dirichlet_sweep_report(P, y_pool, train_prior, central_prior, bundle,
                               spec, class_names, pair_idx, loss, args,
                               out_dir: Path, conf_line: str, model_beta,
                               misspec_line) -> None:
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

    aurc_risk_d = {n: np.zeros((S, N)) for n in names}
    aurc_regret_d = {n: np.zeros((S, N)) for n in names}
    warned_d = np.zeros((S, N))
    epi_d = np.zeros((S, N, 3))
    rts, _ = _resolve_risk_targets(args.risk_target)
    cov_risk_d = [{n: np.zeros((S, N)) for n in names} for _ in rts]
    cov_regret_d = [{n: np.zeros((S, N)) for n in names}
                    for _ in args.regret_target]
    risk_curves_d = {n: np.zeros((S, N, args.n_eval)) for n in names}
    regret_curves_d = {n: np.zeros((S, N, args.n_eval)) for n in names}
    base_acc_d = {k: np.zeros((S, N))
                  for k in ("bayes_learned", "plugin_true",
                            "plugin_supervised")}
    realized_d = np.zeros((S, N))
    short_eval_all: set[int] = set()
    alphas = np.zeros((N, len(central_prior)))

    for j in range(N):
        prng = np.random.default_rng(prior_seeds[j])
        alpha = sample_target_prior(prng, beta_gen)
        alphas[j] = alpha
        (aurc_risk, aurc_regret, warned, epi_metrics, _short_a, short_e,
         cov_risk, cov_regret, risk_curves, regret_curves, base_acc,
         realized_n) = run_sweep(
            P, y_pool, train_prior, alpha, sizes, T, args.n_eval, loss,
            prng, args.epi_threshold, args.risk_target, args.regret_target,
            sampler=args.sampler, beta=model_beta, sup_beta=model_beta,
            adapt_replace=False, progress_desc=f"prior {j + 1}/{N}")
        short_eval_all |= short_e
        for n in names:
            aurc_risk_d[n][:, j] = aurc_risk[n].mean(axis=1)
            aurc_regret_d[n][:, j] = aurc_regret[n].mean(axis=1)
            risk_curves_d[n][:, j] = risk_curves[n].mean(axis=1)
            regret_curves_d[n][:, j] = regret_curves[n].mean(axis=1)
        warned_d[:, j] = warned.mean(axis=1)
        epi_d[:, j] = epi_metrics.mean(axis=1)
        for ti in range(len(rts)):
            for n in names:
                cov_risk_d[ti][n][:, j] = cov_risk[ti][n].mean(axis=1)
        for ei in range(len(args.regret_target)):
            for n in names:
                cov_regret_d[ei][n][:, j] = cov_regret[ei][n].mean(axis=1)
        for k in base_acc_d:
            base_acc_d[k][:, j] = base_acc[k].mean(axis=1)
        realized_d[:, j] = realized_n.mean(axis=1)

    _write_sampled_priors(out_dir, alphas, prior_seeds, beta_gen)
    configure_aggregation(
        "std", f"{N}x{T} runs, ±1 std over {{reps}} priors")

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
        f"n_eval {args.n_eval}   sizes {sizes}",
        *_dirichlet_header_lines(args, misspec_line),
        conf_line,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"central prior: {np.array2string(central_prior, precision=3)}",
        *_sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names),
    ]
    if np.any(realized_d < np.asarray(sizes)[:, None]):
        per_size = ", ".join(
            f"{n}->{realized_d[i].mean():.1f}" for i, n in enumerate(sizes)
            if realized_d[i].mean() < n)
        lines.append(f"note         : adaptation sets truncated to pool "
                     f"availability (no replacement); mean realized n: "
                     f"{per_size}")
    if short_eval_all:
        lines.append("note         : evaluation sets drawn WITH replacement "
                     "where a class's pool fell short (expected in dirichlet "
                     "mode)")

    _sweep_outputs(sizes, args, out_dir, lines, aurc_risk_d, aurc_regret_d,
                   warned_d, epi_d, cov_risk_d, cov_regret_d, risk_curves_d,
                   regret_curves_d, base_acc_d, N)

    make_epi_regret_calibration_figure(sizes, epi_d, args.out_dir)
    print(f"calibration figure and sampled priors written to {out_dir}/: "
          f"epi_vs_regret_calibration.png, sampled_priors.txt")


def _single_outputs(args, out_dir: Path, lines: list[str], risk_curves,
                    regret_curves, accs, epi_metrics, cov_risk, cov_regret,
                    rep_label: str = "trials") -> None:
    """Append the single-size metric tables to ``lines``, write/print the
    report, and build the coverage-curve figures. Shared by the fixed-prior
    mode (replicate axis = trials) and the dirichlet mode (replicate axis =
    sampled priors; arrays hold per-prior means and ``configure_aggregation``
    is set by the caller)."""
    names = list(REJECT_LABELS.keys())
    aurc_risk = {n: risk_curves[n].mean(axis=1) for n in names}
    aurc_regret = {n: regret_curves[n].mean(axis=1) for n in names}
    # Generalized curves and their areas: a rescaling of the above by the
    # coverage, so no re-ranking is needed.
    gen_risk_curves = {n: generalize_curve(risk_curves[n]) for n in names}
    gen_regret_curves = {n: generalize_curve(regret_curves[n]) for n in names}
    augrc_risk = {n: gen_risk_curves[n].mean(axis=1) for n in names}
    augrc_regret = {n: gen_regret_curves[n].mean(axis=1) for n in names}
    # Areas over the high-coverage window only: a slice of the same curves.
    aurc50_risk = {n: truncated_area(risk_curves[n]) for n in names}
    aurc50_regret = {n: truncated_area(regret_curves[n]) for n in names}
    rts, rt_descs = _resolve_risk_targets(args.risk_target)

    lines.append("-" * 76)
    lines.append(f"{'predictor':<44}{'test acc':>12}{'std':>10}")
    lines.append("-" * 76)
    for name, label in PREDICTOR_LABELS.items():
        v = accs[name]
        lines.append(f"{label:<44}{np.mean(v):>12.4f}{np.std(v):>10.4f}")
    lines.append("-" * 76)
    lines.append(f"{'reject-option predictor':<46}{'AuRC risk':>14}{'AuRC regret':>14}")
    lines.append("-" * 76)
    for name in names:
        lines.append(f"{REJECT_LABELS[name]:<46}"
                     f"{aurc_risk[name].mean():>8.4f} ± {aurc_risk[name].std():.4f}"
                     f"{aurc_regret[name].mean():>8.4f} ± {aurc_regret[name].std():.4f}")
    lines.append("-" * 76)
    lines.append(AURC50_NOTE)
    lines.append(f"{'reject-option predictor':<46}{'AuRC50 risk':>14}"
                 f"{'AuRC50 regret':>14}")
    lines.append("-" * 76)
    for name in names:
        lines.append(
            f"{REJECT_LABELS[name]:<46}"
            f"{aurc50_risk[name].mean():>8.4f} ± {aurc50_risk[name].std():.4f}"
            f"{aurc50_regret[name].mean():>8.4f} ± {aurc50_regret[name].std():.4f}")
    lines.append(AURC50_CAVEAT)
    lines.append("-" * 76)
    lines.append("area under the generalized curves (normalized by n_eval, not "
                 "by the accepted count: not on the AuRC scale above)")
    lines.append(f"{'reject-option predictor':<46}{'AuGRC risk':>14}"
                 f"{'AuGRC regret':>14}")
    lines.append("-" * 76)
    for name in names:
        lines.append(f"{REJECT_LABELS[name]:<46}"
                     f"{augrc_risk[name].mean():>8.4f} ± {augrc_risk[name].std():.4f}"
                     f"{augrc_regret[name].mean():>8.4f} ± {augrc_regret[name].std():.4f}")
    lines.append("-" * 76)
    ref_note = ("  ('ref' = per-trial full-coverage risk of the true-prior "
                "reference)" if args.risk_target is None else "")
    lines.append(f"coverage at target (mean±std over {rep_label}){ref_note}")
    header = f"{'reject-option predictor':<46}"
    header += "".join(f"{'risk<=' + d:>14}" for d in rt_descs)
    header += "".join(f"{f'regret<={e:g}':>14}" for e in args.regret_target)
    lines.append(header)
    lines.append("-" * 76)
    for name in names:
        row = f"{REJECT_LABELS[name]:<46}"
        for cov in (*cov_risk, *cov_regret):
            row += f"{cov[name].mean():>8.3f}±{cov[name].std():.3f}"
        lines.append(row)
    lines.append("-" * 76)
    lines.append("epistemic-uncertainty metrics of the Bayesian predictor "
                 f"(threshold={args.epi_threshold:g})")
    for label, col in (("avg epistemic uncertainty", 0),
                       ("avg regret (full coverage)", 1),
                       ("portion with negligible epistemic uncertainty", 2)):
        lines.append(f"  {label:<48}"
                     f"{epi_metrics[:, col].mean():>9.4f} ± {epi_metrics[:, col].std():.4f}")
    lines.append("=" * 76)
    report = "\n".join(lines)
    (out_dir / "real_reject_option_report.txt").write_text(report + "\n")
    print(report)

    make_curve_figures(risk_curves, regret_curves, aurc_risk, aurc_regret,
                       args.n_eval, args.out_dir)
    make_gen_curve_figures(gen_risk_curves, gen_regret_curves,
                           augrc_risk, augrc_regret, args.n_eval, args.out_dir)
    print(f"\nreport and figures written to {out_dir}/: "
          f"real_reject_option_report.txt, risk_coverage.png, "
          f"regret_coverage.png, gen_risk_coverage.png, "
          f"gen_regret_coverage.png")


def run_dirichlet_single_report(P, y_pool, train_prior, central_prior, bundle,
                                spec, class_names, pair_idx, loss, args,
                                out_dir: Path, conf_line: str, model_beta,
                                misspec_line) -> None:
    """Single-size experiment repeated over N sampled target priors: the
    dirichlet-mode counterpart of the fixed-prior non-sweep path, collapsing
    each draw's trials to per-prior means before aggregation."""
    names = list(REJECT_LABELS.keys())
    N, T = args.trials_prior, args.trials
    Y = len(central_prior)
    master = np.random.default_rng(args.seed)
    prior_seeds = master.integers(1 << 32, size=N)
    beta_gen = args.dirichlet * central_prior

    risk_curves_d = {n: np.zeros((N, args.n_eval)) for n in names}
    regret_curves_d = {n: np.zeros((N, args.n_eval)) for n in names}
    accs_d = {k: np.zeros(N) for k in PREDICTOR_LABELS}
    epi_d = np.zeros((N, 3))
    rts, _ = _resolve_risk_targets(args.risk_target)
    cov_risk_d = [{n: np.zeros(N) for n in names} for _ in rts]
    cov_regret_d = [{n: np.zeros(N) for n in names}
                    for _ in args.regret_target]
    learned_d = np.zeros((N, Y))
    realized_d = np.zeros(N)
    alphas = np.zeros((N, Y))
    short_eval_all: set[int] = set()
    ident_count, ident_first = 0, None

    with _progress(total=N * T, desc="priors x trials") as bar:
        for j in range(N):
            prng = np.random.default_rng(prior_seeds[j])
            alpha = sample_target_prior(prng, beta_gen)
            alphas[j] = alpha
            rc = {n: np.zeros((T, args.n_eval)) for n in names}
            gc = {n: np.zeros((T, args.n_eval)) for n in names}
            ref_risks = np.zeros(T)
            epi_t = np.zeros((T, 3))
            acc_t = {k: np.zeros(T) for k in PREDICTOR_LABELS}
            learned_t = np.zeros((T, Y))
            for t in range(T):
                rng = np.random.default_rng(prng.integers(1 << 32))
                res = run_real_trial(
                    P, y_pool, train_prior, alpha, args.n_test, args.n_eval,
                    loss, rng, epi_threshold=args.epi_threshold,
                    sampler=args.sampler, beta=model_beta,
                    sup_beta=model_beta, adapt_replace=False)
                for name, (h, u) in res["predictors"].items():
                    risk, regret = selective_curves(
                        res["loss"][h, res["y_ev"]], res["losses_ref"], u)
                    rc[name][t], gc[name][t] = risk, regret
                if res["oracle"] is not None:
                    rc["oracle"][t], gc["oracle"][t] = res["oracle"]
                for k, a in res["accuracy"].items():
                    acc_t[k][t] = a
                epi_t[t] = res["epi"]
                ref_risks[t] = res["losses_ref"].mean()
                learned_t[t] = res["learned_prior"]
                realized_d[j] += res["n_realized"] / T
                short_eval_all |= res["short_eval"]
                if res["ident_warn"] is not None:
                    ident_count += 1
                    if ident_first is None:
                        ident_first = res["ident_warn"]
                bar.update(1)
            for name in names:
                risk_curves_d[name][j] = rc[name].mean(axis=0)
                regret_curves_d[name][j] = gc[name].mean(axis=0)
                for ti, rt in enumerate(rts):
                    cov_risk_d[ti][name][j] = np.mean([
                        coverage_at_target(rc[name][t],
                                           ref_risks[t] if rt is None else rt)
                        for t in range(T)])
                for ei, eps in enumerate(args.regret_target):
                    cov_regret_d[ei][name][j] = np.mean([
                        coverage_at_target(gc[name][t], eps)
                        for t in range(T)])
            for k in accs_d:
                accs_d[k][j] = acc_t[k].mean()
            epi_d[j] = epi_t.mean(axis=0)
            learned_d[j] = learned_t.mean(axis=0)

    _write_sampled_priors(out_dir, alphas, prior_seeds, beta_gen)
    configure_aggregation(
        "std", f"{N}x{T} runs, ±1 std over {{reps}} priors")

    lines = [
        "=" * 76,
        "Reject-option / test-prior adaptation on real data "
        "(sampled target priors)",
        "=" * 76,
        f"timestamp    : {datetime.now().isoformat(timespec='seconds')}",
        f"command      : {' '.join(sys.argv)}",
        f"base model   : {spec.display_name} ({bundle['dataset']}), "
        f"arch {bundle['arch']}",
        *calibration_lines(bundle, class_names),
        f"pool size    : {len(y_pool)}   priors {N} x trials {T}   "
        f"n_test {args.n_test}   n_eval {args.n_eval}",
        *_dirichlet_header_lines(args, misspec_line),
        conf_line,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"central prior: {np.array2string(central_prior, precision=3)}",
        f"learned prior: {np.array2string(learned_d.mean(axis=0), precision=3)}  "
        f"(posterior mean, over priors x trials)",
        *_sampled_prior_lines(alphas, prior_seeds, pair_idx, class_names),
    ]
    if np.any(realized_d < args.n_test):
        lines.append(f"note         : adaptation sets truncated to pool "
                     f"availability (no replacement); mean realized n_test = "
                     f"{realized_d.mean():.1f} of {args.n_test}")
    if short_eval_all:
        lines.append("note         : evaluation sets drawn WITH replacement "
                     "where a class's pool fell short (expected in dirichlet "
                     "mode)")
    if ident_count:
        lines.append(f"!!! IDENTIFIABILITY WARNING (fired in "
                     f"{ident_count}/{N * T} runs) !!!")
        lines.append(f"    {ident_first}")

    _single_outputs(args, out_dir, lines, risk_curves_d, regret_curves_d,
                    accs_d, epi_d, cov_risk_d, cov_regret_d,
                    rep_label="sampled priors")

    make_epi_regret_calibration_figure([args.n_test], epi_d[None, :, :],
                                       args.out_dir)
    print(f"calibration figure and sampled priors written to {out_dir}/: "
          f"epi_vs_regret_calibration.png, sampled_priors.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=str,
                        help="Path to a model.pt bundle from run_base_predictor_exp.py.")
    parser.add_argument("out_dir", type=str,
                        help="Directory receiving the report and figures.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=500,
                        help="Unlabeled adaptation-set size.")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Labeled evaluation-set size. Default: the maximum "
                             "all-distinct size at the target prior (the "
                             "adaptation set is drawn first, the rest go to "
                             "evaluation); in dirichlet mode, where that "
                             "maximum is undefined, a fixed 1000 drawn with "
                             "replacement where needed. Pass an integer to "
                             "pin it.")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep the adaptation-set size and plot AuRC-vs-n instead of "
             "the single-size curves.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[50, 100, 200, 500, 1000, 2000],
        help="Adaptation-set sizes (the swept variable) for --sweep.")
    parser.add_argument(
        "--epi-threshold", type=float, default=0.001,
        help="Epistemic uncertainty below this value counts as negligible "
             "in the reported portion metric.")
    parser.add_argument(
        "--optimal-rejection", action="store_true",
        help="Also evaluate the oracle reject-option baseline (best attainable "
             "selective risk/regret, ranked by the actual per-example loss and "
             "regret). It is label-aware, so it is off by default.")
    parser.add_argument(
        "--risk-target", type=float, nargs="+", default=None,
        help="Risk budget(s) for the coverage-at-target metric; the metric is "
             "computed for each value given. Default: a single budget, the "
             "per-trial full-coverage risk of the true-prior reference "
             "predictor.")
    parser.add_argument(
        "--regret-target", type=float, nargs="+", default=[0.002],
        help="Regret budget(s) for the coverage-at-target metric; the metric "
             "is computed for each value given.")
    parser.add_argument("--test-prior", type=float, nargs="+", default=None,
                        help="Explicit target test prior (Y floats, summing to "
                             "1). Default: train prior with the confusable pair "
                             "skewed asymmetrically.")
    parser.add_argument("--pair-ratio", type=float, nargs=2, default=(1.0, 7.0),
                        help="Asymmetric split of the confusable pair's mass "
                             "between its two classes (default 1 7).")
    parser.add_argument("--confusable-pair", type=int, nargs=2, default=None,
                        metavar=("I", "J"),
                        help="Two 0-based class indices to treat as the "
                             "confusable pair, overriding the dataset registry "
                             "default (e.g. --confusable-pair 0 4).")
    parser.add_argument("--pair-rest-ratio", type=float, nargs=2, default=None,
                        metavar=("A", "B"),
                        help="Split of total target-prior mass between the "
                             "confusable pair (A/(A+B)) and the remaining "
                             "classes (B/(A+B)); the rest is spread "
                             "proportionally to the training prior. Default: "
                             "keep the pair's training combined mass.")
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
             "total concentration and p the central target prior built by "
             "--pair-ratio / --confusable-pair / --pair-rest-ratio / "
             "--test-prior. Larger s concentrates the draws around p; "
             "s -> inf recovers the fixed-prior experiment.")
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_oracle(args.optimal_rejection)

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

    # Resolve the confusable pair: explicit --confusable-pair, else registry.
    if args.confusable_pair is not None:
        i, j = args.confusable_pair
        if not (0 <= i < Y and 0 <= j < Y):
            sys.exit(f"error: --confusable-pair indices must be in [0, {Y})")
        if i == j:
            sys.exit("error: --confusable-pair needs two distinct classes")
        pair_idx, pair_source = (i, j), "user override"
    elif spec.confusable_pair is not None:
        pair_idx = tuple(class_names.index(c) for c in spec.confusable_pair)
        pair_source = "registry default"
    else:
        pair_idx, pair_source = None, "none"

    if args.pair_rest_ratio is not None:
        a, b = args.pair_rest_ratio
        if a < 0 or b < 0 or a + b <= 0:
            sys.exit("error: --pair-rest-ratio needs two non-negative floats "
                     "with A + B > 0")
        if pair_idx is None:
            sys.exit("error: --pair-rest-ratio needs a confusable pair, but the "
                     "dataset has no registry pair; pass --confusable-pair too")

    if args.test_prior is not None:
        if args.confusable_pair is not None or args.pair_rest_ratio is not None:
            print("warning: --test-prior overrides --confusable-pair / "
                  "--pair-rest-ratio / --pair-ratio")
        target_prior = np.asarray(args.test_prior, dtype=float)
        if len(target_prior) != Y or not np.isclose(target_prior.sum(), 1.0):
            sys.exit(f"error: --test-prior must be {Y} floats summing to 1")
    else:
        target_prior = default_target_prior(
            train_prior, pair_idx, args.pair_ratio, args.pair_rest_ratio)
        if args.pair_rest_ratio is not None:
            a, b = args.pair_rest_ratio
            if b == 0:
                print("warning: --pair-rest-ratio puts all mass on the pair; "
                      "the eval set will have no decoy classes (degenerate for "
                      "the epistemic-vs-total contrast)")
            if a == 0:
                print("warning: --pair-rest-ratio puts zero mass on the pair; "
                      "the confusable pair will be absent from the eval set")

    conf_line = confusable_report_line(pair_idx, class_names, pair_source)
    print(conf_line)

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
                     "--test-prior / --pair-rest-ratio.")
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
    n_adapt = max(args.sizes) if args.sweep else args.n_test
    pool_counts = np.bincount(y_pool, minlength=Y)
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
                     f"{names}. Adjust --test-prior / --confusable-pair.")
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
                            f"{pair_source})"),
        "n_eval_resolved": f"{args.n_eval}"
                           + ((" (dirichlet-mode default)" if dirichlet_mode
                               else " (auto max at target prior)")
                              if n_eval_auto else " (explicit)"),
    }
    if dirichlet_mode:
        extra["model_prior"] = (
            "matched: beta = s * central prior (well specified)"
            if misspec_line is None else
            f"symmetric beta = {args.beta:g} per class (MISSPECIFIED)")
    ignored = {"n_test"} if args.sweep else {"sizes"}
    if not dirichlet_mode:
        ignored |= {"dirichlet", "trials_prior"}
    save_run_args(
        args,
        "run_real_reject_option_exp_sweep_args.txt" if args.sweep
        else "run_real_reject_option_exp_args.txt",
        extra=extra,
        ignored=ignored,
    )

    if args.sweep:
        if dirichlet_mode:
            run_dirichlet_sweep_report(P, y_pool, train_prior, target_prior,
                                       bundle, spec, class_names, pair_idx,
                                       loss, args, out_dir, conf_line,
                                       model_beta, misspec_line)
        else:
            run_sweep_report(P, y_pool, train_prior, target_prior, bundle,
                             spec, class_names, loss, args, out_dir,
                             conf_line)
        return

    if dirichlet_mode:
        run_dirichlet_single_report(P, y_pool, train_prior, target_prior,
                                    bundle, spec, class_names, pair_idx, loss,
                                    args, out_dir, conf_line, model_beta,
                                    misspec_line)
        return

    # --- trials -------------------------------------------------------------
    names = list(REJECT_LABELS.keys())
    risk_curves = {n: np.zeros((args.trials, args.n_eval)) for n in names}
    regret_curves = {n: np.zeros((args.trials, args.n_eval)) for n in names}
    accs: dict[str, list[float]] = {}
    epi_metrics = np.zeros((args.trials, 3))
    ref_risks = np.zeros(args.trials)
    learned_priors = []
    shortfalls, ident_warnings = set(), []
    master_rng = np.random.default_rng(args.seed)

    with _progress(total=args.trials, desc="trials") as bar:
        for t in range(args.trials):
            rng = np.random.default_rng(master_rng.integers(1 << 32))
            res = run_real_trial(P, y_pool, train_prior, target_prior,
                                 args.n_test, args.n_eval, loss, rng,
                                 epi_threshold=args.epi_threshold,
                                 sampler=args.sampler, beta=args.beta)
            for name, (h, u) in res["predictors"].items():
                risk, regret = selective_curves(
                    res["loss"][h, res["y_ev"]], res["losses_ref"], u)
                risk_curves[name][t] = risk
                regret_curves[name][t] = regret
            if res["oracle"] is not None:
                risk_curves["oracle"][t], regret_curves["oracle"][t] = res["oracle"]
            for k, a in res["accuracy"].items():
                accs.setdefault(k, []).append(a)
            epi_metrics[t] = res["epi"]
            ref_risks[t] = res["losses_ref"].mean()
            learned_priors.append(res["learned_prior"])
            shortfalls |= res["short"]
            if res["ident_warn"] is not None:
                ident_warnings.append(res["ident_warn"])
            bar.update(1)

    rts, _ = _resolve_risk_targets(args.risk_target)
    cov_risk = [
        {n: np.array([
            coverage_at_target(risk_curves[n][t],
                               ref_risks[t] if rt is None else rt)
            for t in range(args.trials)]) for n in names}
        for rt in rts]
    cov_regret = [
        {n: np.array([
            coverage_at_target(regret_curves[n][t], eps)
            for t in range(args.trials)]) for n in names}
        for eps in args.regret_target]
    learned_prior_mean = np.mean(learned_priors, axis=0)

    # --- report -------------------------------------------------------------
    lines = [
        "=" * 76,
        "Reject-option / test-prior adaptation on real data",
        "=" * 76,
        f"timestamp    : {datetime.now().isoformat(timespec='seconds')}",
        f"command      : {' '.join(sys.argv)}",
        f"base model   : {spec.display_name} ({bundle['dataset']}), "
        f"arch {bundle['arch']}",
        *calibration_lines(bundle, class_names),
        f"pool size    : {len(y_pool)}   trials {args.trials}   "
        f"n_test {args.n_test}   n_eval {args.n_eval}",
        f"prior beta   : {args.beta:g} per class (symmetric Dirichlet)",
        conf_line,
        f"train prior  : {np.array2string(train_prior, precision=3)}",
        f"target prior : {np.array2string(target_prior, precision=3)}",
        f"learned prior: {np.array2string(learned_prior_mean, precision=3)}  "
        f"(posterior mean, over trials)",
    ]
    if shortfalls:
        pretty = ", ".join(class_names[c] for c in sorted(shortfalls))
        lines.append(f"note         : classes resampled WITH replacement "
                     f"(pool too small at this target prior): {pretty}")
    if ident_warnings:
        lines.append(f"!!! IDENTIFIABILITY WARNING (fired in "
                     f"{len(ident_warnings)}/{args.trials} trials) !!!")
        lines.append(f"    {ident_warnings[0]}")
    _single_outputs(args, out_dir, lines, risk_curves, regret_curves, accs,
                    epi_metrics, cov_risk, cov_regret)


if __name__ == "__main__":
    main()
