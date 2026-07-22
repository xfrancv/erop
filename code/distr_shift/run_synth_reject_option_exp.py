"""Reject-option experiment: selective prediction under Bayesian prior adaptation.

Extends ``run_synth_bayesian_learning_exp.py`` (same setting: unsupervised test-prior adaptation
under label shift) with three reject-option predictors, each a pair of a base
predictor ``h(x)`` and an uncertainty score ``u(x)``:

1. **Bayesian, total uncertainty** — Bayesian learning predictor with
   ``u(x) = T_hat(x, D)``, the conditional risk of ``h(x, D)`` under the
   posterior-averaged label distribution.
2. **Bayesian, epistemic uncertainty** — the same base predictor with
   ``u(x) = T_hat(x, D) - A_hat(x, D)``, where the aleatoric part ``A_hat``
   averages the per-draw minimal conditional risk over the MCMC samples.
3. **Plugin, supervised prior (reference)** — plugin label-shift predictor
   using the prior counted from the *labels* of the adaptation set, with
   ``u(x)`` its estimated conditional risk.  A supervised reference baseline.

Unlike ``run_synth_bayesian_learning_exp.py``, both modes score on a **fixed labeled evaluation
set** disjoint from the ``n_test`` adaptation examples, so the supervised prior
baseline carries no in-sample bias.  Predictors are ranked by ``u`` on the
evaluation set and judged by risk-coverage and regret-coverage curves (regret
is measured against the plugin predictor given the *true* test prior) and the
areas under them, ``AuRC = (1/n) sum_k metric(k)``.

Run with::

    python run_synth_reject_option_exp.py                 # curves + AuRC table
    python run_synth_reject_option_exp.py --sweep         # AuRC vs. n_test

Figures are written to ``figures/`` and result tables are printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prior_shift import (
    BaseModel,
    GaussianClassConditionalModel,
    bayes_decision,
    corrected_posterior,
    estimate_prior_em,
    load_experiment_config,
    sample_prior_posterior,
    zero_one_loss_matrix,
)
from run_synth_bayesian_learning_exp import _progress, accuracy, save_run_args

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.json"
TRAIN_PRIOR = np.array([0.25, 0.25, 0.25, 0.25])
TEST_PRIOR = np.array([0.60, 0.20, 0.15, 0.05])

# The reject-option predictors (order fixed for tables and figures). The
# oracle is the best-attainable envelope (label-aware, metric-specific ranking);
# it is not a deployable predictor but marks the lower bound for each metric.
REJECT_LABELS = {
    "bayes_total": "Bayesian, total uncertainty",
    "bayes_epistemic": "Bayesian, epistemic uncertainty",
    "em_plugin": "EM plugin (learned prior)",
    "oracle": "Oracle (best attainable)",
}
REJECT_COLORS = {
    "bayes_total": "C1",
    "bayes_epistemic": "C2",
    "em_plugin": "C3",
    "oracle": "C7",
}


def configure_oracle(enabled: bool) -> None:
    """Include the oracle baseline in the reject-option set, or drop it.

    The evaluation loops, report tables and figure builders all key off the
    module-level ``REJECT_LABELS``, so enabling/disabling the oracle is done by
    keeping or removing its entry -- call this once from ``main`` before any
    evaluation runs. The oracle is off by default (``--optimal-rejection``
    turns it on) because it is a label-aware reference, not a deployable
    predictor.
    """
    if not enabled:
        REJECT_LABELS.pop("oracle", None)
        REJECT_COLORS.pop("oracle", None)


# Replicate-axis aggregation used by every figure builder: how the shaded band
# is computed from the replicate axis and how titles describe it. The default
# reproduces the historical behaviour (s.e.m. over trials); the real-data
# dirichlet mode switches to std over sampled priors via
# ``configure_aggregation`` (the ``{reps}`` placeholder receives the replicate
# count the figure was given).
_AGG_BAND = "sem"
_AGG_DESC = "mean ± s.e.m., {reps} trials"


def configure_aggregation(band: str = "sem",
                          desc: str = "mean ± s.e.m., {reps} trials") -> None:
    """Set the figures' replicate-axis band ('sem' or 'std') and title text."""
    global _AGG_BAND, _AGG_DESC
    if band not in ("sem", "std"):
        raise ValueError(f"band must be 'sem' or 'std', got {band!r}")
    _AGG_BAND, _AGG_DESC = band, desc


def _band(arr: np.ndarray, axis: int, reps: int) -> np.ndarray:
    """Half-width of the shaded band along the replicate axis."""
    s = arr.std(axis=axis)
    return s / np.sqrt(reps) if _AGG_BAND == "sem" else s


def _agg_desc(reps: int) -> str:
    """Title fragment describing the aggregation, e.g. 'mean ± s.e.m., 10
    trials'."""
    return _AGG_DESC.format(reps=reps)


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


def em_plugin_predictor(
    post_adapt: np.ndarray,   # (m, Y) = p_tr(y | x) of the unlabeled adaptation inputs
    post_eval: np.ndarray,    # (n, Y) = p_tr(y | x) of the evaluation inputs
    train_prior: np.ndarray,  # (Y,)
    loss: np.ndarray,         # (Y, Y) with loss[yhat, y]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM-plugin reject-option predictor ``(h_em, u_em, em_prior)``.

    Estimates the test prior by EM on the unlabeled adaptation posteriors (the
    MLE point estimate; no Dirichlet regularisation), forms the plugin
    label-shift posterior ``q_em`` on the evaluation set, and returns the Bayes
    decision ``h_em`` and its conditional-Bayes-risk uncertainty ``u_em =
    min_yhat sum_y q_em(y|x) loss[yhat, y]``. Being a point estimate it has no
    epistemic component (total == aleatoric), so it contributes one curve. The
    learned prior is returned too, for the accuracy-vs-n reporting.
    """
    em_prior = estimate_prior_em(post_adapt, train_prior)
    q_em = corrected_posterior(post_eval, train_prior, em_prior)
    cond_risk = q_em @ loss.T
    return cond_risk.argmin(axis=1), cond_risk.min(axis=1), em_prior


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


def oracle_curves(losses_pred, losses_ref):
    """Best-attainable selective curves for a base predictor.

    The risk-coverage curve is ranked by the predictor's *actual* per-example
    loss and the regret-coverage curve by its *actual* per-example regret --
    two metric-specific, label-aware orderings. Sorting by the realised loss (or
    regret) minimises the selective metric at every coverage, so these are the
    lower envelopes: no ranking rule can do better on the given evaluation
    sample. Returns ``(risk, regret)``; each is taken from the call whose
    ranking matches that metric.
    """
    risk, _ = selective_curves(losses_pred, losses_ref, losses_pred)
    _, regret = selective_curves(losses_pred, losses_ref, losses_pred - losses_ref)
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

# Shared header note and caveat for the AuRC50 report blocks (both scripts,
# both modes). The caveat goes under the numbers, where they are read.
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


def run_reject_trial(
    model: GaussianClassConditionalModel,
    m_train: int,
    n_test: int,
    n_eval: int,
    rng: np.random.Generator,
    mcmc_kwargs: dict | None = None,
):
    """One trial: train, adapt on n_test unlabeled inputs, score on n_eval.

    Returns a dict with, per reject-option predictor, the predictions and
    uncertainty scores on the evaluation set, plus the reference losses,
    evaluation labels, accuracies, and MCMC diagnostics.
    """
    Y = model.num_classes
    loss = zero_one_loss_matrix(Y)
    mcmc_kwargs = mcmc_kwargs or {}

    # --- training ---
    X_tr, y_tr = model.sample(m_train, TRAIN_PRIOR, rng)
    base = BaseModel.fit(X_tr, y_tr, num_classes=Y)

    # --- adaptation pool: inputs feed the MCMC ---
    X_te, y_te = model.sample(n_test, TEST_PRIOR, rng)
    est_post_te = base.posterior(X_te)
    mcmc = sample_prior_posterior(
        est_post_te, base.train_prior, rng=rng, **mcmc_kwargs
    )

    # --- fixed labeled evaluation set (disjoint from the adaptation pool) ---
    X_ev, y_ev = model.sample(n_eval, TEST_PRIOR, rng)
    est_post_ev = base.posterior(X_ev)

    # Bayesian predictor: mixture posterior, total and aleatoric uncertainty.
    bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
        est_post_ev, base.train_prior, mcmc.samples, loss
    )
    cond_risk_bayes = bayes_post @ loss.T          # (n, Y) risk of each yhat
    h_bayes = cond_risk_bayes.argmin(axis=1)
    total = cond_risk_bayes.min(axis=1)            # T_hat = risk of h(x, D)

    # True-prior plugin reference for the regret.
    h_true = bayes_decision(
        corrected_posterior(est_post_ev, base.train_prior, TEST_PRIOR), loss)
    losses_ref = loss[h_true, y_ev]

    # EM-plugin baseline: MLE test prior + plugin Bayes reject-option predictor.
    h_em, u_em, _em_prior = em_plugin_predictor(
        est_post_te, est_post_ev, base.train_prior, loss)

    predictors = {
        "bayes_total": (h_bayes, total),
        "bayes_epistemic": (h_bayes, total - aleatoric),
        "em_plugin": (h_em, u_em),
    }

    acc = {name: accuracy(h, y_ev) for name, (h, _) in predictors.items()}
    acc["plugin_true_prior (regret reference)"] = accuracy(h_true, y_ev)
    acc["opt_bayes_test (upper bound)"] = accuracy(
        bayes_decision(model.true_posterior(X_ev, TEST_PRIOR), loss), y_ev)

    return {
        "predictors": predictors,
        "loss": loss,
        "y_ev": y_ev,
        "losses_ref": losses_ref,
        "accuracy": acc,
        "mcmc": mcmc,
        "learned_prior": mcmc.posterior_mean,
    }


def curves_from_trial(res: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Risk/regret curves of every reject-option predictor for one trial,
    plus the oracle envelope (metric-specific ranking of the Bayesian
    predictor)."""
    out = {}
    for name, (h, u) in res["predictors"].items():
        losses_pred = res["loss"][h, res["y_ev"]]
        out[name] = selective_curves(losses_pred, res["losses_ref"], u)
    if "oracle" in REJECT_LABELS:
        h_bayes = res["predictors"]["bayes_total"][0]
        losses_bayes = res["loss"][h_bayes, res["y_ev"]]
        out["oracle"] = oracle_curves(losses_bayes, res["losses_ref"])
    return out


# ---------------------------------------------------------------------------
# Non-sweep mode: full curves at a single adaptation-set size.
# ---------------------------------------------------------------------------

def run_single_experiment(model, args, master_rng) -> None:
    names = list(REJECT_LABELS.keys())
    risk_curves = {n: np.zeros((args.trials, args.n_eval)) for n in names}
    regret_curves = {n: np.zeros((args.trials, args.n_eval)) for n in names}
    accs: dict[str, list[float]] = {}
    prior_errs: list[float] = []
    ident_warnings: list[str] = []
    epi_metrics = np.zeros((args.trials, 3))   # avg epi, avg regret, portion
    ref_risks = np.zeros(args.trials)          # full-coverage risk of the reference

    with _progress(total=args.trials, desc="trials") as bar:
        for t in range(args.trials):
            rng = np.random.default_rng(master_rng.integers(1 << 32))
            res = run_reject_trial(
                model, args.m_train, args.n_test, args.n_eval, rng,
                mcmc_kwargs={"sampler": args.sampler})
            for name, (risk, regret) in curves_from_trial(res).items():
                risk_curves[name][t] = risk
                regret_curves[name][t] = regret
            h_b, epi_u = res["predictors"]["bayes_epistemic"]
            epi_metrics[t] = epistemic_metrics(
                epi_u, res["loss"][h_b, res["y_ev"]], res["losses_ref"],
                args.epi_threshold)
            ref_risks[t] = res["losses_ref"].mean()
            for k, a in res["accuracy"].items():
                accs.setdefault(k, []).append(a)
            prior_errs.append(
                float(np.abs(res["learned_prior"] - TEST_PRIOR).sum()))
            w = res["mcmc"].identifiability_warning()
            if w is not None:
                ident_warnings.append(w)
            bar.update(1)

    # AuRC per trial (mean over ranks), then aggregated over trials.
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
    # Coverage-at-target per trial (computed per trial, then aggregated:
    # threshold crossings are nonlinear, so the order matters). One entry per
    # requested target.
    rts, rt_descs = _resolve_risk_targets(args.risk_target)
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

    # ---- report ----
    print("=" * 76)
    print("Reject-option predictors under Bayesian test-prior adaptation")
    print("=" * 76)
    print(f"trials={args.trials}  m_train={args.m_train}  "
          f"n_test={args.n_test}  n_eval={args.n_eval}")
    print(f"train prior     : {np.array2string(TRAIN_PRIOR, precision=3)}")
    print(f"true test prior : {np.array2string(TEST_PRIOR, precision=3)}")
    print(f"prior L1 error  : {np.mean(prior_errs):.3f} +/- "
          f"{np.std(prior_errs):.3f}   (learned vs true, over trials)")
    if ident_warnings:
        print(f"!!! IDENTIFIABILITY WARNING (fired in {len(ident_warnings)}/"
              f"{args.trials} trials) !!!")
        print(f"    {ident_warnings[0]}")
    print("-" * 76)
    print(f"{'predictor (base, full coverage)':<46}{'test acc':>12}{'std':>10}")
    print("-" * 76)
    for k, v in accs.items():
        label = REJECT_LABELS.get(k, k)
        print(f"{label:<46}{np.mean(v):>12.4f}{np.std(v):>10.4f}")
    print("-" * 76)
    print(f"{'reject-option predictor':<46}{'AuRC risk':>14}{'AuRC regret':>14}")
    print("-" * 76)
    for name in names:
        print(f"{REJECT_LABELS[name]:<46}"
              f"{aurc_risk[name].mean():>8.4f} ± {aurc_risk[name].std():.4f}"
              f"{aurc_regret[name].mean():>8.4f} ± {aurc_regret[name].std():.4f}")
    print("-" * 76)
    print(AURC50_NOTE)
    print(f"{'reject-option predictor':<46}{'AuRC50 risk':>14}"
          f"{'AuRC50 regret':>14}")
    print("-" * 76)
    for name in names:
        print(f"{REJECT_LABELS[name]:<46}"
              f"{aurc50_risk[name].mean():>8.4f} ± {aurc50_risk[name].std():.4f}"
              f"{aurc50_regret[name].mean():>8.4f} ± {aurc50_regret[name].std():.4f}")
    print(AURC50_CAVEAT)
    print("-" * 76)
    print("area under the generalized curves (normalized by n_eval, not by the "
          "accepted count: not on the AuRC scale above)")
    print(f"{'reject-option predictor':<46}{'AuGRC risk':>14}{'AuGRC regret':>14}")
    print("-" * 76)
    for name in names:
        print(f"{REJECT_LABELS[name]:<46}"
              f"{augrc_risk[name].mean():>8.4f} ± {augrc_risk[name].std():.4f}"
              f"{augrc_regret[name].mean():>8.4f} ± {augrc_regret[name].std():.4f}")
    print("-" * 76)
    ref_note = ("  ('ref' = per-trial full-coverage risk of the true-prior "
                "reference)" if args.risk_target is None else "")
    print(f"coverage at target (mean±std over trials){ref_note}")
    header = f"{'reject-option predictor':<46}"
    header += "".join(f"{'risk<=' + d:>14}" for d in rt_descs)
    header += "".join(f"{f'regret<={e:g}':>14}" for e in args.regret_target)
    print(header)
    print("-" * 76)
    for name in names:
        row = f"{REJECT_LABELS[name]:<46}"
        for cov in (*cov_risk, *cov_regret):
            row += f"{cov[name].mean():>8.3f}±{cov[name].std():.3f}"
        print(row)
    print("-" * 76)
    print("epistemic-uncertainty metrics of the Bayesian predictor "
          f"(threshold={args.epi_threshold:g})")
    for label, col in (("avg epistemic uncertainty", 0),
                       ("avg regret (full coverage)", 1),
                       ("portion with negligible epistemic uncertainty", 2)):
        print(f"  {label:<48}"
              f"{epi_metrics[:, col].mean():>9.4f} ± {epi_metrics[:, col].std():.4f}")
    print("=" * 76)

    make_curve_figures(risk_curves, regret_curves, aurc_risk, aurc_regret,
                       args.n_eval, args.out_dir)
    make_gen_curve_figures(gen_risk_curves, gen_regret_curves,
                           augrc_risk, augrc_regret, args.n_eval, args.out_dir)
    print(f"figures written to {args.out_dir}/: risk_coverage.png, "
          f"regret_coverage.png, gen_risk_coverage.png, "
          f"gen_regret_coverage.png")


def make_curve_figures(
    risk_curves: dict, regret_curves: dict,
    aurc_risk: dict, aurc_regret: dict,
    n_eval: int, out_dir: str,
    metrics: tuple[str, str] = ("selective risk", "selective regret"),
    area_label: str = "AuRC",
    fnames: tuple[str, str] = ("risk_coverage", "regret_coverage"),
) -> None:
    """One figure per metric, written to the run root (non-sweep mode).

    ``metrics``, ``area_label`` and ``fnames`` select the flavour: the defaults
    draw the selective curves; ``make_gen_curve_figures`` passes the generalized
    ones.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverage = np.arange(1, n_eval + 1) / n_eval
    trials = next(iter(risk_curves.values())).shape[0]

    for curves, aurc, metric, fname, is_regret in (
        (risk_curves, aurc_risk, metrics[0], fnames[0], False),
        (regret_curves, aurc_regret, metrics[1], fnames[1], True),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in REJECT_LABELS:
            mean = curves[name].mean(axis=0)
            sem = _band(curves[name], 0, trials)
            label = (f"{REJECT_LABELS[name]}  "
                     f"({area_label} {aurc[name].mean():.4f} ± "
                     f"{aurc[name].std():.4f})")
            ax.plot(coverage, mean, lw=1.8, color=REJECT_COLORS[name], label=label)
            ax.fill_between(coverage, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if is_regret:
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("coverage")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.capitalize()}-coverage curve "
                     f"({_agg_desc(trials)})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{fname}.png", dpi=130)
        plt.close(fig)


def make_gen_curve_figures(
    gen_risk_curves: dict, gen_regret_curves: dict,
    augrc_risk: dict, augrc_regret: dict,
    n_eval: int, out_dir: str,
) -> None:
    """Generalized counterpart of ``make_curve_figures``."""
    make_curve_figures(
        gen_risk_curves, gen_regret_curves, augrc_risk, augrc_regret,
        n_eval, out_dir,
        metrics=("generalized risk", "generalized regret"),
        area_label="AuGRC",
        fnames=("gen_risk_coverage", "gen_regret_coverage"))


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials, n_eval = next(iter(risk_curves.values())).shape
    coverage = np.arange(1, n_eval + 1) / n_eval
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, curves, metric, is_regret in (
        (axes[0], risk_curves, metrics[0], False),
        (axes[1], regret_curves, metrics[1], True),
    ):
        for name in REJECT_LABELS:
            mean = curves[name].mean(axis=0)
            sem = _band(curves[name], 0, trials)
            area = curves[name].mean(axis=1)     # per-trial area
            ax.plot(coverage, mean, lw=1.8, color=REJECT_COLORS[name],
                    label=f"{REJECT_LABELS[name]}  "
                          f"({area_label} {area.mean():.4f} ± {area.std():.4f})")
            ax.fill_between(coverage, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if is_regret:
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("coverage")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.capitalize()}-coverage curve, "
                     f"$n_\\mathrm{{test}}$ = {n_test}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{suptitle_prefix} at n_test = {n_test} "
                 f"({_agg_desc(trials)})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sub = Path(out_dir) / "coverage_curves"
    sub.mkdir(parents=True, exist_ok=True)
    path = str(sub / f"{fname_prefix}_n{n_test}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


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


# ---------------------------------------------------------------------------
# Sweep mode: AuRC as a function of the adaptation-set size.
# ---------------------------------------------------------------------------

def run_sweep_experiment(model, args, master_rng) -> None:
    Y = model.num_classes
    loss = zero_one_loss_matrix(Y)
    sizes = sorted(args.sizes)
    n_max = sizes[-1]
    names = list(REJECT_LABELS.keys())
    aurc_risk = {n: np.zeros((len(sizes), args.trials)) for n in names}
    aurc_regret = {n: np.zeros((len(sizes), args.trials)) for n in names}
    warned = np.zeros((len(sizes), args.trials), dtype=bool)
    epi_metrics = np.zeros((len(sizes), args.trials, 3))  # avg epi, avg regret, portion
    rts, rt_descs = _resolve_risk_targets(args.risk_target)
    cov_risk = [{n: np.zeros((len(sizes), args.trials)) for n in names}
                for _ in rts]
    cov_regret = [{n: np.zeros((len(sizes), args.trials)) for n in names}
                  for _ in args.regret_target]
    # Full per-size curves, kept for the per-n coverage-curve figures.
    risk_curves = {n: np.zeros((len(sizes), args.trials, args.n_eval))
                   for n in names}
    regret_curves = {n: np.zeros((len(sizes), args.trials, args.n_eval))
                     for n in names}
    with _progress(total=args.trials * len(sizes), desc="sweep") as bar:
        for t in range(args.trials):
            rng = np.random.default_rng(master_rng.integers(1 << 32))

            X_tr, y_tr = model.sample(args.m_train, TRAIN_PRIOR, rng)
            base = BaseModel.fit(X_tr, y_tr, num_classes=Y)

            # Adaptation pool used as nested prefixes across sizes.
            X_pool, y_pool = model.sample(n_max, TEST_PRIOR, rng)
            est_post_pool = base.posterior(X_pool)

            # Fixed labeled evaluation set shared by all sizes in this trial.
            X_ev, y_ev = model.sample(args.n_eval, TEST_PRIOR, rng)
            est_post_ev = base.posterior(X_ev)
            h_true = bayes_decision(
                corrected_posterior(est_post_ev, base.train_prior, TEST_PRIOR),
                loss)
            losses_ref = loss[h_true, y_ev]
            resolved_rts = [float(losses_ref.mean()) if rt is None else rt
                            for rt in rts]

            for i, n in enumerate(sizes):
                mcmc = sample_prior_posterior(
                    est_post_pool[:n], base.train_prior, rng=rng,
                    sampler=args.sampler)
                warned[i, t] = mcmc.identifiability_warning() is not None

                bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
                    est_post_ev, base.train_prior, mcmc.samples, loss)
                cond_risk_bayes = bayes_post @ loss.T
                h_bayes = cond_risk_bayes.argmin(axis=1)
                total = cond_risk_bayes.min(axis=1)

                h_em, u_em, _em_prior = em_plugin_predictor(
                    est_post_pool[:n], est_post_ev, base.train_prior, loss)

                predictors = {
                    "bayes_total": (h_bayes, total),
                    "bayes_epistemic": (h_bayes, total - aleatoric),
                    "em_plugin": (h_em, u_em),
                }
                # Score each predictor, then the oracle envelope (whose risk
                # and regret curves use separate, metric-specific rankings).
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
                    for ei, eps in enumerate(args.regret_target):
                        cov_regret[ei][name][i, t] = coverage_at_target(regret, eps)
                epi_metrics[i, t] = epistemic_metrics(
                    total - aleatoric, loss[h_bayes, y_ev], losses_ref,
                    args.epi_threshold)
                bar.update(1)

    # Generalized curves and their areas: a rescaling of the selective curves by
    # the coverage, so no re-ranking is needed.
    gen_risk_curves = {n: generalize_curve(risk_curves[n]) for n in names}
    gen_regret_curves = {n: generalize_curve(regret_curves[n]) for n in names}
    augrc_risk = {n: gen_risk_curves[n].mean(axis=-1) for n in names}
    augrc_regret = {n: gen_regret_curves[n].mean(axis=-1) for n in names}
    # Areas over the high-coverage window only: a slice of the same curves.
    aurc50_risk = {n: truncated_area(risk_curves[n]) for n in names}
    aurc50_regret = {n: truncated_area(regret_curves[n]) for n in names}

    # ---- report ----
    print("=" * 76)
    print("AuRC vs. number of unlabeled test examples")
    print("=" * 76)
    print(f"trials={args.trials}  m_train={args.m_train}  "
          f"n_eval={args.n_eval}  sizes={sizes}")
    for metric, aurc in (("risk", aurc_risk), ("regret", aurc_regret)):
        print("-" * 76)
        print(f"AuRC ({metric})")
        print(f"{'n_test':>8}{'warn':>8}"
              + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            row = f"{n:>8}{warned[i].mean():>8.2f}"
            row += "".join(f"{aurc[name][i].mean():>24.4f}" for name in names)
            print(row)
    for metric, aurc50 in (("risk", aurc50_risk), ("regret", aurc50_regret)):
        print("-" * 76)
        print(f"AuRC50 ({metric})  ({AURC50_NOTE})")
        print(f"{'n_test':>8}"
              + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            print(f"{n:>8}"
                  + "".join(f"{aurc50[name][i].mean():>24.4f}" for name in names))
    print(AURC50_CAVEAT)
    for metric, augrc in (("risk", augrc_risk), ("regret", augrc_regret)):
        print("-" * 76)
        print(f"AuGRC ({metric})  (normalized by n_eval; not on the AuRC scale)")
        print(f"{'n_test':>8}"
              + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            print(f"{n:>8}"
                  + "".join(f"{augrc[name][i].mean():>24.4f}" for name in names))
    risk_fig_descs = ["reference risk" if rt is None else d
                      for rt, d in zip(rts, rt_descs)]
    regret_fig_descs = [f"{e:g}" for e in args.regret_target]
    blocks = [(f"coverage @ risk <= {d}"
               + (" (per-trial reference full-coverage risk)" if rt is None else ""),
               cov) for rt, d, cov in zip(rts, rt_descs, cov_risk)]
    blocks += [(f"coverage @ regret <= {e:g}", cov)
               for e, cov in zip(args.regret_target, cov_regret)]
    for label, cov in blocks:
        print("-" * 76)
        print(label)
        print(f"{'n_test':>8}"
              + "".join(f"{REJECT_LABELS[n][:22]:>24}" for n in names))
        for i, n in enumerate(sizes):
            print(f"{n:>8}"
                  + "".join(f"{cov[name][i].mean():>24.3f}" for name in names))
    print("-" * 76)
    print("Epistemic-uncertainty metrics of the Bayesian predictor "
          f"(threshold={args.epi_threshold:g})")
    print(f"{'n_test':>8}{'avg epi':>14}{'avg regret':>14}{'portion negl':>14}")
    for i, n in enumerate(sizes):
        print(f"{n:>8}{epi_metrics[i, :, 0].mean():>14.4f}"
              f"{epi_metrics[i, :, 1].mean():>14.4f}"
              f"{epi_metrics[i, :, 2].mean():>14.3f}")
    print("=" * 76)
    if warned.any():
        print("!!! IDENTIFIABILITY WARNING: 'warn' = fraction of trials where "
              "the learned prior was only weakly identifiable (see README).")

    make_sweep_figure(sizes, aurc_risk, aurc_regret, args.trials, args.out_dir)
    make_gen_sweep_figure(sizes, augrc_risk, augrc_regret, args.trials,
                          args.out_dir)
    make_trunc_sweep_figure(sizes, aurc50_risk, aurc50_regret, args.trials,
                            args.out_dir)
    make_epistemic_metrics_figure(
        sizes, epi_metrics, args.epi_threshold, args.out_dir)
    make_cov_target_figure(sizes, cov_risk, cov_regret, args.trials,
                           risk_fig_descs, regret_fig_descs, args.out_dir)
    for i, n in enumerate(sizes):
        make_curves_at_n_figure(
            {name: risk_curves[name][i] for name in names},
            {name: regret_curves[name][i] for name in names},
            n, args.out_dir)
        make_gen_curves_at_n_figure(
            {name: gen_risk_curves[name][i] for name in names},
            {name: gen_regret_curves[name][i] for name in names},
            n, args.out_dir)
    print(f"figures written to {args.out_dir}/aurc_vs_n_test.png, "
          f"{args.out_dir}/gen_aurc_vs_n_test.png, "
          f"{args.out_dir}/{trunc_sweep_fname()}.png, "
          f"{args.out_dir}/epistemic_metrics_vs_n_test.png, "
          f"{args.out_dir}/cov_at_target_vs_n_test.png and "
          f"{args.out_dir}/coverage_curves/"
          f"[gen_]coverage_curves_n<n_test>.png (two per size)")


def make_sweep_figure(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict,
    trials: int, out_dir: str,
    metrics: tuple[str, str] = ("AuRC (selective risk)",
                                "AuRC (selective regret)"),
    fname: str = "aurc_vs_n_test",
    ylabels: tuple[str, str] | None = None,
) -> None:
    """Area under the coverage curves vs. the adaptation-set size.

    ``metrics`` and ``fname`` select the flavour: the defaults plot the AuRC of
    the selective curves; ``make_gen_sweep_figure`` plots the AuGRC of the
    generalized ones. The two areas are on different scales (the AuGRC weights
    sum to ~1/2), so they get separate figures rather than shared axes.

    ``ylabels`` defaults to ``metrics`` and overrides just the y-axis text, for
    flavours whose full name is too long for the title (the truncated one spells
    its coverage window out there, and keeps a short name for the title).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylabels = ylabels or metrics
    x = np.asarray(sizes, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, aurc, metric, ylabel, is_regret in (
        (axes[0], aurc_risk, metrics[0], ylabels[0], False),
        (axes[1], aurc_regret, metrics[1], ylabels[1], True),
    ):
        for name in REJECT_LABELS:
            mean = aurc[name].mean(axis=1)
            sem = _band(aurc[name], 1, trials)
            ax.plot(x, mean, lw=1.8, marker="o", color=REJECT_COLORS[name],
                    label=REJECT_LABELS[name])
            ax.fill_between(x, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if is_regret:
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of unlabeled adaptation examples $n$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs. test-set size ({_agg_desc(trials)})")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/{fname}.png", dpi=130)
    plt.close(fig)


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(sizes, dtype=float)
    trials = epi_metrics.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- Panel 1: regret and epistemic uncertainty overlaid ----
    ax = axes[0]
    for col, label, color in (
        (1, "average regret (full coverage)", "C1"),
        (0, "average epistemic uncertainty", "C2"),
    ):
        mean = epi_metrics[:, :, col].mean(axis=1)
        sem = _band(epi_metrics[:, :, col], 1, trials)
        ax.plot(x, mean, lw=1.8, marker="o", color=color, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axhline(0.0, color="0.4", ls="--", lw=1)
    ax.set_ylabel("0/1-loss units")
    ax.legend(fontsize=8)

    # ---- Panel 2: portion with negligible epistemic uncertainty ----
    ax = axes[1]
    mean = epi_metrics[:, :, 2].mean(axis=1)
    sem = _band(epi_metrics[:, :, 2], 1, trials)
    ax.plot(x, mean, lw=1.8, marker="o", color="C0")
    ax.fill_between(x, mean - sem, mean + sem, color="C0", alpha=0.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"portion with epistemic uncertainty < {threshold:g}")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of unlabeled adaptation examples $n$")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"({_agg_desc(trials)})")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/epistemic_metrics_vs_n_test.png", dpi=130)
    plt.close(fig)


def make_cov_target_figure(
    sizes: list[int], cov_risk: list[dict], cov_regret: list[dict],
    trials: int, risk_descs: list[str], regret_descs: list[str], out_dir: str,
) -> None:
    """Grid of panels vs. n: one column per target, risk targets on the top
    row and regret targets on the bottom row.

    ``cov_risk`` / ``cov_regret`` are lists (one entry per target) of dicts
    mapping predictor name to a (len(sizes), trials) coverage array.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(sizes, dtype=float)
    ncols = max(len(cov_risk), len(cov_regret))
    fig, axes = plt.subplots(2, ncols, figsize=(5.8 * ncols, 9.2),
                             squeeze=False)

    for r, covs, descs, metric in (
        (0, cov_risk, risk_descs, "risk"),
        (1, cov_regret, regret_descs, "regret"),
    ):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(covs):
                ax.axis("off")
                continue
            for name in REJECT_LABELS:
                mean = covs[c][name].mean(axis=1)
                sem = _band(covs[c][name], 1, trials)
                ax.plot(x, mean, lw=1.8, marker="o", color=REJECT_COLORS[name],
                        label=REJECT_LABELS[name])
                ax.fill_between(x, mean - sem, mean + sem,
                                color=REJECT_COLORS[name], alpha=0.2)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("number of unlabeled adaptation examples $n$")
            ax.set_ylabel("coverage at target")
            ax.set_title(f"coverage @ {metric} <= {descs[c]}")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Coverage at target vs. test-set size "
                 f"({_agg_desc(trials)})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out_dir}/cov_at_target_vs_n_test.png", dpi=130)
    plt.close(fig)


def _resolve_risk_targets(risk_target_arg):
    """Resolve ``--risk-target`` into parallel ``(values, descriptions)``.

    ``None`` (flag omitted) means a single implicit target: the per-trial
    full-coverage risk of the true-prior reference predictor, described as
    ``"ref"`` in tables and ``"reference risk"`` in figures.
    """
    if risk_target_arg is None:
        return [None], ["ref"]
    return list(risk_target_arg), [f"{v:g}" for v in risk_target_arg]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--m-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000,
                        help="Unlabeled adaptation-set size (non-sweep mode).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep the adaptation-set size and plot AuRC-vs-n instead of "
             "the single-size curves.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000],
        help="Adaptation-set sizes (the swept variable) for --sweep.")
    parser.add_argument(
        "--n-eval", type=int, default=2000,
        help="Size of the fixed labeled evaluation set (both modes).")
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
             "predictor (self-calibrating across configs).")
    parser.add_argument(
        "--regret-target", type=float, nargs="+", default=[0.002],
        help="Regret budget(s) for the coverage-at-target metric; the metric "
             "is computed for each value given.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="JSON file with the synthetic-generator setting.")
    parser.add_argument(
        "--sampler", choices=("mh", "gibbs"), default="mh",
        help="Posterior sampler for the test prior: random-walk "
             "Metropolis-Hastings (mh, default) or the latent-variable "
             "Gibbs sampler (gibbs).")
    args = parser.parse_args()

    configure_oracle(args.optimal_rejection)

    cfg = load_experiment_config(args.config)
    model = cfg.model
    global TRAIN_PRIOR, TEST_PRIOR
    TRAIN_PRIOR, TEST_PRIOR = cfg.train_prior, cfg.test_prior
    print(f"config: {cfg.name}  ({args.config})")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    args_file = save_run_args(
        args,
        "run_synth_reject_option_exp_sweep_args.txt" if args.sweep
        else "run_synth_reject_option_exp_args.txt",
        extra={
            "config_name": cfg.name,
            "train_prior": np.array2string(TRAIN_PRIOR, precision=4),
            "test_prior": np.array2string(TEST_PRIOR, precision=4),
        },
        # Both modes use --n-eval here; only the swept size variable differs.
        ignored={"n_test"} if args.sweep else {"sizes"},
    )
    print(f"arguments written to {args_file}")

    master_rng = np.random.default_rng(args.seed)

    if args.sweep:
        run_sweep_experiment(model, args, master_rng)
    else:
        run_single_experiment(model, args, master_rng)


if __name__ == "__main__":
    main()
