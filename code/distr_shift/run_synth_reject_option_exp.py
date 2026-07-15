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
    "oracle": "Oracle (best attainable)",
}
REJECT_COLORS = {
    "bayes_total": "C1",
    "bayes_epistemic": "C2",
    "oracle": "C7",
}


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

    predictors = {
        "bayes_total": (h_bayes, total),
        "bayes_epistemic": (h_bayes, total - aleatoric),
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
                model, args.m_train, args.n_test, args.n_eval, rng)
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
    print(f"figures written to {args.out_dir}/risk_coverage.png and "
          f"{args.out_dir}/regret_coverage.png")


def make_curve_figures(
    risk_curves: dict, regret_curves: dict,
    aurc_risk: dict, aurc_regret: dict,
    n_eval: int, out_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coverage = np.arange(1, n_eval + 1) / n_eval
    trials = next(iter(risk_curves.values())).shape[0]

    for curves, aurc, metric, fname in (
        (risk_curves, aurc_risk, "selective risk", "risk_coverage"),
        (regret_curves, aurc_regret, "selective regret", "regret_coverage"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in REJECT_LABELS:
            mean = curves[name].mean(axis=0)
            sem = curves[name].std(axis=0) / np.sqrt(trials)
            label = (f"{REJECT_LABELS[name]}  "
                     f"(AuRC {aurc[name].mean():.4f} ± {aurc[name].std():.4f})")
            ax.plot(coverage, mean, lw=1.8, color=REJECT_COLORS[name], label=label)
            ax.fill_between(coverage, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if metric == "selective regret":
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("coverage")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.capitalize()}-coverage curve "
                     f"(mean ± s.e.m., {trials} trials)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{fname}.png", dpi=130)
        plt.close(fig)


def make_curves_at_n_figure(
    risk_curves: dict, regret_curves: dict, n_test: int, out_dir: str,
) -> str:
    """Side-by-side risk- and regret-coverage curves for one adaptation-set
    size of a sweep. ``risk_curves`` / ``regret_curves`` map predictor name to
    a (trials, n_eval) array of per-trial curves at that size. The size
    appears in the panel titles and in the file name; the written path is
    returned.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials, n_eval = next(iter(risk_curves.values())).shape
    coverage = np.arange(1, n_eval + 1) / n_eval
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, curves, metric in (
        (axes[0], risk_curves, "selective risk"),
        (axes[1], regret_curves, "selective regret"),
    ):
        for name in REJECT_LABELS:
            mean = curves[name].mean(axis=0)
            sem = curves[name].std(axis=0) / np.sqrt(trials)
            aurc = curves[name].mean(axis=1)     # per-trial AuRC
            ax.plot(coverage, mean, lw=1.8, color=REJECT_COLORS[name],
                    label=f"{REJECT_LABELS[name]}  "
                          f"(AuRC {aurc.mean():.4f} ± {aurc.std():.4f})")
            ax.fill_between(coverage, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if metric == "selective regret":
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xlabel("coverage")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric.capitalize()}-coverage curve, "
                     f"$n_\\mathrm{{test}}$ = {n_test}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Coverage curves at n_test = {n_test} "
                 f"(mean ± s.e.m., {trials} trials)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sub = Path(out_dir) / "coverage_curves"
    sub.mkdir(parents=True, exist_ok=True)
    path = str(sub / f"coverage_curves_n{n_test}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


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
                    est_post_pool[:n], base.train_prior, rng=rng)
                warned[i, t] = mcmc.identifiability_warning() is not None

                bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
                    est_post_ev, base.train_prior, mcmc.samples, loss)
                cond_risk_bayes = bayes_post @ loss.T
                h_bayes = cond_risk_bayes.argmin(axis=1)
                total = cond_risk_bayes.min(axis=1)

                predictors = {
                    "bayes_total": (h_bayes, total),
                    "bayes_epistemic": (h_bayes, total - aleatoric),
                }
                # Score each predictor, then the oracle envelope (whose risk
                # and regret curves use separate, metric-specific rankings).
                curve_set = {
                    name: selective_curves(loss[h, y_ev], losses_ref, u)
                    for name, (h, u) in predictors.items()
                }
                curve_set["oracle"] = oracle_curves(loss[h_bayes, y_ev], losses_ref)
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
    make_epistemic_metrics_figure(
        sizes, epi_metrics, args.epi_threshold, args.out_dir)
    make_cov_target_figure(sizes, cov_risk, cov_regret, args.trials,
                           risk_fig_descs, regret_fig_descs, args.out_dir)
    for i, n in enumerate(sizes):
        make_curves_at_n_figure(
            {name: risk_curves[name][i] for name in names},
            {name: regret_curves[name][i] for name in names},
            n, args.out_dir)
    print(f"figures written to {args.out_dir}/aurc_vs_n_test.png, "
          f"{args.out_dir}/epistemic_metrics_vs_n_test.png, "
          f"{args.out_dir}/cov_at_target_vs_n_test.png and "
          f"{args.out_dir}/coverage_curves/coverage_curves_n<n_test>.png "
          f"(one per size)")


def make_sweep_figure(
    sizes: list[int], aurc_risk: dict, aurc_regret: dict,
    trials: int, out_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(sizes, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, aurc, metric in (
        (axes[0], aurc_risk, "AuRC (selective risk)"),
        (axes[1], aurc_regret, "AuRC (selective regret)"),
    ):
        for name in REJECT_LABELS:
            mean = aurc[name].mean(axis=1)
            sem = aurc[name].std(axis=1) / np.sqrt(trials)
            ax.plot(x, mean, lw=1.8, marker="o", color=REJECT_COLORS[name],
                    label=REJECT_LABELS[name])
            ax.fill_between(x, mean - sem, mean + sem,
                            color=REJECT_COLORS[name], alpha=0.2)
        if "regret" in metric:
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of unlabeled adaptation examples $n$")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs. test-set size (mean ± s.e.m., {trials} trials)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/aurc_vs_n_test.png", dpi=130)
    plt.close(fig)


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
        sem = epi_metrics[:, :, col].std(axis=1) / np.sqrt(trials)
        ax.plot(x, mean, lw=1.8, marker="o", color=color, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axhline(0.0, color="0.4", ls="--", lw=1)
    ax.set_ylabel("0/1-loss units")
    ax.legend(fontsize=8)

    # ---- Panel 2: portion with negligible epistemic uncertainty ----
    ax = axes[1]
    mean = epi_metrics[:, :, 2].mean(axis=1)
    sem = epi_metrics[:, :, 2].std(axis=1) / np.sqrt(trials)
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
                 f"(mean ± s.e.m., {trials} trials)")
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
                sem = covs[c][name].std(axis=1) / np.sqrt(trials)
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
                 f"(mean ± s.e.m., {trials} trials)")
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
    args = parser.parse_args()

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
