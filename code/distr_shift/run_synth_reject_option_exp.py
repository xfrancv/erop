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

# The three reject-option predictors (order fixed for tables and figures).
REJECT_LABELS = {
    "bayes_total": "Bayesian, total uncertainty",
    "bayes_epistemic": "Bayesian, epistemic uncertainty",
    "plugin_supervised": "Plugin, supervised prior (reference)",
}
REJECT_COLORS = {
    "bayes_total": "C1",
    "bayes_epistemic": "C2",
    "plugin_supervised": "C4",
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

    # --- adaptation pool: inputs feed the MCMC, labels only the supervised prior ---
    X_te, y_te = model.sample(n_test, TEST_PRIOR, rng)
    est_post_te = base.posterior(X_te)
    mcmc = sample_prior_posterior(
        est_post_te, base.train_prior, rng=rng, **mcmc_kwargs
    )
    supervised_prior = np.bincount(y_te, minlength=Y).astype(float)
    supervised_prior /= supervised_prior.sum()

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

    # Supervised plugin predictor and its conditional-risk uncertainty.
    post_sup = corrected_posterior(est_post_ev, base.train_prior, supervised_prior)
    cond_risk_sup = post_sup @ loss.T
    h_sup = cond_risk_sup.argmin(axis=1)

    # True-prior plugin reference for the regret.
    h_true = bayes_decision(
        corrected_posterior(est_post_ev, base.train_prior, TEST_PRIOR), loss)
    losses_ref = loss[h_true, y_ev]

    predictors = {
        "bayes_total": (h_bayes, total),
        "bayes_epistemic": (h_bayes, total - aleatoric),
        "plugin_supervised": (h_sup, cond_risk_sup.min(axis=1)),
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
    """Risk/regret curves of every reject-option predictor for one trial."""
    out = {}
    for name, (h, u) in res["predictors"].items():
        losses_pred = res["loss"][h, res["y_ev"]]
        out[name] = selective_curves(losses_pred, res["losses_ref"], u)
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

            for i, n in enumerate(sizes):
                mcmc = sample_prior_posterior(
                    est_post_pool[:n], base.train_prior, rng=rng)
                warned[i, t] = mcmc.identifiability_warning() is not None

                bayes_post, aleatoric = bayesian_posterior_and_aleatoric(
                    est_post_ev, base.train_prior, mcmc.samples, loss)
                cond_risk_bayes = bayes_post @ loss.T
                h_bayes = cond_risk_bayes.argmin(axis=1)
                total = cond_risk_bayes.min(axis=1)

                supervised_prior = np.bincount(
                    y_pool[:n], minlength=Y).astype(float)
                supervised_prior /= supervised_prior.sum()
                post_sup = corrected_posterior(
                    est_post_ev, base.train_prior, supervised_prior)
                cond_risk_sup = post_sup @ loss.T

                predictors = {
                    "bayes_total": (h_bayes, total),
                    "bayes_epistemic": (h_bayes, total - aleatoric),
                    "plugin_supervised": (cond_risk_sup.argmin(axis=1),
                                          cond_risk_sup.min(axis=1)),
                }
                for name, (h, u) in predictors.items():
                    risk, regret = selective_curves(loss[h, y_ev], losses_ref, u)
                    aurc_risk[name][i, t] = risk.mean()
                    aurc_regret[name][i, t] = regret.mean()
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
    print(f"figures written to {args.out_dir}/aurc_vs_n_test.png and "
          f"{args.out_dir}/epistemic_metrics_vs_n_test.png")


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
        ax.set_xlabel("number of unlabeled test examples $n$")
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
    """Three panels vs. n: avg regret, avg epistemic uncertainty, negligible portion.

    ``epi_metrics`` is (len(sizes), trials, 3) with columns (avg epistemic
    uncertainty, avg regret, portion below ``threshold``).  All three are
    properties of the Bayesian learned-prior predictor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(sizes, dtype=float)
    trials = epi_metrics.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    panels = (
        (axes[0], epi_metrics[:, :, 1], "average regret (full coverage)", "C1"),
        (axes[1], epi_metrics[:, :, 0], "average epistemic uncertainty", "C2"),
        (axes[2], epi_metrics[:, :, 2],
         f"portion with epistemic uncertainty < {threshold:g}", "C0"),
    )
    for ax, arr, label, color in panels:
        mean = arr.mean(axis=1)
        sem = arr.std(axis=1) / np.sqrt(trials)
        ax.plot(x, mean, lw=1.8, marker="o", color=color)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
        if "regret" in label:
            ax.axhline(0.0, color="0.4", ls="--", lw=1)
        if label.startswith("portion"):
            ax.set_ylim(-0.02, 1.02)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of unlabeled test examples $n$")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Epistemic-uncertainty metrics of the Bayesian predictor "
                 f"(mean ± s.e.m., {trials} trials)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/epistemic_metrics_vs_n_test.png", dpi=130)
    plt.close(fig)


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
