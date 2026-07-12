"""Numerical experiment: Bayesian label-prior adaptation under label shift.

We train a logistic-regression base model on data drawn with a *training*
prior, then adapt to a *test* prior that is significantly different, using only
the unlabeled test inputs.  The learned-prior Bayesian predictor is compared
against oracle Bayes predictors and plugin baselines.

Run with::

    python run_experiment.py

The synthetic-generator setting (number of Gaussians, their means and
covariances, and the train/test label priors) is read from a JSON file given
by ``--config``; the default ``configs/default.json`` reproduces the original
experiment. Figures are written to ``figures/`` and a results table is printed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from prior_shift import (
    BaseModel,
    GaussianClassConditionalModel,
    bayes_decision,
    corrected_posterior,
    load_experiment_config,
    posterior_label_probabilities,
    sample_prior_posterior,
    zero_one_loss_matrix,
)

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


def save_run_args(
    args,
    filename: str,
    extra: dict | None = None,
    ignored: set[str] | None = None,
) -> Path:
    """Write the run's argument setting to ``args.out_dir / filename``.

    ``filename`` differs per script *and* per mode (sweep vs. single), so runs
    of the two experiment scripts into a shared ``--out-dir`` never overwrite
    each other's record.  ``extra`` holds values resolved after parsing (the
    config name, the priors it supplied) that are not argparse arguments.

    ``ignored`` names the arguments the current mode does not read; they are
    omitted rather than recorded at their (unused) defaults, which would invite
    a reader to reconstruct the run wrongly.  Which arguments those are depends
    on both the script and the mode, so the caller decides.
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
        lines += ["", "[resolved from config]"]
        lines += [f"{k:<{width}}: {v}" for k, v in extra.items()]
    path.write_text("\n".join(lines) + "\n")
    return path


# The generator setting (Gaussians + priors) comes from a JSON config; the
# default reproduces the original experiment: uniform training prior, strongly
# imbalanced test prior.  ``main()`` overwrites these two module-level priors
# from the loaded config before any experiment code runs.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.json"
TRAIN_PRIOR = np.array([0.25, 0.25, 0.25, 0.25])
TEST_PRIOR = np.array([0.60, 0.20, 0.15, 0.05])


def accuracy(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(pred == y_true))


def run_single_trial(
    model: GaussianClassConditionalModel,
    m_train: int,
    n_test: int,
    rng: np.random.Generator,
    mcmc_kwargs: dict | None = None,
):
    """Run one full train/adapt/evaluate cycle. Returns a results dict."""
    Y = model.num_classes
    loss = zero_one_loss_matrix(Y)
    mcmc_kwargs = mcmc_kwargs or {}

    # --- training ---
    X_tr, y_tr = model.sample(m_train, TRAIN_PRIOR, rng)
    base = BaseModel.fit(X_tr, y_tr, num_classes=Y)

    # --- test data (labels used only for evaluation) ---
    X_te, y_te = model.sample(n_test, TEST_PRIOR, rng)
    est_post_te = base.posterior(X_te)  # p_hat_tr(y | x) on the test inputs

    # --- Bayesian prior learning from the unlabeled test inputs ---
    mcmc = sample_prior_posterior(
        est_post_te, base.train_prior, rng=rng, **mcmc_kwargs
    )
    bayes_post = posterior_label_probabilities(
        est_post_te, base.train_prior, mcmc.samples
    )

    # Supervised prior estimate: empirical test-label frequencies (available
    # only because the data is synthetic -- this is a supervised reference).
    supervised_prior = np.bincount(y_te, minlength=Y).astype(float)
    supervised_prior /= supervised_prior.sum()

    # --- predictors ---
    preds = {
        # 1. Optimal Bayes for the test distribution (true conditionals + true prior).
        "opt_bayes_test": bayes_decision(
            model.true_posterior(X_te, TEST_PRIOR), loss),
        # 2. Plugin with the (estimated) training prior = raw logistic classifier.
        "plugin_train_prior": bayes_decision(
            corrected_posterior(est_post_te, base.train_prior, base.train_prior),
            loss),
        # 3. Plugin with the true test prior (oracle prior).
        "plugin_true_test_prior": bayes_decision(
            corrected_posterior(est_post_te, base.train_prior, TEST_PRIOR),
            loss),
        # 4. Plugin with the test prior estimated from supervised test labels.
        "plugin_supervised_prior": bayes_decision(
            corrected_posterior(est_post_te, base.train_prior, supervised_prior),
            loss),
        # 5. Proposed: Bayesian predictor with prior learned from test inputs.
        "bayes_learned_prior": bayes_decision(bayes_post, loss),
    }
    acc = {name: accuracy(p, y_te) for name, p in preds.items()}

    return {
        "accuracy": acc,
        "mcmc": mcmc,
        "learned_prior": mcmc.posterior_mean,
        "train_prior_est": base.train_prior,
        "X_te": X_te,
        "y_te": y_te,
        "base": base,
        "model": model,
    }


PREDICTOR_LABELS = {
    "opt_bayes_test": "Optimal Bayes, true test prior (upper bound)",
    "plugin_true_test_prior": "Plugin, true test prior (oracle)",
    "plugin_supervised_prior": "Plugin, supervised prior estimate",
    "bayes_learned_prior": "Bayesian, learned prior (proposed)",
    "plugin_train_prior": "Plugin, training prior (no adaptation)",
}

# Colors shared across figures, keyed by predictor name so the two figure
# builders stay consistent when the predictor set changes.
PREDICTOR_COLORS = {
    "opt_bayes_test": "C2",
    "plugin_true_test_prior": "C0",
    "plugin_supervised_prior": "C4",
    "bayes_learned_prior": "C1",
    "plugin_train_prior": "C3",
}


def run_size_sweep(
    model: GaussianClassConditionalModel,
    m_train: int,
    sizes: list[int],
    trials: int,
    master_rng: np.random.Generator,
    n_eval: int = 2000,
    mcmc_kwargs: dict | None = None,
):
    """Accuracy (and learned-prior error) as a function of the test-set size.

    The sweep variable ``n`` is the number of *unlabeled* test examples the
    prior is learned from.  To isolate its effect, every predictor is scored on
    a single **fixed** labeled evaluation set of size ``n_eval`` that is drawn
    once per trial and never used for learning.  Consequently the oracle and
    no-adaptation baselines are constant in ``n`` (they do not use the learned
    prior), and only the proposed predictor moves — climbing from the unadapted
    plugin toward the oracle as ``n`` grows.

    Within a trial the ``n`` unlabeled examples are nested prefixes of one
    pool of size ``max(sizes)``, so neighbouring sizes share draws and the
    curves reflect ``n`` rather than re-sampling noise.

    Returns ``(acc, prior_err, warned)`` where ``acc[name]`` is a
    ``(len(sizes), trials)`` array of accuracies, ``prior_err`` is the matching
    ``(len(sizes), trials)`` array of learned-prior L1 errors, and ``warned``
    is a ``(len(sizes), trials)`` bool array marking runs where the MCMC
    identifiability diagnostic fired.
    """
    Y = model.num_classes
    loss = zero_one_loss_matrix(Y)
    mcmc_kwargs = mcmc_kwargs or {}
    sizes = sorted(sizes)
    n_max = sizes[-1]

    names = list(PREDICTOR_LABELS.keys())
    acc = {name: np.zeros((len(sizes), trials)) for name in names}
    prior_err = np.zeros((len(sizes), trials))
    warned = np.zeros((len(sizes), trials), dtype=bool)

    # One step per MCMC fit (trials x sizes), the dominant cost of the sweep.
    bar = _progress(total=trials * len(sizes), desc="sweep")

    for t in range(trials):
        rng = np.random.default_rng(master_rng.integers(1 << 32))

        X_tr, y_tr = model.sample(m_train, TRAIN_PRIOR, rng)
        base = BaseModel.fit(X_tr, y_tr, num_classes=Y)

        # Pool of test examples used (as prefixes) to adapt the prior. The
        # inputs feed the unsupervised MCMC; the labels feed the supervised
        # frequency estimate. Sharing the pool makes the two methods comparable
        # at each n.
        X_pool, y_pool = model.sample(n_max, TEST_PRIOR, rng)
        est_post_pool = base.posterior(X_pool)

        # Fixed labeled evaluation set, shared across all sizes in this trial.
        X_ev, y_ev = model.sample(n_eval, TEST_PRIOR, rng)
        est_post_ev = base.posterior(X_ev)

        # Predictors that do NOT depend on the learned prior are constant in n;
        # compute them once on the evaluation set.
        acc_opt_test = accuracy(
            bayes_decision(model.true_posterior(X_ev, TEST_PRIOR), loss), y_ev)
        acc_plugin_train = accuracy(
            bayes_decision(
                corrected_posterior(est_post_ev, base.train_prior, base.train_prior),
                loss), y_ev)
        acc_plugin_test = accuracy(
            bayes_decision(
                corrected_posterior(est_post_ev, base.train_prior, TEST_PRIOR),
                loss), y_ev)

        for i, n in enumerate(sizes):
            # Learn the prior from the first n unlabeled inputs...
            mcmc = sample_prior_posterior(
                est_post_pool[:n], base.train_prior, rng=rng, **mcmc_kwargs
            )
            # ...then apply it on the fixed evaluation set.
            bayes_post = posterior_label_probabilities(
                est_post_ev, base.train_prior, mcmc.samples
            )
            acc["bayes_learned_prior"][i, t] = accuracy(
                bayes_decision(bayes_post, loss), y_ev)
            # Supervised baseline: empirical prior from the first n test labels.
            supervised_prior = np.bincount(y_pool[:n], minlength=Y).astype(float)
            supervised_prior /= supervised_prior.sum()
            acc["plugin_supervised_prior"][i, t] = accuracy(
                bayes_decision(
                    corrected_posterior(est_post_ev, base.train_prior, supervised_prior),
                    loss), y_ev)
            acc["opt_bayes_test"][i, t] = acc_opt_test
            acc["plugin_train_prior"][i, t] = acc_plugin_train
            acc["plugin_true_test_prior"][i, t] = acc_plugin_test
            prior_err[i, t] = float(np.abs(mcmc.posterior_mean - TEST_PRIOR).sum())
            warned[i, t] = mcmc.identifiability_warning() is not None
            bar.update(1)

    bar.close()
    return acc, prior_err, warned


def make_sweep_figure(
    sizes: list[int], acc: dict, prior_err: np.ndarray, out_dir: str
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(sizes)
    x = np.asarray(sizes, dtype=float)
    trials = prior_err.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel 1: accuracy vs number of unlabeled test examples ----------
    ax = axes[0]
    # Flat baselines drawn as reference lines; the learned prior as the curve
    # that should climb from the unadapted plugin toward the oracle as n grows.
    style = {
        "opt_bayes_test": dict(color="C2", ls="--"),
        "plugin_true_test_prior": dict(color="C0", ls="--"),
        "plugin_supervised_prior": dict(color="C4", ls="-", marker="s"),
        "bayes_learned_prior": dict(color="C1", ls="-", marker="o"),
        "plugin_train_prior": dict(color="C3", ls=":"),
    }
    # Both n-dependent adaptation curves get a shaded s.e.m. band.
    banded = {"bayes_learned_prior", "plugin_supervised_prior"}
    for name in PREDICTOR_LABELS:
        mean = acc[name].mean(axis=1)
        sem = acc[name].std(axis=1) / np.sqrt(trials)
        ax.plot(x, mean, label=PREDICTOR_LABELS[name], lw=1.8, **style[name])
        if name in banded:
            ax.fill_between(x, mean - sem, mean + sem,
                            color=PREDICTOR_COLORS[name], alpha=0.2)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of unlabeled test examples $n$")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Accuracy vs. test-set size (mean ± s.e.m., {trials} trials)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)

    # ---- Panel 2: learned-prior L1 error vs n ----------------------------
    ax = axes[1]
    mean = prior_err.mean(axis=1)
    sem = prior_err.std(axis=1) / np.sqrt(trials)
    ax.plot(x, mean, color="C1", lw=1.8, marker="o", label="learned prior")
    ax.fill_between(x, mean - sem, mean + sem, color="C1", alpha=0.2)
    # 1/sqrt(n) reference (rate of a well-behaved estimator), anchored at the
    # first point, to make the convergence rate legible.
    ref = mean[0] * np.sqrt(x[0]) / np.sqrt(x)
    ax.plot(x, ref, color="0.5", ls="--", lw=1.2, label=r"$\propto 1/\sqrt{n}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of unlabeled test examples $n$")
    ax.set_ylabel(r"prior $L_1$ error  $\|\hat\alpha - \alpha^\star\|_1$")
    ax.set_title("Learned-prior error vs. test-set size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_vs_n_test.png", dpi=130)
    plt.close(fig)


def make_figures(detailed: dict, agg_acc: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = detailed["model"]
    X_te, y_te = detailed["X_te"], detailed["y_te"]
    mcmc = detailed["mcmc"]
    Y = model.num_classes
    colors = plt.cm.tab10(np.arange(Y))

    # ---- Figure 1: data + learned vs true prior -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for y in range(Y):
        pts = X_te[y_te == y]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5,
                   color=colors[y], label=f"class {y}")
    ax.set_title("Test data (colored by true class)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    idx = np.arange(Y)
    w = 0.27
    ax.bar(idx - w, TRAIN_PRIOR, w, label="train prior", color="0.7")
    ax.bar(idx, TEST_PRIOR, w, label="true test prior", color="C0")
    # Posterior mean +/- std of the learned prior.
    lp_mean = mcmc.samples.mean(axis=0)
    lp_std = mcmc.samples.std(axis=0)
    ax.bar(idx + w, lp_mean, w, yerr=lp_std, capsize=3,
           label="learned prior (post. mean)", color="C1")
    ax.set_xticks(idx); ax.set_xticklabels([f"y={y}" for y in range(Y)])
    ax.set_ylabel("probability")
    ax.set_title("Prior recovery")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/data_and_prior.png", dpi=130)
    plt.close(fig)

    # ---- Figure 2: MCMC diagnostics -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for y in range(Y):
        ax.plot(mcmc.all_alpha[:, y], lw=0.6, color=colors[y], label=f"$\\alpha_{y}$")
        ax.axhline(TEST_PRIOR[y], color=colors[y], ls="--", lw=1)
    ax.set_title("MCMC trace of the prior (dashed = true test prior)")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\alpha(y)$")
    ax.legend(fontsize=8, ncol=Y)

    ax = axes[1]
    for y in range(Y):
        ax.hist(mcmc.samples[:, y], bins=40, alpha=0.5,
                color=colors[y], density=True, label=f"$\\alpha_{y}$")
        ax.axvline(TEST_PRIOR[y], color=colors[y], ls="--", lw=1.5)
    ax.set_title("Posterior of the prior (dashed = true)")
    ax.set_xlabel(r"$\alpha(y)$"); ax.set_ylabel("density")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/mcmc_diagnostics.png", dpi=130)
    plt.close(fig)

    # ---- Figure 3: accuracy comparison ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(PREDICTOR_LABELS.keys())
    means = np.array([agg_acc[n][0] for n in names])
    stds = np.array([agg_acc[n][1] for n in names])
    bar_colors = [PREDICTOR_COLORS[n] for n in names]
    ax.barh(np.arange(len(names)), means, xerr=stds, capsize=4, color=bar_colors)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([PREDICTOR_LABELS[n] for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("test accuracy")
    ax.set_xlim(means.min() - 0.05, 1.0)
    for i, (mu, sd) in enumerate(zip(means, stds)):
        ax.text(mu + sd + 0.004, i, f"{mu:.3f}", va="center", fontsize=8)
    ax.set_title("Test accuracy by predictor (mean ± std over trials)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_comparison.png", dpi=130)
    plt.close(fig)


def run_sweep_experiment(model, args, master_rng) -> None:
    """Driver for the accuracy-vs-test-set-size curve."""
    sizes = sorted(args.sizes)
    acc, prior_err, warned = run_size_sweep(
        model, args.m_train, sizes, args.trials, master_rng, n_eval=args.n_eval
    )

    print("=" * 74)
    print("Accuracy vs. number of unlabeled test examples")
    print("=" * 74)
    print(f"trials={args.trials}  m_train={args.m_train}  "
          f"n_eval={args.n_eval}  sizes={sizes}")
    print(f"true test prior : {np.array2string(TEST_PRIOR, precision=3)}")
    print("-" * 74)
    header = f"{'n_test':>8}{'prior L1':>12}{'warn':>8}"
    header += "".join(f"{n[:14]:>16}" for n in
                      ["opt(test)", "plugin(test)", "plugin(sup)",
                       "bayes(learn)", "plugin(train)"])
    print(header)
    print("-" * 74)
    cols = ["opt_bayes_test", "plugin_true_test_prior", "plugin_supervised_prior",
            "bayes_learned_prior", "plugin_train_prior"]
    for i, n in enumerate(sizes):
        row = f"{n:>8}{prior_err[i].mean():>12.3f}{warned[i].mean():>8.2f}"
        row += "".join(f"{acc[name][i].mean():>16.4f}" for name in cols)
        print(row)
    print("=" * 74)
    if warned.any():
        print("!!! IDENTIFIABILITY WARNING !!!")
        print("    'warn' = fraction of trials where the posterior of the "
              "prior was >=3x wider than")
        print("    label-counting at the same n -- the prior is only weakly "
              "identifiable from")
        print("    unlabeled data (near-identical class conditionals or too "
              "little data; see README).")

    make_sweep_figure(sizes, acc, prior_err, args.out_dir)
    print(f"figure written to {args.out_dir}/accuracy_vs_n_test.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--m-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep the number of unlabeled test examples and plot the "
             "accuracy-vs-n curve instead of the single-size experiment.")
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000],
        help="Unlabeled test-set sizes (the swept variable) for --sweep.")
    parser.add_argument(
        "--n-eval", type=int, default=2000,
        help="Size of the fixed labeled evaluation set used to score every "
             "point of the --sweep curve.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="JSON file with the synthetic-generator setting: Gaussian means "
             "and covariances plus the train/test label priors.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    model = cfg.model
    global TRAIN_PRIOR, TEST_PRIOR
    TRAIN_PRIOR, TEST_PRIOR = cfg.train_prior, cfg.test_prior
    print(f"config: {cfg.name}  ({args.config})")

    # Create the output directory (including any parents) so figure saving does
    # not fail when a nested --out-dir does not yet exist.
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    args_file = save_run_args(
        args,
        "run_experiment_sweep_args.txt" if args.sweep else "run_experiment_args.txt",
        extra={
            "config_name": cfg.name,
            "train_prior": np.array2string(TRAIN_PRIOR, precision=4),
            "test_prior": np.array2string(TEST_PRIOR, precision=4),
        },
        # The single-size mode scores on the test set itself: no --n-eval, and
        # --sizes is the sweep's variable.
        ignored={"n_test"} if args.sweep else {"sizes", "n_eval"},
    )
    print(f"arguments written to {args_file}")

    master_rng = np.random.default_rng(args.seed)

    if args.sweep:
        run_sweep_experiment(model, args, master_rng)
        return

    # Accumulate accuracies across trials.
    accs: dict[str, list[float]] = {}
    prior_errs: list[float] = []
    ident_warnings: list[str] = []
    detailed = None
    bar = _progress(total=args.trials, desc="trials")
    for t in range(args.trials):
        rng = np.random.default_rng(master_rng.integers(1 << 32))
        res = run_single_trial(model, args.m_train, args.n_test, rng)
        for name, a in res["accuracy"].items():
            accs.setdefault(name, []).append(a)
        prior_errs.append(float(np.abs(res["learned_prior"] - TEST_PRIOR).sum()))
        w = res["mcmc"].identifiability_warning()
        if w is not None:
            ident_warnings.append(w)
        if t == 0:
            detailed = res
        bar.update(1)
    bar.close()

    agg_acc = {name: (float(np.mean(v)), float(np.std(v))) for name, v in accs.items()}

    # ---- report ----
    print("=" * 68)
    print("Bayesian label-prior adaptation under label shift")
    print("=" * 68)
    print(f"trials={args.trials}  m_train={args.m_train}  n_test={args.n_test}")
    print(f"train prior     : {np.array2string(TRAIN_PRIOR, precision=3)}")
    print(f"true test prior : {np.array2string(TEST_PRIOR, precision=3)}")
    print(f"learned prior   : {np.array2string(detailed['learned_prior'], precision=3)}"
          f"  (trial 0 posterior mean)")
    print(f"MCMC acceptance : {detailed['mcmc'].acceptance_rate:.3f}")
    print(f"prior L1 error  : {np.mean(prior_errs):.3f} +/- {np.std(prior_errs):.3f}"
          f"   (learned vs true, over trials)")
    if ident_warnings:
        print(f"!!! IDENTIFIABILITY WARNING (fired in {len(ident_warnings)}/"
              f"{args.trials} trials) !!!")
        print(f"    {ident_warnings[0]}")
    print("-" * 68)
    print(f"{'predictor':<44}{'test acc':>12}{'std':>10}")
    print("-" * 68)
    for name in PREDICTOR_LABELS:
        mu, sd = agg_acc[name]
        print(f"{PREDICTOR_LABELS[name]:<44}{mu:>12.4f}{sd:>10.4f}")
    print("=" * 68)

    make_figures(detailed, agg_acc, args.out_dir)
    print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
