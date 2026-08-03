"""The adaptation-size sweep: the experiment's inner loop.

One call to :func:`run_sweep` runs every trial at every adaptation-set size for
*one* target prior, and returns a :class:`SweepResult` -- the whole run's
numbers on a common ``(len(sizes), replicates)`` layout. The command-line
driver (``rejopt_eval.py``) calls it once for a fixed target prior, or once per
sampled prior in dirichlet mode; the reporting and figure layers consume the
result and never re-derive it.

Pure numpy plus the ``prior_shift`` method modules -- no plotting, no argparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mcmc import sample_prior_posterior
from .predictors import bayes_decision, corrected_posterior
from .reject_option import (
    REJECT_LABELS,
    accuracy,
    bayesian_posterior_and_aleatoric,
    coverage_at_target,
    epistemic_metrics,
    generalize_curve,
    selective_curves,
    truncated_area,
)
from .sampling import resample_to_prior, split_adapt_eval

try:  # progress bars are optional -- the sweep runs without tqdm installed.
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
