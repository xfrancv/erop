"""Metropolis-Hastings sampling of the test label prior from unlabeled data.

Model
-----
Under the label-shift assumption the test posterior is a re-weighting of the
(known) training posterior by the unknown test prior ``alpha(y) = p_te(y)``::

    p(y | x, theta) proportional to  p_tr(y | x) * alpha(y) / p_tr(y)

with theta = (alpha(1), ..., alpha(Y)) on the probability simplex.

The marginal test density of a single input is (up to the unparameterised
constant ``p_tr(x)``)::

    p(x | theta) proportional to  sum_y  p_tr(y | x) / p_tr(y) * alpha(y)
                               =  sum_y  R(x, y) * alpha(y)

where ``R(x, y) = p_tr(y | x) / p_tr(y)``.  Hence the log data likelihood is

    log p(D | theta) = sum_i log( sum_y R(x_i, y) alpha(y) ) + const .

With a Dirichlet(beta) prior on theta the log posterior (up to a constant) is

    log p(theta | D) = sum_y (beta_y - 1) log alpha(y)
                     + sum_i log( sum_y R(x_i, y) alpha(y) ) .

This module samples theta from ``p(theta | D)`` with a random-walk
Metropolis-Hastings chain.  To respect the simplex constraint we reparameterise
theta with an additive-logistic (softmax) map from an unconstrained
``z in R^{Y-1}`` and include the change-of-variables Jacobian, so the chain
targets the correct density on the simplex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def _softmax_last_fixed(z: np.ndarray) -> np.ndarray:
    """Map z in R^{Y-1} to a point on the Y-simplex (last logit fixed to 0)."""
    logits = np.concatenate([z, [0.0]])
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


@dataclass
class MCMCResult:
    samples: np.ndarray          # (num_samples, Y) posterior draws of alpha
    posterior_mean: np.ndarray   # (Y,) mean of alpha over the draws
    acceptance_rate: float
    all_alpha: np.ndarray        # (num_iters, Y) full chain (incl. burn-in) for diagnostics
    ident_ratio: np.ndarray      # (Y,) posterior std / std of counting the same n labels

    def identifiability_warning(self, ratio_threshold: float = 3.0) -> str | None:
        """Ill-conditioning diagnostic; a warning message, or None if healthy.

        Label-shift prior estimation is identifiable only if the class
        conditionals are distinguishable.  When two of them are (nearly)
        identical the likelihood is flat along the direction that trades their
        prior mass, so the posterior of ``alpha`` spreads into a ridge and the
        posterior mean along it follows the Dirichlet prior, not the data.

        The signature we test for: the posterior std of ``alpha(y)`` should be
        comparable to ``sqrt(alpha(1-alpha)/n)`` — the error of estimating the
        prior by *counting* ``n`` labels — when the problem is well conditioned
        (observed ratio ~1-1.5).  A ratio above ``ratio_threshold`` for any
        class means the unlabeled data is far less informative than labels
        would be, i.e. the prior is only weakly identifiable.
        """
        flagged = np.flatnonzero(self.ident_ratio > ratio_threshold)
        if flagged.size == 0:
            return None
        classes = ", ".join(f"y={y}" for y in flagged)
        ratios = np.array2string(self.ident_ratio[flagged], precision=1)
        return (
            f"test prior only weakly identifiable from the unlabeled data: "
            f"posterior std of alpha({classes}) is {ratios}x the std of "
            f"counting the same number of labels (threshold "
            f"{ratio_threshold:g}x). Likely cause: near-identical class "
            f"conditionals (or too little data); along the flat directions "
            f"the learned prior follows the Dirichlet prior, not the data."
        )


def sample_prior_posterior(
    train_posterior: np.ndarray,   # (n, Y) = p_tr(y | x_i)
    train_prior: np.ndarray,       # (Y,)  = p_tr(y)
    *,
    beta: np.ndarray | None = None,
    num_iters: int = 20000,
    burn_in: int = 5000,
    thin: int = 5,
    step_size: float = 0.15,
    rng: np.random.Generator | None = None,
) -> MCMCResult:
    """Run random-walk Metropolis-Hastings for the test prior ``alpha``."""
    if rng is None:
        rng = np.random.default_rng()

    n, Y = train_posterior.shape
    if beta is None:
        beta = np.ones(Y)
    beta = np.asarray(beta, dtype=float)

    # R(x_i, y) = p_tr(y | x_i) / p_tr(y).  Precompute once.
    R = train_posterior / train_prior[None, :]
    logR = np.log(np.clip(R, 1e-300, None))

    def log_target(z: np.ndarray) -> float:
        alpha = _softmax_last_fixed(z)
        log_alpha = np.log(alpha)
        # Data term: sum_i log sum_y R_iy alpha_y  (log-sum-exp for stability).
        per_sample = logsumexp(logR + log_alpha[None, :], axis=1)
        data = per_sample.sum()
        # Dirichlet log prior (up to constant): sum_y (beta_y - 1) log alpha_y.
        prior = np.sum((beta - 1.0) * log_alpha)
        # Change-of-variables Jacobian for the additive-logistic map:
        # log|d alpha / d z| = sum_y log alpha_y.
        jacobian = np.sum(log_alpha)
        return data + prior + jacobian

    z = np.zeros(Y - 1)           # start at the uniform prior
    lt = log_target(z)
    chain = np.empty((num_iters, Y))
    n_accept = 0

    for t in range(num_iters):
        z_prop = z + step_size * rng.standard_normal(Y - 1)
        lt_prop = log_target(z_prop)
        if np.log(rng.random()) < lt_prop - lt:
            z, lt = z_prop, lt_prop
            n_accept += 1
        chain[t] = _softmax_last_fixed(z)

    kept = chain[burn_in::thin]
    posterior_mean = kept.mean(axis=0)
    # Identifiability diagnostic: posterior std relative to the std of the
    # label-counting estimator sqrt(alpha(1-alpha)/n) at the same sample size.
    counting_std = np.sqrt(np.clip(posterior_mean * (1 - posterior_mean), 1e-12, None) / n)
    return MCMCResult(
        samples=kept,
        posterior_mean=posterior_mean,
        acceptance_rate=n_accept / num_iters,
        all_alpha=chain,
        ident_ratio=kept.std(axis=0) / counting_std,
    )


def posterior_label_probabilities(
    train_posterior: np.ndarray,   # (n, Y) = p_tr(y | x_i)
    train_prior: np.ndarray,       # (Y,)
    alpha_samples: np.ndarray,     # (S, Y) posterior draws of the test prior
) -> np.ndarray:
    """Bayesian label posterior ``hat p(y | x, D)`` averaged over prior samples.

    For each draw ``alpha`` the per-input label distribution is the normalised
    re-weighting ``p(y | x, alpha) proportional to R(x, y) alpha(y)``.  We
    average these normalised distributions over the posterior draws.
    """
    R = train_posterior / train_prior[None, :]           # (n, Y)
    # weighted[s] = R * alpha_s  -> normalise over y -> average over s.
    # Vectorised over samples with an accumulator to bound memory.
    n, Y = R.shape
    acc = np.zeros((n, Y))
    for alpha in alpha_samples:
        w = R * alpha[None, :]
        w /= w.sum(axis=1, keepdims=True)
        acc += w
    return acc / len(alpha_samples)
