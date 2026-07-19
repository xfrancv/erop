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

This module samples theta from ``p(theta | D)`` with one of two chains,
selected by the ``sampler`` argument of :func:`sample_prior_posterior`:

``"mh"`` (default)
    Random-walk Metropolis-Hastings.  To respect the simplex constraint we
    reparameterise theta with an additive-logistic (softmax) map from an
    unconstrained ``z in R^{Y-1}`` and include the change-of-variables
    Jacobian, so the chain targets the correct density on the simplex.

``"gibbs"``
    Latent-variable Gibbs sampler (see
    'tasks/latent_variable_sampling_dirichlet_posterior_dollar_math.md').
    Each likelihood term ``sum_y R(x_i, y) alpha(y)`` is the marginal of a
    latent class ``z_i`` with ``p(z_i = y | alpha) proportional to
    R(x_i, y) alpha(y)``; conditioned on the latent assignments the posterior
    of alpha is conjugate, ``alpha | z ~ Dirichlet(beta + counts(z))``.  The
    sampler alternates the two exact conditionals, so every move is accepted.
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
    sampler: str = "mh",
) -> MCMCResult:
    """Sample the test-prior posterior ``p(alpha | D)``.

    ``sampler`` selects the chain: ``"mh"`` runs the random-walk
    Metropolis-Hastings sampler (default), ``"gibbs"`` the latent-variable
    Gibbs sampler.  Both target the same posterior; ``step_size`` only
    applies to ``"mh"`` and is ignored by ``"gibbs"``, whose moves are
    exact conditional draws (``acceptance_rate`` is reported as 1).

    ``beta`` is the Dirichlet concentration: a (Y,) vector, or a scalar for a
    symmetric Dirichlet(beta, ..., beta).  Default: Dirichlet(1).  Note that
    the total pseudo-count is ``sum(beta)``, so with many classes Dirichlet(1)
    is far from uninformative — its per-class marginal Beta(1, Y-1) all but
    rules out a single class carrying large mass, and it overwhelms the
    likelihood of small unlabeled samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, Y = train_posterior.shape
    if beta is None:
        beta = np.ones(Y)
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 0:
        beta = np.full(Y, float(beta))

    # R(x_i, y) = p_tr(y | x_i) / p_tr(y).  Precompute once.
    R = train_posterior / train_prior[None, :]

    if sampler == "gibbs":
        chain, acceptance_rate = _run_gibbs(R, beta, num_iters, rng)
    elif sampler == "mh":
        chain, acceptance_rate = _run_mh(R, beta, num_iters, step_size, rng)
    else:
        raise ValueError(f"unknown sampler {sampler!r}; expected 'mh' or 'gibbs'")

    kept = chain[burn_in::thin]
    posterior_mean = kept.mean(axis=0)
    # Identifiability diagnostic: posterior std relative to the std of the
    # label-counting estimator sqrt(alpha(1-alpha)/n) at the same sample size.
    counting_std = np.sqrt(np.clip(posterior_mean * (1 - posterior_mean), 1e-12, None) / n)
    return MCMCResult(
        samples=kept,
        posterior_mean=posterior_mean,
        acceptance_rate=acceptance_rate,
        all_alpha=chain,
        ident_ratio=kept.std(axis=0) / counting_std,
    )


def _run_mh(
    R: np.ndarray,
    beta: np.ndarray,
    num_iters: int,
    step_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis-Hastings chain; returns (chain, acceptance rate)."""
    Y = R.shape[1]
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

    return chain, n_accept / num_iters


def _run_gibbs(
    R: np.ndarray,
    beta: np.ndarray,
    num_iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Latent-variable Gibbs chain; returns (chain, acceptance rate = 1).

    Alternates the two exact conditionals

        z_i | alpha  ~ Categorical( R(x_i, y) alpha(y) / sum_k R(x_i, k) alpha(k) )
        alpha | z    ~ Dirichlet( beta + counts(z) )

    which leave the augmented posterior ``p(alpha, z | D)`` invariant; the
    marginal over alpha is exactly ``p(alpha | D)``.
    """
    n, Y = R.shape
    alpha = np.full(Y, 1.0 / Y)   # start at the uniform prior, as the MH chain
    chain = np.empty((num_iters, Y))

    for t in range(num_iters):
        # Latent assignments: one categorical draw per observation, vectorised
        # via inverse-CDF sampling on the row-normalised weights R * alpha.
        W = R * alpha[None, :]
        cdf = W.cumsum(axis=1)
        u = rng.random(n) * cdf[:, -1]
        z = np.minimum((cdf < u[:, None]).sum(axis=1), Y - 1)
        # Conjugate update of alpha given the class counts.
        counts = np.bincount(z, minlength=Y)
        alpha = rng.dirichlet(beta + counts)
        chain[t] = alpha

    return chain, 1.0


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
