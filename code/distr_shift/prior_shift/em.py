"""EM estimate of the test label prior from unlabeled data (label shift).

The maximum-likelihood point-estimate counterpart of the MCMC sampler in
``mcmc.py``: both target the same unlabeled log-likelihood

    log p(D | theta) = sum_i log( sum_y R(x_i, y) alpha(y) ) + const ,
    R(x, y) = p_tr(y | x) / p_tr(y),   theta = alpha = p_te(y) ,

but where ``sample_prior_posterior`` draws from the (Dirichlet-regularised)
posterior over ``alpha``, this returns its MLE via the classic
Saerens-Latinne-Decaestecker (2002) EM iteration:

    E-step:  alpha_i(y)  proportional to  p_tr(y | x_i) * hat p(y) / p_tr(y)
    M-step:  hat p(y)   <-  (1/m) sum_i alpha_i(y)

which monotonically increases the log-likelihood above.  The estimate is then
plugged into ``corrected_posterior`` to build the plugin Bayes reject-option
predictor.
"""

from __future__ import annotations

import numpy as np


def estimate_prior_em(
    train_posterior: np.ndarray,   # (n, Y) = p_tr(y | x_i)
    train_prior: np.ndarray,       # (Y,)  = p_tr(y)
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """MLE of the test label prior ``hat p(y)`` from unlabeled inputs.

    ``train_posterior`` are the calibrated training posteriors ``p_tr(y | x_i)``
    of the unlabeled adaptation inputs; ``train_prior`` is ``p_tr(y)``.  Iterates
    the Saerens EM E-/M-steps from ``init`` (default: the uniform prior) until the
    increase in the log-likelihood between two steps falls below ``tol`` or
    ``max_iter`` steps are taken.  Returns the ``(Y,)`` estimate on the simplex.
    """
    R = train_posterior / train_prior[None, :]     # R(x_i, y) = p_tr(y|x)/p_tr(y)
    n, Y = R.shape

    p = (np.full(Y, 1.0 / Y) if init is None
         else np.asarray(init, dtype=float).copy())

    prev_ll = -np.inf
    for _ in range(max_iter):
        w = R * p[None, :]                         # unnormalised alpha_i(y)
        s = w.sum(axis=1, keepdims=True)           # = p(x_i) / p_tr(x_i)
        ll = float(np.log(np.clip(s, 1e-300, None)).sum())
        alpha = w / np.clip(s, 1e-300, None)       # E-step: responsibilities
        p = alpha.mean(axis=0)                      # M-step: prior update
        if ll - prev_ll < tol:
            break
        prev_ll = ll
    return p
