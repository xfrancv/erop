"""Predictors and the Bayes decision rule for an arbitrary loss matrix.

For a label posterior ``q(y | x)`` the Bayes-optimal prediction under a loss
``loss[yhat, y] = ell(yhat, y)`` is

    h(x) = argmin_{yhat} sum_y q(y | x) loss[yhat, y] .

For the 0/1 loss this reduces to ``argmax_y q(y | x)``.
"""

from __future__ import annotations

import numpy as np


def zero_one_loss_matrix(num_classes: int) -> np.ndarray:
    """(Y, Y) matrix with ``loss[yhat, y] = 0`` if equal else 1."""
    return 1.0 - np.eye(num_classes)


def bayes_decision(posterior: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Return argmin_yhat expected loss for each row of ``posterior``.

    ``posterior`` is (n, Y); ``loss`` is (Y, Y) with rows indexed by yhat.
    """
    risk = posterior @ loss.T          # (n, Y): expected loss of each yhat
    return risk.argmin(axis=1)


def corrected_posterior(
    train_posterior: np.ndarray,   # (n, Y) = p_tr(y | x)
    train_prior: np.ndarray,       # (Y,)
    target_prior: np.ndarray,      # (Y,)
) -> np.ndarray:
    """Plugin label-shift posterior for a chosen ``target_prior``.

    ``p(y | x) proportional to p_tr(y | x) * target_prior(y) / p_tr(y)``.
    """
    w = train_posterior * (target_prior / train_prior)[None, :]
    return w / w.sum(axis=1, keepdims=True)
