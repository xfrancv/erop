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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, Y: int) -> np.ndarray:
    """(Y, Y) matrix with rows = true class, columns = predicted class."""
    cm = np.zeros((Y, Y), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def format_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    """Confusion matrix as text, rows true / columns predicted."""
    short = [n[:10] for n in class_names]
    width = max(10, max(len(s) for s in short) + 1)
    lines = [" " * width + "".join(f"{s:>{width}}" for s in short)
             + "   (rows = true, cols = predicted)"]
    for i, name in enumerate(short):
        lines.append(f"{name:>{width}}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    return "\n".join(lines)


def per_class_error(y_true: np.ndarray, y_pred: np.ndarray, Y: int) -> np.ndarray:
    """Per-class 0/1 error ``Err(k) = mean_{i: y_i = k} [ y_pred_i != k ]``.

    Returns a (Y,) array; ``Err(k) = 1 - recall_k`` and is ``nan`` for classes
    absent from ``y_true``. Equivalently one minus the row-normalised diagonal
    of ``confusion_matrix``.
    """
    err = np.full(Y, np.nan)
    for k in range(Y):
        mask = y_true == k
        if mask.any():
            err[k] = float(np.mean(y_pred[mask] != k))
    return err


def format_per_class_error(err: np.ndarray, class_names: list[str],
                           counts: np.ndarray | None = None) -> str:
    """Per-class error as text, one line per class (with optional support n)."""
    width = max(10, max(len(n) for n in class_names) + 1)
    header = f"{'class':>{width}}{'Err(k)':>10}"
    if counts is not None:
        header += f"{'n':>8}"
    lines = [header]
    for k, name in enumerate(class_names):
        val = "   n/a" if np.isnan(err[k]) else f"{err[k]:.4f}"
        row = f"{name:>{width}}{val:>10}"
        if counts is not None:
            row += f"{int(counts[k]):>8}"
        lines.append(row)
    return "\n".join(lines)
