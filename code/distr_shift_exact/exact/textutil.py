"""Text formatting shared by the two report-writing drivers."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, Y: int) -> np.ndarray:
    """(Y, Y) matrix with rows = true class, columns = predicted class."""
    cm = np.zeros((Y, Y), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_error(y_true: np.ndarray, y_pred: np.ndarray, Y: int) -> np.ndarray:
    """``Err(k) = P(yhat != k | y = k)``; NaN for classes absent from ``y_true``.

    This is what selects the two "hard" classes of ``theta_3`` (README S7).
    """
    err = np.full(Y, np.nan)
    for k in range(Y):
        mask = y_true == k
        if mask.any():
            err[k] = float(np.mean(y_pred[mask] != k))
    return err


def format_confusion(cm: np.ndarray, class_names: list[str],
                     max_classes: int = 20) -> str:
    """Confusion matrix as text; elided for datasets with many classes."""
    if len(class_names) > max_classes:
        return (f"({len(class_names)} classes -- confusion matrix omitted; "
                f"see the per-class error table below)")
    short = [n[:10] for n in class_names]
    width = max(10, max(len(s) for s in short) + 1)
    lines = [" " * width + "".join(f"{s:>{width}}" for s in short)
             + "   (rows = true, cols = predicted)"]
    for i, name in enumerate(short):
        lines.append(f"{name:>{width}}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    return "\n".join(lines)


def format_per_class_error(err: np.ndarray, class_names: list[str],
                           counts: np.ndarray | None = None) -> str:
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


def format_prior_table(ps, class_names: list[str], max_classes: int = 12) -> str:
    """The prior set as a table with the S7 TV column."""
    tv = ps.tv_to_train
    if ps.Y <= max_classes:
        head = f"{'idx':>4} {'TV':>7}  " + "".join(
            f"{n[:7]:>8}" for n in class_names) + "   label"
        lines = [head]
        for i, t in enumerate(ps.theta):
            lines.append(f"{i:>4} {tv[i]:>7.4f}  "
                         + "".join(f"{v:>8.4f}" for v in t)
                         + f"   {ps.labels[i]}")
    else:
        lines = [f"{'idx':>4} {'TV':>7}  {'min':>8}{'max':>8}   label"]
        for i, t in enumerate(ps.theta):
            lines.append(f"{i:>4} {tv[i]:>7.4f}  {t.min():>8.5f}{t.max():>8.5f}"
                         f"   {ps.labels[i]}")
    pw = ps.pairwise_tv()
    off = pw[np.triu_indices(ps.C, k=1)]
    lines.append(f"min pairwise TV = {off.min():.4f}, "
                 f"max = {off.max():.4f}  (guard: >= 0.01)")
    return "\n".join(lines)


def format_calibration(temperature: np.ndarray, bias: np.ndarray,
                       class_names: list[str], max_classes: int = 20) -> str:
    """The fitted BCTS map ``softmax(z_k / T_k + b_k)``, per class.

    Elided to five-number summaries for datasets with many classes, the way
    ``format_confusion`` is.
    """
    def summary(name, v, fmt):
        return (f"  {name} : min {v.min():{fmt}}  median {np.median(v):{fmt}}"
                f"  max {v.max():{fmt}}")

    if len(class_names) > max_classes:
        return "\n".join([
            summary("T_k", temperature, ".4f"),
            summary("b_k", bias, "+.4f"),
            f"  ({len(class_names)} classes -- per-class table omitted)"])
    width = max(10, max(len(n) for n in class_names) + 1)
    lines = [f"{'class':>{width}}{'T_k':>10}{'b_k':>10}"]
    for k, name in enumerate(class_names):
        lines.append(f"{name:>{width}}{temperature[k]:>10.4f}{bias[k]:>+10.4f}")
    return "\n".join(lines)
