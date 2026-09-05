"""The split policy of README S6.1, in one place.

Uniform across datasets:

* **development** = the dataset's official ``train`` split. It is divided
  class-stratified into a **fit** part (0.80, trains the network weights) and a
  **validation** part (0.20, model selection + BCTS calibration + the
  per-class error rates that define ``theta_3``).
* **evaluation** = the official ``test`` split, with the official ``val`` split
  merged in when the dataset has one. Never seen during training or
  calibration; every test set ``D'`` of S6.3 is drawn from it.

``theta_tr`` is the empirical class frequency of the **fit** part, because that
is the prior the network was actually fit under and the whole re-weighting
``theta_y / p_tr(y)`` is only valid against it (S6.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from data_tools.loaders import Dataset


@dataclass
class Splits:
    """The three arrays plus the labels that describe how they were made."""

    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    eval_desc: str
    num_classes: int

    @property
    def train_prior(self) -> np.ndarray:
        """``theta_tr``: empirical class frequency of the fit part."""
        counts = np.bincount(self.y_fit, minlength=self.num_classes).astype(float)
        return counts / counts.sum()

    @property
    def eval_class_counts(self) -> np.ndarray:
        return np.bincount(self.y_eval, minlength=self.num_classes)


def make_splits(ds: Dataset, val_fraction: float = 0.2, seed: int = 0) -> Splits:
    X_dev, y_dev = ds.splits["train"]
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_dev, y_dev, test_size=val_fraction, stratify=y_dev, random_state=seed)

    if "val" in ds.splits:
        X_eval = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
        y_eval = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
        eval_desc = "official val + test merged"
    else:
        X_eval, y_eval = ds.splits["test"]
        eval_desc = "official test split"

    return Splits(X_fit, y_fit, X_val, y_val, X_eval, y_eval,
                  eval_desc, ds.num_classes)
