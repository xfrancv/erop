"""The base discriminative model estimated from supervised training data.

We fit multinomial logistic regression to obtain an estimate of the training
class posterior ``p_tr(y | x)`` and use the empirical label frequencies as an
estimate of the training prior ``p_tr(y)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class BaseModel:
    """Wraps a fitted logistic-regression posterior and an empirical prior."""

    clf: LogisticRegression
    train_prior: np.ndarray  # (Y,) estimate of p_tr(y)
    num_classes: int

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, num_classes: int) -> "BaseModel":
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        # Empirical training prior, robust to unseen classes.
        counts = np.bincount(y, minlength=num_classes).astype(float)
        train_prior = counts / counts.sum()
        # Reorder the classifier's probability columns into 0..Y-1 order.
        return cls(clf=clf, train_prior=train_prior, num_classes=num_classes)

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Estimated training posterior ``p_tr(y | x)`` as an (n, Y) matrix."""
        proba = self.clf.predict_proba(X)
        # Map sklearn's class ordering to 0..Y-1.
        full = np.zeros((X.shape[0], self.num_classes))
        full[:, self.clf.classes_] = proba
        return full
