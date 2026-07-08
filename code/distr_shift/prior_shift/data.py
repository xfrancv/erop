"""Synthetic data model for the prior-shift (label-shift) experiment.

Each class ``y`` has a fixed 2-D Gaussian class-conditional density
``p(x | y) = N(x; mu_y, Sigma_y)`` that is *shared* by the training and test
distributions.  Only the label prior ``p(y)`` differs between train and test,
which is exactly the label-shift assumption ``p_tr(x | y) = p_te(x | y)``.

The class conditionals are known in closed form, so the Bayes-optimal predictor
for any prior can be computed exactly and used as a baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def cov_from_axes(sx: float, sy: float, theta: float) -> np.ndarray:
    """Covariance with std devs (sx, sy) along axes rotated by ``theta``."""
    R = _rotation(theta)
    return R @ np.diag([sx**2, sy**2]) @ R.T


@dataclass
class GaussianClassConditionalModel:
    """A generative model with Gaussian class conditionals and a label prior.

    Attributes
    ----------
    means : list of (2,) arrays, one mean per class.
    covs  : list of (2, 2) arrays, one covariance per class.
    """

    means: list[np.ndarray]
    covs: list[np.ndarray]

    @property
    def num_classes(self) -> int:
        return len(self.means)

    @classmethod
    def default(cls) -> "GaussianClassConditionalModel":
        """Four classes placed at the corners of a square.

        The separation is tuned so that the Bayes error under a *uniform*
        prior is ~10% (see ``scripts``/README for the tuning check).
        """
        s = 1.4
        means = [
            np.array([s, s]),
            np.array([s, -s]),
            np.array([-s, s]),
            np.array([-s, -s]),
        ]
        covs = [
            cov_from_axes(1.0, 0.6, 0.6),
            cov_from_axes(0.7, 1.1, -0.3),
            cov_from_axes(1.2, 0.8, 0.9),
            cov_from_axes(0.9, 0.9, 0.0),
        ]
        return cls(means=means, covs=covs)

    def class_conditional_pdf(self, X: np.ndarray) -> np.ndarray:
        """Return the (n, Y) matrix of densities ``p(x_i | y)``."""
        return np.column_stack(
            [multivariate_normal(self.means[y], self.covs[y]).pdf(X)
             for y in range(self.num_classes)]
        )

    def class_conditional_logpdf(self, X: np.ndarray) -> np.ndarray:
        """Return the (n, Y) matrix of log densities ``log p(x_i | y)``."""
        return np.column_stack(
            [multivariate_normal(self.means[y], self.covs[y]).logpdf(X)
             for y in range(self.num_classes)]
        )

    def sample(self, n: int, prior: np.ndarray, rng: np.random.Generator):
        """Draw ``n`` i.i.d. samples ``(x, y)`` from ``p(x | y) prior(y)``."""
        prior = np.asarray(prior, dtype=float)
        counts = rng.multinomial(n, prior)
        X_parts, y_parts = [], []
        for y, c in enumerate(counts):
            if c == 0:
                continue
            X_parts.append(rng.multivariate_normal(self.means[y], self.covs[y], c))
            y_parts.append(np.full(c, y, dtype=int))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        # Shuffle so class blocks are not contiguous.
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def true_posterior(self, X: np.ndarray, prior: np.ndarray) -> np.ndarray:
        """Exact posterior ``p(y | x)`` for a given label ``prior``.

        Returns an (n, Y) matrix whose rows sum to 1.
        """
        prior = np.asarray(prior, dtype=float)
        log_joint = self.class_conditional_logpdf(X) + np.log(prior)[None, :]
        log_joint -= log_joint.max(axis=1, keepdims=True)
        joint = np.exp(log_joint)
        return joint / joint.sum(axis=1, keepdims=True)
