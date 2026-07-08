"""Loading a synthetic-generator setting from a JSON configuration file.

A configuration fully specifies the synthetic experiment: the mixture of 2-D
Gaussian class conditionals plus the training and test label priors.  Example::

    {
      "name": "default-4-gaussians",
      "means": [[1.4, 1.4], [1.4, -1.4], [-1.4, 1.4], [-1.4, -1.4]],
      "covs": [
        {"sx": 1.0, "sy": 0.6, "theta": 0.6},
        [[0.53, 0.0], [0.0, 1.17]]
      ],
      "train_prior": [0.25, 0.25, 0.25, 0.25],
      "test_prior":  [0.60, 0.20, 0.15, 0.05]
    }

Each covariance is either a full 2x2 matrix (nested list) or the readable
axis-aligned-rotated form ``{"sx": .., "sy": .., "theta": ..}`` with standard
deviations (sx, sy) along axes rotated by ``theta`` radians.  The number of
classes is implied by the length of ``means``; ``covs`` and both priors must
have the same length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import GaussianClassConditionalModel, cov_from_axes


@dataclass
class ExperimentConfig:
    """A generative model plus the train/test label priors."""

    name: str
    model: GaussianClassConditionalModel
    train_prior: np.ndarray  # (Y,)
    test_prior: np.ndarray   # (Y,)


def _parse_prior(raw, num_classes: int, field: str) -> np.ndarray:
    prior = np.asarray(raw, dtype=float)
    if prior.shape != (num_classes,):
        raise ValueError(
            f"'{field}' must have one entry per class "
            f"({num_classes}), got shape {prior.shape}.")
    if np.any(prior <= 0.0):
        raise ValueError(f"'{field}' entries must be strictly positive.")
    if not np.isclose(prior.sum(), 1.0, atol=1e-6):
        raise ValueError(f"'{field}' must sum to 1, got {prior.sum():.6f}.")
    return prior / prior.sum()  # exact normalisation


def _parse_cov(raw, index: int) -> np.ndarray:
    if isinstance(raw, dict):
        try:
            cov = cov_from_axes(raw["sx"], raw["sy"], raw["theta"])
        except KeyError as exc:
            raise ValueError(
                f"covs[{index}]: the dict form requires keys "
                f"'sx', 'sy', 'theta'; missing {exc}.") from None
    else:
        cov = np.asarray(raw, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError(f"covs[{index}] must be a 2x2 matrix, got {cov.shape}.")
    if not np.allclose(cov, cov.T):
        raise ValueError(f"covs[{index}] must be symmetric.")
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(f"covs[{index}] must be positive definite.") from None
    return cov


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate a JSON generator setting into an ExperimentConfig."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    for field in ("means", "covs", "train_prior", "test_prior"):
        if field not in raw:
            raise ValueError(f"config {path}: missing required field '{field}'.")

    means = [np.asarray(m, dtype=float) for m in raw["means"]]
    Y = len(means)
    if Y < 2:
        raise ValueError(f"config {path}: need at least 2 classes, got {Y}.")
    for i, m in enumerate(means):
        if m.shape != (2,):
            raise ValueError(
                f"config {path}: means[{i}] must be 2-D, got shape {m.shape}.")
    if len(raw["covs"]) != Y:
        raise ValueError(
            f"config {path}: 'covs' has {len(raw['covs'])} entries "
            f"but 'means' implies {Y} classes.")
    covs = [_parse_cov(c, i) for i, c in enumerate(raw["covs"])]

    return ExperimentConfig(
        name=raw.get("name", path.stem),
        model=GaussianClassConditionalModel(means=means, covs=covs),
        train_prior=_parse_prior(raw["train_prior"], Y, "train_prior"),
        test_prior=_parse_prior(raw["test_prior"], Y, "test_prior"),
    )
