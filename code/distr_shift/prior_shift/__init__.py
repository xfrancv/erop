"""Bayesian label-prior adaptation from unlabeled test data (label shift)."""

from .base_model import BaseModel
from .config import ExperimentConfig, load_experiment_config
from .data import GaussianClassConditionalModel
from .em import estimate_prior_em
from .mcmc import (
    MCMCResult,
    posterior_label_probabilities,
    sample_prior_posterior,
)
from .predictors import (
    bayes_decision,
    confusion_matrix,
    corrected_posterior,
    format_confusion,
    format_per_class_error,
    per_class_error,
    zero_one_loss_matrix,
)

__all__ = [
    "BaseModel",
    "ExperimentConfig",
    "load_experiment_config",
    "GaussianClassConditionalModel",
    "estimate_prior_em",
    "MCMCResult",
    "sample_prior_posterior",
    "posterior_label_probabilities",
    "bayes_decision",
    "corrected_posterior",
    "confusion_matrix",
    "format_confusion",
    "per_class_error",
    "format_per_class_error",
    "zero_one_loss_matrix",
]
