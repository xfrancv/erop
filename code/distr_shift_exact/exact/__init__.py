"""Exact Bayesian label-prior adaptation with a finite prior set.

Layout mirrors the README's sections:

``splits``       S6.1 split policy
``calibration``  S6.1 BCTS calibration, NLL and equal-mass ECE
``priors``       S7 generation and serialisation of the finite prior set Theta
``inference``    S2/S5 the exact log-space posterior, T/A/E, and the predictors
``protocol``     S6.2-S6.4 grid, trial sampling, budgets
``sweep``        the inner loop that turns trials into pooled triplets
``metrics``      S4 selective curves, summary scalars, and the trial bootstrap
"""

from .calibration import (
    calibration_summary,
    ece_equal_mass,
    fit_bcts,
    fit_temperature,
    log_softmax_np,
    nll,
)
from .inference import (
    TrialInference,
    identifiability_table,
    infer_trial,
    log_weights,
    plugin_for_prior,
)
from .metrics import (
    DEFAULT_COVERAGE,
    REJECTORS,
    REJECTOR_KEYS,
    CellResult,
    Pool,
    Rejector,
    evaluate_pool,
)
from .priors import (
    PriorSet,
    build_prior_set,
    dirichlet_prior_set,
    read_prior_set,
    total_variation,
    write_prior_set,
)
from .protocol import (
    BUDGET_B,
    N_MAX,
    N_MIN,
    SIZE_GRID,
    ThetaStarDrawer,
    Trial,
    TrialSampler,
    n_trials,
    resolve_grid,
    subsample_layout,
)
from .splits import Splits, make_splits
from .sweep import CellDiagnostics, run_cell

__all__ = [
    "BUDGET_B", "CellDiagnostics", "CellResult", "DEFAULT_COVERAGE", "N_MAX",
    "N_MIN", "Pool", "PriorSet", "REJECTORS", "REJECTOR_KEYS", "Rejector",
    "SIZE_GRID", "Splits", "ThetaStarDrawer", "Trial", "TrialInference",
    "TrialSampler", "build_prior_set", "calibration_summary",
    "dirichlet_prior_set", "ece_equal_mass", "evaluate_pool", "fit_bcts",
    "fit_temperature", "identifiability_table", "infer_trial", "log_softmax_np",
    "log_weights", "make_splits", "n_trials", "nll", "plugin_for_prior",
    "read_prior_set", "resolve_grid",
    "run_cell", "subsample_layout", "total_variation", "write_prior_set",
]
