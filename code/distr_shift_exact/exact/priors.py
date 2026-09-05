"""Build and serialise the finite prior set ``Theta`` of README S7.

``Theta`` is an *output of the training run*, not a hand-written config: two of
its entries depend on the training class frequencies and on the per-class
validation error rates, so it is emitted by ``base_predictor_training.py``
after calibration and read back by ``rejopt_eval.py``.

The default generator produces ``C = 3 + S`` priors with
``S = min(5, ceil(Y / 2))``:

===========================  ====================================================
``theta_1`` train prior      ``theta_tr``, the fit-part class frequency. Index 1
                             by convention so that the lowest-index tie-break of
                             S3 degenerates to the train-prior plugin at m = 0.
``theta_2`` uniform          ``1 / Y``.
``theta_3`` hard-class spike ``tau`` on each of the two highest-validation-error
                             classes, the rest redistributed in proportion to
                             ``theta_tr``.
``theta_4..``  doubling      one prior per class for the ``S`` most frequent
                             classes, each doubling that single class.
===========================  ====================================================

Guards (all asserted here, S7): every entry positive and summing to one, and no
two elements within total variation ``TV_TOL`` of each other -- which is also
what removes ``theta_2`` on an exactly balanced dataset, where it coincides
with ``theta_1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAU_DEFAULT = 0.2
TV_TOL = 1e-2
# Pairwise TV within this factor of TV_TOL is reported as a near-degeneracy:
# the construction survived the guard, but only just, and priors that close
# together are not distinguishable from an unlabeled sample of the sizes in the
# S6.2 grid. See ``identifiability_report`` in :mod:`exact.inference`.
TV_WARN_FACTOR = 2.0


@dataclass
class PriorSet:
    """``Theta`` plus the label and TV-to-train-prior of each element."""

    theta: np.ndarray            # (C, Y), rows sum to 1
    labels: list[str]
    train_prior: np.ndarray      # (Y,)
    dropped: list[str]           # human-readable notes about dropped duplicates

    @property
    def C(self) -> int:
        return self.theta.shape[0]

    @property
    def Y(self) -> int:
        return self.theta.shape[1]

    @property
    def tv_to_train(self) -> np.ndarray:
        return np.array([total_variation(t, self.train_prior) for t in self.theta])

    def pairwise_tv(self) -> np.ndarray:
        C = self.C
        M = np.zeros((C, C))
        for a in range(C):
            for b in range(a + 1, C):
                M[a, b] = M[b, a] = total_variation(self.theta[a], self.theta[b])
        return M

    def log_theta(self) -> np.ndarray:
        return np.log(self.theta)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _spike(train_prior: np.ndarray, classes: np.ndarray, tau: float) -> np.ndarray:
    """``tau`` on each class in ``classes``; the rest in proportion to the prior."""
    Y = len(train_prior)
    rest = np.setdiff1d(np.arange(Y), classes)
    denom = train_prior[rest].sum()
    theta = np.empty(Y)
    theta[classes] = tau
    theta[rest] = (1.0 - tau * len(classes)) * train_prior[rest] / denom
    return theta


def _double_one_class(train_prior: np.ndarray, c: int) -> np.ndarray:
    """Double class ``c``; renormalise the others so the vector still sums to 1."""
    theta = train_prior * (1.0 - 2.0 * train_prior[c]) / (1.0 - train_prior[c])
    theta[c] = 2.0 * train_prior[c]
    return theta


def build_prior_set(train_prior: np.ndarray, val_error: np.ndarray,
                    tau: float = TAU_DEFAULT) -> PriorSet:
    """Generate ``Theta`` from the training prior and per-class validation error.

    ``val_error`` may contain NaN for classes absent from the validation split;
    those classes simply never win the "hardest class" ranking.
    """
    train_prior = np.asarray(train_prior, dtype=float)
    Y = len(train_prior)
    assert Y >= 3, "theta_3 (hard-class spike) requires Y >= 3"
    assert 2 * tau < 1, f"theta_3 requires 2*tau < 1, got tau={tau}"
    assert np.all(train_prior > 0), "every class must be present in the fit part"

    thetas = [train_prior.copy()]
    labels = ["train"]

    thetas.append(np.full(Y, 1.0 / Y))
    labels.append("uniform")

    # Hardest two classes by validation error. Ties (and the all-NaN case) are
    # broken by lowest class index via the stable sort, so Theta is a
    # deterministic function of the training run.
    err = np.where(np.isnan(val_error), -np.inf, val_error)
    hard = np.argsort(-err, kind="stable")[:2]
    thetas.append(_spike(train_prior, np.sort(hard), tau))
    labels.append(f"spike(tau={tau:g}) on classes {sorted(int(c) for c in hard)}")

    # One doubling prior per class, over a *proper subset* of the classes:
    # doubling every class at once and renormalising returns the training prior
    # exactly, which is why S7 loops over the S most frequent classes only.
    S = min(5, int(np.ceil(Y / 2)))
    freq_order = np.argsort(-train_prior, kind="stable")[:S]
    for c in freq_order:
        if 2.0 * train_prior[c] >= 1.0:
            continue
        thetas.append(_double_one_class(train_prior, int(c)))
        labels.append(f"double class {int(c)}")

    theta = np.stack(thetas)
    return _apply_guards(theta, labels, train_prior)


def dirichlet_prior_set(train_prior: np.ndarray, n: int, concentration: float,
                        rng: np.random.Generator) -> PriorSet:
    """``Theta`` of ``n`` draws from ``Dir(concentration * theta_tr)``.

    Not part of the default protocol; a way to build a larger, less structured
    prior set for the sensitivity study of Appendix A.
    """
    theta = rng.dirichlet(concentration * np.asarray(train_prior, float), size=n)
    labels = [f"dirichlet draw {i}" for i in range(n)]
    return _apply_guards(theta, labels, np.asarray(train_prior, float))


def _apply_guards(theta: np.ndarray, labels: list[str],
                  train_prior: np.ndarray) -> PriorSet:
    """Assert the S7 guards, drop near-duplicates, and re-index."""
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-9), \
        f"prior rows must sum to 1: {theta.sum(axis=1)}"
    assert np.all(theta > 0), "every prior entry must be strictly positive"

    keep, dropped = [], []
    for i in range(len(theta)):
        clash = next((j for j in keep
                      if total_variation(theta[i], theta[j]) < TV_TOL * (1 - 1e-9)),
                     None)
        if clash is None:
            keep.append(i)
        else:
            dropped.append(
                f"theta[{i}] ({labels[i]}) dropped: TV to theta[{clash}] "
                f"({labels[clash]}) = {total_variation(theta[i], theta[clash]):.2e}"
                f" < {TV_TOL:g}")

    ps = PriorSet(theta[keep], [labels[i] for i in keep], train_prior, dropped)
    assert ps.C >= 2, (
        "Theta collapsed to fewer than 2 distinct priors -- the experiment is "
        "vacuous. Raise tau or use --prior-set dirichlet.")
    return ps


# --- the priors file ------------------------------------------------------

_HEADER = "# index  label  TV(theta, theta_tr)  theta_1 ... theta_Y"


def write_prior_set(ps: PriorSet, path: Path) -> None:
    """Write the human-readable, re-parseable priors file of S7."""
    tv = ps.tv_to_train
    pw = ps.pairwise_tv()
    off = pw[np.triu_indices(ps.C, k=1)]
    lines = [
        "# Theta: the finite prior set (README S7)",
        f"# C = {ps.C} priors over Y = {ps.Y} classes",
        "# p(theta) = 1/C uniform, for the model and for drawing theta_*",
        f"# min pairwise TV = {off.min():.4f} (guard: >= {TV_TOL:g})",
        *[f"# {d}" for d in ps.dropped],
        _HEADER,
    ]
    if off.min() < TV_TOL * TV_WARN_FACTOR:
        lines.insert(-1, (
            f"# WARNING: the closest pair of priors is only TV = {off.min():.4f} "
            f"apart; such priors are hard to tell apart from an unlabeled sample "
            f"of the sizes in the S6.2 grid. See the identifiability table in "
            f"the rejopt_eval report."))
    for i, (t, lab) in enumerate(zip(ps.theta, ps.labels)):
        vec = " ".join(f"{v:.10g}" for v in t)
        lines.append(f"{i}\t{lab}\t{tv[i]:.6f}\t{vec}")
    path.write_text("\n".join(lines) + "\n")


def read_prior_set(path: Path, train_prior: np.ndarray) -> PriorSet:
    """Read back a priors file written by :func:`write_prior_set`."""
    thetas, labels, dropped = [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            if "dropped:" in line:
                dropped.append(line.lstrip("# "))
            continue
        _, label, _tv, vec = line.split("\t")
        thetas.append(np.fromstring(vec, sep=" "))
        labels.append(label)
    theta = np.stack(thetas)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-6), \
        f"{path}: prior rows do not sum to 1"
    return PriorSet(theta, labels, np.asarray(train_prior, float), dropped)
