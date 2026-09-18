"""The admissible test priors ``Theta`` of C3.3.

``Y`` sliding-pair priors: element ``i`` puts ``tau`` on classes ``i`` and
``i + 1 (mod Y)`` and spreads ``1 - 2 tau`` uniformly over the other ``Y - 2``.
Every element is the same distance from every other by construction, so a
per-``theta_*`` breakdown compares like with like.

**The training prior is not a member.** The parent S7 construction makes
``theta_1 = theta_tr``; here it is deliberately absent, so every test batch is
shifted and the non-adapted base predictor is beatable on all of them (C9.1).
``theta_tr`` is still needed -- and still shipped -- because the re-weighting
``theta_y / p_tr(y)`` divides by it.

Two consequences of dropping it, both intended:

* the S3 tie-break convention "lowest index degenerates to the train-prior
  plugin" no longer holds. Lowest-index is kept purely as a deterministic rule;
  it now selects an arbitrary shifted prior, which only matters on an exact
  likelihood tie (measure zero for a batch of one or more real images).
* the S6.4 "shifted trials only" stratification is vacuous and is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAU_DEFAULT = 0.35
TV_TOL = 1e-2


@dataclass
class PriorSet:
    """``Theta`` plus a label per element and the training prior it shifts from."""

    theta: np.ndarray            # (C, Y), rows sum to 1, all entries > 0
    labels: list[str]
    train_prior: np.ndarray      # (Y,)

    @property
    def C(self) -> int:
        return self.theta.shape[0]

    @property
    def Y(self) -> int:
        return self.theta.shape[1]

    @property
    def log_theta(self) -> np.ndarray:
        return np.log(self.theta)

    @property
    def log_p_theta(self) -> np.ndarray:
        """``log p(theta)``: uniform over ``Theta`` (C3.3), for the model and
        for drawing ``theta_*``."""
        return np.full(self.C, -np.log(self.C))

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


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def pair_prior_set(train_prior: np.ndarray, tau: float = TAU_DEFAULT) -> PriorSet:
    """``Theta``: one sliding-pair prior per class, training prior excluded."""
    train_prior = np.asarray(train_prior, dtype=float)
    Y = len(train_prior)
    assert Y >= 3, (
        f"a pair prior needs a third class to hold the remaining mass, Y = {Y}")
    assert 0.0 < tau < 0.5, (
        f"tau must lie in (0, 1/2), got {tau:g}: tau = 0 puts zero mass on the "
        f"pair and tau = 1/2 puts zero on every other class, and log theta is "
        f"-inf either way")

    rest = (1.0 - 2.0 * tau) / (Y - 2)
    thetas, labels = [], []
    for i in range(Y):
        j = (i + 1) % Y
        theta = np.full(Y, rest)
        theta[[i, j]] = tau
        thetas.append(theta)
        labels.append(f"pair(tau={tau:g}) on classes [{i}, {j}]")

    ps = PriorSet(np.stack(thetas), labels, train_prior)
    _guards(ps, tau)
    return ps


def _guards(ps: PriorSet, tau: float) -> None:
    assert np.allclose(ps.theta.sum(axis=1), 1.0, atol=1e-12), \
        f"prior rows must sum to 1: {ps.theta.sum(axis=1)}"
    assert np.all(ps.theta > 0), "every prior entry must be strictly positive"
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    assert off.min() >= TV_TOL, (
        f"tau = {tau:g} collapses Theta: its closest pair of priors is only "
        f"TV = {off.min():.4f} apart, under the guard of {TV_TOL:g}. "
        f"tau = 1/Y = {1 / ps.Y:.4f} makes every element the uniform prior; "
        f"move tau away from it.")
    assert ps.tv_to_train.min() >= TV_TOL, (
        "an element of Theta coincides with the training prior, which C3.3 "
        "excludes by construction")


# --- the priors file ------------------------------------------------------

_HEADER = "# index  label  TV(theta, theta_tr)  theta_1 ... theta_Y"


def write_prior_set(ps: PriorSet, path: Path) -> None:
    tv = ps.tv_to_train
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    lines = [
        "# Theta: the admissible test priors of the challenge (C3.3)",
        f"# C = {ps.C} priors over Y = {ps.Y} classes",
        "# p(theta) = 1/C uniform, for the model and for drawing theta_*",
        "# the TRAINING PRIOR IS NOT A MEMBER: every test batch is shifted",
        f"# min pairwise TV = {off.min():.4f} (guard: >= {TV_TOL:g})",
        f"# TV to theta_tr in [{tv.min():.4f}, {tv.max():.4f}]",
        _HEADER,
    ]
    for i, (t, lab) in enumerate(zip(ps.theta, ps.labels)):
        vec = " ".join(f"{v:.10g}" for v in t)
        lines.append(f"{i}\t{lab}\t{tv[i]:.6f}\t{vec}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_prior_set(path: Path, train_prior: np.ndarray) -> PriorSet:
    thetas, labels = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _, label, _tv, vec = line.split("\t")
        thetas.append(np.fromstring(vec, sep=" "))
        labels.append(label)
    theta = np.stack(thetas)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-6), \
        f"{path}: prior rows do not sum to 1"
    return PriorSet(theta, labels, np.asarray(train_prior, float))
