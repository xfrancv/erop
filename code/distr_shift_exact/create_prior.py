"""Build a sliding-pair prior set ``Theta`` for ``rejopt_eval.py --priors``.

An alternative to the S7 construction of ``base_predictor_training.py``: instead
of priors whose shape is derived from the training frequencies and the per-class
validation errors, every element puts the *same* mass ``tau`` on an adjacent
pair of classes and spreads the rest uniformly. The result is a prior set whose
elements are equally spaced by construction, which makes the per-``theta_*``
breakdown of S6.4 comparable across elements: they differ only in *which* pair
is enriched, not in how far the shift goes.

With ``Y`` classes the file holds ``Y + 1`` priors::

    theta[0]        the training prior, copied from the input file. Index 0 by
                    the S7 convention: it is the element the 'shifted' stratum
                    excludes and the one the m = 0 tie-break of S3 returns.
    theta[1 + i]    ``tau`` on classes ``i`` and ``i + 1 (mod Y)``, and
                    ``(1 - 2 tau) / (Y - 2)`` on each of the others, for
                    ``i = 0 .. Y - 1``.

The pair wraps at the end (the last element enriches classes ``Y - 1`` and
``0``), which is what makes the count ``Y`` rather than ``Y - 1`` and keeps
every class enriched by exactly two elements.

``tau`` must lie in ``(0, 1/2)``: at ``1/2`` the remaining classes get zero mass,
which S7 forbids and which sends ``log theta`` to ``-inf``. Note ``tau = 1 / Y``
makes every element the uniform prior; the script refuses that and anything near
it, since a ``Theta`` of indistinguishable elements makes the experiment vacuous.

Run with::

    python create_prior.py runs/tissuemnist/priors.txt 0.2 \\
        runs/tissuemnist/priors_manual.txt
    python rejopt_eval.py runs/tissuemnist \\
        --priors runs/tissuemnist/priors_manual.txt \\
        --out-dir runs/tissuemnist/rejopt_manual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from exact.priors import TV_TOL, PriorSet, read_prior_set, write_prior_set


def pair_prior_set(train_prior: np.ndarray, tau: float) -> PriorSet:
    """``Theta`` of the training prior plus one element per adjacent pair."""
    train_prior = np.asarray(train_prior, dtype=float)
    Y = len(train_prior)
    assert Y >= 3, (
        f"a pair prior needs a third class to hold the remaining mass, but "
        f"Y = {Y}")
    assert 0.0 < tau < 0.5, (
        f"tau must lie in (0, 1/2), got {tau:g}: S7 forbids the zero entries "
        f"that tau = 0 puts on the pair and tau = 1/2 puts on every other "
        f"class")

    rest = (1.0 - 2.0 * tau) / (Y - 2)
    thetas, labels = [train_prior.copy()], ["train"]
    for i in range(Y):
        j = (i + 1) % Y
        theta = np.full(Y, rest)
        theta[[i, j]] = tau
        thetas.append(theta)
        labels.append(f"pair(tau={tau:g}) on classes [{i}, {j}]")

    theta = np.stack(thetas)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-12), \
        f"prior rows must sum to 1: {theta.sum(axis=1)}"
    assert np.all(theta > 0), "every prior entry must be strictly positive (S7)"
    return PriorSet(theta, labels, train_prior, dropped=[])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("priors", type=str,
                   help="the run's generated priors.txt, e.g. "
                        "runs/tissuemnist/priors.txt; only its theta[0] (the "
                        "training prior) and the number of classes are used")
    p.add_argument("tau", type=float,
                   help="mass on each class of the enriched pair, in (0, 1/2)")
    p.add_argument("out", type=str, help="priors file to write")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # read_prior_set only uses train_prior to fill PriorSet.train_prior, which
    # feeds the TV column; we take it from the file itself, so pass a placeholder
    # and never touch the TV of this intermediate set.
    src = read_prior_set(Path(args.priors), train_prior=np.ones(1))
    train_prior = src.theta[0]
    if src.labels[0] != "train":
        print(f"!! {args.priors}: theta[0] is labelled {src.labels[0]!r}, not "
              f"'train'. By the S7 convention index 0 is the training prior, "
              f"and it is what is copied into theta[0] of the output.",
              file=sys.stderr)

    ps = pair_prior_set(train_prior, args.tau)
    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    assert off.min() >= TV_TOL, (
        f"tau = {args.tau:g} collapses Theta: its closest pair of priors is "
        f"TV = {off.min():.4f} apart, under the S7 guard of {TV_TOL:g}. "
        f"tau = 1/Y = {1 / ps.Y:.4f} makes every element the uniform prior; "
        f"move tau away from it.")

    out = Path(args.out)
    write_prior_set(ps, out)

    print(f"{out}: C = {ps.C} priors over Y = {ps.Y} classes, tau = "
          f"{args.tau:g}, {(1 - 2 * args.tau) / (ps.Y - 2):.6g} on each of the "
          f"other {ps.Y - 2}")
    print(f"  min pairwise TV = {off.min():.4f}, "
          f"TV to theta_tr in [{ps.tv_to_train[1:].min():.4f}, "
          f"{ps.tv_to_train.max():.4f}]")
    for i, (lab, tv) in enumerate(zip(ps.labels, ps.tv_to_train)):
        print(f"  theta[{i}]  TV to theta_tr = {tv:.4f}   {lab}")


if __name__ == "__main__":
    main()
