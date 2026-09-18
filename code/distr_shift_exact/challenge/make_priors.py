#!/usr/bin/env python3
"""Build ``Theta``, the admissible test priors of the challenge (C3.3).

``Y`` sliding-pair priors, ``tau`` on an adjacent pair of classes and the rest
spread uniformly. **The training prior is deliberately not a member**, so every
test batch is shifted and the non-adapted base predictor is beatable on all of
them (C9.1). This replaces the hand-trimmed file the plan inherited: the shipped
``Theta`` now has a command line behind it.

    python make_priors.py out/model priors_tissuemnist_challenge.txt
    python make_priors.py out/model priors.txt --tau 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chal.priors import TAU_DEFAULT, pair_prior_set, write_prior_set


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path,
                   help="output directory of train_base_model.py; only its "
                        "train_prior is used")
    p.add_argument("out", type=Path, help="priors file to write")
    p.add_argument("--tau", type=float, default=TAU_DEFAULT,
                   help=f"mass on each class of the enriched pair, in (0, 1/2) "
                        f"(default {TAU_DEFAULT})")
    args = p.parse_args()

    train_prior = np.load(args.model_dir / "log_post.npz")["train_prior"]
    ps = pair_prior_set(train_prior, args.tau)
    write_prior_set(ps, args.out)

    off = ps.pairwise_tv()[np.triu_indices(ps.C, k=1)]
    rest = (1 - 2 * args.tau) / (ps.Y - 2)
    print(f"{args.out}: C = {ps.C} priors over Y = {ps.Y} classes")
    print(f"  tau = {args.tau:g} on the pair, {rest:.6g} on each of the other "
          f"{ps.Y - 2}")
    print(f"  the training prior is NOT a member (C3.3)")
    print(f"  min pairwise TV = {off.min():.4f}")
    print(f"  TV to theta_tr in [{ps.tv_to_train.min():.4f}, "
          f"{ps.tv_to_train.max():.4f}]")
    for i, (lab, tv) in enumerate(zip(ps.labels, ps.tv_to_train)):
        print(f"  theta[{i}]  TV to theta_tr = {tv:.4f}   {lab}")


if __name__ == "__main__":
    main()
