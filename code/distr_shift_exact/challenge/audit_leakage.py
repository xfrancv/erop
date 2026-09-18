#!/usr/bin/env python3
"""Measure what a competitor can learn from the *sampling*, not the images.

A batch draws a class from ``theta_*`` and then an image uniformly from that
class's pool, so one slot picks a particular image ``X`` with probability

    P(X | theta) = theta_{y_X} / n_{y_X}

where ``n_y`` is the number of images of class ``y`` in the pool. Averaging over
``theta`` uniform on ``Theta``, every class has ``E[theta_y] = 1/8`` -- each is
enriched by exactly two of the eight priors -- so the ``theta`` washes out. The
``1 / n_y`` does not. On an imbalanced pool the number of times an ``img_id``
appears in ``test_batches.csv`` is therefore a direct signal of its class, and it
costs a competitor nothing: the counts are in the released files, and the labels
needed to calibrate ``P(n | y)`` are in ``dev.csv``.

This script measures the size of that channel:

1. **marginal** -- mean occurrences per class. Flat means the channel is shut.
2. **accuracy** -- what the occurrence count adds to the base posterior.
3. **score** -- what it does to the metric when folded into the intended
   solution. The danger sign is a submission that beats the reference predictor,
   i.e. scores below zero, using nothing but the provided data.

Run it on every generated dataset before launch:

    python audit_leakage.py out/kaggle
    python audit_leakage.py out/kaggle --usage Private
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from chal.metric import score
from chal.predictors import Problem, rank_confidence

Y = 8


def occurrence_table(counts: np.ndarray, labels: np.ndarray, n_max: int
                     ) -> np.ndarray:
    """``P(n | y)`` with a Laplace prior, as a competitor would estimate it."""
    tab = np.zeros((Y, n_max + 1))
    for c in range(Y):
        tab[c] = np.bincount(counts[labels == c], minlength=n_max + 1)
    return (tab + 0.5) / (tab.sum(1, keepdims=True) + 0.5 * (n_max + 1))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kaggle_dir", type=Path)
    p.add_argument("--usage", default="Private",
                   choices=("Public", "Private", "Ignored"))
    args = p.parse_args()
    d = args.kaggle_dir

    tb = pd.read_csv(d / "test_batches.csv")
    sol = pd.read_csv(d / "solution.csv")
    # The public files carry no image key (C9.7). row_image.csv is local-only
    # and exists precisely so this audit can still measure the channel a
    # competitor would have to reconstruct by matching identical posteriors.
    tb = tb.merge(pd.read_csv(d / "row_image.csv"), on="row_id")
    occ = tb.groupby("img_id").size()

    rows = sol.merge(tb[["row_id", "img_id"]], on="row_id")
    uniq = rows.drop_duplicates("img_id")
    un = occ.reindex(uniq["img_id"]).to_numpy()
    uy = uniq["label"].to_numpy()

    pub = pd.read_csv(d / "test_batches.csv", nrows=1).columns.tolist()
    print(f"{d}\n{'=' * 70}")
    print(f"published test_batches.csv columns: {pub}")
    print(f"  image key published: "
          f"{'YES - channel handed over' if any('img' in c for c in pub) else 'NO - must be reconstructed by matching posteriors'}\n")
    print("1. MARGINAL CHANNEL -- mean occurrences of an img_id, by true class")
    means = np.array([un[uy == c].mean() for c in range(Y)])
    print("   " + "  ".join(f"{c}:{v:.2f}" for c, v in enumerate(means)))
    spread = means.max() / means.min()
    print(f"   spread {spread:.2f}x   "
          f"{'FLAT - channel shut' if spread < 1.15 else 'LEAKS - class is readable from the count'}")

    # --- 2. what the count adds to the base posterior --------------------
    prob = Problem(d)
    order = prob.batches["row_id"].to_numpy()
    lab = sol.set_index("row_id").loc[order, "label"].to_numpy()
    img_of_row = tb.set_index("row_id").loc[prob.row_id, "img_id"]
    n_row = occ.reindex(img_of_row).to_numpy()
    n_max = int(un.max())
    tab = occurrence_table(np.clip(un, 0, n_max), uy, n_max)
    log_tab = np.log(tab[:, np.clip(n_row, 0, n_max)].T)

    base = prob.log_post.argmax(1)
    with_cnt = (prob.log_post + log_tab).argmax(1)
    print(f"\n2. ACCURACY on all {len(lab):,} test rows")
    print(f"   base posterior alone        : {(base == lab).mean():.4f}")
    print(f"   base posterior x count      : {(with_cnt == lab).mean():.4f}"
          f"   ({(with_cnt == lab).mean() - (base == lab).mean():+.4f})")

    # --- 3. what it does to the metric -----------------------------------
    inf = prob.inference
    ratio = (prob.log_theta - prob.log_train_prior)[None, :, :]
    r = logsumexp(prob.log_post[:, None, :] + ratio, axis=2)
    g = prob.log_p_theta[None, :] + np.add.reduceat(r, prob.starts, axis=0)
    pth = np.exp(g - logsumexp(g, axis=1, keepdims=True))
    rb = np.repeat(np.arange(len(prob.starts)),
                   np.diff(np.append(prob.starts, prob.n)))
    loglab = logsumexp(np.log(pth)[rb][:, :, None]
                       + (prob.log_post[:, None, :] + ratio - r[:, :, None]),
                       axis=1)
    loglab -= logsumexp(loglab, axis=1, keepdims=True)

    s = sol[sol.Usage == args.usage].drop(columns=["Usage"])
    print(f"\n3. SCORE on the {args.usage} split (lower is better; "
          f"below zero beats the reference)")
    for name, extra in (("intended solution", None),
                        ("intended + occurrence count", log_tab)):
        lp = loglab if extra is None else loglab + extra
        lp = lp - logsumexp(lp, axis=1, keepdims=True)
        pr = np.exp(lp)
        pred = pr.argmax(1)
        tie = 1.0 - pr[np.arange(len(pred)), pred]
        sub = pd.DataFrame({"row_id": prob.row_id, "pred": pred,
                            "confidence": rank_confidence(inf.epistemic, tie)})
        sb = sub[sub.row_id.isin(s.row_id)]
        val = score(s.copy(), sb.copy(), "row_id")
        print(f"   {name:<30}{val:+.4f}   accuracy {(pred == lab).mean():.4f}"
              f"{'   <-- BEATS THE REFERENCE' if val < 0 else ''}")

    print(f"\n   reference predictor accuracy  "
          f"{(sol['pred_ref'].to_numpy() == sol['label'].to_numpy()).mean():.4f}")

    # --- 4. the residual channel ------------------------------------------
    # Balancing the pool only removes the *marginal* 1/n_y term. Which priors
    # an image was drawn under still carries information: an img_id that keeps
    # turning up in batches enriched on classes {i, i+1} is more likely to be
    # class i or i+1. Evidence for class y is sum_j log theta^(c_j)_y over the
    # batches j the image appears in. This uses the TRUE theta_*, so it is an
    # upper bound -- a competitor would have to estimate it per batch, which is
    # hopeless at m = 1 and easy at m = 100.
    meta = pd.read_csv(d / "batch_meta.csv")
    theta = pd.read_csv(d / "test_priors.csv")[[f"p{i}" for i in range(Y)]].to_numpy()
    tstar = dict(zip(meta["id_test"], meta["theta_star_index"]))
    occ_theta = tb.assign(c=tb["id_test"].map(tstar))
    ev = np.zeros((len(occ), Y))
    idx = pd.Index(occ.index)
    np.add.at(ev, idx.get_indexer(occ_theta["img_id"]),
              np.log(theta)[occ_theta["c"].to_numpy()])
    ev_row = ev[idx.get_indexer(img_of_row)]

    print("\n4. RESIDUAL CHANNEL -- which priors an image was drawn under")
    print("   (upper bound: uses the true theta_*, which a competitor must estimate)")
    for name, extra in (("base posterior x prior-evidence", ev_row),
                        ("base x count x prior-evidence", ev_row + log_tab)):
        acc = ((prob.log_post + extra).argmax(1) == lab).mean()
        print(f"   {name:<34}accuracy {acc:.4f}   "
              f"({acc - (base == lab).mean():+.4f})")
    lp = loglab + ev_row + log_tab
    lp -= logsumexp(lp, axis=1, keepdims=True)
    pr = np.exp(lp); pred = pr.argmax(1)
    tie = 1.0 - pr[np.arange(len(pred)), pred]
    sub = pd.DataFrame({"row_id": prob.row_id, "pred": pred,
                        "confidence": rank_confidence(inf.epistemic, tie)})
    val = score(s.copy(), sub[sub.row_id.isin(s.row_id)].copy(), "row_id")
    print(f"   intended + count + prior-evidence  {val:+.4f}   "
          f"accuracy {(pred == lab).mean():.4f}"
          f"{'   <-- BEATS THE REFERENCE' if val < 0 else ''}")


if __name__ == "__main__":
    main()
