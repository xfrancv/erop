#!/usr/bin/env python3
"""Write the hard variant's Kaggle upload (tasks/hard_variant.md).

Reads ``hard_make_data.py``'s arrays and the reference network's calibrated
posteriors from ``train_base_model.py --hard-data``, computes the reference
predictor, and writes three groups of files.

**Published** (upload to Kaggle; ``row k`` of a CSV is ``image k`` of its
``.npy``):

``train.csv``, ``train_images.npy``          labeled training images and their
                                             location
``dev_test.csv``                             one row per development batch
``dev_test_batches.csv``, ``dev_images.npy`` one row per development image
``dev_solution.csv``                         labels and the reference predictor
``dev_sample_submission.csv``
``test.csv``                                 one row per test batch
``test_batches.csv``, ``test_images.npy``    one row per test image
``sample_submission.csv``

The sample submissions predict the most frequent training class with a
constant confidence. They show the format and nothing else: a baseline built
on the organisers' network would hand competitors that network's output on
every test image.

**Given to Kaggle, hidden from students:** ``solution.csv``.

**Never uploaded** -- ``organiser/``:

``organiser/test/``, ``organiser/dev/``  each a directory in the easy variant's
    layout (``predictions.csv``, ``test_batches.csv``, ``test_priors.csv``,
    ``train_prior.csv``, ``batch_meta.csv``, ``solution.csv``) holding the
    reference network's posteriors and the 9 location priors, so
    ``baseline_solutions.py``, ``evaluate.py``, ``compare_baselines.py`` and
    ``run_baselines.sh`` run on the hard variant unchanged
``organiser/row_source.csv``  which original image and rotation every released
    development and test row came from
``manifest.json``

The location priors in ``organiser/*/test_priors.csv`` are recomputed from the
published ``train.csv`` and checked against the priors file: they are what a
competitor can estimate, exactly.

    python hard_package.py out/hard/data out/hard/model out/hard/kaggle
    python baseline_solutions.py out/hard/kaggle/organiser/test out/hard/submissions
    ./run_baselines.sh out/hard/kaggle/organiser/test out/hard/submissions
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from chal import data as chdata
from chal.generate import reference_and_base
from chal.priors import read_prior_set


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="output of hard_make_data.py")
    p.add_argument("model_dir", type=Path,
                   help="output of train_base_model.py --hard-data")
    p.add_argument("out_dir", type=Path, help="directory receiving the upload")
    p.add_argument("--priors", type=Path, default=None,
                   help="the 9 location priors (default: data_dir/priors.txt)")
    p.add_argument("--seed", type=int, default=20260919,
                   help="shuffles the order of the training rows")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    Y = chdata.NUM_CLASSES

    hd = np.load(args.data_dir / "hard_data.npz")
    lp = np.load(args.model_dir / "log_post.npz", allow_pickle=True)
    assert "hard_data_id" in lp.files, (
        f"{args.model_dir}/log_post.npz is an easy-variant model; train with "
        f"train_base_model.py --hard-data {args.data_dir}")
    assert str(lp["hard_data_id"]) == str(hd["run_id"]), (
        f"the model was trained on data run {lp['hard_data_id']}, but "
        f"{args.data_dir} holds run {hd['run_id']}; retrain or point at the "
        f"right directory")
    if bool(lp["capped"]):
        print("!! the model was trained with --max-fit: a smoke run, NOT for "
              "the competition")

    train_prior = lp["train_prior"]
    log_train_prior = np.log(train_prior)
    ps = read_prior_set(args.priors or args.data_dir / "priors.txt", train_prior)
    theta = ps.theta
    L = ps.C
    assert np.allclose(theta, hd["priors"], atol=1e-9), \
        "the priors file does not match hard_data.npz"

    # --- the training data, as published -----------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(hd["train_label"]))
    train_label = hd["train_label"][perm]
    train_location = hd["train_location"][perm]
    # A competitor's estimate of the location priors is the class frequency
    # of each location. It must be the prior the batches were drawn from.
    est = np.stack([np.bincount(train_label[train_location == l], minlength=Y)
                    for l in range(L)]).astype(float)
    est /= est.sum(axis=1, keepdims=True)
    assert np.allclose(est, theta, atol=1e-9), \
        "location class frequencies in train.csv differ from the priors file"

    # --- the reference predictor on every released row (C5) ---------------
    def reference(split: str):
        lp_rows = lp[f"log_post_{split}_rows"].astype(np.float64)
        assert len(lp_rows) == len(hd[f"{split}_label"]), \
            f"{split}: posteriors and rows have different lengths"
        pred_ref, _unc, base_pred, base_conf = reference_and_base(
            lp_rows, log_train_prior, theta[hd[f"{split}_location"]])
        return lp_rows, pred_ref, base_pred, base_conf

    dev_lp, dev_ref, dev_base, _ = reference("dev")
    test_lp, test_ref, test_base, _ = reference("test")
    usages = [str(u) for u in hd["usages"]]
    test_usage = np.array(usages, dtype=object)[hd["test_usage"]]

    # --- published ------------------------------------------------------------
    pub = out_dir
    pub.mkdir(parents=True, exist_ok=True)
    np.save(pub / "train_images.npy", hd["train_images"][perm])
    pd.DataFrame({"label": train_label, "location": train_location}).to_csv(
        pub / "train.csv", index=False)

    majority = int(np.bincount(train_label, minlength=Y).argmax())

    def batch_files(prefix: str, split: str):
        rows = pd.DataFrame({"row_id": hd[f"{split}_row_id"],
                             "id_test": hd[f"{split}_id_test"],
                             "slot": hd[f"{split}_slot"]})
        batches = pd.DataFrame({"id_test": hd[f"{split}_batch_id"],
                                "m": hd[f"{split}_batch_m"]})
        sample = pd.DataFrame({"row_id": rows["row_id"],
                               "pred": majority, "confidence": 0.0})
        batches.to_csv(pub / f"{prefix}test.csv", index=False)
        rows.to_csv(pub / f"{prefix}test_batches.csv", index=False)
        np.save(pub / f"{split}_images.npy", hd[f"{split}_images"])
        sample.sort_values("row_id").to_csv(
            pub / f"{prefix}sample_submission.csv", index=False)
        return rows

    dev_rows = batch_files("dev_", "dev")
    test_rows = batch_files("", "test")

    dev_solution = dev_rows.assign(m=hd["dev_m"], label=hd["dev_label"],
                                   pred_ref=dev_ref)
    dev_solution.to_csv(pub / "dev_solution.csv", index=False)

    # --- hidden: Kaggle's solution file --------------------------------------
    solution = test_rows.assign(m=hd["test_m"], label=hd["test_label"],
                                pred_ref=test_ref, Usage=test_usage)
    solution.to_csv(pub / "solution.csv", index=False)

    # --- organiser only -------------------------------------------------------
    org = pub / "organiser"
    for split, rows, sol, lp_rows in (
            ("test", test_rows, solution, test_lp),
            ("dev", dev_rows, dev_solution, dev_lp)):
        d = org / split
        d.mkdir(parents=True, exist_ok=True)
        rows.to_csv(d / "test_batches.csv", index=False)
        sol.to_csv(d / "solution.csv", index=False)
        # Full precision, no float_format: true_plugin must reproduce pred_ref
        # exactly from this file (see prepare_kaggle_data.py).
        post = np.exp(lp_rows)
        pd.DataFrame({"row_id": rows["row_id"],
                      **{f"p{y}": post[:, y] for y in range(Y)}}).to_csv(
            d / "predictions.csv", index=False)
        pd.DataFrame({"id": np.arange(L),
                      **{f"p{y}": est[:, y] for y in range(Y)}}).to_csv(
            d / "test_priors.csv", index=False)
        pd.DataFrame({f"p{y}": [train_prior[y]] for y in range(Y)}).to_csv(
            d / "train_prior.csv", index=False)
        meta = {"id_test": hd[f"{split}_batch_id"], "m": hd[f"{split}_batch_m"]}
        if split == "test":
            meta["usage"] = np.array(usages, dtype=object)[hd["test_batch_usage"]]
        meta["theta_star_index"] = hd[f"{split}_batch_location"]
        meta["dup_fraction"] = hd[f"{split}_batch_dup"]
        pd.DataFrame(meta).to_csv(d / "batch_meta.csv", index=False)

    pd.concat([
        pd.DataFrame({"split": "test", "row_id": hd["test_row_id"],
                      "source": "val+test", "source_index": hd["test_source"],
                      "quarter_turns": hd["test_rot"]}),
        pd.DataFrame({"split": "dev", "row_id": hd["dev_row_id"],
                      "source": "train", "source_index": hd["dev_source"],
                      "quarter_turns": hd["dev_rot"]}),
    ]).to_csv(org / "row_source.csv", index=False)

    def acc(pred, label):
        return float((pred == label).mean())

    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_run_id": str(hd["run_id"]),
        "data_seed": int(hd["seed"]),
        "noise_std": float(hd["noise_std"]),
        "model_capped": bool(lp["capped"]),
        "locations": int(L),
        "location_sizes": [int(v) for v in hd["location_sizes"]],
        "n_train": int(len(train_label)),
        "dev": {"batches": int(len(hd["dev_batch_id"])),
                "rows": int(len(dev_rows))},
        "test": {"batches": int(len(hd["test_batch_id"])),
                 "rows": int(len(test_rows)),
                 "rows_per_usage": {u: int((test_usage == u).sum())
                                    for u in usages}},
        "batches_per_location": [
            int(v) for v in np.bincount(hd["test_batch_location"], minlength=L)],
        "reference_accuracy": {"dev": acc(dev_ref, hd["dev_label"]),
                               "test": acc(test_ref, hd["test_label"])},
        "non_adapted_accuracy": {"dev": acc(dev_base, hd["dev_label"]),
                                 "test": acc(test_base, hd["test_label"])},
        "sample_submission": f"class {majority}, constant confidence",
        "images_published": True,
    }
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(train_label):,} training images in {L} locations; "
          f"{manifest['dev']['batches']:,} development batches "
          f"({manifest['dev']['rows']:,} rows); "
          f"{manifest['test']['batches']:,} test batches "
          f"({manifest['test']['rows']:,} rows)")
    print(f"reference predictor accuracy     : dev "
          f"{manifest['reference_accuracy']['dev']:.4f}   test "
          f"{manifest['reference_accuracy']['test']:.4f}")
    print(f"non-adapted network accuracy     : dev "
          f"{manifest['non_adapted_accuracy']['dev']:.4f}   test "
          f"{manifest['non_adapted_accuracy']['test']:.4f}")
    print("\nupload to Kaggle : train.csv train_images.npy test.csv "
          "test_batches.csv test_images.npy\n                   "
          "sample_submission.csv dev_test.csv dev_test_batches.csv "
          "dev_images.npy\n                   dev_solution.csv "
          "dev_sample_submission.csv")
    print("give Kaggle only : solution.csv")
    print("keep local       : organiser/ manifest.json")
    print(f"\nbaselines        : python baseline_solutions.py {org}/test "
          f"<subs_dir>")


if __name__ == "__main__":
    main()
