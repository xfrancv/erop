#!/usr/bin/env python3
"""Train and calibrate the base predictor students are given (C3.1).

The base model estimates ``p_tr(y | x)``; the whole method re-weights it by
``theta_y / p_tr(y)``, which is only valid if the network was actually fit under
the prior ``p_tr`` this script records. Two consequences drive the design:

* **No class reweighting during training** -- no class-balanced sampler, no
  class-weighted loss. Asserted, not merely intended (S6.1).
* **Calibration is not optional.** Label-shift correction inherits every
  calibration error of the base model directly, so BCTS (per-class temperature,
  ``softmax(z_k / T_k + b_k)``) is fit by LBFGS on the **calibration** split and
  NLL/ECE are reported before and after.

The calibration split and the split released to students as ``dev.csv`` are
**disjoint** (C9.4): calibrating and then releasing the same data would make
students' offline score estimates optimistic by a method-dependent amount.

Outputs in ``out_dir``:

``model.pt``      weights, the calibration map, ``theta_tr``, normalisation
``log_post.npz``  calibrated ``log p_tr(y | x)`` for the calibration, student
                  development and evaluation splits, plus labels and the eval
                  pool assignment -- so no later script needs torch
``report.txt``    splits, calibration numbers, per-class error
``learning_curves.png``

**Hard variant** (``--hard-data``, ``tasks/hard_variant.md``). The model is
the reference predictor's network. It is trained on the *released* training
images of ``hard_make_data.py`` -- rotated and noised, with the location labels
ignored -- split by class into a weight-fitting part and a validation part
(``--cal-fraction`` of the training data) that selects the epoch and fits BCTS.
Both parts have the pooled class frequency, so the calibrated posterior is the
posterior under it. ``log_post.npz`` then holds the calibrated posterior of
every released development and test *row*, since each row is its own
transformed array; no ``images.npz`` is written.

Run with::

    python train_base_model.py out/model
    python train_base_model.py out/hard/model --hard-data out/hard/data
    python train_base_model.py out/model --epochs 30 --device cuda
    python train_base_model.py out/smoke --epochs 2 --max-fit 4000   # smoke only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from chal import data as chdata
from chal.calibration import (
    calibration_summary,
    fit_bcts,
    fit_temperature,
    log_softmax_np,
)
from chal.splits import USAGES, make_splits

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


def make_model(num_classes: int, in_channels: int = 1) -> nn.Module:
    """ResNet-18 with the small-input stem (3x3 conv, stride 1, no max-pool)."""
    from torchvision.models import resnet18
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                        padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def to_tensor(x: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """uint8 (N, H, W) -> normalised float32 (N, 1, H, W)."""
    x = x.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x[:, None, :, :].copy())


@torch.no_grad()
def collect_logits(model: nn.Module, X: torch.Tensor, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def collect_logits_np(model: nn.Module, X: np.ndarray, mean: float, std: float,
                      device: torch.device, chunk: int = 16384) -> np.ndarray:
    """``collect_logits`` on uint8 images, normalised a chunk at a time.

    The hard variant scores ~136 000 released rows; as one float32 tensor they
    would take half a gigabyte for nothing.
    """
    return np.concatenate([
        collect_logits(model, to_tensor(X[i:i + chunk], mean, std), device)
        for i in range(0, len(X), chunk)])


def per_class_error(y: np.ndarray, pred: np.ndarray, Y: int) -> np.ndarray:
    err = np.full(Y, np.nan)
    for c in range(Y):
        mask = y == c
        if mask.any():
            err[c] = float((pred[mask] != c).mean())
    return err


def make_curves_figure(history: dict, best_epoch: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["fit_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, label, tr, va in (
            (axes[0], "loss", history["fit_loss"], history["cal_loss"]),
            (axes[1], "classification error", history["fit_err"], history["cal_err"])):
        ax.plot(epochs, tr, lw=1.8, color="C0", marker="o", ms=3,
                label="fit part (subsample)")
        ax.plot(epochs, va, lw=1.8, color="C1", marker="s", ms=3,
                label="calibration part")
        ax.axvline(best_epoch, color="0.4", ls="--", lw=1,
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Base predictor: training curves")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="directory receiving all outputs")
    p.add_argument("--data-root", type=Path, default=chdata.DATA_ROOT)
    p.add_argument("--hard-data", type=Path, default=None,
                   help="output directory of hard_make_data.py: train the hard "
                        "variant's reference network on its released training "
                        "images and score its development and test rows")
    p.add_argument("--cal-fraction", type=float, default=0.10,
                   help="portion of development used for model selection and "
                        "BCTS (default 0.10, C3.1); with --hard-data, the "
                        "portion of the training data")
    p.add_argument("--dev-fraction", type=float, default=0.10,
                   help="portion of development released to students as "
                        "dev.csv (default 0.10, C3.1)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--calibration", choices=("bcts", "temperature", "none"),
                   default="bcts",
                   help="post-hoc calibration on the calibration split: 'bcts' "
                        "(default, S6.1), 'temperature' (scalar T, the Guo et "
                        "al. ablation) or 'none' (raw softmax)")
    p.add_argument("--curve-sample", type=int, default=10000,
                   help="fit-part examples scored each epoch for the learning "
                        "curve (default 10000); model selection uses the "
                        "calibration split, so this is diagnostic only")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads (0 = leave torch's default)")
    p.add_argument("--max-fit", type=int, default=0,
                   help="cap the fit part at this many examples (0 = no cap). "
                        "For smoke runs; a capped run is marked in report.txt "
                        "and must not be used for the competition.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data and splits (C3.1) -------------------------------------------
    Y = chdata.NUM_CLASSES
    hard = None
    if args.hard_data is not None:
        hard = np.load(Path(args.hard_data) / "hard_data.npz")
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            hard["train_images"], hard["train_label"],
            test_size=args.cal_fraction, stratify=hard["train_label"],
            random_state=args.seed)
        # name -> (images, labels) of everything scored after training
        targets = {"dev_rows": (hard["dev_images"], hard["dev_label"]),
                   "test_rows": (hard["test_images"], hard["test_label"])}
    else:
        ds = chdata.load(args.data_root)
        sp = make_splits(ds, cal_fraction=args.cal_fraction,
                         dev_fraction=args.dev_fraction, seed=args.seed)
        X_fit, y_fit, X_cal, y_cal = sp.X_fit, sp.y_fit, sp.X_cal, sp.y_cal
        targets = {"dev": (sp.X_dev, sp.y_dev), "eval": (sp.X_eval, sp.y_eval)}
    capped = bool(args.max_fit and args.max_fit < len(y_fit))
    if capped:
        # Stratified cap, so theta_tr is unchanged by the subsampling.
        rng = np.random.default_rng(args.seed)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y_fit == c))[
                :max(1, round(args.max_fit * (y_fit == c).mean()))]
            for c in range(Y)])
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    xf = X_fit.astype(np.float32) / 255.0
    norm_mean = float(xf.mean())
    norm_std = float(xf.std()) + 1e-7
    del xf

    Xt_fit = to_tensor(X_fit, norm_mean, norm_std)
    Xt_cal = to_tensor(X_cal, norm_mean, norm_std)
    yt_fit = torch.from_numpy(y_fit)
    yt_cal = torch.from_numpy(y_cal)

    train_prior = np.bincount(y_fit, minlength=Y).astype(float)
    train_prior /= train_prior.sum()
    assert np.all(train_prior > 0), (
        "some class is absent from the fit part; theta_y / p_tr(y) would "
        "divide by zero")

    # --- training ----------------------------------------------------------
    model = make_model(Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # S6.1: the re-weighting theta_y / p_tr(y) is only valid if the network was
    # fit under p_tr itself, so nothing here may reweight classes. Asserted
    # rather than assumed: plain unweighted CE and a uniform shuffle.
    assert criterion.weight is None, "class-weighted loss breaks the S6.1 contract"
    assert criterion.reduction == "mean"

    @torch.no_grad()
    def evaluate(X, y):
        model.eval()
        losses, preds = 0.0, []
        for i in range(0, len(X), 512):
            xb, yb = X[i:i + 512].to(device), y[i:i + 512].to(device)
            logits = model(xb)
            losses += F.cross_entropy(logits, yb, reduction="sum").item()
            preds.append(logits.argmax(dim=1).cpu())
        preds = torch.cat(preds)
        return losses / len(X), float((preds != y).float().mean())

    history = {k: [] for k in ("fit_loss", "fit_err", "cal_loss", "cal_err")}
    best_err, best_epoch, best_state = float("inf"), 0, None
    n_fit = len(Xt_fit)

    # The fit-part curve is diagnostic only -- model selection uses the
    # calibration split -- so it is measured on a fixed subsample. Scoring all
    # 132k fit images every epoch would add about a quarter to the wall time
    # for a line that a 10k sample draws just as well.
    curve_n = min(args.curve_sample, n_fit)
    curve_idx = torch.from_numpy(
        np.random.default_rng(args.seed).choice(n_fit, curve_n, replace=False))
    Xt_curve, yt_curve = Xt_fit[curve_idx], yt_fit[curve_idx]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_fit)
        batches = range(0, n_fit, args.batch_size)
        if tqdm is not None:
            batches = tqdm(batches, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for i in batches:
            idx = perm[i:i + args.batch_size]
            xb, yb = Xt_fit[idx].to(device), yt_fit[idx].to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        tr_loss, tr_err = evaluate(Xt_curve, yt_curve)
        ca_loss, ca_err = evaluate(Xt_cal, yt_cal)
        for k, v in zip(history, (tr_loss, tr_err, ca_loss, ca_err)):
            history[k].append(v)
        print(f"epoch {epoch:3d}/{args.epochs}  fit loss {tr_loss:.4f} "
              f"err {tr_err:.4f}   cal loss {ca_loss:.4f} err {ca_err:.4f}",
              flush=True)
        if ca_err < best_err:
            best_err, best_epoch = ca_err, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # --- calibration on the calibration split (S6.1) -----------------------
    cal_logits = collect_logits(model, Xt_cal, device)
    # Every mode ends up as the same affine map on the logits, so there is one
    # code path downstream: temperature is (Y,) throughout.
    temperature, bias = np.ones(Y), np.zeros(Y)
    if args.calibration == "bcts":
        temperature, bias = fit_bcts(torch.from_numpy(cal_logits).float(), yt_cal)
    elif args.calibration == "temperature":
        temperature = np.full(
            Y, fit_temperature(torch.from_numpy(cal_logits).float(), yt_cal))
    assert np.all(temperature > 0), "non-positive temperature from calibration"
    scale = 1.0 / temperature

    def calibrate(logits: np.ndarray) -> np.ndarray:
        return log_softmax_np(logits * scale + bias)

    calib = calibration_summary(cal_logits, y_cal, scale, bias)

    # --- calibrated posteriors for every split a later script needs --------
    lp_cal = calibrate(cal_logits)
    lps = {name: calibrate(collect_logits_np(model, X, norm_mean, norm_std,
                                             device))
           for name, (X, _y) in targets.items()}
    for name, lp in (("cal", lp_cal), *lps.items()):
        assert np.all(np.isfinite(lp)), f"non-finite log-posterior on {name}"

    common = dict(
        log_post_cal=lp_cal.astype(np.float32), y_cal=y_cal,
        train_prior=train_prior,
        class_names=np.array(chdata.CLASS_NAMES, dtype=object),
        seed=args.seed, capped=capped,
    )
    if hard is None:
        np.savez_compressed(
            out_dir / "log_post.npz", **common,
            log_post_dev=lps["dev"].astype(np.float32), y_dev=sp.y_dev,
            log_post_eval=lps["eval"].astype(np.float32), y_eval=sp.y_eval,
            pool_of_eval=sp.pool_of_eval,
            usages=np.array(USAGES, dtype=object),
        )
        # The raw images too: prepare_kaggle_data.py needs the pixels, and
        # reading them back from here keeps it from having to re-derive the
        # splits.
        np.savez_compressed(out_dir / "images.npz",
                            X_dev=sp.X_dev, X_eval=sp.X_eval)
    else:
        np.savez_compressed(
            out_dir / "log_post.npz", **common,
            **{f"log_post_{name}": lp.astype(np.float32)
               for name, lp in lps.items()},
            # hard_package.py refuses posteriors from a different dataset.
            hard_data_id=str(hard["run_id"]),
        )

    torch.save({
        "model_state": best_state, "num_classes": Y,
        "calibration": args.calibration, "temperature": temperature,
        "calib_scale": scale, "calib_bias": bias,
        "calibration_metrics": calib, "train_prior": train_prior,
        "hard_data": str(args.hard_data) if hard is not None else None,
        "norm_mean": norm_mean, "norm_std": norm_std,
        "best_epoch": best_epoch, "seed": args.seed,
        "cal_fraction": args.cal_fraction, "dev_fraction": args.dev_fraction,
        "max_fit": args.max_fit,
    }, out_dir / "model.pt")

    cal_pred = lp_cal.argmax(axis=1)
    cls_err = per_class_error(y_cal, cal_pred, Y)
    if hard is None:
        pool_counts = np.bincount(sp.pool_of_eval, minlength=len(USAGES))
        split_lines = [
            f"  development -> calibration : {len(y_cal):,}",
            f"  development -> student dev : {len(sp.y_dev):,}",
            f"  evaluation                 : {len(sp.y_eval):,}  "
            f"(official val + test)",
        ] + [f"    pool {u:<8}           : {int(c):,}"
             for u, c in zip(USAGES, pool_counts)]
        err_names = {"dev": "student dev split", "eval": "evaluation split "}
    else:
        split_lines = [
            f"  {'training -> validation':<26} : {len(y_cal):,}   (epoch "
            f"selection and BCTS)",
            f"  {'scored: development rows':<26} : "
            f"{len(targets['dev_rows'][1]):,}",
            f"  {'scored: test rows':<26} : {len(targets['test_rows'][1]):,}",
            f"  (hard variant, data run {hard['run_id']}; images rotated and "
            f"noised as released)",
        ]
        err_names = {"dev_rows": "development rows ",
                     "test_rows": "test rows        "}

    lines = [
        "Base predictor: training and calibration (challenge_polish.md C3.1)"
        + ("\nHARD VARIANT: the reference predictor's network "
           "(tasks/hard_variant.md)" if hard is not None else ""),
        "=" * 78,
        f"timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"command     : {' '.join(sys.argv)}",
        f"device      : {args.device}   seed: {args.seed}",
        f"epochs      : {args.epochs}  batch {args.batch_size}  lr {args.lr:g}",
        f"curve sample: {curve_n:,} fit examples scored per epoch (diagnostic)",
        "-" * 78,
        "splits (C3.1)",
        f"  {'training -> fit' if hard is not None else 'development -> fit':<26}"
        f" : {len(y_fit):,}"
        + ("   (CAPPED by --max-fit; NOT for the competition)" if capped else ""),
    ] + split_lines + [
        f"best epoch  : {best_epoch}  (calibration-split error {best_err:.4f})",
        "-" * 78,
        f"calibration : {args.calibration}"
        + ("   softmax(z_k / T_k + b_k)" if args.calibration == "bcts" else ""),
        f"  NLL : {calib['nll_before']:.4f} -> {calib['nll_after']:.4f}",
        f"  ECE : {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}"
        f"   ({calib['n_ece_bins']} equal-mass bins)",
        "  (label-shift correction is highly sensitive to calibration -- S6.1)",
    ] + ([
        "  !! ECE got worse after calibration; check the calibration split size."
    ] if calib["ece_after"] > calib["ece_before"] + 1e-3 else []) + [
        "-" * 78,
        "theta_tr (fit-part class frequency):",
        "  " + "  ".join(f"{v:.4f}" for v in train_prior),
        "",
        f"classification error, calibration split : "
        f"{float((cal_pred != y_cal).mean()):.4f}",
    ] + [
        f"classification error, {err_names[name]} : "
        f"{float((lps[name].argmax(axis=1) != y).mean()):.4f}"
        for name, (_X, y) in targets.items()
    ] + [
        "",
        "per-class error, calibration split:",
    ] + [
        f"  {c}  {chdata.CLASS_NAMES[c][:44]:<44} {cls_err[c]:.4f}"
        for c in range(Y)
    ] + [""]

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)
    make_curves_figure(history, best_epoch, out_dir)
    print(f"outputs in {out_dir}/: model.pt, log_post.npz, "
          + ("images.npz, " if hard is None else "")
          + "report.txt, learning_curves.png")


if __name__ == "__main__":
    main()
