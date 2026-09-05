"""Train, calibrate, and emit everything ``rejopt_eval.py`` needs (README S6.1).

The base model estimates ``p_tr(y | x)``; the whole method then re-weights it by
``theta_y / p_tr(y)``, which is only valid if the network was actually fit under
the prior ``p_tr`` this script records. Two consequences drive the design:

* **No class reweighting during training.** No class-balanced sampler, no
  class-weighted loss -- asserted, not merely intended (S6.1).
* **Calibration is not optional.** Label-shift correction inherits every
  calibration error of the base model directly, so the script fits
  bias-corrected temperature scaling with a per-class temperature (BCTS,
  ``softmax(z_k / T_k + b_k)``) on the validation split by LBFGS and reports
  validation NLL and ECE (15 equal-mass bins) *before and after*. Those two
  numbers belong in the paper.

The run also emits the prior set ``Theta`` (S7), which is an output of training
rather than a hand-written config: ``theta_1`` is the fit-part class frequency
and ``theta_3`` is built from the per-class validation error rates. S6.1's order
of operations, end to end::

    train -> calibrate -> per-class validation errors -> emit priors file -> evaluate

Outputs in ``out_dir``:

``model.pt``          weights, calibration map, ``theta_tr``, normalisation
``eval_log_post.npz`` calibrated ``log p_tr(y | x)`` on the whole evaluation
                      split, plus its labels -- so ``rejopt_eval.py`` needs no
                      torch and no forward pass at all
``priors.txt``        the prior set Theta (S7)
``report.txt``        splits, calibration numbers, confusion, per-class error
``learning_curves.png``

Run with::

    python base_predictor_training.py fashion_mnist runs/fashion_mnist
    python base_predictor_training.py bloodmnist runs/bloodmnist --epochs 30 --device cuda
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

from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS
from exact.calibration import (
    calibration_summary,
    fit_bcts,
    fit_temperature,
    log_softmax_np,
)
from exact.priors import build_prior_set, write_prior_set
from exact.protocol import resolve_grid
from exact.splits import make_splits
from exact.textutil import (
    confusion_matrix,
    format_calibration,
    format_confusion,
    format_per_class_error,
    format_prior_table,
    per_class_error,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

ARCH_DEFAULTS = {
    "fashion_mnist": "lenet",
    "cifar10": "resnet18-32",
    "cifar100": "resnet18-32",
    "dermamnist": "resnet18-28",
    "bloodmnist": "resnet18-28",
    "tissuemnist": "resnet18-28",
    "organamnist": "resnet18-28",
    "organsmnist": "resnet18-28",
}
ARCH_CHOICES = ("lenet", "resnet18-32", "resnet18-28")


class LeNet(nn.Module):
    """LeNet-5-style CNN for 28x28 inputs (default for Fashion-MNIST)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(arch: str, in_channels: int, num_classes: int) -> nn.Module:
    """Both ResNet variants share the small-input stem (3x3 conv, stride 1, no
    max-pool) and train from scratch; the two names exist only because the
    expected input size differs per dataset (32x32 vs 28x28)."""
    if arch == "lenet":
        return LeNet(in_channels, num_classes)
    if arch in ("resnet18-32", "resnet18-28"):
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                            padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    raise ValueError(f"unknown architecture: {arch}")


def to_tensor(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W[, C]) -> normalised float32 (N, C, H, W)."""
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[:, :, :, None]
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


@torch.no_grad()
def collect_logits(model: nn.Module, X: torch.Tensor, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def make_curves_figure(history: dict, best_epoch: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["fit_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = (
        (axes[0], "loss", history["fit_loss"], history["val_loss"]),
        (axes[1], "classification error", history["fit_err"], history["val_err"]),
    )
    for ax, label, tr, va in panels:
        ax.plot(epochs, tr, lw=1.8, color="C0", marker="o", ms=3, label="fit part")
        ax.plot(epochs, va, lw=1.8, color="C1", marker="s", ms=3, label="validation")
        ax.axvline(best_epoch, color="0.4", ls="--", lw=1,
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Training and validation curves")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS.keys()))
    p.add_argument("out_dir", type=str, help="directory receiving all outputs")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="portion of the development split held out for model "
                        "selection, calibration and the per-class error rates "
                        "of theta_3 (default 0.2, README S6.1)")
    p.add_argument("--arch", choices=ARCH_CHOICES, default=None,
                   help="network architecture; defaults per dataset: "
                        + ", ".join(f"{k}={v}" for k, v in ARCH_DEFAULTS.items()))
    p.add_argument("--device", default="auto",
                   choices=("auto", "cpu", "cuda"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--calibration", choices=("bcts", "temperature", "none"),
                   default="bcts",
                   help="post-hoc calibration on the validation split: "
                        "'bcts' (default, README S6.1: bias-corrected "
                        "temperature scaling with a per-class temperature, "
                        "softmax(z_k / T_k + b_k)), 'temperature' (one scalar "
                        "T, the Guo et al. ablation), or 'none' (raw softmax "
                        "-- the miscalibration ablation)")
    p.add_argument("--tau", type=float, default=0.2,
                   help="spike mass per hard class in theta_3 (S7, default 0.2)")
    p.add_argument("--max-fit", type=int, default=0,
                   help="cap the fit part at this many examples (0 = no cap). "
                        "For smoke runs on a CPU-only machine; a capped run is "
                        "marked as such in report.txt and must not be a run of "
                        "record.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data and splits (S6.1) -------------------------------------------
    ds = load_dataset(args.dataset)
    Y = ds.num_classes
    arch = args.arch or ARCH_DEFAULTS[args.dataset]
    sp = make_splits(ds, val_fraction=args.val_fraction, seed=args.seed)

    X_fit, y_fit = sp.X_fit, sp.y_fit
    if args.max_fit and args.max_fit < len(y_fit):
        # Stratified cap, so theta_tr is unchanged by the subsampling.
        rng = np.random.default_rng(args.seed)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(y_fit == c))[
                :max(1, round(args.max_fit * (y_fit == c).mean()))]
            for c in range(Y)])
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    x = X_fit.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[:, :, :, None]
    norm_mean = x.mean(axis=(0, 1, 2))
    norm_std = x.std(axis=(0, 1, 2)) + 1e-7
    in_channels = x.shape[-1]
    del x

    Xt_fit = to_tensor(X_fit, norm_mean, norm_std)
    Xt_val = to_tensor(sp.X_val, norm_mean, norm_std)
    Xt_eval = to_tensor(sp.X_eval, norm_mean, norm_std)
    yt_fit = torch.from_numpy(y_fit)
    yt_val = torch.from_numpy(sp.y_val)

    train_prior = np.bincount(y_fit, minlength=Y).astype(float)
    train_prior /= train_prior.sum()
    assert np.all(train_prior > 0), (
        "some class is absent from the fit part; theta_y / p_tr(y) would divide "
        "by zero")

    # --- training ----------------------------------------------------------
    model = make_model(arch, in_channels, Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # S6.1: the re-weighting theta_y / p_tr(y) is only valid if the network was
    # fit under p_tr itself, so nothing here may reweight classes. Asserted
    # rather than assumed: a plain unweighted CE loss, and a uniform shuffle
    # (torch.randperm) rather than a class-balanced sampler.
    assert criterion.weight is None, "class-weighted loss breaks the S6.1 contract"
    assert criterion.reduction == "mean"

    @torch.no_grad()
    def evaluate(X, y, temperature=1.0):
        model.eval()
        losses, preds = 0.0, []
        for i in range(0, len(X), 512):
            xb = X[i:i + 512].to(device)
            yb = y[i:i + 512].to(device)
            logits = model(xb) / temperature
            losses += F.cross_entropy(logits, yb, reduction="sum").item()
            preds.append(logits.argmax(dim=1).cpu())
        preds = torch.cat(preds)
        return losses / len(X), float((preds != y).float().mean()), preds.numpy()

    history = {k: [] for k in ("fit_loss", "fit_err", "val_loss", "val_err")}
    best_val_err, best_epoch, best_state = float("inf"), 0, None
    n_fit = len(Xt_fit)

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

        tr_loss, tr_err, _ = evaluate(Xt_fit, yt_fit)
        va_loss, va_err, _ = evaluate(Xt_val, yt_val)
        for k, v in zip(history, (tr_loss, tr_err, va_loss, va_err)):
            history[k].append(v)
        print(f"epoch {epoch:3d}/{args.epochs}  fit loss {tr_loss:.4f} "
              f"err {tr_err:.4f}   val loss {va_loss:.4f} err {va_err:.4f}",
              flush=True)
        if va_err < best_val_err:
            best_val_err, best_epoch = va_err, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # --- calibration on the validation split (S6.1) ------------------------
    val_logits = collect_logits(model, Xt_val, device)
    # Every mode ends up as the same affine map on the logits, so there is one
    # code path downstream: temperature is (Y,) throughout, degenerate (all
    # entries equal) for scalar TS and all-ones for the uncalibrated ablation.
    temperature = np.ones(Y)
    bias = np.zeros(Y)
    if args.calibration == "bcts":
        temperature, bias = fit_bcts(
            torch.from_numpy(val_logits).float(), yt_val)
    elif args.calibration == "temperature":
        temperature = np.full(
            Y, fit_temperature(torch.from_numpy(val_logits).float(), yt_val))
    assert np.all(temperature > 0), "non-positive temperature from calibration"
    scale = 1.0 / temperature

    def calibrate(logits: np.ndarray) -> np.ndarray:
        return log_softmax_np(logits * scale + bias)

    # NLL and ECE before and after -- the two numbers S6.1 puts in the paper.
    calib = calibration_summary(val_logits, sp.y_val, scale, bias)

    # --- per-class validation error -> Theta (S7) --------------------------
    val_pred = calibrate(val_logits).argmax(axis=1)
    val_err_per_class = per_class_error(sp.y_val, val_pred, Y)
    prior_set = build_prior_set(train_prior, val_err_per_class, tau=args.tau)
    write_prior_set(prior_set, out_dir / "priors.txt")

    # --- the evaluation split's calibrated posterior ------------------------
    # Saved once here so rejopt_eval.py never needs torch: every trial is a row
    # gather out of this matrix.
    eval_log_post = calibrate(collect_logits(model, Xt_eval, device))
    assert np.all(np.isfinite(eval_log_post))
    np.savez_compressed(
        out_dir / "eval_log_post.npz",
        log_post=eval_log_post.astype(np.float32),
        y=sp.y_eval.astype(np.int64),
        train_prior=train_prior,
        class_names=np.array(ds.spec.class_names, dtype=object),
        dataset=args.dataset,
        eval_desc=sp.eval_desc,
    )

    eval_pred = eval_log_post.argmax(axis=1)
    eval_err = float((eval_pred != sp.y_eval).mean())
    grid, grid_note = resolve_grid(len(sp.y_eval))

    bundle = {
        "model_state": best_state, "arch": arch, "dataset": args.dataset,
        "num_classes": Y, "in_channels": in_channels,
        "image_shape": tuple(ds.image_shape),
        "calibration": args.calibration, "temperature": temperature,
        "calib_scale": scale, "calib_bias": bias,  # log_softmax(z*scale + bias)
        "calibration_metrics": calib,
        "train_prior": train_prior,
        "val_error_per_class": val_err_per_class,
        "class_names": list(ds.spec.class_names),
        "norm_mean": norm_mean, "norm_std": norm_std,
        "best_epoch": best_epoch, "seed": args.seed,
        "val_fraction": args.val_fraction, "max_fit": args.max_fit,
    }
    torch.save(bundle, out_dir / "model.pt")

    lines = [
        "Base predictor: training, calibration, and the prior set Theta",
        "=" * 78,
        f"timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"command     : {' '.join(sys.argv)}",
        f"dataset     : {ds.spec.display_name} ({args.dataset})",
        f"architecture: {arch}   device: {args.device}   seed: {args.seed}",
        f"epochs      : {args.epochs}  batch {args.batch_size}  lr {args.lr:g}",
        "-" * 78,
        "splits (README S6.1)",
        f"  development -> fit        : {len(y_fit):,}"
        + (f"   (CAPPED by --max-fit {args.max_fit}; NOT a run of record)"
           if args.max_fit and args.max_fit < len(sp.y_fit) else ""),
        f"  development -> validation : {len(sp.y_val):,}",
        f"  evaluation                : {len(sp.y_eval):,}  ({sp.eval_desc})",
        f"  adaptation grid           : {list(grid)}",
    ] + ([f"  !! {grid_note}"] if grid_note else []) + [
        f"best epoch  : {best_epoch}  (validation error {best_val_err:.4f})",
        "-" * 78,
        f"calibration : {args.calibration}"
        + ("   softmax(z_k / T_k + b_k)" if args.calibration == "bcts" else ""),
    ] + ([format_calibration(temperature, bias, list(ds.spec.class_names))]
         if args.calibration != "none" else []) + [
        f"  validation NLL : {calib['nll_before']:.4f} -> {calib['nll_after']:.4f}",
        f"  validation ECE : {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}"
        f"   ({calib['n_ece_bins']} equal-mass bins)",
        "  (label-shift correction is highly sensitive to calibration -- S6.1)",
    ] + ([
        "  !! ECE got worse after calibration; check the validation split size."
    ] if calib["ece_after"] > calib["ece_before"] + 1e-3 else []) + [
        "-" * 78,
        f"theta_tr    : {np.array2string(train_prior, precision=4, threshold=20)}",
        "",
        f"prior set Theta (S7): C = {prior_set.C}",
        format_prior_table(prior_set, list(ds.spec.class_names)),
    ] + ([""] + [f"  {d}" for d in prior_set.dropped] if prior_set.dropped else []) + [
        "-" * 78,
        f"classification error, validation : {float((val_pred != sp.y_val).mean()):.4f}",
        f"classification error, evaluation : {eval_err:.4f}",
        "",
        "confusion matrix, evaluation split:",
        format_confusion(confusion_matrix(sp.y_eval, eval_pred, Y),
                         list(ds.spec.class_names)),
        "",
        "per-class error, validation split (this is what defines theta_3):",
        format_per_class_error(val_err_per_class, list(ds.spec.class_names),
                               np.bincount(sp.y_val, minlength=Y)),
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)

    make_curves_figure(history, best_epoch, out_dir)
    print(f"\noutputs in {out_dir}/: model.pt, eval_log_post.npz, priors.txt, "
          f"report.txt, learning_curves.png")


if __name__ == "__main__":
    main()
