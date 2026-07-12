"""Train and evaluate a neural-network base predictor on a real dataset.

Implements ``tasks/baseline_real_life_data.md``: the trained predictor is the
*base model* that the downstream reject-option / test-prior-adaptation scripts
will consume, so besides the best-epoch weights the output bundle carries the
calibration temperature ``T`` and the estimated training prior ``p_tr(y)``.

Split policy (uniform across datasets):

- The original *training* subset is split, class-stratified, into a **fit
  part** (trains the network) and a **model-selection part** (selects the best
  epoch by validation error, then fits the temperature).
- Datasets with train/test splits use the whole test subset for testing;
  datasets with train/val/test splits use **val + test merged** for testing
  only (the official val split is never seen during training or calibration,
  which also leaves a larger pool for the downstream prior-resampling).

Per epoch the script records loss and classification error on both the fit
part and the model-selection part; after training it fits one temperature
``T`` by minimizing NLL on the model-selection part (T does not change the
argmax, so the reported error/confusion matrix are unaffected — it matters
for the downstream label-shift correction).

Run with::

    python run_base_predictor_exp.py fashion_mnist runs/fashion
    python run_base_predictor_exp.py bloodmnist runs/blood --epochs 30 --device cuda

Outputs in the given directory: ``model.pt`` (weights + T + train prior +
normalization), ``report.txt`` (metrics + confusion matrices), and
``learning_curves.png``.
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

from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS

try:  # progress bars are optional, as in the synthetic experiment scripts.
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Architecture keys and the per-dataset defaults from the task assignment.
ARCH_DEFAULTS = {
    "fashion_mnist": "lenet",
    "cifar10": "resnet18-32",
    "dermamnist": "resnet18-28",
    "bloodmnist": "resnet18-28",
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
    """Build the network. Both ResNet variants share the small-input stem
    (3x3 conv, stride 1, no max-pool) and are trained from scratch; the two
    names exist because the assignment fixes the expected input size per
    dataset (32x32 for CIFAR-10, 28x28 for MedMNIST)."""
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
    """uint8 (N,H,W[,C]) -> normalized float32 (N,C,H,W)."""
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:                       # grayscale -> add channel axis
        x = x[:, :, :, None]
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             device: torch.device, batch_size: int = 512):
    """Return (mean CE loss, classification error, predictions)."""
    model.eval()
    losses, preds = [], []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        logits = model(xb)
        losses.append(F.cross_entropy(logits, yb, reduction="sum").item())
        preds.append(logits.argmax(dim=1).cpu())
    preds = torch.cat(preds)
    loss = sum(losses) / len(X)
    error = float((preds != y).float().mean())
    return loss, error, preds.numpy()


@torch.no_grad()
def collect_logits(model: nn.Module, X: torch.Tensor, device: torch.device,
                   batch_size: int = 512) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """One scalar T minimizing NLL of ``softmax(logits / T)`` (Guo et al. 2017).

    Optimized over log T so T stays positive.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, Y: int) -> np.ndarray:
    """(Y, Y) matrix with rows = true class, columns = predicted class."""
    cm = np.zeros((Y, Y), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def format_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    """Confusion matrix as text, rows true / columns predicted."""
    short = [n[:10] for n in class_names]
    width = max(10, max(len(s) for s in short) + 1)
    lines = [" " * width + "".join(f"{s:>{width}}" for s in short)
             + "   (rows = true, cols = predicted)"]
    for i, name in enumerate(short):
        lines.append(f"{name:>{width}}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    return "\n".join(lines)


def make_curves_figure(history: dict, best_epoch: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = (
        (axes[0], "loss", history["train_loss"], history["val_loss"]),
        (axes[1], "classification error", history["train_err"], history["val_err"]),
    )
    for ax, label, tr, va in panels:
        ax.plot(epochs, tr, lw=1.8, color="C0", marker="o", ms=3, label="training")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()),
                        help="Dataset key as used by download_datasets.py.")
    parser.add_argument("out_dir", type=str,
                        help="Directory receiving all output files.")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Portion of the training subset held out for "
                             "model selection (default 0.2).")
    parser.add_argument("--arch", choices=ARCH_CHOICES, default=None,
                        help="Network architecture; defaults per dataset: "
                             + ", ".join(f"{k}={v}" for k, v in ARCH_DEFAULTS.items()))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Compute device (default cpu).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("error: --device cuda requested but CUDA is not available")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data -------------------------------------------------------------
    ds = load_dataset(args.dataset)
    Y = ds.num_classes
    arch = args.arch or ARCH_DEFAULTS[args.dataset]

    X_train_full, y_train_full = ds.splits["train"]
    # Class-stratified split of the training subset into fit / model-selection.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=args.val_fraction,
        stratify=y_train_full, random_state=args.seed)

    # Test data: the test subset, plus the official val subset when one exists
    # (merged into test only -- never used for training or calibration).
    if "val" in ds.splits:
        X_test = np.concatenate([ds.splits["val"][0], ds.splits["test"][0]])
        y_test = np.concatenate([ds.splits["val"][1], ds.splits["test"][1]])
        test_desc = "official val + test merged"
    else:
        X_test, y_test = ds.splits["test"]
        test_desc = "test subset"

    # Per-channel normalization computed on the fit part and stored in the
    # bundle so the downstream script reproduces the same posterior.
    x = X_fit.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[:, :, :, None]
    norm_mean = x.mean(axis=(0, 1, 2))
    norm_std = x.std(axis=(0, 1, 2)) + 1e-7
    in_channels = x.shape[-1]

    Xt_fit = to_tensor(X_fit, norm_mean, norm_std)
    Xt_val = to_tensor(X_val, norm_mean, norm_std)
    Xt_test = to_tensor(X_test, norm_mean, norm_std)
    yt_fit = torch.from_numpy(y_fit)
    yt_val = torch.from_numpy(y_val)
    yt_test = torch.from_numpy(y_test)

    # Training prior estimate from the fit part (what the model was fit on;
    # identical to the full training subset up to stratification rounding).
    train_prior = np.bincount(y_fit, minlength=Y).astype(float)
    train_prior /= train_prior.sum()

    # --- training ----------------------------------------------------------
    model = make_model(arch, in_channels, Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_err": [], "val_loss": [], "val_err": []}
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
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Per-epoch record: loss + error on the fit part and validation part.
        tr_loss, tr_err, _ = evaluate(model, Xt_fit, yt_fit, device)
        va_loss, va_err, _ = evaluate(model, Xt_val, yt_val, device)
        history["train_loss"].append(tr_loss)
        history["train_err"].append(tr_err)
        history["val_loss"].append(va_loss)
        history["val_err"].append(va_err)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train loss {tr_loss:.4f} err {tr_err:.4f}   "
              f"val loss {va_loss:.4f} err {va_err:.4f}")

        # Best epoch by validation error (first best kept on ties).
        if va_err < best_val_err:
            best_val_err, best_epoch = va_err, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # --- temperature calibration on the model-selection part ---------------
    val_logits = collect_logits(model, Xt_val, device)
    temperature = fit_temperature(val_logits, yt_val)

    # --- final evaluation (T does not change the argmax) --------------------
    _, fit_err, fit_pred = evaluate(model, Xt_fit, yt_fit, device)
    _, test_err, test_pred = evaluate(model, Xt_test, yt_test, device)
    cm_fit = confusion_matrix(y_fit, fit_pred, Y)
    cm_test = confusion_matrix(y_test, test_pred, Y)

    # --- outputs -------------------------------------------------------------
    bundle = {
        "model_state": best_state,
        "arch": arch,
        "dataset": args.dataset,
        "num_classes": Y,
        "in_channels": in_channels,
        "image_shape": tuple(ds.image_shape),
        "temperature": temperature,
        "train_prior": train_prior,
        "class_names": list(ds.spec.class_names),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    torch.save(bundle, out_dir / "model.pt")

    lines = [
        "Base neural-network predictor: training and evaluation report",
        "=" * 72,
        f"timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"command     : {' '.join(sys.argv)}",
        f"dataset     : {ds.spec.display_name} ({args.dataset})",
        f"architecture: {arch}",
        f"device      : {args.device}    seed: {args.seed}",
        f"epochs      : {args.epochs}  batch size: {args.batch_size}  lr: {args.lr:g}",
        f"splits      : fit {len(y_fit)}  model-selection {len(y_val)}  "
        f"test {len(y_test)} ({test_desc})",
        f"best epoch  : {best_epoch}  (validation error {best_val_err:.4f})",
        f"temperature : {temperature:.4f}  (fit on the model-selection part)",
        f"train prior : {np.array2string(train_prior, precision=4)}",
        "-" * 72,
        f"classification error, training (fit part) : {fit_err:.4f}",
        f"classification error, test                : {test_err:.4f}",
        "-" * 72,
        "confusion matrix, training (fit part):",
        format_confusion(cm_fit, list(ds.spec.class_names)),
        "",
        "confusion matrix, test:",
        format_confusion(cm_test, list(ds.spec.class_names)),
        "",
    ]
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report)
    print(report)

    make_curves_figure(history, best_epoch, out_dir)
    print(f"outputs written to {out_dir}/: model.pt, report.txt, "
          f"learning_curves.png")


if __name__ == "__main__":
    main()
