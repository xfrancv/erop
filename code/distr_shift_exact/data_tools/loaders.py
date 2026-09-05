"""Load each downloaded dataset into a common in-memory representation."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .download import DATA_ROOT, download_dataset
from .registry import DATASETS, DatasetSpec

# Canonical split order for reporting.
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class Dataset:
    """A loaded dataset with a uniform interface across the four sources.

    ``splits`` maps a split name to ``(X, y)`` where ``X`` is a uint8 array of
    shape ``(N, H, W)`` (grayscale) or ``(N, H, W, C)`` (RGB) and ``y`` is an
    int array of shape ``(N,)``.
    """

    spec: DatasetSpec
    splits: dict[str, tuple[np.ndarray, np.ndarray]]

    @property
    def image_shape(self) -> tuple[int, ...]:
        return next(iter(self.splits.values()))[0].shape[1:]

    @property
    def num_features(self) -> int:
        return int(np.prod(self.image_shape))

    @property
    def num_classes(self) -> int:
        return len(self.spec.class_names)

    @property
    def is_rgb(self) -> bool:
        return len(self.image_shape) == 3 and self.image_shape[-1] == 3

    def class_counts(self, split: str) -> np.ndarray:
        _, y = self.splits[split]
        return np.bincount(y, minlength=self.num_classes)


# --- Fashion-MNIST (IDX) --------------------------------------------------

def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad IDX image magic {magic}")
        buf = fh.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad IDX label magic {magic}")
        buf = fh.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def _load_idx(spec: DatasetSpec, data_dir: Path) -> dict:
    return {
        "train": (
            _read_idx_images(data_dir / "train-images-idx3-ubyte.gz"),
            _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz"),
        ),
        "test": (
            _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz"),
            _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz"),
        ),
    }


# --- Image folder, e.g. CIFAR-10 / CIFAR-100 -----------------------------

def _leaf_class_dirs(split_dir: Path):
    """Yield the leaf directories that directly contain ``*.png`` files.

    Handles both a flat layout (``<split>/<class>/*.png``, CIFAR-10) and a
    nested one (``<split>/<superclass>/<class>/*.png``, the fast.ai CIFAR-100
    mirror): the label is the *leaf* folder name, so intermediate superclass
    directories (which hold only subdirectories, no images) are skipped.
    """
    for p in sorted(split_dir.rglob("*")):
        if p.is_dir() and next(p.glob("*.png"), None) is not None:
            yield p


def _load_imagefolder(spec: DatasetSpec, data_dir: Path) -> dict:
    """Load an image-folder tree; cache the arrays as an ``.npz``.

    Class folders are mapped to integer labels by their position in
    ``spec.class_names`` (so the index is stable regardless of directory order).
    Decoding tens of thousands of PNGs is slow, so results are cached next to
    the data.
    """
    cache = data_dir / f"{spec.key}_arrays.npz"
    if cache.exists():
        z = np.load(cache)
        return {s: (z[f"{s}_x"], z[f"{s}_y"]) for s in SPLIT_ORDER if f"{s}_x" in z.files}

    import matplotlib.image as mpimg  # native PNG decode, no Pillow needed

    name_to_idx = {name: i for i, name in enumerate(spec.class_names)}
    base = data_dir / spec.archive_dir
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "test"):
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        images, labels = [], []
        for cls_dir in _leaf_class_dirs(split_dir):
            idx = name_to_idx[cls_dir.name]
            for png in sorted(cls_dir.glob("*.png")):
                arr = mpimg.imread(png)  # float32 in [0, 1], (H, W, 3|4)
                arr = (arr[..., :3] * 255.0).round().astype(np.uint8)
                images.append(arr)
                labels.append(idx)
        splits[split] = (np.stack(images), np.asarray(labels, dtype=np.int64))

    np.savez(
        cache,
        **{f"{s}_x": xy[0] for s, xy in splits.items()},
        **{f"{s}_y": xy[1] for s, xy in splits.items()},
    )
    return splits


# --- MedMNIST (.npz) ------------------------------------------------------

def _load_medmnist(spec: DatasetSpec, data_dir: Path) -> dict:
    npz = np.load(data_dir / spec.files[0][1])
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLIT_ORDER:
        img_key, lbl_key = f"{split}_images", f"{split}_labels"
        if img_key not in npz:
            continue
        x = npz[img_key].astype(np.uint8)
        y = npz[lbl_key].astype(np.int64).reshape(-1)
        splits[split] = (x, y)
    return splits


_LOADERS = {"idx": _load_idx, "imagefolder": _load_imagefolder, "medmnist": _load_medmnist}


def load_dataset(
    key: str, data_root: Path = DATA_ROOT, auto_download: bool = True
) -> Dataset:
    """Load a dataset by key, downloading it first if necessary."""
    spec = DATASETS[key]
    data_dir = data_root / key
    if auto_download:
        download_dataset(key, data_root)
    splits = _LOADERS[spec.kind](spec, data_dir)
    # Order splits canonically for stable reporting.
    ordered = {s: splits[s] for s in SPLIT_ORDER if s in splits}
    return Dataset(spec=spec, splits=ordered)
