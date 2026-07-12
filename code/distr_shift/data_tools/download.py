"""Fetch the raw dataset files into ``data/<key>/`` with the standard library."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from .registry import DATASETS, DatasetSpec

# data/ lives next to the code directory (the repo root of run_synth_bayesian_learning_exp.py).
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

_USER_AGENT = "distr_shift-datasets/1.0 (+urllib)"


def _download_file(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a tqdm progress bar (skip if present)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted URLs)
        total = int(resp.headers.get("Content-Length", 0)) or None
        bar = tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {dest.name}", leave=False,
        )
        with open(tmp, "wb") as fh:
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                fh.write(chunk)
                bar.update(len(chunk))
        bar.close()
    tmp.rename(dest)


def _extract_tarball(spec: DatasetSpec, dest_dir: Path) -> None:
    """Extract a ``.tar.gz`` once, into data/<key>/<archive_dir>/."""
    if (dest_dir / spec.archive_dir).is_dir():
        return
    tarball = dest_dir / spec.files[0][1]
    with tarfile.open(tarball, "r:gz") as tar:
        # Guard against path traversal in archive members.
        members = [m for m in tar.getmembers() if not m.name.startswith(("/", ".."))]
        tar.extractall(dest_dir, members=members)  # noqa: S202 (members filtered)


def download_dataset(key: str, data_root: Path = DATA_ROOT) -> Path:
    """Download (and extract, for CIFAR) one dataset; return its directory."""
    spec = DATASETS[key]
    dest_dir = data_root / key
    print(f"[{spec.display_name}] -> {dest_dir}")
    for url, filename in spec.files:
        _download_file(url, dest_dir / filename)
    if spec.kind == "imagefolder":
        _extract_tarball(spec, dest_dir)
    return dest_dir


def download_all(keys: list[str] | None = None, data_root: Path = DATA_ROOT) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    for key in keys or list(DATASETS):
        download_dataset(key, data_root)


def clean(key: str, data_root: Path = DATA_ROOT) -> None:
    """Remove a dataset's local directory (for re-download)."""
    shutil.rmtree(data_root / key, ignore_errors=True)
