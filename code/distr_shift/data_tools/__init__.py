"""Download and analyze the reject-option candidate datasets.

The datasets are the ones proposed in ``tasks/datataset_proposal.md`` for the
epistemic reject-option showcase (all but the iNaturalist/CUB follow-up):

* ``fashion_mnist`` — Fashion-MNIST (28x28 grayscale, 10 classes)
* ``cifar10``       — CIFAR-10 (32x32 RGB, 10 classes)
* ``dermamnist``    — DermaMNIST / MedMNIST v2 (28x28 RGB, 7 classes)
* ``bloodmnist``    — BloodMNIST / MedMNIST v2 (28x28 RGB, 8 classes)

Everything is fetched from public URLs with the standard library and loaded
with NumPy — no torch/torchvision/medmnist dependency.
"""

from __future__ import annotations

from .registry import DATASETS, DatasetSpec
from .loaders import Dataset, load_dataset
from .download import DATA_ROOT, download_dataset

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "Dataset",
    "load_dataset",
    "download_dataset",
    "DATA_ROOT",
]
