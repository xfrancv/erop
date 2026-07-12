"""Download the reject-option candidate datasets into ``data/``.

Fetches, from public URLs, the datasets proposed in
``tasks/datataset_proposal.md`` (all but the iNaturalist/CUB follow-up):
Fashion-MNIST, CIFAR-10, DermaMNIST and BloodMNIST. Everything uses the
standard library plus NumPy — no torch/torchvision/medmnist.

Run with::

    python download_datasets.py                 # all four
    python download_datasets.py cifar10 bloodmnist
    python download_datasets.py --list
"""

from __future__ import annotations

import argparse

from data_tools.download import DATA_ROOT, download_dataset
from data_tools.registry import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", choices=list(DATASETS) + [[]],
                        help="dataset keys to download (default: all)")
    parser.add_argument("--list", action="store_true", help="list dataset keys and exit")
    args = parser.parse_args()

    if args.list:
        for key, spec in DATASETS.items():
            print(f"  {key:<14} {spec.display_name}")
        return

    keys = args.datasets or list(DATASETS)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for key in keys:
        download_dataset(key)
    print(f"\nDone. Data in {DATA_ROOT}")


if __name__ == "__main__":
    main()
