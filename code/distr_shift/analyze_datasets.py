"""Analyze the downloaded datasets and emit a self-contained HTML report each.

For every dataset it reports the split sizes, input dimensionality, number of
classes and per-class balance, and visualizes random example inputs for each
class. Reports (and an index) are written to ``data/reports/`` with all images
embedded as base64 — each HTML file is fully self-contained.

Run with::

    python analyze_datasets.py                  # all four (downloads if missing)
    python analyze_datasets.py fashion_mnist    # one dataset
    python analyze_datasets.py --per-class 12
"""

from __future__ import annotations

import argparse

from data_tools.download import DATA_ROOT
from data_tools.loaders import load_dataset
from data_tools.registry import DATASETS
from data_tools.report import write_index, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", choices=list(DATASETS) + [[]],
                        help="dataset keys to analyze (default: all)")
    parser.add_argument("--per-class", type=int, default=10,
                        help="example images per class in the montage (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of downloading missing data")
    args = parser.parse_args()

    keys = args.datasets or list(DATASETS)
    out_dir = DATA_ROOT / "reports"
    reports = []
    for key in keys:
        print(f"[analyze] {DATASETS[key].display_name}")
        ds = load_dataset(key, auto_download=not args.no_download)
        path = write_report(ds, out_dir, per_class=args.per_class, seed=args.seed)
        print(f"          -> {path}")
        reports.append((ds, path))

    index = write_index(reports, out_dir)
    print(f"\nDone. Open {index}")


if __name__ == "__main__":
    main()
