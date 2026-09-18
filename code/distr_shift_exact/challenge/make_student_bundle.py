#!/usr/bin/env python3
"""Assemble the starter-code bundle that ships to students.

The bundle is **built, not maintained by hand**. One of its files is a copy of
an organiser artifact and must never diverge from it:

``metric.py``   copied from ``chal/metric.py`` -- the same function Kaggle runs,
                so a student's offline number is the leaderboard's number

Everything else comes from ``student/``, which holds the student-facing sources.

**The trained model is not shipped** (C9.5). Competitors receive the base
predictor's calibrated posteriors, not the network and not the pixels, which is
what stops the released images from being matched against the public TissueMNIST
archive to read off the labels.

**What must never enter the bundle.** The intended solution is that the Bayesian
learned-prior rule should be ranked by *epistemic* uncertainty rather than total
uncertainty, and discovering that is the point of the competition. So
``chal/inference.py``, ``chal/predictors.py``, ``optimal_solution.py``,
``baseline_solutions.py`` and ``batch_meta.csv`` stay out, and the builder
asserts that nothing it copies mentions them.

    python make_student_bundle.py out/kaggle_code
    python make_student_bundle.py out/kaggle_code --zip
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Words that would give the answer away if they appeared in the bundle.
FORBIDDEN = (
    "epistemic", "aleatoric", "bayes_total", "bayes_epistemic", "optimal",
    "theta_star", "batch_meta", "true_plugin", "map_plugin", "pth",
)
# Shipping any of these would undo the decision of C9.5 to withhold the pixels
# and the network.
FORBIDDEN_FILES = ("model.pt", "images.npz", "base_model.py", "make_predictions.py")

METRIC_HEADER = '''"""The competition metric: AvgRegAtCoverage.

This is the exact function the organisers run on Kaggle -- ``evaluate.py`` calls
it, so a score you compute offline on the development batches is the score the
leaderboard would give you for those rows.

Read ``score()``'s docstring for what the metric does and for two worked
examples you can check by hand.
"""
'''


def build(out_dir: Path) -> list[Path]:
    # Rebuild from empty. Copying into an existing bundle leaves behind files
    # that have since been removed from student/ -- which is exactly how a
    # withdrawn artifact (the network, a pixel dump) would quietly ship again.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    written = []

    # 1. the student-facing sources, verbatim
    for src in sorted((HERE / "student").glob("*")):
        if src.name.startswith((".", "__")):
            continue
        shutil.copy2(src, out_dir / src.name)
        written.append(out_dir / src.name)

    # 2. the metric, with its organiser-facing module docstring replaced
    metric_src = (HERE / "chal" / "metric.py").read_text()
    body = metric_src.split('"""', 2)[2].lstrip("\n")
    (out_dir / "metric.py").write_text(METRIC_HEADER + "\n" + body)
    written.append(out_dir / "metric.py")

    _audit(out_dir)
    return written


def _audit(out_dir: Path) -> None:
    """Refuse to ship a bundle that leaks the intended solution."""
    hits = []
    for f in sorted(out_dir.glob("*.py")):
        text = f.read_text().lower()
        for word in FORBIDDEN:
            if re.search(rf"\b{re.escape(word)}\b", text):
                hits.append(f"{f.name}: {word!r}")
    assert not hits, (
        "the bundle names something it must not:\n  " + "\n  ".join(hits)
        + "\nThe intended solution has to stay discoverable, not documented.")

    # Every script must be runnable on its own: no imports from the organiser
    # package, which is not shipped.
    bad = [f.name for f in out_dir.glob("*.py")
           if re.search(r"^\s*(from|import)\s+chal\b", f.read_text(), re.M)]
    assert not bad, f"these still import the organiser package: {bad}"

    leaked = [f.name for f in out_dir.iterdir() if f.name in FORBIDDEN_FILES]
    assert not leaked, (
        f"the bundle contains {leaked}, which C9.5 withholds: publishing the "
        f"network or the pixels makes the released images matchable against the "
        f"public TissueMNIST archive")
    assert not any(f.suffix in (".png", ".pt", ".npz") for f in out_dir.iterdir()), \
        "the bundle must contain no model weights and no image data"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="bundle directory to write")
    p.add_argument("--zip", action="store_true",
                   help="also write <out_dir>.zip, ready to upload")
    args = p.parse_args()

    written = build(args.out_dir)
    total = sum(f.stat().st_size for f in written)
    print(f"{args.out_dir}: {len(written)} files, {total / 1e6:.1f} MB")
    for f in written:
        print(f"  {f.name:<28} {f.stat().st_size / 1e3:>9,.1f} kB")
    print("\naudit passed: the bundle names nothing from the intended solution")

    if args.zip:
        z = args.out_dir.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in written:
                zf.write(f, f.name)
        print(f"\nwrote {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
