#!/usr/bin/env python3
"""Generate ``metric-template.ipynb`` from ``chal/metric.py`` (C6.3).

Kaggle runs a notebook, not a package, so the metric has to be a single
self-contained cell. Rather than maintaining that cell by hand next to the copy
``evaluate.py`` imports -- two copies that would eventually disagree, in the one
piece of code that cannot be debugged after launch -- the notebook is
**generated** from ``chal/metric.py``, which is why that module imports nothing
but numpy and pandas.

    python make_metric_notebook.py metric-template.ipynb
    python -m doctest chal/metric.py -v      # the notebook's tests, run locally

**The hard variant** runs the same function with different docstrings: Kaggle
renders them to competitors, and the hard variant must not say that the batches
differ in their label prior. ``chal/metric_hard.py`` is therefore also
generated -- from ``chal/metric.py``, by :func:`hard_source`, which swaps the
docstring sentences and nothing else -- and ``selftest.py`` fails if the two
files drift apart::

    python make_metric_notebook.py --hard-module chal/metric_hard.py
    python make_metric_notebook.py metric-template-hard.ipynb --source chal/metric_hard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = """# Competition metric: AvgRegAtCoverage
#
# GENERATED FILE -- do not edit here. Edit challenge/{source} and re-run
#     python make_metric_notebook.py {out} --source {source}
# so that this cell and the function evaluate.py imports stay the same code.
#
# Kaggle contract: a function named `score` whose first three arguments are
# (solution, submission, row_id_column_name), every argument annotated, and a
# single finite float returned. All solution columns except `Usage` are passed
# through, which is where `m`, `label` and `pred_ref` come from.
"""

FOOTER = '''

if __name__ == "__main__":
    import doctest
    results = doctest.testmod(verbose=False)
    print(f"doctests: {results.attempted - results.failed}"
          f"/{results.attempted} passed")
    assert results.failed == 0
'''


# (easy-variant text, hard-variant text) -- docstring sentences only. Each must
# occur exactly once in chal/metric.py, or hard_source() refuses.
_HARD_SWAPS = (
    ("The competition metric: AvgRegAtCoverage (C6).",
     "The competition metric of the HARD variant: AvgRegAtCoverage.\n\n"
     "GENERATED from ``chal/metric.py`` by ``make_metric_notebook.py "
     "--hard-module``;\ndo not edit. The code is identical; only the docstrings "
     "differ, because Kaggle\nrenders them to competitors and the hard variant "
     "does not say that batches\ndiffer in their label prior "
     "(tasks/hard_variant.md). What follows is the easy\nvariant's description, "
     "kept for the organisers."),
    ("Each test batch is a set of images drawn under one unknown label prior. For",
     "Each test batch is a set of images from one location. For"),
    ("over the kept rows, where ``pred_ref`` is a reference predictor that was\n"
     "    given the true prior of that batch. The reported score averages these seven\n"
     "    per-``m`` numbers with equal weight.",
     "over the kept rows, where ``pred_ref`` is a reference predictor fine-tuned\n"
     "    for the location the batch came from. The reported score averages these\n"
     "    seven per-``m`` numbers with equal weight."),
    ("Scores below zero are possible and are not an error: the competitor and the\n"
     "    reference predictor share the same imperfect calibrated posterior, so the\n"
     "    competitor can occasionally beat it. Ties in ``confidence`` are broken by\n"
     "    ascending ``row_id``, so a constant-confidence submission is still scored\n"
     "    deterministically.",
     "Scores below zero are possible and are not an error: the reference\n"
     "    predictor makes mistakes too, and a competitor can beat it. Ties in\n"
     "    ``confidence`` are broken by ascending ``row_id``, so a\n"
     "    constant-confidence submission is still scored deterministically."),
)


def hard_source(src: str) -> str:
    """``chal/metric_hard.py`` as generated from the text of ``chal/metric.py``."""
    for old, new in _HARD_SWAPS:
        assert src.count(old) == 1, (
            f"chal/metric.py no longer contains, exactly once:\n  {old[:70]}\n"
            f"update _HARD_SWAPS in make_metric_notebook.py")
        src = src.replace(old, new)
    return src


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=Path("metric-template.ipynb"))
    p.add_argument("--source", type=Path, default=Path("chal/metric.py"))
    p.add_argument("--hard-module", type=Path, default=None,
                   help="instead of a notebook, write the hard variant's metric "
                        "module (chal/metric_hard.py) generated from --source")
    args = p.parse_args()

    if args.hard_module is not None:
        args.hard_module.write_text(hard_source(args.source.read_text()))
        print(f"{args.hard_module}: generated from {args.source}")
        return

    src = args.source.read_text()
    # The module docstring describes the file's role in the repository, which is
    # noise inside the notebook; score()'s own docstring is what Kaggle renders.
    body = src.split('"""', 2)[2].lstrip("\n")
    body = body.replace("from __future__ import annotations\n\n", "")
    cell = HEADER.format(source=args.source, out=args.out) + "\n" \
        + body.rstrip() + FOOTER

    nb = {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3",
                           "name": "python3"},
            "language_info": {
                "name": "python", "version": "3.11.0",
                "mimetype": "text/x-python", "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3}},
            "kaggle": {"accelerator": "none", "dataSources": [],
                       "isInternetEnabled": False, "language": "python",
                       "sourceType": "notebook", "isGpuEnabled": False},
        },
        "nbformat": 4, "nbformat_minor": 4,
        "cells": [{"cell_type": "code", "source": cell, "metadata": {},
                   "outputs": [], "execution_count": None}],
    }
    args.out.write_text(json.dumps(nb))
    print(f"{args.out}: one cell, {len(cell.splitlines())} lines, "
          f"from {args.source}")


if __name__ == "__main__":
    main()
