"""Combine several sweep results into one LaTeX-ish summary table.

``rejopt_eval.py`` writes a ``results.json`` per run (alongside the human-facing
``real_reject_option_sweep_report.txt``). This script pulls the headline
reject-option numbers out of a set of those and lays them side by side, one row
per dataset and one 3-column block per requested ``n_test``:

    python summary_table.py \
        --reports runs/bloodmnist/*/results.json \
                  runs/cifar10/*/results.json \
        --sizes 1 10 --output table.txt

Per dataset and size it takes, from the AuRC-of-the-regret numbers, the areas of
the epistemic (``Epist``) and the total-uncertainty Bayesian (``Bayes``)
reject-option predictors, and from the matching win rates the percentage of
sampled priors where *total uncertainty* won -- reported here flipped
(``100 - win%``), i.e. as the win rate of the epistemic predictor, which is the
direction the paper argues.

``--reports`` also accepts the ``real_reject_option_sweep_report.txt`` paths (the
``results.json`` beside a report is preferred automatically), so the wrapper
scripts and older run directories keep working; see :func:`load_run`.

See [tasks/summary_table.md](tasks/summary_table.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Column headers as spelled (truncated) in the legacy report tables.
EPISTEMIC_COL = "Bayesian, epistemic un"
TOTAL_COL = "Bayesian, total uncert"

AURC_TITLE = "AuRC (regret)"
WIN_TITLE = "win% AuRC (regret)"

RESULTS_FILENAME = "results.json"
REPORT_FILENAME = "real_reject_option_sweep_report.txt"

# Row label of the across-sizes summary. The results file spells it 'avg' in the
# area tables and 'all' in the win rates (as the text report does); both are
# normalised to this on load.
AVG_LABEL = "avg"


class ReportError(Exception):
    """A run's results are missing a table, a column, or a requested size."""


# ---------------------------------------------------------------------------
# The normalised record every loader produces
# ---------------------------------------------------------------------------
# {"name": dataset key, "path": Path, "scaled": bool,
#  "aurc": {row label: {"epistemic": float, "total": float}},   # x1000 scale
#  "win_total": {row label: float}}   # win% OF THE TOTAL-UNCERTAINTY predictor
# Areas are carried on the display (x1000) scale, which is what the table prints.


def load_json_run(path: Path) -> dict:
    """Normalised record from a ``results.json`` written by ``rejopt_eval.py``."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path}: not valid JSON ({exc})") from exc

    if "win_rate" not in raw:
        raise ReportError(
            f"{path}: no win rates in this run (they exist only in dirichlet "
            "mode, i.e. runs passing --dirichlet)")
    try:
        areas = raw["areas"]["aurc"]["regret"]
        wins = raw["win_rate"]["areas"]["aurc"]["regret"]
        scale = raw["area_scale"]
    except KeyError as exc:
        raise ReportError(f"{path}: results file is missing {exc} "
                          f"(schema {raw.get('schema', 'unknown')})") from exc

    aurc = {label: {"epistemic": cells["bayes_epistemic"] * scale,
                    "total": cells["bayes_total"] * scale}
            for label, cells in areas.items()}
    win_total = {("all" if label == "all" else label): row["bayes_epistemic"]
                 for label, row in wins.items()}
    win_total[AVG_LABEL] = win_total.pop("all", win_total.get(AVG_LABEL))

    return {
        "name": raw.get("dataset") or path.parent.parent.name,
        "path": path,
        "scaled": scale != 1,
        "aurc": aurc,
        "win_total": win_total,
    }


# ---------------------------------------------------------------------------
# Legacy: parse the fixed-width text report
# ---------------------------------------------------------------------------
# Kept so run directories produced before results.json still summarise. New runs
# always take the JSON path above; this can be deleted once every run of record
# has been regenerated.

def header_spans(header: str) -> dict[str, tuple[int, int]]:
    """Map each column name in a fixed-width table header to its character
    span.  Names may contain spaces ('Bayesian, total uncert'), so columns are
    split on runs of two-or-more spaces; a column's field runs from the end of
    the previous name to the end of its own name, since the report right-aligns
    every value on the header's last character."""
    spans: dict[str, tuple[int, int]] = {}
    start = 0
    for m in re.finditer(r"\S+(?: \S+)*", header):
        # A single space belongs to a name, two or more separate columns.
        spans[m.group().strip()] = (start, m.end())
        start = m.end()
    return spans


def read_table(lines: list[str], title: str, path: Path) -> dict[str, dict[str, str]]:
    """Locate the table whose title line starts with ``title`` and return it as
    ``{row label: {column name: raw cell text}}``.  Row labels are the first
    column ('1', '10', ..., 'avg'/'all'), cells are stripped strings so an
    empty field (the win% self-column) stays distinguishable from a zero."""
    for i, line in enumerate(lines):
        if line.startswith(title):
            break
    else:
        raise ReportError(f"{path}: no table titled {title!r} "
                          "(reports predating the win statistics lack it)")

    header = lines[i + 1]
    spans = header_spans(header)
    label_col, *value_cols = spans

    table: dict[str, dict[str, str]] = {}
    for line in lines[i + 2:]:
        if not line.strip() or line.startswith(("-", "=")):
            break
        if line.startswith("  note:"):  # trailing commentary inside a table
            continue
        lo, hi = spans[label_col]
        label = line[lo:hi].strip()
        if not label:
            continue
        table[label] = {c: line[slice(*spans[c])].strip() for c in value_cols}
    return table


def dataset_name(lines: list[str], path: Path) -> str:
    """Dataset key for the row label, taken from the recorded command line
    ('rejopt_eval.py runs/<key>/model.pt ...').  Falls back to
    the run directory holding the report."""
    for line in lines:
        if line.startswith("command"):
            args = line.split(":", 1)[1].split()
            for arg in args[1:]:
                if arg.endswith(".pt"):
                    return Path(arg).parent.name
            break
    return path.parent.parent.name


def load_text_run(path: Path) -> dict:
    """Normalised record parsed out of a legacy fixed-width text report."""
    lines = path.read_text().splitlines()
    aurc_tbl = read_table(lines, AURC_TITLE, path)
    win_tbl = read_table(lines, WIN_TITLE, path)
    scaled = any(line.startswith(AURC_TITLE) and "[x1000]" in line
                 for line in lines)

    def norm(label: str) -> str:
        return AVG_LABEL if label in (AVG_LABEL, "all") else label

    aurc: dict[str, dict[str, float]] = {}
    for label, row in aurc_tbl.items():
        for col in (EPISTEMIC_COL, TOTAL_COL):
            if col not in row:
                raise ReportError(
                    f"{path}: {AURC_TITLE!r} table has no column {col!r}")
        aurc[norm(label)] = {"epistemic": float(row[EPISTEMIC_COL]),
                             "total": float(row[TOTAL_COL])}
    win_total = {norm(label): float(row[EPISTEMIC_COL])
                 for label, row in win_tbl.items()
                 if row.get(EPISTEMIC_COL)}

    return {
        "name": dataset_name(lines, path),
        "path": path,
        "scaled": scaled,
        "aurc": aurc,
        "win_total": win_total,
    }


def load_run(path: Path) -> dict:
    """Load one run's numbers, preferring the machine-readable results file.

    ``path`` may point at a ``results.json``, at a
    ``real_reject_option_sweep_report.txt`` (the results file beside it is used
    when present), or at a run directory.
    """
    if path.is_dir():
        candidates = [path / RESULTS_FILENAME, path / REPORT_FILENAME]
    elif path.name == RESULTS_FILENAME:
        candidates = [path]
    else:
        # A text report: prefer its results.json sibling, fall back to parsing.
        candidates = [path.parent / RESULTS_FILENAME, path]

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            return load_json_run(candidate)
        return load_text_run(candidate)
    raise ReportError(f"{path}: no {RESULTS_FILENAME} or {REPORT_FILENAME} found")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def cells(run: dict, size: str, precision: int) -> list[str]:
    """The ('Epist', 'Bayes', 'Win') triple for one dataset at one size."""
    label = AVG_LABEL if size in (AVG_LABEL, "all") else size
    if label not in run["aurc"]:
        raise ReportError(
            f"{run['path']}: no AuRC row for n_test={size} "
            f"(available: {', '.join(run['aurc'])})")
    if label not in run["win_total"]:
        raise ReportError(
            f"{run['path']}: no win rate for n_test={size} "
            f"(available: {', '.join(run['win_total'])})")

    areas = run["aurc"][label]
    # The run counts wins of the total-uncertainty predictor; the summary
    # reports the complementary rate, i.e. wins of the epistemic one.
    win = 100.0 - run["win_total"][label]
    return [f"{areas['epistemic']:.{precision}f}",
            f"{areas['total']:.{precision}f}",
            f"{win:.1f}"]


def render(runs: list[dict], sizes: list[str], precision: int) -> str:
    """Lay the per-dataset triples out as an aligned LaTeX table body."""
    rows = [[r["name"]] + [c for s in sizes for c in cells(r, s, precision)]
            for r in runs]
    sub = ["dataset"] + ["Epist", "Bayes", r"Win [\%]"] * len(sizes)

    widths = [max(len(r[i]) for r in [sub] + rows) for i in range(len(sub))]

    def join(cells_: list[str]) -> str:
        padded = [c.ljust(w) for c, w in zip(cells_, widths)]
        return " & ".join(padded).rstrip() + r" \\"

    # The n= banner spans each dataset's 3 columns, so pad it to their combined
    # width plus the two ' & ' separators swallowed by the \multicolumn.
    top = ["".ljust(widths[0])]
    for k, size in enumerate(sizes):
        block = sum(widths[1 + 3 * k: 4 + 3 * k]) + 6
        top.append((r"\multicolumn{3}{c}{n=" + size + "}").ljust(block))
    lines = [" & ".join(top).rstrip() + r" \\", join(sub)]
    lines += [join(r) for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", nargs="+", required=True,
                        help=f"one run per dataset row: a {RESULTS_FILENAME}, a "
                             f"{REPORT_FILENAME}, or a run directory")
    parser.add_argument("--sizes", nargs="+", required=True,
                        help="n_test values to keep, one 3-column block each")
    parser.add_argument("--output", required=True,
                        help="output file for the summary table")
    parser.add_argument("--precision", type=int, default=4,
                        help="decimals for the AuRC columns (default 4)")
    args = parser.parse_args()

    try:
        runs = [load_run(Path(p)) for p in args.reports]
        table = render(runs, args.sizes, args.precision)
    except (ReportError, OSError) as exc:
        sys.exit(f"error: {exc}")

    if len({r["scaled"] for r in runs}) > 1:
        scaled = [r["path"].name for r in runs if r["scaled"]]
        print("warning: mixing runs with and without the 'x1000' AuRC "
              f"scaling; scaled: {', '.join(map(str, scaled))}", file=sys.stderr)

    out = Path(args.output)
    out.write_text(table)
    print(table, end="")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
