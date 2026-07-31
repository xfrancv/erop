"""Combine several sweep reports into one LaTeX-ish summary table.

``rejopt_eval.py --sweep`` writes a
``real_reject_option_sweep_report.txt`` per dataset, holding a stack of
fixed-width metric tables indexed by the adaptation-set size ``n_test``. This
script pulls the headline reject-option numbers out of a set of such reports
and lays them side by side, one row per dataset and one 3-column block per
requested ``n_test``:

    python summary_table.py \
        --reports runs/bloodmnist/*/real_reject_option_sweep_report.txt \
                  runs/cifar10/*/real_reject_option_sweep_report.txt \
        --sizes 1 10 --output table.txt

Per dataset and size it takes, from the ``AuRC (regret)`` table, the areas of
the epistemic (``Epist``) and the total-uncertainty Bayesian (``Bayes``)
reject-option predictors, and from the matching ``win% AuRC (regret)`` table
the percentage of sampled priors where *total uncertainty* won -- reported
here flipped (``100 - win%``), i.e. as the win rate of the epistemic
predictor, which is the direction the paper argues.

See [tasks/summary_table.md](tasks/summary_table.md).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Column headers as spelled (truncated) in the report tables.
EPISTEMIC_COL = "Bayesian, epistemic un"
TOTAL_COL = "Bayesian, total uncert"

AURC_TITLE = "AuRC (regret)"
WIN_TITLE = "win% AuRC (regret)"


class ReportError(Exception):
    """A report file is missing a table, a column, or a requested size."""


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


def parse_report(path: Path) -> dict:
    """Extract the dataset name and the two regret tables from one report."""
    lines = path.read_text().splitlines()
    aurc = read_table(lines, AURC_TITLE, path)
    win = read_table(lines, WIN_TITLE, path)
    for title in lines:
        if title.startswith(AURC_TITLE):
            scaled = "[x1000]" in title
            break
    return {
        "name": dataset_name(lines, path),
        "aurc": aurc,
        "win": win,
        "scaled": scaled,
        "path": path,
    }


def row_of(table: dict[str, dict[str, str]], size: str, kind: str,
           path: Path) -> dict[str, str]:
    """Fetch a table row by its ``n_test`` label.  The summary line is spelled
    'avg' in the AuRC tables but 'all' in the win% ones; either spelling picks
    the right row in both."""
    aliases = {"avg": "all", "all": "avg"}
    for label in (size, aliases.get(size)):
        if label in table:
            return table[label]
    raise ReportError(f"{path}: {kind!r} table has no row for n_test={size} "
                      f"(available: {', '.join(table)})")


def cells(report: dict, size: str, precision: int) -> list[str]:
    """The ('Epist', 'Bayes', 'Win') triple for one dataset at one size."""
    path = report["path"]
    aurc_row = row_of(report["aurc"], size, AURC_TITLE, path)
    win_row = row_of(report["win"], size, WIN_TITLE, path)
    for col in (EPISTEMIC_COL, TOTAL_COL):
        if col not in aurc_row:
            raise ReportError(f"{path}: {AURC_TITLE!r} table has no column {col!r}")
    win_raw = win_row.get(EPISTEMIC_COL, "")
    if not win_raw:
        raise ReportError(
            f"{path}: {WIN_TITLE!r} table has no value for n_test={size} "
            f"under {EPISTEMIC_COL!r}")

    epist = float(aurc_row[EPISTEMIC_COL])
    bayes = float(aurc_row[TOTAL_COL])
    # The report counts wins of the total-uncertainty predictor; the summary
    # reports the complementary rate, i.e. wins of the epistemic one.
    win = 100.0 - float(win_raw)
    return [f"{epist:.{precision}f}", f"{bayes:.{precision}f}", f"{win:.1f}"]


def render(reports: list[dict], sizes: list[str], precision: int) -> str:
    """Lay the per-dataset triples out as an aligned LaTeX table body."""
    rows = [[r["name"]] + [c for s in sizes for c in cells(r, s, precision)]
            for r in reports]
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
                        help="sweep report .txt files, one per dataset row")
    parser.add_argument("--sizes", nargs="+", required=True,
                        help="n_test values to keep, one 3-column block each")
    parser.add_argument("--output", required=True,
                        help="output file for the summary table")
    parser.add_argument("--precision", type=int, default=4,
                        help="decimals for the AuRC columns (default 4)")
    args = parser.parse_args()

    try:
        reports = [parse_report(Path(p)) for p in args.reports]
        table = render(reports, args.sizes, args.precision)
    except (ReportError, OSError) as exc:
        sys.exit(f"error: {exc}")

    if len({r["scaled"] for r in reports}) > 1:
        scaled = [r["path"].name for r in reports if r["scaled"]]
        print("warning: mixing reports with and without the '[x1000]' AuRC "
              f"scaling; scaled: {', '.join(map(str, scaled))}", file=sys.stderr)

    out = Path(args.output)
    out.write_text(table)
    print(table, end="")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
