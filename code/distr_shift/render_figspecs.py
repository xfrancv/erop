"""Re-render saved figure specs -- restyle paper figures without re-running the
experiments.

Every ``*.png`` the reject-option scripts write drops a reusable
``*.figspec.json`` (+ ``.npz``) beside it (see ``figspec.py``). This script
loads those specs and re-renders them, optionally under a different matplotlib
style sheet and/or to a vector format for LaTeX -- so tuning a figure's look is
a fast, data-free loop:

    # re-render every spec under figures/ back to PNG (sanity check / no-op)
    python render_figspecs.py figures/

    # restyle all specs with a paper style sheet and emit PDF alongside
    python render_figspecs.py figures/ --style styles/paper.mplstyle --format pdf

    # one figure, a couple of formats
    python render_figspecs.py figures/risk_coverage.figspec.json --format pdf svg

``--style`` centralises presentation (fonts, sizes, line widths, colour cycle,
grid) in one rcParams file, decoupled from the data. Any file matplotlib's
``plt.style.use`` accepts works; pass it more than once to layer them. The
editable text of a single figure (labels, titles, colours) lives in its
``.figspec.json`` and can be hand-edited before re-rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import figspec


def find_specs(paths: list[str]) -> list[Path]:
    """Resolve CLI paths to a sorted, de-duplicated list of ``.figspec.json``
    files: a directory contributes every spec under it (recursively), a file is
    taken as-is (any of the stem / PNG / JSON forms)."""
    found: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.update(p.rglob("*.figspec.json"))
        else:
            found.add(figspec._stem(p).with_name(
                figspec._stem(p).name + ".figspec.json"))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="Spec files (any of stem/.png/.figspec.json) or "
                             "directories to search recursively.")
    parser.add_argument("--style", nargs="+", default=None,
                        help="matplotlib style sheet(s) to apply, layered "
                             "left-to-right (plt.style.use).")
    parser.add_argument("--format", nargs="+", default=["png"],
                        help="Output format(s): png pdf svg ... (default png). "
                             "Each spec is rendered once per format.")
    parser.add_argument("--dpi", type=int, default=None,
                        help="Raster dpi override (default: each spec's own).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Write outputs here instead of beside each spec "
                             "(the stem's base name is kept).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.style:
        plt.style.use(args.style)

    specs = find_specs(args.paths)
    if not specs:
        parser.error("no .figspec.json files found under the given paths")

    for json_path in specs:
        spec = figspec.load_figspec(json_path)
        fig = figspec.render_figure(spec)
        stem = figspec._stem(json_path)
        out_base = (Path(args.out_dir) / stem.name if args.out_dir else stem)
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for fmt in args.format:
            out = f"{out_base}.{fmt}"
            fig.savefig(out, dpi=args.dpi or spec.dpi)
            print(f"rendered {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
