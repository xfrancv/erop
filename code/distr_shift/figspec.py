"""A tiny declarative figure layer that separates *what* a figure shows from
*how* it is drawn (Route A of the figure-persistence design).

Every figure in the reject-option experiment scripts is, structurally, a small
grid of panels; each panel holds a handful of line series (a centre curve with
an optional shaded band), some horizontal reference lines, and the usual axis
furniture (labels, title, log/linear scale, ticks, legend, grid). This module
captures exactly that as plain dataclasses -- a :class:`FigureSpec` -- and gives
three operations on it:

* :func:`render_figure` -- draw the spec with matplotlib (the single source of
  truth for *how* things look), returning a ``Figure``;
* :func:`save_figspec` -- persist the spec next to its PNG as a human-readable
  ``<stem>.figspec.json`` (the editable text: labels, colours, titles, scales)
  plus a companion ``<stem>.figspec.npz`` holding the numeric curve arrays
  exactly (not stringified into the JSON);
* :func:`write` -- the convenience used by the experiment scripts: render, save
  the PNG, and drop the spec, in one call.

The point is that restyling a figure for a paper no longer means re-running the
experiment: load the saved spec (:func:`load_figspec`) and re-render it -- see
``render_figspecs.py`` -- optionally under a different matplotlib style sheet.

The JSON is deliberately the *editable* half: open it and change a label, a
colour or a title by hand, then re-render. The ``.npz`` holds the curves, which
you would not hand-edit. Together they round-trip losslessly through
``save_figspec`` / ``load_figspec``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# JSON tag marking a value that lives in the companion .npz rather than inline.
_NP_TAG = "__np__"


# ---------------------------------------------------------------------------
# The spec: a figure is a grid of panels; a panel is a set of series + axes.
# ---------------------------------------------------------------------------

@dataclass
class Series:
    """One plotted element of a panel.

    ``kind="line"`` draws ``center`` vs. ``x`` and, when ``lower``/``upper`` are
    given, a shaded band between them (the mean/median line and its uncertainty
    band). ``kind="scatter"`` draws ``x`` vs. ``center`` as markers (``center``
    carries the y values), using the ``size``/``edgecolors``/``zorder`` fields.
    ``color`` is anything matplotlib accepts: a cycle name (``"C1"``), a grey
    level (``"0.4"``), or an explicit ``[r, g, b, a]`` list.
    """
    x: np.ndarray
    center: np.ndarray
    label: str | None = None
    color: Any = "C0"
    kind: str = "line"
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    marker: str | None = None
    linestyle: str = "-"
    linewidth: float = 1.8
    band_alpha: float = 0.2
    # scatter-only
    size: float = 36.0
    alpha: float = 1.0
    edgecolors: str = "none"
    zorder: int | None = None


@dataclass
class HLine:
    """A horizontal reference line (e.g. the zero-regret baseline)."""
    y: float
    color: str = "0.4"
    linestyle: str = "--"
    linewidth: float = 1.0


@dataclass
class Panel:
    """One axes: its series, reference lines and axis furniture.

    ``axis_off=True`` yields an empty (hidden) cell, used to pad a grid whose
    last row is not full.
    """
    series: list[Series] = field(default_factory=list)
    hlines: list[HLine] = field(default_factory=list)
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    xscale: str = "linear"
    xticks: list[float] | None = None
    ylim: list[float] | None = None
    grid: bool = True
    grid_which: str = "major"
    grid_alpha: float = 0.25
    legend: bool = False
    legend_loc: str | None = None
    legend_fontsize: float = 8.0
    axis_off: bool = False


@dataclass
class FigureSpec:
    """A whole figure: an ``nrows x ncols`` grid of panels (row-major) plus the
    figure-level furniture. ``dpi`` is remembered so a re-render reproduces the
    original raster resolution unless overridden."""
    panels: list[Panel]
    nrows: int = 1
    ncols: int = 1
    figsize: list[float] = field(default_factory=lambda: [8.0, 5.0])
    suptitle: str | None = None
    tight_rect: list[float] | None = None
    dpi: int = 130


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_series(ax, s: Series) -> None:
    if s.kind == "scatter":
        kw = {} if s.zorder is None else {"zorder": s.zorder}
        ax.scatter(s.x, s.center, s=s.size, color=s.color, alpha=s.alpha,
                   edgecolors=s.edgecolors, label=s.label, **kw)
        return
    ax.plot(s.x, s.center, lw=s.linewidth, color=s.color, ls=s.linestyle,
            marker=s.marker, label=s.label)
    if s.lower is not None and s.upper is not None:
        ax.fill_between(s.x, s.lower, s.upper, color=s.color, alpha=s.band_alpha)


def _draw_panel(ax, panel: Panel, mticker) -> None:
    if panel.axis_off:
        ax.axis("off")
        return
    for s in panel.series:
        _draw_series(ax, s)
    for h in panel.hlines:
        ax.axhline(h.y, color=h.color, ls=h.linestyle, lw=h.linewidth)
    if panel.xscale != "linear":
        ax.set_xscale(panel.xscale)
    if panel.xticks is not None:
        ax.set_xticks(panel.xticks)
        if panel.xscale == "log":
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    if panel.ylim is not None:
        ax.set_ylim(*panel.ylim)
    if panel.xlabel is not None:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel is not None:
        ax.set_ylabel(panel.ylabel)
    if panel.title is not None:
        ax.set_title(panel.title)
    if panel.legend:
        if panel.legend_loc is None:
            ax.legend(fontsize=panel.legend_fontsize)
        else:
            ax.legend(fontsize=panel.legend_fontsize, loc=panel.legend_loc)
    if panel.grid:
        ax.grid(True, which=panel.grid_which, alpha=panel.grid_alpha)


def render_figure(spec: FigureSpec):
    """Draw ``spec`` and return the matplotlib ``Figure`` (not yet saved)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(spec.nrows, spec.ncols,
                             figsize=tuple(spec.figsize), squeeze=False)
    for idx, panel in enumerate(spec.panels):
        r, c = divmod(idx, spec.ncols)
        _draw_panel(axes[r][c], panel, mticker)
    if spec.suptitle is not None:
        fig.suptitle(spec.suptitle)
    if spec.tight_rect is not None:
        fig.tight_layout(rect=tuple(spec.tight_rect))
    else:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# (De)serialization: JSON for the structure/text, .npz for the arrays.
# ---------------------------------------------------------------------------

def _to_jsonable(obj: Any, put) -> Any:
    """Recursively convert a spec into JSON-able data, offloading every ndarray
    to the array store via ``put`` (which returns a ``{_NP_TAG: key}`` stub)."""
    if isinstance(obj, np.ndarray):
        return put(obj)
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name), put)
                for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, put) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _from_jsonable(cls, data: Any, get):
    """Inverse of :func:`_to_jsonable` for a known dataclass ``cls``: rebuild the
    dataclass tree, resolving ``{_NP_TAG: key}`` stubs back to arrays via
    ``get``."""
    # Field name -> the dataclass type to rebuild for nested spec objects.
    nested = {
        (FigureSpec, "panels"): Panel,
        (Panel, "series"): Series,
        (Panel, "hlines"): HLine,
    }
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        child = nested.get((cls, f.name))
        if child is not None and isinstance(val, list):
            kwargs[f.name] = [_from_jsonable(child, v, get) for v in val]
        else:
            kwargs[f.name] = _resolve(val, get)
    return cls(**kwargs)


def _resolve(val: Any, get):
    """Turn any ``{_NP_TAG: key}`` stubs inside a plain (non-dataclass) value
    back into arrays."""
    if isinstance(val, dict) and _NP_TAG in val:
        return get(val[_NP_TAG])
    if isinstance(val, list):
        return [_resolve(v, get) for v in val]
    return val


def _stem(path: str | Path) -> Path:
    """Normalise any of ``foo``, ``foo.png``, ``foo.figspec.json`` to the shared
    stem ``foo`` (the base both companion files hang off)."""
    p = Path(path)
    name = p.name
    for suffix in (".figspec.json", ".figspec.npz", ".png", ".pdf", ".svg"):
        if name.endswith(suffix):
            return p.with_name(name[: -len(suffix)])
    return p


def save_figspec(spec: FigureSpec, path: str | Path) -> Path:
    """Write ``spec`` as ``<stem>.figspec.json`` (+ ``.npz`` for its arrays).

    ``path`` may be the stem, the PNG path, or the JSON path itself -- all three
    resolve to the same stem. Returns the JSON path.
    """
    stem = _stem(path)
    arrays: dict[str, np.ndarray] = {}

    def put(a: np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = np.asarray(a)
        return {_NP_TAG: key}

    obj = _to_jsonable(spec, put)
    json_path = stem.with_name(stem.name + ".figspec.json")
    json_path.write_text(json.dumps(obj, indent=2) + "\n")
    npz_path = stem.with_name(stem.name + ".figspec.npz")
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    elif npz_path.exists():  # a prior run left arrays; this spec has none
        npz_path.unlink()
    return json_path


def load_figspec(path: str | Path) -> FigureSpec:
    """Reconstruct a :class:`FigureSpec` saved by :func:`save_figspec`. ``path``
    may be the stem, the JSON path, or the sibling PNG path."""
    stem = _stem(path)
    obj = json.loads(stem.with_name(stem.name + ".figspec.json").read_text())
    npz_path = stem.with_name(stem.name + ".figspec.npz")
    store = np.load(npz_path) if npz_path.exists() else {}

    def get(key: str):
        return np.asarray(store[key])

    return _from_jsonable(FigureSpec, obj, get)


def write(spec: FigureSpec, png_path: str | Path) -> str:
    """Render ``spec`` to ``png_path`` at its own dpi and save the reusable spec
    beside it (``<stem>.figspec.json`` / ``.npz``). Returns ``str(png_path)``.

    This is the one call the experiment scripts make per figure: the PNG stays
    the primary output, and the spec rides along for later restyling.
    """
    import matplotlib.pyplot as plt

    fig = render_figure(spec)
    fig.savefig(str(png_path), dpi=spec.dpi)
    plt.close(fig)
    save_figspec(spec, png_path)
    return str(png_path)
