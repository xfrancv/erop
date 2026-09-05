"""Render a self-contained HTML analysis report for a loaded dataset."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .loaders import Dataset  # noqa: E402

# Imported lazily inside the function so that data_tools stays independent of
# the exact/ package for anyone who only wants the raw dataset report.


def _fig_to_data_uri(fig, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _sample_grid(ds: Dataset, split: str, per_class: int, rng: np.random.Generator) -> str:
    """One row of example images per class (from ``split``), as a data URI."""
    X, y = ds.splits[split]
    n_cls = ds.num_classes
    fig, axes = plt.subplots(
        n_cls, per_class,
        figsize=(per_class * 0.9, n_cls * 0.95),
        squeeze=False,
    )
    cmap = None if ds.is_rgb else "gray"
    for c in range(n_cls):
        idx = np.where(y == c)[0]
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False) if len(idx) else []
        for j in range(per_class):
            ax = axes[c][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(pick):
                ax.imshow(X[pick[j]], cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
            else:
                ax.axis("off")
            if j == 0:
                name = ds.spec.class_names[c]
                label = name if len(name) <= 22 else name[:20] + "…"
                ax.set_ylabel(f"{c}: {label}", rotation=0, ha="right", va="center",
                              fontsize=8, labelpad=6)
    fig.suptitle(f"{ds.spec.display_name} — examples per class ({split})", fontsize=10)
    fig.subplots_adjust(left=0.32, wspace=0.05, hspace=0.05, top=0.95)
    return _fig_to_data_uri(fig)


def _class_balance_chart(ds: Dataset) -> str:
    """Grouped bar chart of per-class counts across splits (as fractions)."""
    splits = list(ds.splits)
    n_cls = ds.num_classes
    fig, ax = plt.subplots(figsize=(max(6, n_cls * 0.7), 3.2))
    width = 0.8 / len(splits)
    x = np.arange(n_cls)
    for i, split in enumerate(splits):
        counts = ds.class_counts(split)
        frac = counts / counts.sum()
        ax.bar(x + i * width, frac, width, label=split)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels([str(c) for c in range(n_cls)])
    ax.set_xlabel("class index")
    ax.set_ylabel("fraction of split")
    ax.set_title("Class balance per split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _fig_to_data_uri(fig)


def _split_table(ds: Dataset) -> str:
    rows = []
    for split, (X, _) in ds.splits.items():
        rows.append(f"<tr><td>{html.escape(split)}</td><td>{len(X):,}</td></tr>")
    total = sum(len(X) for X, _ in ds.splits.values())
    rows.append(f"<tr class='total'><td>total</td><td>{total:,}</td></tr>")
    return "\n".join(rows)


def _class_table(ds: Dataset) -> str:
    splits = list(ds.splits)
    head = "".join(f"<th>{html.escape(s)}</th>" for s in splits)
    body = []
    for c in range(ds.num_classes):
        cells = "".join(f"<td>{ds.class_counts(s)[c]:,}</td>" for s in splits)
        marker = ""
        if ds.spec.confusable_pair and ds.spec.class_names[c] in ds.spec.confusable_pair:
            marker = " <span class='pair'>confusable pair</span>"
        body.append(
            f"<tr><td>{c}</td><td class='cname'>{html.escape(ds.spec.class_names[c])}"
            f"{marker}</td>{cells}</tr>"
        )
    return head, "\n".join(body)


def _protocol_section(ds: Dataset, val_fraction: float = 0.2) -> str:
    """The README S6.1 splits as this project will actually resolve them.

    Shows the evaluation-split size that S6.2's ``m_max <= |eval| / 10`` assert
    bounds, and the per-class evaluation counts that govern how often a query
    recurs inside its own adaptation set (S6.3): the expected number of
    duplicates for a query of class y is ``m * theta_y / |eval_y|``, so it is
    the *rarest* class under the *most spiked* prior that decides, not the
    overall eval size.
    """
    from exact.protocol import EVAL_SIZE_RATIO, SIZE_GRID

    n_dev = len(ds.splits["train"][0])
    n_fit = int(round(n_dev * (1 - val_fraction)))
    n_val = n_dev - n_fit
    if "val" in ds.splits:
        n_eval = len(ds.splits["val"][0]) + len(ds.splits["test"][0])
        eval_desc = "official val + test merged"
    else:
        n_eval = len(ds.splits["test"][0])
        eval_desc = "official test split"

    m_allowed = n_eval // EVAL_SIZE_RATIO
    dropped = [m for m in SIZE_GRID if m > m_allowed]
    grid_note = (
        f"<p class='warn'>The S6.2 grid is truncated to "
        f"m &le; {m_allowed}: {dropped} dropped.</p>" if dropped else
        f"<p>The full S6.2 grid fits (m<sub>max</sub> = {max(SIZE_GRID)} "
        f"&le; {m_allowed}).</p>")

    # Worst-case duplicate rate: rarest evaluation class, at the tau = 0.2 spike.
    counts = ds.class_counts("val") + ds.class_counts("test") \
        if "val" in ds.splits else ds.class_counts("test")
    rarest = int(np.argmin(counts))
    m_max = max([m for m in SIZE_GRID if m <= m_allowed] or [0])
    dup_uniform = m_max * (1.0 / ds.num_classes) / max(counts[rarest], 1)
    dup_spike = m_max * 0.2 / max(counts[rarest], 1)

    rows = "\n".join(
        f"<tr><td>{c}</td><td class='cname'>{html.escape(ds.spec.class_names[c])}"
        f"</td><td>{counts[c]:,}</td>"
        f"<td>{m_max * 0.2 / max(counts[c], 1):.3f}</td></tr>"
        for c in range(ds.num_classes))

    return f"""
<h2>Protocol splits (README §6.1)</h2>
<table>
<tr><th>role</th><th>source</th><th># examples</th></tr>
<tr><td>development → fit</td><td>{1 - val_fraction:g} of official train</td>
    <td>{n_fit:,}</td></tr>
<tr><td>development → validation</td><td>{val_fraction:g} of official train</td>
    <td>{n_val:,}</td></tr>
<tr class='total'><td>evaluation</td><td>{eval_desc}</td><td>{n_eval:,}</td></tr>
</table>
{grid_note}

<h3>Duplicate contamination (§6.3)</h3>
<p>A query drawn with replacement can recur inside its own adaptation set. The
expected number of recurrences is <code>m · θ<sub>y</sub> / |eval<sub>y</sub>|</code>,
so it is set by the <em>per-class</em> evaluation counts, not by the overall
size that §6.2's assert bounds. At m = {m_max} the rarest class
(<em>{html.escape(ds.spec.class_names[rarest])}</em>, {counts[rarest]:,}
examples) sees {dup_uniform:.3f} expected duplicates under a uniform prior and
<strong>{dup_spike:.3f}</strong> under the τ = 0.2 spike of §7.</p>
<table><tr><th>#</th><th>label</th><th>|eval<sub>y</sub>|</th>
<th>E[dups] at θ<sub>y</sub>=0.2</th></tr>
{rows}
</table>
"""


_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 960px; padding: 2rem 1.5rem; color: #1a1a1a; line-height: 1.5; }
h1 { margin-bottom: 0.2rem; } h2 { margin-top: 2rem; border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }
.desc { color: #444; font-style: italic; }
.meta { color: #888; font-size: 0.85rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; }
th { background: #f4f4f4; } td.cname { text-align: left; } td:first-child { text-align: left; }
tr.total td { font-weight: bold; background: #fafafa; }
.pair { background: #ffe8b3; color: #7a5200; font-size: 0.72rem; padding: 1px 6px; border-radius: 4px; }
.stats { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
.stat { background: #f7f9fb; border: 1px solid #e2e8ee; border-radius: 8px; padding: 0.7rem 1.1rem; }
.stat .n { font-size: 1.6rem; font-weight: 700; } .stat .l { color: #667; font-size: 0.8rem; }
.warn { background: #fff4e5; border-left: 4px solid #e8a33d; padding: 0.6rem 0.9rem; }
img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 6px; }
"""


def render_report(ds: Dataset, per_class: int = 10, seed: int = 0) -> str:
    """Build a full self-contained HTML report string for one dataset."""
    rng = np.random.default_rng(seed)
    shape = "×".join(str(d) for d in ds.image_shape)
    kind = "RGB" if ds.is_rgb else "grayscale"
    split_split = " / ".join(f"{len(X):,} {s}" for s, (X, _) in ds.splits.items())
    grid = _sample_grid(ds, next(iter(ds.splits)), per_class, rng)
    balance = _class_balance_chart(ds)
    head, cbody = _class_table(ds)
    pair = ""
    if ds.spec.confusable_pair:
        a, b = ds.spec.confusable_pair
        pair = (f"<p><strong>Target confusable pair:</strong> "
                f"{html.escape(a)} vs. {html.escape(b)}.</p>")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(ds.spec.display_name)} — dataset analysis</title>
<style>{_STYLE}</style></head><body>
<h1>{html.escape(ds.spec.display_name)}</h1>
<p class="meta">Generated {now} · key <code>{ds.spec.key}</code></p>
<p class="desc">{html.escape(ds.spec.description)}</p>
{pair}

<div class="stats">
  <div class="stat"><div class="n">{sum(len(X) for X, _ in ds.splits.values()):,}</div>
       <div class="l">examples total</div></div>
  <div class="stat"><div class="n">{shape}</div><div class="l">image ({kind})</div></div>
  <div class="stat"><div class="n">{ds.num_features:,}</div><div class="l">features (flattened)</div></div>
  <div class="stat"><div class="n">{ds.num_classes}</div><div class="l">classes</div></div>
</div>

<h2>Splits</h2>
<p>{split_split}</p>
<table><tr><th>split</th><th># examples</th></tr>
{_split_table(ds)}
</table>

{_protocol_section(ds)}

<h2>Classes &amp; per-split balance</h2>
<table><tr><th>#</th><th>label</th>{head}</tr>
{cbody}
</table>
<img alt="class balance" src="{balance}">

<h2>Example inputs</h2>
<p>{per_class} random examples per class from the
<code>{next(iter(ds.splits))}</code> split.</p>
<img alt="examples per class" src="{grid}">
</body></html>
"""


def write_report(ds: Dataset, out_dir: Path, per_class: int = 10, seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ds.spec.key}.html"
    out_path.write_text(render_report(ds, per_class=per_class, seed=seed), encoding="utf-8")
    return out_path


def write_index(reports: list[tuple[Dataset, Path]], out_dir: Path) -> Path:
    """A small index.html linking the per-dataset reports."""
    items = []
    for ds, path in reports:
        total = sum(len(X) for X, _ in ds.splits.values())
        items.append(
            f"<li><a href='{path.name}'>{html.escape(ds.spec.display_name)}</a> — "
            f"{total:,} examples, {ds.num_classes} classes, "
            f"{'×'.join(str(d) for d in ds.image_shape)}</li>"
        )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join(items)
    out_path = out_dir / "index.html"
    out_path.write_text(
        f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reject-option candidate datasets</title><style>{_STYLE}</style></head><body>
<h1>Reject-option candidate datasets</h1>
<p class="meta">Generated {now}</p>
<p class="desc">Analysis reports for the datasets proposed in
<code>tasks/datataset_proposal.md</code>.</p>
<ul>{body}</ul></body></html>""",
        encoding="utf-8",
    )
    return out_path
