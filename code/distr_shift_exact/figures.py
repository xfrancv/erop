"""The S6.5 figures, built from a ``results.json`` produced by ``rejopt_eval.py``.

Four panels per stratum, all against the adaptation size ``m`` with bootstrap
95% bands:

1. **AuRC** vs m -- rejectors 1-5.
2. **AuRegC** vs m -- rejectors 1-4; row 5 is the regret reference and is
   identically zero, so it is drawn as a grey baseline rather than a competitor.
3. **Reg@c** vs m -- rejectors 1-4.
4. **Accuracy at full coverage** vs m -- the four predictors, with the
   train-prior and true-prior lines flat in ``m`` because they never see ``D``.

Panel 4 is the full-coverage endpoint of panel 1 and so carries no independent
evidence; it is kept because the floor/ceiling reading is much easier there. Two
caveats belong in the caption rather than in the plot: the "ceiling" is only a
ceiling in expectation, since both it and the adapted predictor are built on the
same imperfect ``p_tr(y | x)`` (which is exactly why selective regret can go
negative, S4.1); and the bands *widen* to the right, because the bootstrap unit
is the trial and ``N(m)`` falls from ``N_max`` to ``N_min`` across the grid even
though the pooled budget ``B`` is held constant.

The x-axis is symlog with a linear threshold of 1, so the ``m = 0`` anchor point
of S6.2 is drawn rather than dropped as a plain log axis would do.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "bayes_total":     dict(color="#1f4e9c", marker="o", label="Bayesian, total"),
    "bayes_epistemic": dict(color="#d1495b", marker="s", label="Bayesian, epistemic"),
    "map_plugin":      dict(color="#2a9d8f", marker="^", label="MAP plugin"),
    "train_plugin":    dict(color="#8d6e63", marker="v", label="train-prior plugin"),
    "true_plugin":     dict(color="#6c757d", marker="D", label="true-prior plugin (oracle)",
                            ls="--"),
    "bayes_aleatoric": dict(color="#e9a13b", marker="*", label="Bayesian, aleatoric",
                            ls=":"),
}

PANELS = (
    ("aurc", "AuRC", ("bayes_total", "bayes_epistemic", "map_plugin",
                      "train_plugin", "true_plugin")),
    ("auregc", "AuRegC", ("bayes_total", "bayes_epistemic", "map_plugin",
                          "train_plugin")),
    ("regret_at_c", "Reg@c", ("bayes_total", "bayes_epistemic", "map_plugin",
                              "train_plugin")),
    ("accuracy_full", "accuracy at full coverage",
     ("bayes_total", "map_plugin", "train_plugin", "true_plugin")),
)


def _log_x_axis(ax, sizes):
    """Log-spaced m axis that still shows the ``m = 0`` anchor of S6.2.

    A plain log axis drops ``m = 0`` entirely, so the scale is symlog with a
    linear threshold of 1. Its default limits leave the whole negative half of
    the linear region empty, hence the explicit left limit; crowded grid points
    (100 next to 200) get every other label dropped.
    """
    ax.set_xscale("symlog", linthresh=1, linscale=0.4)
    ax.set_xlim(-0.35, max(sizes) * 1.25)
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(axis="x", labelsize=8.5)
    # Thin out labels that would collide. The grid is log-spaced, so "close" has
    # to be measured in the *transformed* axis, not in m: 100 and 200 sit almost
    # on top of each other while 1 and 2 do not.
    tf = ax.get_xaxis().get_transform().transform
    lo, hi = tf(np.array(ax.get_xlim()))
    pos = (tf(np.asarray(sizes, float)) - lo) / (hi - lo)
    last = -1.0
    for tick, x in zip(ax.get_xticklabels(), pos):
        if x - last < 0.055:
            tick.set_visible(False)
        else:
            last = x


def _series(cells: list[dict], key: str, scalar: str):
    m = np.array([c["m"] for c in cells], dtype=float)
    v = np.array([c["scalars"][key][scalar] for c in cells])
    lo = hi = None
    if all(c.get("ci", {}).get(key) for c in cells):
        lo = np.array([c["ci"][key][scalar][0] for c in cells])
        hi = np.array([c["ci"][key][scalar][1] for c in cells])
    return m, v, lo, hi


def _draw_panel(ax, cells, scalar, ylabel, keys, coverage, with_aleatoric):
    if with_aleatoric and scalar != "accuracy_full":
        keys = tuple(keys) + ("bayes_aleatoric",)
    for key in keys:
        if key not in cells[0]["scalars"]:
            continue
        m, v, lo, hi = _series(cells, key, scalar)
        st = dict(STYLE[key])
        ls = st.pop("ls", "-")
        ax.plot(m, v, ls=ls, lw=1.7, ms=4.5, **st)
        if lo is not None:
            ax.fill_between(m, lo, hi, color=st["color"], alpha=0.15, lw=0)
    if scalar in ("auregc", "regret_at_c"):
        ax.axhline(0.0, color="#6c757d", lw=1.2, ls="--", zorder=0)
    ax.set_xlabel("adaptation set size $m$")
    ax.set_ylabel(f"Reg@{coverage:g}" if scalar == "regret_at_c" else ylabel)
    ax.grid(True, alpha=0.25)
    _log_x_axis(ax, [c["m"] for c in cells])


def figure_for_stratum(res: dict, name: str, with_aleatoric: bool = False):
    st = res["strata"][name]
    cells = st["cells"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for ax, (scalar, ylabel, keys) in zip(axes.ravel(), PANELS):
        _draw_panel(ax, cells, scalar, ylabel, keys, res["coverage"],
                    with_aleatoric)
    axes[0, 0].legend(fontsize=8, loc="best")
    conv = ("theta_* uniform over Theta minus theta_1 (shifted trials only)"
            if name == "shifted" else
            "theta_* uniform over Theta (unstratified)" if name == "marginal"
            else f"theta_* fixed to {st['label']}")
    n_lo = min(c["n_trials"] for c in cells)
    n_hi = max(c["n_trials"] for c in cells)
    fig.suptitle(
        f"{res['dataset']} -- selective risk and regret vs adaptation size\n"
        f"{conv};  B = {res['budget']} pooled triplets per curve;  "
        f"N(m) = {n_hi}..{n_lo} trials;  95% bootstrap bands over trials",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_risk_coverage(res: dict, name: str, sizes=(0, 10, 100)):
    """Risk-coverage curves at a few adaptation sizes (not in S6.5; diagnostic).

    Rows 1 and 2 share the base predictor ``H(x, D)``, so their curves must meet
    at full coverage; rows 3-5 have their own base predictors and need not.
    """
    cells = {c["m"]: c for c in res["strata"][name]["cells"]}
    shown = [m for m in sizes if m in cells] or [max(cells)]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.6 * len(shown), 4.0),
                             squeeze=False)
    for ax, m in zip(axes[0], shown):
        cell = cells[m]
        cov = np.array(cell["coverage_grid"])
        for key in ("bayes_total", "bayes_epistemic", "map_plugin",
                    "train_plugin", "true_plugin"):
            st = dict(STYLE[key])
            ls = st.pop("ls", "-")
            st.pop("marker", None)
            ax.plot(cov, cell["curves"][key]["risk"], ls=ls, lw=1.6, **st)
        ax.set_title(f"$m = {m}$")
        ax.set_xlabel("coverage")
        ax.set_ylabel("selective risk")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{res['dataset']} -- risk-coverage curves, stratum '{name}'",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def figure_theta_breakdown(res: dict, scalar: str = "aurc",
                           key: str = "bayes_total"):
    """The S6.4 supplementary panel: one line per ``theta_*``, plus the marginal.

    Pooling mixes trials with different ``theta_*``; this is the breakdown that
    shows whether a low-coverage region is populated by hard *examples* or by
    whole easy *trials*.
    """
    strata = [n for n in res["strata"] if n.startswith("theta[")]
    if not strata:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cmap = plt.get_cmap("viridis")
    for ax, sc, ylab in ((axes[0], "aurc", "AuRC"),
                         (axes[1], "auregc", "AuRegC")):
        for i, n in enumerate(strata):
            cells = res["strata"][n]["cells"]
            m, v, _, _ = _series(cells, key, sc)
            c = int(n[len("theta["):-1])
            ax.plot(m, v, lw=1.5, marker="o", ms=3.5,
                    color=cmap(i / max(1, len(strata) - 1)),
                    label=f"$\\theta_{{{c}}}$: {res['theta_labels'][c][:22]}")
        for n, style in (("marginal", dict(color="k", ls="--")),
                         ("shifted", dict(color="k", ls=":"))):
            if n in res["strata"]:
                cells = res["strata"][n]["cells"]
                m, v, _, _ = _series(cells, key, sc)
                ax.plot(m, v, lw=2.0, label=n, **style)
        if sc == "auregc":
            ax.axhline(0.0, color="#6c757d", lw=1.0, ls="--", zorder=0)
        ax.set_xlabel("adaptation set size $m$")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        _log_x_axis(ax, [c["m"] for c in cells])
    axes[1].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(
        f"{res['dataset']} -- {STYLE[key]['label']}, broken down by "
        f"$\\theta_*$ (README S6.4 supplementary)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def write_all_figures(res: dict, out_dir: Path,
                      with_aleatoric: bool = False) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        if fig is None:
            return
        p = out_dir / name
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    for name in res["strata"]:
        safe = name.replace("[", "_").replace("]", "")
        save(figure_for_stratum(res, name, with_aleatoric), f"panels_{safe}.png")
    save(figure_risk_coverage(res, "shifted"), "risk_coverage_shifted.png")
    save(figure_theta_breakdown(res), "theta_breakdown.png")
    return paths


if __name__ == "__main__":
    import json
    import sys

    path = Path(sys.argv[1])
    res = json.loads(path.read_text())
    for p in write_all_figures(res, path.parent):
        print(p)
