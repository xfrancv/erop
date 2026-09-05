"""The adaptation + reject-option experiment (README S3-S6).

Consumes a training run directory and produces, per dataset, the four panels of
S6.5 with bootstrap bands, a text report, and a ``results.json`` that is the
machine interface for any cross-dataset table.

Everything is exact: ``Theta`` is finite, so the posterior over priors is a
normalised ``C``-vector rather than a sampler's output, and the leave-one-out
identity of S5 turns one test set of ``m + 1`` examples into ``m + 1`` triplets
for the price of one weight matrix. No torch, no forward passes -- the training
run already saved the calibrated ``log p_tr(y | x)`` of the whole evaluation
split.

Strata (S6.4). Pooling mixes trials with different ``theta_*``, so the run
produces several:

``shifted``    ``theta_*`` uniform over ``Theta \\ {theta_1}`` -- the main
               panels. S6.4 phrases this as conditioning the pool on
               ``theta_* != theta_1``; drawing conditionally is the same
               distribution but keeps ``N(m)`` and the budget ``B`` at their
               nominal values, which post-hoc filtering would not.
``marginal``   ``theta_*`` uniform over all of ``Theta`` -- the unstratified
               supplementary curve, and the only stratum in which ``theta_*``
               and the model's ``p(theta)`` agree (Appendix A.5).
``theta[c]``   one stratum per element of ``Theta`` -- the supplementary
               breakdown.
``dirichlet``  optional (``--dirichlet-scale``): ``theta_* ~ Dir(s theta_tr)``,
               so ``theta_* not in Theta`` almost surely while the model still
               uses ``Theta``. This is the misspecified second arm Appendix A.1
               calls for; without it the epistemic-calibration claim is not
               falsifiable.

Run with::

    python rejopt_eval.py runs/fashion_mnist
    python rejopt_eval.py runs/cifar100 --dirichlet-scale 20 --bootstrap-reps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from exact.inference import identifiability_table
from exact.metrics import (
    BOOTSTRAP_REPS,
    DEFAULT_COVERAGE,
    REJECTORS,
    evaluate_pool,
)
from exact.priors import read_prior_set
from exact.protocol import (
    BUDGET_B,
    N_MAX,
    N_MIN,
    SIZE_GRID,
    ThetaStarDrawer,
    TrialSampler,
    resolve_grid,
)
from exact.sweep import run_cell

RESULTS_SCHEMA = 1


def cell_seed(seed: int, stratum: str, m: int) -> int:
    """Deterministic per-cell seed.

    Python's ``hash`` of a string is salted per process, so it cannot be used
    here: the same command would give different trials on every run.
    """
    key = f"{seed}|{stratum}|{m}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _json_safe(obj):
    """NaN/inf -> None, so results.json is valid JSON for non-Python readers."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

SCALAR_LABELS = {
    "aurc": "AuRC",
    "auregc": "AuRegC",
    "regret_at_c": "Reg@c",
    "accuracy_full": "acc@1.0",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str,
                   help="training run directory holding eval_log_post.npz and "
                        "priors.txt")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output directory (default: <run_dir>/rejopt)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE,
                   help=f"coverage budget c for Reg@c (S4.2, default "
                        f"{DEFAULT_COVERAGE})")
    p.add_argument("--budget", type=int, default=BUDGET_B,
                   help=f"pooled ranking budget B (S6.4, default {BUDGET_B})")
    p.add_argument("--n-min", type=int, default=N_MIN)
    p.add_argument("--n-max", type=int, default=N_MAX)
    p.add_argument("--sizes", type=int, nargs="*", default=None,
                   help=f"adaptation sizes (default {list(SIZE_GRID)})")
    p.add_argument("--strict-grid", action="store_true",
                   help="fail instead of truncating when the S6.2 assert "
                        "m_max <= |eval|/10 does not hold")
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS,
                   help=f"bootstrap replicates for the main stratum (S6.4, "
                        f"default {BOOTSTRAP_REPS}); the resampling unit is the "
                        f"trial, never the triplet")
    p.add_argument("--supp-bootstrap-reps", type=int, default=200,
                   help="replicates for the supplementary strata (default 200; "
                        "0 disables their bands)")
    p.add_argument("--no-supplementary", action="store_true",
                   help="run only the main 'shifted' stratum")
    p.add_argument("--dirichlet-scale", type=float, default=None,
                   help="also run the misspecified arm of Appendix A.1 with "
                        "theta_* ~ Dir(s * theta_tr) for this s")
    p.add_argument("--no-figures", action="store_true")
    return p


def _stratum_drawers(theta: np.ndarray, train_prior: np.ndarray,
                     args) -> list[tuple[str, ThetaStarDrawer]]:
    drawers = [("shifted", ThetaStarDrawer(theta, "shifted"))]
    if not args.no_supplementary:
        drawers.append(("marginal", ThetaStarDrawer(theta, "marginal")))
        for c in range(len(theta)):
            drawers.append((f"theta[{c}]",
                            ThetaStarDrawer(theta, "fixed", index=c)))
    if args.dirichlet_scale:
        drawers.append(("dirichlet", ThetaStarDrawer(
            theta, "dirichlet", train_prior=train_prior,
            concentration=args.dirichlet_scale)))
    return drawers


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "rejopt"
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = np.load(run_dir / "eval_log_post.npz", allow_pickle=True)
    log_post = blob["log_post"].astype(np.float64)
    y_eval = blob["y"]
    train_prior = blob["train_prior"]
    class_names = list(blob["class_names"])
    dataset = str(blob["dataset"])
    eval_desc = str(blob["eval_desc"])
    Y = log_post.shape[1]

    ps = read_prior_set(run_dir / "priors.txt", train_prior)
    theta, C = ps.theta, ps.C
    log_theta = np.log(theta)
    log_p_theta = np.full(C, -np.log(C))       # S7: p(theta) = 1/C, uniform
    log_train_prior = np.log(train_prior)

    grid, grid_note = resolve_grid(len(y_eval), tuple(args.sizes or SIZE_GRID),
                                   strict=args.strict_grid)
    if grid_note:
        print(f"!! {grid_note}", file=sys.stderr)

    sampler = TrialSampler(y_eval, Y)

    # How fast can the posterior over Theta possibly separate its elements?
    # S6.2 asserts it concentrates exponentially in m, but the rate is set by
    # the KL between the induced marginals over x, which the S7 constructions
    # do not control -- so measure it rather than assume it.
    kl = identifiability_table(log_post, y_eval, log_train_prior, theta, Y)
    off = kl[~np.eye(C, dtype=bool)]
    kl_min = float(off.min())
    m_needed = float("inf") if kl_min <= 0 else 1.0 / kl_min

    results = {
        "schema": RESULTS_SCHEMA,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "dataset": dataset,
        "run_dir": str(run_dir),
        "num_classes": Y,
        "class_names": class_names,
        "eval_size": int(len(y_eval)),
        "eval_desc": eval_desc,
        "eval_error": float((log_post.argmax(axis=1) != y_eval).mean()),
        "grid": list(grid),
        "grid_note": grid_note,
        "budget": args.budget,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "coverage": args.coverage,
        "seed": args.seed,
        "train_prior": train_prior.tolist(),
        "theta": theta.tolist(),
        "theta_labels": ps.labels,
        "theta_tv_to_train": ps.tv_to_train.tolist(),
        "theta_pairwise_tv": ps.pairwise_tv().tolist(),
        "identifiability": {
            "kl": kl.tolist(),
            "min_offdiag_kl": kl_min,
            "m_for_separation": m_needed,
        },
        "rejectors": [{"key": r.key, "label": r.label, "optional": r.optional}
                      for r in REJECTORS],
        "strata": {},
    }

    for name, drawer in _stratum_drawers(theta, train_prior, args):
        reps = (args.bootstrap_reps if name == "shifted"
                else args.supp_bootstrap_reps)
        cells = []
        print(f"[{dataset}] stratum {name} ({drawer.label})", flush=True)
        for m in grid:
            rng = np.random.default_rng(cell_seed(args.seed, name, m))
            pool, diag = run_cell(
                m, drawer, sampler, log_post, log_train_prior, train_prior,
                log_theta, log_p_theta, rng,
                budget=args.budget, n_min=args.n_min, n_max=args.n_max)
            # Full risk-coverage curves are kept only for the two strata that
            # get their own figure; storing them for all C+2 strata would
            # bloat results.json for no reader.
            cell = evaluate_pool(pool, coverage=args.coverage, reps=reps,
                                 rng=rng,
                                 keep_curves=name in ("shifted", "marginal"))
            entry = {
                "m": m, "n_trials": cell.n_trials, "budget": cell.budget,
                "scalars": cell.scalars, "ci": cell.ci,
                "curves": cell.curves, "coverage_grid": cell.coverage_grid,
                "diagnostics": asdict(diag),
            }
            cells.append(entry)
            s = cell.scalars
            print(f"   m={m:<4d} N={cell.n_trials:<5d} B={cell.budget:<5d} "
                  f"AuRC(bayes)={s['bayes_total']['aurc']:.4f} "
                  f"AuRegC(bayes)={s['bayes_total']['auregc']:+.4f} "
                  f"acc(bayes)={s['bayes_total']['accuracy_full']:.4f} "
                  f"dup={diag.dup_fraction:.3f}", flush=True)
        results["strata"][name] = {"label": drawer.label, "mode": drawer.mode,
                                   "bootstrap_reps": reps, "cells": cells}

    (out_dir / "results.json").write_text(
        json.dumps(_json_safe(results), indent=1, allow_nan=False))
    report = format_report(results)
    (out_dir / "report.txt").write_text(report)
    print("\n" + report)

    if not args.no_figures:
        from figures import write_all_figures
        paths = write_all_figures(results, out_dir)
        print("figures: " + ", ".join(p.name for p in paths))
    print(f"\noutputs in {out_dir}/")


# --- text report ----------------------------------------------------------

def _fmt_ci(cell: dict, key: str, scalar: str, digits: int = 4) -> str:
    v = cell["scalars"][key][scalar]
    if key in cell.get("ci", {}) and cell["ci"][key]:
        lo, hi = cell["ci"][key][scalar]
        return f"{v:+.{digits}f} [{lo:+.{digits}f},{hi:+.{digits}f}]"
    return f"{v:+.{digits}f}"


def format_report(res: dict) -> str:
    L = [
        "Exact Bayesian label-prior adaptation: reject-option evaluation",
        "=" * 78,
        f"timestamp : {res['timestamp']}",
        f"command   : {res['command']}",
        f"dataset   : {res['dataset']}   Y = {res['num_classes']}",
        f"eval split: {res['eval_size']:,} ({res['eval_desc']}), "
        f"base error {res['eval_error']:.4f}",
        f"grid      : {res['grid']}",
    ]
    if res["grid_note"]:
        L.append(f"  !! {res['grid_note']}")
    L += [
        f"budget    : B = {res['budget']}, N in [{res['n_min']}, {res['n_max']}], "
        f"coverage c = {res['coverage']}",
        "-" * 78,
        f"prior set Theta: C = {len(res['theta'])}",
    ]
    for i, (lab, tv) in enumerate(zip(res["theta_labels"],
                                      res["theta_tv_to_train"])):
        L.append(f"  theta[{i}]  TV to theta_tr = {tv:.4f}   {lab}")

    ident = res["identifiability"]
    L += [
        "",
        "identifiability of Theta from *unlabeled* data",
        "  Per-example drift of the log-likelihood ratio,",
        "  K[a,b] = E_{x~theta_a}[ log w(x,theta_a) - log w(x,theta_b) ].",
        "  Separating the closest pair needs ~1/K adaptation examples.",
        f"  min off-diagonal drift = {ident['min_offdiag_kl']:.3e}"
        + (f"  ->  ~{ident['m_for_separation']:.0f} examples"
           if ident["m_for_separation"] is not None
           and np.isfinite(ident["m_for_separation"]) else ""),
    ]
    grid_max = max(res["grid"])
    if ident["min_offdiag_kl"] <= 0:
        L.append(
            "  !! NEGATIVE drift: for some ordered pair the likelihood moves "
            "towards the WRONG prior.\n"
            "     With a perfectly calibrated p_tr(y|x) this quantity is a KL "
            "and cannot be negative,\n"
            "     so this says the calibration error outweighs the prior shift "
            "for that pair and the\n"
            "     MAP estimate converges to the wrong element of Theta. Check "
            "the ECE in the training report.")
    elif (ident["m_for_separation"] is None
          or ident["m_for_separation"] > grid_max):
        L.append(
            f"  !! that exceeds m_max = {grid_max}: the closest pair of priors "
            f"cannot be\n     separated anywhere on this grid, so flat curves "
            f"here are a property of\n     Theta, not of the method. See README "
            f"S7 'On comparability across datasets'.")

    for name, st in res["strata"].items():
        L += ["-" * 78,
              f"stratum '{name}' ({st['label']}), "
              f"{st['bootstrap_reps']} bootstrap replicates"]
        for scalar in ("aurc", "auregc", "regret_at_c", "accuracy_full"):
            lab = SCALAR_LABELS[scalar]
            if scalar == "regret_at_c":
                lab = f"Reg@{res['coverage']:g}"
            L.append(f"\n  {lab} vs m  (95% percentile bootstrap CI, "
                     f"trial as the resampling unit)")
            keys = [r["key"] for r in res["rejectors"]
                    if not (scalar in ("auregc", "regret_at_c")
                            and r["key"] == "true_plugin")]
            head = f"{'m':>5} {'N':>5} " + "".join(f"{k[:22]:>26}" for k in keys)
            L.append(head)
            for cell in st["cells"]:
                row = f"{cell['m']:>5} {cell['n_trials']:>5} "
                row += "".join(f"{_fmt_ci(cell, k, scalar):>26}" for k in keys)
                L.append(row)

        L.append("\n  diagnostics per m")
        L.append(f"{'m':>5} {'dup frac':>10} {'dup mean':>10} "
                 f"{'P(MAP=true)':>12} {'E[p(th*|D)]':>12} "
                 f"{'mean T':>9} {'mean A':>9} {'mean E':>9}")
        def _num(v, w=12):
            # theta_* off the grid (the dirichlet arm) has no index in Theta, so
            # P(MAP = true) and p(theta_* | D) are undefined there.
            return f"{'n/a':>{w}}" if v is None or (
                isinstance(v, float) and not math.isfinite(v)) else f"{v:>{w}.4f}"

        for cell in st["cells"]:
            d = cell["diagnostics"]
            L.append(f"{cell['m']:>5} {_num(d['dup_fraction'], 10)}"
                     f"{_num(d['dup_mean'], 11)}{_num(d['map_correct'], 13)}"
                     f"{_num(d['post_true'], 13)}{_num(d['mean_total'], 10)}"
                     f"{_num(d['mean_aleatoric'], 10)}"
                     f"{_num(d['mean_epistemic'], 10)}")
        worst = max(c["diagnostics"]["dup_fraction"] for c in st["cells"])
        if worst > 0.05:
            L.append(f"  !! realised duplicate rate reaches {worst:.3f} "
                     f"(S6.3 expects ~0.05); the query recurs inside its own\n"
                     f"     adaptation set that often, which the S6.2 assert on "
                     f"|eval| does not bound -- it is the\n"
                     f"     per-class count |eval_y| that matters.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
