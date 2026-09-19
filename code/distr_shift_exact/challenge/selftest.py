#!/usr/bin/env python3
"""Checks that need no data and no network. Run before spending compute.

Three things are worth brute-forcing rather than trusting:

1. **The inference.** ``chal/inference.py`` computes ``H``, ``T``, ``A`` and
   ``E`` with chunked logsumexps over an ``(n, C, Y)`` array. Here the same
   quantities are recomputed from the definitions in S2 with explicit Python
   loops over ``theta`` and ``y``, on small random problems.
2. **The protocol bookkeeping.** ``N(m)``, the row counts of C4, and the
   guarantee that slots run ``0..m-1``.
3. **The metric.** ``score()`` against a direct implementation, including the
   tie-break, the coverage rounding, and the sign of the ranking.
4. **The hard variant.** The location linear program, the image
   transformation, the balanced batch dealing, and that ``chal/metric_hard.py``
   is still exactly what ``make_metric_notebook.py`` generates from
   ``chal/metric.py``.

    python selftest.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from chal.inference import batch_starts_from_ids, infer, plugin_for_prior
from chal.metric import score
from chal.predictors import rank_confidence
from chal.priors import pair_prior_set
from chal.protocol import SIZE_GRID, n_batches

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# --- 1. the inference, against the S2 definitions --------------------------

def brute_force(log_post, log_train_prior, log_theta, log_p_theta):
    """``H``, ``T``, ``A``, ``E``, MAP -- straight from S2, loops and all."""
    n, Y = log_post.shape
    C = len(log_theta)
    post = np.exp(log_post)
    theta = np.exp(log_theta)
    p_tr = np.exp(log_train_prior)
    p_theta = np.exp(log_p_theta)

    # w(x_i, theta_c) = sum_y (theta_cy / p_tr(y)) p_tr(y | x_i)     (S2.1)
    w = np.array([[sum(theta[c, y] / p_tr[y] * post[i, y] for y in range(Y))
                   for c in range(C)] for i in range(n)])
    # p_te(y | x_i, theta_c)
    plug = np.array([[[theta[c, y] / p_tr[y] * post[i, y] / w[i, c]
                       for y in range(Y)] for c in range(C)] for i in range(n)])

    # p(theta | D) ~ p(theta) prod_j w(x_j, theta)   -- the whole batch (C3.2)
    unnorm = np.array([p_theta[c] * np.prod(w[:, c]) for c in range(C)])
    pth = unnorm / unnorm.sum()

    label_post = np.array([[sum(pth[c] * plug[i, c, y] for c in range(C))
                            for y in range(Y)] for i in range(n)])
    H = label_post.argmax(axis=1)
    T = 1.0 - label_post[np.arange(n), H]

    h = plug.argmax(axis=2)                                    # (n, C)
    A = np.array([sum(pth[c] * (1.0 - plug[i, c, h[i, c]]) for c in range(C))
                  for i in range(n)])
    map_c = int(unnorm.argmax())
    map_pred = h[:, map_c]
    map_unc = np.array([1.0 - plug[i, map_c, h[i, map_c]] for i in range(n)])
    return H, T, A, T - A, map_pred, map_unc


def test_inference() -> None:
    print("inference (S2) against a brute-force implementation")
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(12):
        n = int(rng.integers(1, 9))
        Y = int(rng.integers(3, 7))
        C = int(rng.integers(2, 6))
        log_post = np.log(rng.dirichlet(np.ones(Y) * 0.7, size=n))
        p_tr = rng.dirichlet(np.ones(Y) * 3.0)
        theta = rng.dirichlet(np.ones(Y) * 0.8, size=C)
        log_p_theta = np.full(C, -np.log(C))

        got = infer(log_post, np.array([0]), np.log(p_tr), np.log(theta),
                    log_p_theta)
        H, T, A, E, mp, mu = brute_force(log_post, np.log(p_tr), np.log(theta),
                                         log_p_theta)
        worst = max(worst, np.abs(got.total - T).max(), np.abs(got.aleatoric - A).max())
        if not (np.array_equal(got.bayes_pred, H)
                and np.allclose(got.total, T) and np.allclose(got.aleatoric, A)
                and np.allclose(got.epistemic, E)
                and np.array_equal(got.map_pred, mp)
                and np.allclose(got.map_unc, mu)):
            check(f"trial {trial} (n={n}, Y={Y}, C={C})", False)
            return
    check("H, T, A, E and the MAP plugin on 12 random problems", True,
          f"max abs error {worst:.2e}")

    # E >= 0 always: h(x, theta) minimises the per-theta conditional risk (S2.4).
    rng = np.random.default_rng(7)
    log_post = np.log(rng.dirichlet(np.ones(5) * 0.4, size=200))
    p_tr = rng.dirichlet(np.ones(5) * 3.0)
    theta = rng.dirichlet(np.ones(5) * 0.5, size=6)
    out = infer(log_post, np.arange(0, 200, 10), np.log(p_tr), np.log(theta),
                np.full(6, -np.log(6)))
    check("E(x, D) >= 0 on 200 rows", out.epistemic.min() >= 0.0,
          f"min {out.epistemic.min():.2e}")
    check("T = A + E exactly", np.allclose(out.total, out.aleatoric + out.epistemic))

    # A batch of one, under a Theta of one prior, is the plugin for that prior.
    theta1 = rng.dirichlet(np.ones(5) * 2.0, size=1)
    single = infer(log_post, np.arange(200), np.log(p_tr), np.log(theta1),
                   np.zeros(1))
    pred, unc = plugin_for_prior(log_post, np.log(p_tr), theta1[0])
    check("C = 1 collapses to the plugin predictor",
          np.array_equal(single.bayes_pred, pred) and np.allclose(single.total, unc))
    check("C = 1 leaves no epistemic uncertainty",
          float(np.abs(single.epistemic).max()) < 1e-12)

    # The full-batch posterior does not depend on which row is called the query,
    # which is the identity C3.2 rests on: leave-one-out and full-batch give the
    # same H, T, A, E.
    perm = rng.permutation(10)
    a = infer(log_post[:10], np.array([0]), np.log(p_tr), np.log(theta),
              np.full(6, -np.log(6)))
    b = infer(log_post[:10][perm], np.array([0]), np.log(p_tr), np.log(theta),
              np.full(6, -np.log(6)))
    check("the batch posterior is permutation-invariant",
          np.allclose(a.total[perm], b.total) and np.allclose(a.pth, b.pth))


# --- 2. the protocol -------------------------------------------------------

def test_protocol() -> None:
    print("\nprotocol (C4)")
    counts = {m: n_batches(m) for m in SIZE_GRID}
    check("N(m) matches the C4 table",
          list(counts.values()) == [2000, 1000, 400, 200, 200, 200, 200],
          str(list(counts.values())))
    rows = {m: counts[m] * m for m in SIZE_GRID}
    check("B_m matches the C4 table",
          list(rows.values()) == [2000, 2000, 2000, 2000, 4000, 10000, 20000])
    check("4200 batches and 42 000 rows per usage",
          sum(counts.values()) == 4200 and sum(rows.values()) == 42000,
          f"{sum(counts.values())} batches, {sum(rows.values())} rows")

    from chal.generate import draw_rows
    ps = pair_prior_set(np.full(8, 1 / 8), 0.35)
    rng = np.random.default_rng(0)
    y_pool = np.repeat(np.arange(8), 200)
    drawn = draw_rows(y_pool, ps.theta, rng, 8, grid=(1, 2, 5, 10),
                      n_min=4, scale=8)
    ok = True
    for b in range(drawn.n_batches):
        s = drawn.slot[drawn.gen_batch == b]
        ok &= np.array_equal(s, np.arange(drawn.batch_m[b]))
    check("slots run 0..m-1 in every batch", bool(ok))
    check("max(slot) + 1 == m in every batch",
          all(drawn.slot[drawn.gen_batch == b].max() + 1 == drawn.batch_m[b]
              for b in range(drawn.n_batches)))
    starts = batch_starts_from_ids(drawn.gen_batch)
    check("batches are contiguous and sorted", len(starts) == drawn.n_batches)

    # theta_* is drawn uniformly over Theta: every element should appear.
    drawn2 = draw_rows(y_pool, ps.theta, np.random.default_rng(1), 8,
                       grid=(5,), n_min=400, scale=400)
    seen = np.bincount(drawn2.batch_theta, minlength=ps.C)
    check("theta_* covers all of Theta", seen.min() > 0, f"counts {seen.tolist()}")


# --- 3. the metric ---------------------------------------------------------

def naive_score(sol: pd.DataFrame, sub: pd.DataFrame, coverage: float) -> float:
    """The metric restated with Python loops, for comparison."""
    merged = sol.merge(sub, on="row_id")
    per_m = []
    for m in sorted(merged["m"].unique()):
        g = merged[merged["m"] == m]
        recs = sorted(((-r.confidence, r.row_id, r) for r in g.itertuples()),
                      key=lambda t: (t[0], t[1]))
        k = max(1, int(np.ceil(coverage * len(recs))))
        tot = 0.0
        for _, _, r in recs[:k]:
            tot += (r.pred != r.label) - (r.pred_ref != r.label)
        per_m.append(tot / k)
    return float(np.mean(per_m))


def test_metric() -> None:
    print("\nmetric (C6)")
    rng = np.random.default_rng(3)
    sizes = (1, 2, 5)
    n_per = 60
    rows = []
    for m in sizes:
        for i in range(n_per):
            rows.append({"row_id": len(rows), "id_test": len(rows) // m, "m": m,
                         "label": int(rng.integers(8)),
                         "pred_ref": int(rng.integers(8))})
    sol = pd.DataFrame(rows)
    sub = pd.DataFrame({"row_id": sol["row_id"],
                        "pred": rng.integers(0, 8, len(sol)),
                        "confidence": rng.random(len(sol))})
    got = score(sol.copy(), sub.copy(), "row_id", expected_sizes=sizes)
    want = naive_score(sol, sub, 0.8)
    check("score() matches a loop implementation", abs(got - want) < 1e-12,
          f"{got:+.6f} vs {want:+.6f}")

    # A submission that copies the reference has regret exactly zero, whatever
    # the confidences are. This is the true_plugin check, in miniature.
    sub_ref = sub.copy()
    sub_ref["pred"] = sol["pred_ref"]
    check("copying pred_ref scores exactly 0",
          score(sol.copy(), sub_ref.copy(), "row_id", expected_sizes=sizes) == 0.0)

    # Constant confidence must still be deterministic: ties break on row_id.
    flat = sub.copy()
    flat["confidence"] = 1.0
    a = score(sol.copy(), flat.copy(), "row_id", expected_sizes=sizes)
    b = score(sol.copy(), flat.sample(frac=1.0, random_state=1).copy(), "row_id",
              expected_sizes=sizes)
    check("constant confidence is scored deterministically", a == b, f"{a:+.6f}")

    # Higher confidence must mean "kept first": hiding the errors must help.
    wrong = (sub["pred"].to_numpy() != sol["label"].to_numpy())
    good = sub.copy()
    good["confidence"] = np.where(wrong, 0.0, 1.0)
    bad = sub.copy()
    bad["confidence"] = np.where(wrong, 1.0, 0.0)
    s_good = score(sol.copy(), good.copy(), "row_id", expected_sizes=sizes)
    s_bad = score(sol.copy(), bad.copy(), "row_id", expected_sizes=sizes)
    check("low confidence on errors scores better", s_good < s_bad,
          f"{s_good:+.6f} < {s_bad:+.6f}")

    # rank_confidence must reproduce the lexicographic order it was given.
    e = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2], size=50)
    t = rng.random(50)
    conf = rank_confidence(e, t)
    want_order = np.lexsort((t, e))
    check("rank_confidence reproduces (E, then T) ascending",
          np.array_equal(np.argsort(-conf, kind="stable"), want_order))


# --- 4. the hard variant ----------------------------------------------------

def test_hard() -> None:
    print("\nhard variant (tasks/hard_variant.md)")
    from pathlib import Path

    from chal.generate import draw_rows
    from chal.locations import assign_locations, plan_locations
    from chal.transform import transform
    from make_metric_notebook import hard_source

    # The location LP, on class counts shaped like TissueMNIST's but smaller.
    D = np.array([4800, 700, 530, 1390, 1060, 690, 3530, 2210])
    ps = pair_prior_set(np.full(8, 1 / 8), 0.35)
    plan = plan_locations(D, ps.theta, eps=0.01, n_min=500)
    check("every training image lands in exactly one location",
          np.array_equal(plan.counts.sum(axis=0), D))
    check("locations 0..7 honour n_min", plan.sizes[:-1].min() >= 500,
          f"sizes {plan.sizes.tolist()}")
    check("every prior clears the floor (up to rounding)",
          plan.priors.min() >= 0.01 - 2.0 / plan.sizes.min(),
          f"min entry {plan.priors.min():.4f}")
    check("the priors moved by at most delta (up to rounding)",
          plan.tv_change.max() <= plan.delta + 8.0 / plan.sizes[:-1].min(),
          f"max TV moved {plan.tv_change.max():.4f}, delta {plan.delta:.4f}")
    tighter = plan_locations(D, ps.theta, eps=0.01, n_min=300)
    check("a smaller n_min needs no larger a change",
          tighter.delta <= plan.delta + 1e-3,
          f"{tighter.delta:.4f} <= {plan.delta:.4f}")
    y = np.repeat(np.arange(8), D)
    loc = assign_locations(y, plan, np.random.default_rng(0))
    freq = np.stack([np.bincount(y[loc == l], minlength=8)
                     for l in range(plan.L)])
    check("class frequency per location is exactly the location's prior",
          np.allclose(freq / freq.sum(1, keepdims=True), plan.priors))

    # The transformation.
    rng = np.random.default_rng(1)
    X = rng.integers(0, 256, size=(300, 28, 28), dtype=np.uint8)
    Xt, k = transform(X, np.random.default_rng(2), noise_std=0.0)
    check("rotations are 90, 180 or 270 degrees, never 0",
          set(np.unique(k).tolist()) <= {1, 2, 3} and len(np.unique(k)) == 3)
    check("without noise, undoing the rotation restores the image",
          all(np.array_equal(np.rot90(Xt[i], -k[i]), X[i]) for i in range(300)))
    Xn, kn = transform(X, np.random.default_rng(2), noise_std=2.0)
    diff = np.stack([np.rot90(Xn[i], -kn[i]) for i in range(300)]).astype(int) - X
    check("noise is small, zero-mean and uint8",
          Xn.dtype == np.uint8 and abs(diff.mean()) < 0.05
          and 1.5 < diff.std() < 2.5,
          f"mean {diff.mean():+.3f}, std {diff.std():.3f}")
    Xa, _ = transform(X, np.random.default_rng(5), 2.0)
    Xb, _ = transform(X, np.random.default_rng(5), 2.0)
    check("the transformation is reproducible from its seed",
          np.array_equal(Xa, Xb))

    # Balanced dealing: every prior the same number of batches at every size.
    y_pool = np.repeat(np.arange(8), 200)
    drawn = draw_rows(y_pool, plan.priors, np.random.default_rng(3), 8,
                      grid=(1, 5, 10), n_min=20, scale=40, balanced=True)
    ok = all(len(set(np.bincount(drawn.batch_theta[drawn.batch_m == m],
                                 minlength=plan.L).tolist())) == 1
             for m in (1, 5, 10))
    check("balanced: each location the same number of batches per size", ok,
          f"{np.bincount(drawn.batch_theta, minlength=plan.L).tolist()}")

    # The hard metric is generated, never edited.
    here = Path(__file__).resolve().parent
    check("chal/metric_hard.py is exactly generated from chal/metric.py",
          (here / "chal" / "metric_hard.py").read_text()
          == hard_source((here / "chal" / "metric.py").read_text()),
          "regenerate: python make_metric_notebook.py --hard-module "
          "chal/metric_hard.py")
    from chal.metric_hard import score as score_hard
    rng = np.random.default_rng(4)
    sol = pd.DataFrame({"row_id": np.arange(90), "id_test": np.arange(90) // 3,
                        "m": np.repeat([1, 2, 5], 30),
                        "label": rng.integers(0, 8, 90),
                        "pred_ref": rng.integers(0, 8, 90)})
    sub = pd.DataFrame({"row_id": np.arange(90), "pred": rng.integers(0, 8, 90),
                        "confidence": rng.random(90)})
    check("the hard metric scores exactly as the easy one",
          score_hard(sol.copy(), sub.copy(), "row_id", expected_sizes=(1, 2, 5))
          == score(sol.copy(), sub.copy(), "row_id", expected_sizes=(1, 2, 5)))


def main() -> None:
    print("challenge self-test -- no data, no network\n")
    test_inference()
    test_protocol()
    test_metric()
    test_hard()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all self-tests passed")


if __name__ == "__main__":
    main()
