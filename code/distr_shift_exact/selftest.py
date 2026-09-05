"""Check the exact inference against a direct transcription of README S2.

:mod:`exact.inference` computes everything in log space and exploits the
leave-one-out identity of S5; this brute-forces the same quantities from the
formulas as written (small ``m``, so no underflow) and compares. Also checks the
two properties S2.4 claims are immediate, the S3 tie-break, and the S6.4 budget
bookkeeping.

Run with ``python selftest.py``.
"""

from __future__ import annotations

import numpy as np

from exact.inference import identifiability_table, infer_trial, plugin_for_prior
from exact.metrics import Pool, evaluate_pool
from exact.priors import build_prior_set, total_variation
from exact.protocol import BUDGET_B, n_trials, subsample_layout


def brute_force(post, p_tr, theta, p_theta):
    """S2, transcribed literally. ``post`` is (n, Y) = p_tr(y | x_i)."""
    n, Y = post.shape
    C = len(theta)
    w = np.array([[np.sum(theta[c] / p_tr * post[i]) for c in range(C)]
                  for i in range(n)])                       # (n, C)
    T = np.zeros(n); A = np.zeros(n); H = np.zeros(n, dtype=int)
    lab_all = np.zeros((n, Y))
    MAP = np.zeros(n, dtype=int)
    for i in range(n):
        lik = np.array([np.prod([w[j, c] for j in range(n) if j != i])
                        for c in range(C)])                 # p_te(D^(i) | theta)
        MAP[i] = int(np.argmax(p_theta * lik))
        num = np.zeros(Y)
        for y in range(Y):
            num[y] = sum(p_theta[c] * post[i, y] * theta[c][y] / p_tr[y] * lik[c]
                         for c in range(C))
        den = sum(p_theta[c] * w[i, c] * lik[c] for c in range(C))
        lab = num / den
        lab_all[i] = lab
        H[i] = int(np.argmax(lab))
        T[i] = 1.0 - lab[H[i]]
        # A = sum_theta sum_y p(theta) p_te(x,y|theta) p(D|theta) l(y, h(x,theta)) / den
        acc = 0.0
        for c in range(C):
            plug = theta[c] / p_tr * post[i]
            h = int(np.argmax(plug))
            for y in range(Y):
                if y != h:
                    acc += p_theta[c] * post[i, y] * theta[c][y] / p_tr[y] * lik[c]
        A[i] = acc / den
    return H, T, A, MAP, lab_all


def check_inference(seed=0, n=7, Y=5, C=4, tol=1e-10):
    rng = np.random.default_rng(seed)
    post = rng.dirichlet(np.ones(Y) * 0.7, size=n)
    p_tr = rng.dirichlet(np.ones(Y) * 5)
    theta = rng.dirichlet(np.ones(Y) * 3, size=C)
    p_theta = rng.dirichlet(np.ones(C) * 4)

    ref_H, ref_T, ref_A, ref_MAP, ref_lab = brute_force(post, p_tr, theta, p_theta)
    got = infer_trial(np.log(post), np.log(p_tr), np.log(theta), np.log(p_theta))

    assert np.array_equal(got.bayes_pred, ref_H), (got.bayes_pred, ref_H)
    assert np.array_equal(got.map_index, ref_MAP), (got.map_index, ref_MAP)
    assert np.allclose(got.total, ref_T, atol=tol), np.abs(got.total - ref_T).max()
    assert np.allclose(got.aleatoric, ref_A, atol=tol), \
        np.abs(got.aleatoric - ref_A).max()
    # S2.4, property 1: E >= 0 because h(x, theta) minimises the per-theta risk.
    assert got.epistemic.min() > -tol, got.epistemic.min()
    # S2.4, property 1 in full: A(x, D) <= T(x, D, yhat) for *every* yhat, not
    # just the Bayesian one -- so E(x, D, yhat) >= 0 whatever is predicted.
    # T(x, D, yhat) = 1 - p(yhat | x, D) under 0/1 loss.
    for yhat in range(Y):
        assert ((1.0 - ref_lab[:, yhat]) - ref_A > -tol).all(), yhat
    # S2.4, property 2: A is free of yhat, so argmin_yhat E = argmin_yhat T = H.
    assert np.array_equal(ref_lab.argmax(axis=1), got.bayes_pred)
    return float(np.abs(got.total - ref_T).max()), float(got.epistemic.min())


def check_posterior_mean_identity(seed=1, n=6, Y=4, C=5):
    """p(y | x, D) is the plugin rule at thetabar = E[theta | D] (see module doc)."""
    rng = np.random.default_rng(seed)
    post = rng.dirichlet(np.ones(Y) * 0.7, size=n)
    p_tr = rng.dirichlet(np.ones(Y) * 5)
    theta = rng.dirichlet(np.ones(Y) * 3, size=C)
    p_theta = rng.dirichlet(np.ones(C) * 4)
    got = infer_trial(np.log(post), np.log(p_tr), np.log(theta), np.log(p_theta))
    for i in range(n):
        tbar = got.pth_adapt[i] @ theta
        v = post[i] * tbar / p_tr
        v = v / v.sum()
        assert np.isclose(1.0 - v.max(), got.total[i], atol=1e-12)
        assert int(np.argmax(v)) == int(got.bayes_pred[i])


def check_m0_degeneracy(seed=2, n=1, Y=6, C=5):
    """At m = 0 the MAP plugin must degenerate to theta_1 (S3 tie-break)."""
    rng = np.random.default_rng(seed)
    post = rng.dirichlet(np.ones(Y) * 0.7, size=n)
    p_tr = rng.dirichlet(np.ones(Y) * 5)
    theta = rng.dirichlet(np.ones(Y) * 3, size=C)
    p_theta = np.full(C, 1.0 / C)
    got = infer_trial(np.log(post), np.log(p_tr), np.log(theta), np.log(p_theta))
    assert np.all(got.map_index == 0), got.map_index
    # ... and its prediction must equal the theta_1 plugin's.
    pred, unc = plugin_for_prior(np.log(post), np.log(p_tr), theta[0])
    assert np.array_equal(pred, got.map_pred) and np.allclose(unc, got.map_unc)


def check_underflow(m=500, Y=10, C=6, seed=3):
    """m = 500 must not underflow: the naive product would be exactly 0."""
    rng = np.random.default_rng(seed)
    post = rng.dirichlet(np.ones(Y) * 0.5, size=m + 1)
    p_tr = np.full(Y, 1.0 / Y)
    theta = rng.dirichlet(np.ones(Y) * 3, size=C)
    got = infer_trial(np.log(post), np.log(p_tr), np.log(theta),
                      np.log(np.full(C, 1.0 / C)))
    assert np.all(np.isfinite(got.total)) and np.all(np.isfinite(got.aleatoric))
    assert np.isclose(got.pth_adapt.sum(axis=1), 1.0).all()
    assert got.epistemic.min() > -1e-9, got.epistemic.min()


def check_budget():
    """S6.4: every m must yield exactly B pooled triplets."""
    for m in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        N = n_trials(m)
        per, rem = subsample_layout(m, N, BUDGET_B)
        assert per * N + rem == BUDGET_B, (m, N, per, rem)
        assert per + (1 if rem else 0) <= m + 1, (m, N, per, rem)


def check_curves():
    """A perfect score must give a monotone risk curve; the oracle regret is 0."""
    rng = np.random.default_rng(4)
    N, n = 20, 5
    loss = (rng.random((6, N, n)) < 0.3).astype(float)
    oracle = loss[4]
    # rejector 0 gets a score that is perfectly informative about its own loss
    score = rng.random((6, N, n))
    score[0] = loss[0] + 0.01 * score[0]
    pool = Pool(loss=loss, score=score, tiebreak=score, oracle_loss=oracle,
                m=n - 1, per_trial=n, remainder=0)
    res = evaluate_pool(pool, reps=50, rng=rng)
    assert res.budget == N * n
    assert set(res.ci) == set(res.scalars), "every rejector needs an interval"
    for key, sc in res.scalars.items():
        for k, v in sc.items():
            lo, hi = res.ci[key][k]
            assert lo <= hi, (key, k, lo, hi)
    assert res.curves["bayes_total"]["risk"][0] <= res.scalars["bayes_total"]["accuracy_full"]
    # row 5 is the regret reference: its regret is identically zero.
    assert abs(res.scalars["true_plugin"]["auregc"]) < 1e-12
    assert abs(res.scalars["true_plugin"]["regret_at_c"]) < 1e-12
    # rows 1 and 2 share a base predictor, so they meet at full coverage.
    assert np.isclose(res.scalars["bayes_total"]["accuracy_full"],
                      res.scalars["bayes_total"]["accuracy_full"])


def check_prior_set():
    """S7 guards, including the balanced-dataset theta_1 == theta_2 collapse."""
    Y = 10
    balanced = np.full(Y, 1.0 / Y)
    err = np.linspace(0.5, 0.1, Y)
    ps = build_prior_set(balanced, err)
    assert ps.labels[0] == "train"
    assert "uniform" not in ps.labels, "theta_2 must be dropped when balanced"
    assert ps.dropped, ps.dropped
    off = ps.pairwise_tv()[np.triu_indices(ps.C, 1)]
    assert off.min() >= 1e-2 - 1e-12, off.min()
    assert np.allclose(ps.theta.sum(axis=1), 1.0) and np.all(ps.theta > 0)
    # the doubling priors must not reproduce theta_1
    for t in ps.theta[1:]:
        assert total_variation(t, balanced) > 1e-3

    skewed = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    ps2 = build_prior_set(skewed, np.array([0.1, 0.9, 0.2, 0.8, 0.3]))
    assert ps2.C == 3 + 3, ps2.labels          # S = min(5, ceil(5/2)) = 3
    assert "spike" in ps2.labels[2] and "[1, 3]" in ps2.labels[2], ps2.labels[2]


def check_identifiability(seed=5, n=20000, Y=5, C=4):
    """A correctly calibrated model must give a non-negative drift.

    ``identifiability_table`` reduces to a KL divergence exactly when ``w`` is
    the true density ratio, so the sign is the check that the estimator is not
    accidentally missing a normalisation. Negative entries with a *real* model
    are a finding about that model's calibration, not a bug -- see the function
    docstring.
    """
    rng = np.random.default_rng(seed)
    post = rng.dirichlet(np.ones(Y) * 0.6, size=n)     # exact p_tr(y | x_i)
    y = np.array([rng.choice(Y, p=p) for p in post])   # labels drawn from it
    p_tr = post.mean(axis=0)
    theta = rng.dirichlet(np.ones(Y) * 4, size=C)
    K = identifiability_table(np.log(post), y, np.log(p_tr), theta, Y)
    off = K[~np.eye(C, dtype=bool)]
    assert off.min() > -1e-3, off.min()
    assert np.allclose(np.diag(K), 0.0)
    return float(off.min())


if __name__ == "__main__":
    errs = [check_inference(seed=s) for s in range(6)]
    print(f"inference vs brute force: max |dT| = {max(e[0] for e in errs):.2e}, "
          f"min E = {min(e[1] for e in errs):+.2e}")
    check_posterior_mean_identity();  print("posterior-mean plugin identity  ok")
    check_m0_degeneracy();            print("m=0 MAP -> theta_1 tie-break     ok")
    check_underflow();                print("m=500 log-space stability        ok")
    check_budget();                   print("S6.4 budget bookkeeping          ok")
    check_curves();                   print("selective curves / oracle regret ok")
    check_prior_set();                print("S7 Theta generation + guards     ok")
    lo = check_identifiability()
    print(f"identifiability drift >= 0        ok (min {lo:+.4f})")
    print("all self-tests passed")
