"""Autonomous target test-prior selection for the real-data experiments.

Implements the search proposed in ``tasks/target_label_prior_selection.md``.
Given the calibrated pool posteriors ``P``, the pool labels ``y`` (label shift
is *simulated* on real data, so the pool labels are available to the
experiment designer -- this selects the benchmark, it is not part of the
method under test) and the training prior, the search picks a target prior
``alpha`` in the existing three-knob family (confusable pair, ``pair_ratio``,
``pair_rest_ratio`` -> :func:`default_target_prior`) such that

- **R1 (eval size)**: the auto evaluation size at ``alpha`` is at least
  ``n_min`` after the ``n_adapt`` adaptation examples are drawn first, with no
  with-replacement resampling -- guaranteed by per-class caps
  ``alpha[c] <= (N_c - 1) / (n_adapt + n_min)`` (Step 1);
- **R2 (separation)**: the evaluation set contains both prior-sensitive
  examples (decisions that flip when the pair split moves within its
  plausible range -> epistemic uncertainty and regret) and prior-insensitive
  hard examples (aleatoric decoys), so the total- and epistemic-uncertainty
  rankings actually differ (Steps 0, 2, 3).

The search itself is closed-form (no MCMC): pair identifiability is predicted
from the Fisher information of the mixture likelihood along the pair-split
direction (Step 0), and candidate priors are scored by the *flip test* --
plugin decisions under the candidate split vs. its mirrored counterpart
(Step 3). The final MCMC validation gate (Step 4) lives in
``run_real_reject_option_exp.py``, which owns the experiment machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .predictors import bayes_decision, corrected_posterior

IDENT_THRESHOLD = 3.0        # same "weakly identifiable" bar as MCMCResult
PAIR_RATIO_GRID = ((1.0, 3.0), (1.0, 5.0), (1.0, 7.0), (1.0, 9.0))
PAIR_REST_GRID = (None, (3.0, 7.0), (1.0, 1.0), (7.0, 3.0))
MIN_PAIR_MASS = 0.05         # discard candidates whose projection ate the pair
MIN_PAIR_SKEW = 2.0          # ... or flattened its asymmetry below 2:1


# --------------------------------------------------------------------------
# Pure target-prior helpers (moved from run_real_reject_option_exp.py so the
# search can use them without importing the experiment script).
# --------------------------------------------------------------------------

def default_target_prior(train_prior, pair_idx, pair_ratio=(1.0, 7.0),
                         pair_rest_ratio=None):
    """Target prior obtained by re-weighting the confusable pair.

    The confusable pair ``(i, j)`` is skewed within itself by ``pair_ratio``
    and, optionally, rescaled against the remaining classes by
    ``pair_rest_ratio``:

    - ``pair_rest_ratio=(a, b)`` sets the pair's total mass to ``a/(a+b)`` and
      the remaining classes' total to ``b/(a+b)``; the latter is distributed
      *proportionally to the training prior* over the non-pair classes.
    - ``pair_rest_ratio=None`` keeps the pair's training combined mass
      ``p_tr[i] + p_tr[j]``, which leaves every non-pair class at its training
      probability exactly -- reproducing the original behaviour.

    With ``pair_idx=None`` there is no pair to skew, so the target is the
    (normalised) training prior itself -- no shift.
    """
    q = np.array(train_prior, dtype=float)
    if pair_idx is None:
        return q / q.sum()
    i, j = pair_idx

    rest_mask = np.ones(len(q), dtype=bool)
    rest_mask[[i, j]] = False
    Q_rest = q[rest_mask].sum()

    if pair_rest_ratio is None:
        pair_total = q[i] + q[j]
    else:
        a, b = pair_rest_ratio
        pair_total = a / (a + b)
    # With no non-pair classes (pair covers everything) the rest has nowhere to
    # go, so the pair must hold all the mass.
    if Q_rest == 0:
        pair_total = 1.0
    rest_total = 1.0 - pair_total

    p = np.zeros_like(q)
    if rest_total > 0 and Q_rest > 0:
        p[rest_mask] = rest_total * q[rest_mask] / Q_rest
    r = np.array(pair_ratio, dtype=float)
    r = r / r.sum()
    p[i], p[j] = pair_total * r[0], pair_total * r[1]
    return p


def target_counts(m, target_prior):
    """Per-class integer counts for an ``m``-sample draw at ``target_prior``.

    Floor of ``m * target_prior``, with the rounding remainder handed to the
    largest fractional parts so the counts sum to exactly ``m``. Deterministic
    in ``(m, target_prior)`` -- used both to draw and to size the pools.
    """
    tp = np.asarray(target_prior, dtype=float)
    counts = np.floor(m * tp).astype(int)
    remainder = int(m - counts.sum())
    if remainder > 0:
        frac = m * tp - counts
        counts[np.argsort(-frac)[:remainder]] += 1
    return counts


def max_distinct_eval(eval_avail, target_prior):
    """Largest all-distinct evaluation size at ``target_prior`` given per-class
    availability ``eval_avail``: ``floor(min_c eval_avail[c] / target[c])`` over
    classes with positive target mass (0 if there are none)."""
    tp = np.asarray(target_prior, dtype=float)
    caps = [eval_avail[c] / tp[c] for c in range(len(tp)) if tp[c] > 0]
    return int(np.floor(min(caps))) if caps else 0


# --------------------------------------------------------------------------
# Step 1 -- feasible set for the eval-size guarantee (R1)
# --------------------------------------------------------------------------

def eval_size_caps(pool_counts, n_adapt: int, n_min: int) -> np.ndarray:
    """Per-class prior caps guaranteeing ``n_eval >= n_min``.

    Class ``c`` must supply ``~(n_adapt + n_eval) * alpha[c]`` distinct
    examples, so ``alpha[c] <= (N_c - 1) / (n_adapt + n_min)`` (the -1 absorbs
    the rounding of :func:`target_counts`). A class absent from the pool gets
    cap 0.
    """
    counts = np.asarray(pool_counts, dtype=float)
    return np.maximum(counts - 1.0, 0.0) / (n_adapt + n_min)


def project_to_caps(alpha: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Project a prior onto the box ``alpha <= caps`` on the simplex.

    Water-filling: clamp the violating classes to their caps and rescale the
    unclamped ones to restore the unit sum, repeating until a fixed point
    (rescaling can push a previously-fine class over its cap). Converges in at
    most ``Y`` passes. Requires ``caps.sum() >= 1``.
    """
    a = np.asarray(alpha, dtype=float).copy()
    a /= a.sum()
    caps = np.asarray(caps, dtype=float)
    if caps.sum() < 1.0:
        raise ValueError(
            f"infeasible eval-size guarantee: the per-class caps sum to "
            f"{caps.sum():.3f} < 1 -- the pool is too small for this "
            f"n_adapt + n_min; reduce the adaptation size or the eval floor")
    clamped = np.zeros(len(a), dtype=bool)
    for _ in range(len(a)):
        over = ~clamped & (a > caps + 1e-12)
        if not over.any():
            break
        a[over] = caps[over]
        clamped |= over
        free = 1.0 - a[clamped].sum()
        rest = ~clamped
        rest_sum = a[rest].sum()
        if rest_sum <= 0:
            a[rest] = free / max(rest.sum(), 1)
        else:
            a[rest] *= free / rest_sum
    return a


def guaranteed_n_eval(alpha, pool_counts, n_adapt: int) -> int:
    """Exact auto ``n_eval`` at ``alpha``: the adaptation counts are removed
    from the pool first, then the largest all-distinct evaluation size is
    computed -- the same arithmetic the experiment script uses."""
    adapt = target_counts(n_adapt, alpha)
    avail = np.maximum(0, np.asarray(pool_counts) - adapt)
    return max_distinct_eval(avail, alpha)


# --------------------------------------------------------------------------
# Step 0 -- predicted identifiability (no MCMC)
# --------------------------------------------------------------------------

def importance_weights(y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Per-example weights re-weighting the pool to the target prior ``alpha``.

    An evaluation set at ``alpha`` draws class ``c`` with probability
    ``alpha[c]``, uniformly within the class, so a pool example of class ``c``
    carries weight ``alpha[c] / N_c`` (normalised to sum 1).
    """
    counts = np.bincount(y, minlength=len(alpha)).astype(float)
    w = np.zeros(len(y))
    present = counts[y] > 0
    w[present] = alpha[y[present]] / counts[y[present]]
    s = w.sum()
    if s <= 0:
        raise ValueError("importance weights vanish: the target prior puts no "
                         "mass on any class present in the pool")
    return w / s


def pair_fisher_info(P, train_prior, alpha, weights) -> np.ndarray:
    """(Y, Y) matrix of Fisher informations along every pair-split direction.

    With ``R_y(x) = p_tr(y|x) / p_tr(y)`` and the mixture likelihood
    ``log p(x|alpha) = log sum_y R_y(x) alpha_y``, moving mass between classes
    ``i`` and ``j`` (``alpha(t) = alpha + t (e_i - e_j)``) has per-example
    score ``(R_i - R_j) / (R . alpha)``, so

        I_ij = E_w[ (R_i - R_j)^2 / (R . alpha)^2 ]
             = M_ii + M_jj - 2 M_ij,   M = G^T diag(w) G,  G = R / (R . alpha)

    where the expectation is the ``weights``-average over the pool.
    """
    R = np.asarray(P) / np.asarray(train_prior)[None, :]
    denom = R @ np.asarray(alpha)
    G = R / denom[:, None]
    M = (G * weights[:, None]).T @ G
    d = np.diag(M)
    return d[:, None] + d[None, :] - 2.0 * M


def predicted_ident_ratio(fisher_ij: float, alpha, pair) -> float:
    """Predicted MCMC ``ident_ratio`` for the pair split, closed form.

    The posterior std of the split from ``n`` unlabeled examples is
    ``~1/sqrt(n * I_ij)``; the label-counting std of ``alpha[c]`` is
    ``sqrt(alpha_c (1 - alpha_c) / n)`` (the benchmark
    ``MCMCResult.ident_ratio`` uses). Their ratio is ``n``-free:

        ratio(c) = 1 / sqrt(I_ij * alpha_c (1 - alpha_c)),  c in {i, j}

    and the pair's predicted ratio is the max over its two classes. Because
    ``1/(n I_ij)`` lower-bounds the directional variance ``d^T F^{-1} d / n``
    (Cauchy-Schwarz), this ratio is a *lower bound* on the observed one -- the
    gate ``ratio >= 3`` is conservative in the safe direction.
    """
    i, j = pair
    var = [alpha[c] * (1.0 - alpha[c]) for c in (i, j)]
    if fisher_ij <= 0:
        return np.inf
    return float(max(1.0 / np.sqrt(fisher_ij * v) if v > 0 else np.inf
                     for v in var))


def rank_pairs_by_identifiability(P, y, train_prior, alpha_ref,
                                  allowed_pairs) -> list[tuple[tuple[int, int], float]]:
    """Predicted ident ratio for every allowed pair at the reference prior,
    sorted least-identifiable (largest ratio) first."""
    w = importance_weights(y, alpha_ref)
    fisher = pair_fisher_info(P, train_prior, alpha_ref, w)
    scored = [(pair, predicted_ident_ratio(fisher[pair], alpha_ref, pair))
              for pair in allowed_pairs]
    return sorted(scored, key=lambda t: -t[1])


# --------------------------------------------------------------------------
# Step 3 -- the flip test (candidate scoring without MCMC)
# --------------------------------------------------------------------------

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    idx = int(np.searchsorted(cw, 0.5 * cw[-1]))
    return float(values[order[min(idx, len(values) - 1)]])


def flip_test_scores(P, y, train_prior, alpha, pair, loss,
                     risk_threshold: float = 0.25) -> dict:
    """Score a candidate prior by the flip test: ``E``, ``A``, ``G``, ``S``.

    When the pair split is weakly identifiable the MCMC posterior spreads
    roughly between the candidate split and its mirror (``alpha`` with the
    pair entries swapped). The flip set ``F`` holds the pool examples whose
    plugin decision differs between the two -- exactly the examples whose
    decision is prior-sensitive (epistemic uncertainty, and regret when the
    wrong split is used). The decoy set ``D`` holds the prior-insensitive
    examples whose committed-decision conditional risk at ``alpha`` exceeds
    ``risk_threshold`` (aleatoric uncertainty no split changes).

    All statistics are importance-weighted to ``alpha`` (an expectation over
    the evaluation set the experiment will draw):

    - ``E`` -- weighted fraction of ``F`` (epistemic mass);
    - ``A`` -- weighted fraction of ``D`` (aleatoric decoy mass);
    - ``G`` -- weighted expectation (over the whole eval distribution, so the
      flip mass counts) of the loss gap between the mirrored-split and
      candidate-split decisions on ``F`` (the expected full-coverage regret
      when the learned split lands on the wrong mode);
    - ``S`` -- weighted fraction of ``D`` whose conditional risk exceeds the
      weighted-median conditional risk in ``F`` (decoy strength: do the decoys
      out-rank the pair examples in total uncertainty, so the total and
      epistemic orderings actually cross?).
    """
    i, j = pair
    alpha = np.asarray(alpha, dtype=float)
    alpha_mirror = alpha.copy()
    alpha_mirror[i], alpha_mirror[j] = alpha[j], alpha[i]

    q = corrected_posterior(P, train_prior, alpha)
    h = bayes_decision(q, loss)
    h_mirror = bayes_decision(
        corrected_posterior(P, train_prior, alpha_mirror), loss)
    flip = h != h_mirror
    cond_risk = (q @ loss.T).min(axis=1)
    decoy = ~flip & (cond_risk > risk_threshold)

    w = importance_weights(y, alpha)
    E = float(w[flip].sum())
    A = float(w[decoy].sum())

    if flip.any():
        gap = loss[h_mirror[flip], y[flip]] - loss[h[flip], y[flip]]
        G = float(np.sum(w[flip] * gap))
        med = _weighted_median(cond_risk[flip], w[flip])
    else:
        G, med = 0.0, np.inf
    S = (float(w[decoy & (cond_risk > med)].sum() / A)
         if A > 0 and np.isfinite(med) else 0.0)
    return {"E": E, "A": A, "G": G, "S": S}


# --------------------------------------------------------------------------
# Steps 0-3 assembled: the grid search
# --------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """One feasible candidate prior with its flip-test scores."""
    alpha: np.ndarray
    pair: tuple[int, int]
    pair_ratio: tuple[float, float]
    pair_rest_ratio: tuple[float, float] | None
    n_eval: int                 # guaranteed auto n_eval at this prior
    ident_pred: float           # predicted ident ratio at this prior
    E: float = 0.0
    A: float = 0.0
    G: float = 0.0
    S: float = 0.0

    def describe(self) -> str:
        rest = ("keep" if self.pair_rest_ratio is None
                else f"{self.pair_rest_ratio[0]:g}:{self.pair_rest_ratio[1]:g}")
        return (f"pair=({self.pair[0]},{self.pair[1]}) "
                f"ratio={self.pair_ratio[0]:g}:{self.pair_ratio[1]:g} "
                f"rest={rest}")


@dataclass
class SearchResult:
    """Outcome of the target-prior search.

    ``ranked`` holds the surviving candidates best-first; empty when the
    Step-0 gate failed (``gate_passed False``) or no candidate met even the
    relaxed constraints. ``relaxed`` names the constraints that had to be
    dropped (``None`` when the full selection rule was satisfied).
    """
    ranked: list[ScoredCandidate]
    gate_passed: bool
    gate_table: list[tuple[tuple[int, int], float]]  # pairs by predicted ratio
    search_pair: tuple[int, int] | None
    relaxed: str | None
    recommended_n_test: int | None
    report_lines: list[str] = field(default_factory=list)


def recommended_n_test(fisher_ij: float, alpha, pair) -> int | None:
    """Largest adaptation size keeping the epistemic signal alive.

    The split posterior width from ``n`` unlabeled examples is
    ``~1/sqrt(n * I_ij)``; the mirrored split sits ``Delta = |alpha_i -
    alpha_j|`` away. Requiring the mirror to stay within ~2 posterior stds
    gives ``n <= 4 / (I_ij * Delta^2)``. ``None`` means unbounded (the split
    is exactly flat).
    """
    i, j = pair
    delta = abs(float(alpha[i]) - float(alpha[j]))
    if fisher_ij <= 0 or delta <= 0:
        return None
    return int(np.floor(4.0 / (fisher_ij * delta * delta)))


def _candidate_pairs(gate_table, registry_pair, top_k=3):
    """The registry pair plus the ``top_k`` least-identifiable pairs.

    The reference-prior ranking is only a *screening* -- the identifiability
    threshold is enforced per candidate, at the candidate's own prior, because
    the predicted ratio is strongly ``alpha``-dependent (a skewed prior can be
    weakly identified where the reference prior is not).
    """
    pairs = []
    if registry_pair is not None:
        reg = tuple(sorted(registry_pair))
        if reg in dict(gate_table):
            pairs.append(reg)
    for p, _r in gate_table[:top_k]:
        if p not in pairs:
            pairs.append(p)
    return pairs


def search_target_prior(
    P: np.ndarray,               # (N, Y) calibrated pool posteriors
    y: np.ndarray,               # (N,) pool labels
    train_prior: np.ndarray,     # (Y,)
    *,
    n_adapt: int,
    n_min: int = 300,
    loss: np.ndarray,
    registry_pair: tuple[int, int] | None = None,
    adjacent_pairs_only: bool = False,
    ident_threshold: float = IDENT_THRESHOLD,
    e_min: float = 0.05,
    a_min: float = 0.05,
    s_min: float = 0.5,
    risk_threshold: float = 0.25,
    class_names: list[str] | None = None,
) -> SearchResult:
    """Run Steps 0-3 of the target-prior search (closed form, no MCMC).

    Returns the ranked feasible candidates; the caller is expected to confirm
    the winner with the MCMC validation gate (Step 4) before using it.
    Raises :class:`ValueError` when the eval-size guarantee is infeasible for
    ``(n_adapt, n_min)`` on this pool.
    """
    Y = len(train_prior)
    pool_counts = np.bincount(y, minlength=Y)
    caps = eval_size_caps(pool_counts, n_adapt, n_min)
    if caps.sum() < 1.0:
        raise ValueError(
            f"pool of {len(y)} examples cannot guarantee n_eval >= {n_min} "
            f"with n_adapt = {n_adapt}: per-class caps sum to "
            f"{caps.sum():.3f} < 1; reduce --n-test/--sizes or N_min")

    def pname(c):
        return class_names[c] if class_names else str(c)

    lines = [f"target-prior search: n_adapt={n_adapt}, n_min={n_min}, "
             f"pool={len(y)}"]

    # ---- Step 0: gate ----------------------------------------------------
    alpha_ref = project_to_caps(np.asarray(train_prior, dtype=float), caps)
    usable = (pool_counts > 0) & (caps > 0)
    allowed = [(i, j) for i in range(Y) for j in range(i + 1, Y)
               if usable[i] and usable[j]
               and (not adjacent_pairs_only or j == i + 1)
               and caps[i] + caps[j] >= MIN_PAIR_MASS]
    if not allowed:
        raise ValueError("no admissible class pair (pool/caps too restrictive)")
    gate_table = rank_pairs_by_identifiability(
        P, y, train_prior, alpha_ref, allowed)
    top_pair, top_ratio = gate_table[0]
    lines.append(f"gate screening: least identifiable pair at the reference "
                 f"prior is {pname(top_pair[0])} / {pname(top_pair[1])}, "
                 f"predicted ident ratio {top_ratio:.1f} "
                 f"(threshold {ident_threshold:g}, applied per candidate)")
    if registry_pair is not None:
        reg = tuple(sorted(registry_pair))
        reg_ratio = dict(gate_table).get(reg)
        if reg_ratio is not None:
            lines.append(f"gate screening: registry pair {pname(reg[0])} / "
                         f"{pname(reg[1])} predicted ratio {reg_ratio:.1f}")

    pairs = _candidate_pairs(gate_table, registry_pair)

    # ---- Steps 1-3: grid, projection, scoring ----------------------------
    # The identifiability gate is enforced per candidate: the predicted ratio
    # is strongly alpha-dependent, so a pair that looks identified at the
    # reference prior can be weakly identified at a skewed feasible one.
    candidates: list[ScoredCandidate] = []
    best_ident = 0.0
    seen: set[tuple] = set()
    for pair in pairs:
        i, j = pair
        for base_ratio in PAIR_RATIO_GRID:
            # Give the small share to the smaller-cap class so the skew never
            # fights the caps.
            ratio = base_ratio if caps[i] <= caps[j] else base_ratio[::-1]
            for rest in PAIR_REST_GRID:
                raw = default_target_prior(train_prior, pair, ratio, rest)
                alpha = project_to_caps(raw, caps)
                key = tuple(np.round(alpha, 6))
                if key in seen:
                    continue
                seen.add(key)
                pair_mass = alpha[i] + alpha[j]
                lo, hi = sorted((alpha[i], alpha[j]))
                if pair_mass < MIN_PAIR_MASS or lo <= 0 or hi / lo < MIN_PAIR_SKEW:
                    continue           # projection destroyed the intent
                n_eval = guaranteed_n_eval(alpha, pool_counts, n_adapt)
                if n_eval < n_min:
                    continue           # exact post-check (rounding edge)
                w = importance_weights(y, alpha)
                fisher = pair_fisher_info(P, train_prior, alpha, w)[i, j]
                ident = predicted_ident_ratio(fisher, alpha, pair)
                best_ident = max(best_ident, ident)
                if ident < ident_threshold:
                    continue           # split identified -> no epistemic mass
                scores = flip_test_scores(P, y, train_prior, alpha, pair,
                                          loss, risk_threshold)
                candidates.append(ScoredCandidate(
                    alpha=alpha, pair=pair, pair_ratio=tuple(ratio),
                    pair_rest_ratio=rest, n_eval=n_eval, ident_pred=ident,
                    **scores))

    if not candidates:
        lines.append(
            f"gate FAILED: no feasible candidate prior is weakly "
            f"identifiable (best predicted ident ratio "
            f"{best_ident:.1f} < {ident_threshold:g}) -- the base model "
            f"resolves every pair, so no target prior can produce an "
            f"epistemic-vs-total contrast. Levers outside the prior: "
            f"smaller --n-test, or a lower-capacity / less-well-fit base "
            f"model.")
        return SearchResult(ranked=[], gate_passed=False,
                            gate_table=gate_table, search_pair=None,
                            relaxed=None, recommended_n_test=None,
                            report_lines=lines)

    # ---- selection rule --------------------------------------------------
    tiers = [
        (None, lambda c: c.E >= e_min and c.A >= a_min and c.S >= s_min),
        ("S", lambda c: c.E >= e_min and c.A >= a_min),
        ("S,A", lambda c: c.E >= e_min),
        ("S,A,E", lambda c: True),
    ]
    ranked, relaxed = [], None
    for name, keep in tiers:
        ranked = sorted((c for c in candidates if keep(c)), key=lambda c: -c.G)
        if ranked:
            relaxed = name
            break

    lines.append(f"candidates: {len(candidates)} feasible "
                 f"(grid {len(pairs)} pair(s) x {len(PAIR_RATIO_GRID)} ratios "
                 f"x {len(PAIR_REST_GRID)} rest splits, after projection, the "
                 f"n_eval >= {n_min} check and the ident >= "
                 f"{ident_threshold:g} gate)")
    if relaxed:
        lines.append(f"note: constraint(s) {relaxed} could not be met and "
                     f"were relaxed")
    header = (f"{'pair':>10}{'ratio':>8}{'rest':>8}{'n_eval':>8}"
              f"{'ident':>8}{'E':>8}{'A':>8}{'G':>9}{'S':>7}")
    lines.append(header)
    for c in ranked[:10]:
        rest = ("keep" if c.pair_rest_ratio is None
                else f"{c.pair_rest_ratio[0]:g}:{c.pair_rest_ratio[1]:g}")
        lines.append(
            f"{f'{c.pair[0]},{c.pair[1]}':>10}"
            f"{f'{c.pair_ratio[0]:g}:{c.pair_ratio[1]:g}':>8}{rest:>8}"
            f"{c.n_eval:>8}{c.ident_pred:>8.1f}{c.E:>8.3f}{c.A:>8.3f}"
            f"{c.G:>9.4f}{c.S:>7.2f}")

    rec = None
    if ranked:
        best = ranked[0]
        w = importance_weights(y, best.alpha)
        fisher = pair_fisher_info(P, train_prior, best.alpha, w)[best.pair]
        rec = recommended_n_test(fisher, best.alpha, best.pair)
        pretty = np.array2string(best.alpha, precision=3)
        lines.append(f"selected: {best.describe()} -> alpha = {pretty}")
        lines.append(f"recommended n_test <= "
                     f"{'unbounded' if rec is None else rec} "
                     f"(keeps the mirrored split within ~2 posterior stds)")

    return SearchResult(ranked=ranked, gate_passed=True,
                        gate_table=gate_table, search_pair=top_pair,
                        relaxed=relaxed, recommended_n_test=rec,
                        report_lines=lines)
