"""Drawing label-shifted samples from a labeled pool.

The sampling layer of the reject-option experiment: how a target prior is built
from per-class weights, how many examples of each class an ``m``-sample draw at
a given prior wants, and how the labeled pool is split into the adaptation and
evaluation sets that ``prior_shift.sweep`` consumes.

Pure numpy: these functions raise on bad input rather than exiting, so the
command-line driver owns every error message the user sees.
"""

from __future__ import annotations

import numpy as np


def target_prior_from_weights(Y, classes, weights, rest_weight):
    """Target prior from a subset of classes, their weights, and a fallback.

    ``classes`` and ``weights`` are positionally aligned sequences of length
    ``k <= Y``; every class *not* named in ``classes`` gets ``rest_weight``
    individually (a per-class weight, not a total mass to share, so the
    unnamed classes are always equiprobable among themselves). The result is
    the normalised weight vector

        p(c) = w(c) / sum_c' w(c'),

    with ``w(c) = weights[i]`` where ``c == classes[i]``, else ``rest_weight``.
    With ``k == Y`` no class is unnamed and ``rest_weight`` is unused.

    Validation of the arguments happens at the call site (argument parsing),
    which owns the error messages; this only assumes them well formed.
    """
    w = np.full(Y, float(rest_weight if rest_weight is not None else 0.0))
    w[np.asarray(classes, dtype=int)] = np.asarray(weights, dtype=float)
    return w / w.sum()


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


def resample_to_prior(source_idx, labels, target_prior, m, rng,
                      replace_short=True):
    """Draw ``m`` indices from ``source_idx`` so labels follow ``target_prior``.

    Sampling is without replacement per class where the pool is large enough.
    A class whose request exceeds its pool is flagged in the returned ``short``
    set and either drawn WITH replacement (``replace_short=True``, the default)
    or truncated to the whole class pool (``replace_short=False`` -- the
    dirichlet-mode adaptation rule, where duplicated inputs would double-count
    evidence in the MCMC likelihood), in which case the returned index set is
    shorter than ``m``. A class the target wants but that has *no* examples in
    ``source_idx`` is skipped and named in the returned ``absent`` set instead
    of raising -- callers guard against this upstream, so in normal operation
    ``absent`` is empty.
    """
    Y = len(target_prior)
    counts = target_counts(m, target_prior)

    chosen, short, absent = [], set(), set()
    for c in range(Y):
        if counts[c] == 0:
            continue
        pool_c = source_idx[labels[source_idx] == c]
        if len(pool_c) == 0:
            absent.add(c)
            continue
        take = counts[c]
        replace = take > len(pool_c)
        if replace:
            short.add(c)
            if not replace_short:
                take, replace = len(pool_c), False
        chosen.append(rng.choice(pool_c, size=take, replace=replace))
    idx = (np.concatenate(chosen) if chosen
           else np.empty(0, dtype=np.asarray(source_idx).dtype))
    rng.shuffle(idx)
    return idx, short, absent


def max_distinct_eval(eval_avail, target_prior):
    """Largest all-distinct evaluation size at ``target_prior`` given per-class
    availability ``eval_avail``: ``floor(min_c eval_avail[c] / target[c])`` over
    classes with positive target mass (0 if there are none)."""
    tp = np.asarray(target_prior, dtype=float)
    caps = [eval_avail[c] / tp[c] for c in range(len(tp)) if tp[c] > 0]
    return int(np.floor(min(caps))) if caps else 0


def split_adapt_eval(all_idx, y, target_prior, n_adapt, n_eval, rng,
                     adapt_replace=True):
    """Adaptation-first stratified split of the whole pool (the default,
    disjoint evaluation design -- ``--eval-on-adapt`` draws no evaluation set of
    its own and does not call this).

    Draw ``n_adapt`` adaptation indices at the target prior from the whole pool
    (``resample_to_prior`` draws per class, so this is stratified by
    construction), give the disjoint remainder to evaluation, and draw
    ``n_eval`` evaluation indices from that remainder. With
    ``adapt_replace=False`` the adaptation draw is truncated to pool
    availability instead of resampling with replacement (dirichlet mode); the
    evaluation draw always allows replacement. Returns ``(adapt_idx, eval_idx,
    short_adapt, short_eval, absent)`` where the short sets name the classes
    that fell short on each side.
    """
    adapt_idx, short_a, absent_a = resample_to_prior(
        all_idx, y, target_prior, n_adapt, rng, replace_short=adapt_replace)
    eval_source = np.setdiff1d(all_idx, adapt_idx)
    eval_idx, short_e, absent_e = resample_to_prior(
        eval_source, y, target_prior, n_eval, rng)
    return adapt_idx, eval_idx, short_a, short_e, absent_a | absent_e


def sample_target_prior(rng, beta_gen, max_tries=100):
    """One target prior drawn from ``Dir(beta_gen)``, guarded against the
    numerical underflow of tiny concentrations (numpy returns NaN when every
    gamma draw underflows to zero); such draws are rejected and redrawn.

    Raises ``ValueError`` when no finite draw appears in ``max_tries``, which
    means the concentration is too small to sample from; the driver turns that
    into the ``--dirichlet`` error message.
    """
    for _ in range(max_tries):
        alpha = rng.dirichlet(beta_gen)
        if np.all(np.isfinite(alpha)) and abs(alpha.sum() - 1.0) < 1e-6:
            return alpha
    raise ValueError("could not draw a finite target prior from Dir(s * p); "
                     "--dirichlet is likely too small")
