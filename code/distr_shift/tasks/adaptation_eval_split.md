# Plan

Change how `run_real_reject_option_exp.py` partitions the labeled pool into the
**adaptation** set (used to learn/estimate the test prior) and the
**evaluation** set (used to score every predictor). Replace the current random
50/50 half-split with a **stratified, adaptation-first split** that gives the
adaptation side only as many examples as it needs (`max(--sizes)` in sweep
mode, `--n-test` in single mode) and the **entire remainder** to evaluation, so
the evaluation set is as large as the pool and target prior allow. This reduces
the variance of the reported metrics, which is currently limited by a small
evaluation set.

# Context (current behaviour)

Both modes split the pool by a plain random permutation and then resample each
half to the target prior:

```python
perm = rng.permutation(N)
src_adapt, src_eval = perm[: N // 2], perm[N // 2:]
adapt_idx, _ = resample_to_prior(src_adapt, y, target_prior, n_test/n_max, rng)
eval_idx,  _ = resample_to_prior(src_eval,  y, target_prior, n_eval,       rng)
```

Two problems:

1. **The evaluation set is capped at ~half the pool**, wasting evaluation
   capacity whenever the adaptation size is far smaller than `N/2` (the common
   case: `n_max` of a few hundred vs. a pool of thousands). A small evaluation
   set inflates the variance / error bars of accuracy, AuRC, regret, and
   coverage.
2. **The half-split is not stratified** (despite a code comment in
   `run_real_trial` claiming "Class-stratified disjoint split"). A class that is
   rare in the pool can land entirely in one half, leaving the other half with
   zero examples of it; when the target prior then wants that class,
   `resample_to_prior` calls `rng.choice` on an empty array and raises
   `ValueError: a cannot be empty unless no samples are taken`.

# New split strategy

Per trial, from the whole pool of `N` labeled examples:

1. **Draw the adaptation set first, at the target prior, from the whole pool:**
   `adapt_idx = resample_to_prior(all_pool_idx, y, target_prior, n_adapt, rng)`,
   where `n_adapt = max(--sizes)` in sweep mode and `--n-test` in single mode.
   `resample_to_prior` already draws per class (it takes
   `round(n_adapt·target[c])` examples of each class `c`), so this draw is
   **stratified by construction** — every class contributes its target-prior
   share, and no class is stranded the way a random half-split can strand it.
2. **The evaluation source is the remainder:**
   `eval_source = setdiff(all_pool_idx, adapt_idx)` (disjoint from adaptation —
   no leakage).
3. **Draw the evaluation set from the remainder, at the target prior:**
   `eval_idx = resample_to_prior(eval_source, y, target_prior, n_eval, rng)`.

Sweep mode keeps its existing structure on top of this: the swept sizes are
nested prefixes of `adapt_idx`, all scored on the same `eval_idx`. The split is
still re-drawn every trial (fresh per-trial `rng`), so the adaptation draw — the
meaningful source of variance — varies across trials.

This replaces the random half-split in **both** `run_sweep` and
`run_real_trial`, and removes the inaccurate "Class-stratified" comment (the new
scheme actually is stratified).

# Evaluation-set size (`--n-eval` -> "max" by default)

The largest **all-distinct** evaluation set at the target prior is bounded by
the class with the most target mass per available example:

```
n_eval_max = floor( min_{c : target[c] > 0}  eval_avail[c] / target[c] )
```

where `eval_avail[c]` is the number of class-`c` examples left in
`eval_source` after the adaptation draw. Beyond `n_eval_max`, some class must be
drawn with replacement (duplicate evaluation examples), which is what should be
avoided.

- Make `--n-eval` **default to this maximum** (use a sentinel: `--n-eval`
  unset / `None` → resolve to `n_eval_max`). This is the variance-minimising
  choice and the point of the change.
- If `--n-eval` is given an explicit positive integer, honour it. If it exceeds
  `n_eval_max`, keep the current behaviour (with-replacement fallback, flagged
  by the existing "resampled WITH replacement" note) — do not silently cap.
- **Print the resolved evaluation size** (and note when it is the auto max), and
  record it in the saved-args file.

Note the adaptation and evaluation sets compete for the highest-target class, so
a large `n_adapt` shrinks `n_eval_max` (both need that scarce class). This is
correct and should just be reported, not worked around; the benefit is largest
when `n_adapt << N`.

# Robustness: empty-pool guard in `resample_to_prior`

Independently of the split change, `resample_to_prior` must not crash when a
wanted class has no available examples in the source. Currently:

```python
replace = counts[c] > len(pool_c)
chosen.append(rng.choice(pool_c, size=counts[c], replace=replace))  # crashes if pool_c empty
```

Change so that when `counts[c] > 0` and `len(pool_c) == 0`, the class is
**skipped with a clear warning naming the class** (e.g. add it to a returned
`absent` set, surfaced in the report as "target wants class X but the
{adaptation,evaluation} pool has none of it"), rather than raising the opaque
numpy `ValueError`. The returned index set is then slightly smaller than
requested; report the actual size. (With the new stratified adaptation-first
draw this should essentially never trigger for a present class, but the guard
makes any residual case — e.g. a class entirely absent from the pool, or a
`--test-prior`/`--confusable-pair` naming a class the pool lacks — a clear
message instead of a stack trace.)

# Edge cases

- **`n_adapt` consumes most/all of a scarce high-target class.** Then
  `eval_avail[c]` for that class is small or zero, so `n_eval_max` is small (or
  the guard skips the class for evaluation). This is a genuine pool limitation
  (the pool cannot support both a large adaptation draw and a large evaluation
  draw for the same scarce class); report it via `n_eval_max` and the
  warnings, do not hide it.
- **Target prior with exact zeros** (e.g. `--pair-rest-ratio 1 0`): classes with
  `target[c] == 0` are excluded from the `n_eval_max` min and drawn 0 times;
  unchanged behaviour, no special handling needed beyond skipping them in the
  min.
- **Single-size mode** uses `--n-test` as `n_adapt`; otherwise identical
  treatment (adaptation-first, remainder to evaluation, `n_eval` default max).

# Cross-trial interpretation (document, no code change)

Because the evaluation set is now most of the pool, consecutive trials re-score
nearly the same examples, so the evaluation-sampling component of trial-to-trial
variance nearly vanishes. The reported error bars then reflect **adaptation
variance** (how much the result depends on which adaptation draw was obtained)
rather than a mix of adaptation and evaluation-sampling noise. A visible
consequence: baselines that do not depend on the adaptation draw (the
true-prior oracle, the no-adaptation plugin) will show near-zero spread across
trials. This is expected and arguably desirable; note it in the README so the
tighter/near-zero bands are not mistaken for a bug.

# Tasks

1. In `run_sweep` and `run_real_trial`, replace the random half-split with the
   adaptation-first stratified split above; delete the inaccurate
   "Class-stratified" comment.
2. Add `n_eval` resolution to `n_eval_max` when `--n-eval` is unset; keep an
   explicit value if given; print and save the resolved evaluation size.
3. Add the empty-pool guard to `resample_to_prior` (skip + warn + report actual
   size) and thread the "absent class" reporting through the existing
   `short`/shortfall reporting.
4. Verify: (a) the evaluation set is all-distinct (no with-replacement) at the
   default `n_eval` on the shipped datasets; (b) disjointness of adaptation and
   evaluation indices; (c) the CIFAR-100 sweep command from the bug report runs
   without the `ValueError`; (d) error bars shrink versus the current 50/50
   split on a fixed dataset/seed.
5. Update the module docstring and the README real-data section: the new split,
   the `--n-eval` default-to-max behaviour, and the cross-trial interpretation
   note.

# Naming / defaults note

`--n-eval` keeps its name but changes its **default** from `2000` to "the
maximum all-distinct size at the target prior". If a fixed, reproducible
evaluation size across datasets is ever wanted, it can still be pinned with an
explicit `--n-eval N`. Flag this default change in the README so existing
scripts that relied on the 2000 default are aware.
