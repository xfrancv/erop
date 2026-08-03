# Plan

Make `rejopt_eval.py` **adapt and evaluate on the same examples**: the set of
`n` unlabeled inputs the MCMC learns the test prior from *is* the set every
predictor is scored on. This replaces the current adaptation-first split
(`tasks/adaptation_eval_split.md`), which draws the adaptation set first and
scores on the disjoint remainder.

Supersedes `tasks/adaptation_eval_split.md` for the split itself; that spec's
`resample_to_prior` empty-pool guard and its stratified adaptation draw stay.

# Motivation

1. **It is the deployment setting.** Label-prior adaptation is transductive: a
   batch of unlabeled inputs arrives, the prior is learned *from that batch*,
   and the same batch is what has to be classified. The disjoint split measures
   something nobody does in practice.
2. **It makes the epistemic-uncertainty claim exact.** The epistemic term is by
   construction the *posterior-expected regret on the points it is computed
   for* (`bayesian_posterior_and_aleatoric` averages the corrected posterior
   over the α-samples at each scored input). With a disjoint evaluation set the
   posterior comes from one sample and the regret is realized on another, so
   `epi_vs_regret_calibration.png`'s `y = x` claim holds only up to that extra
   sampling gap. Merging removes the gap: the quantity predicted and the
   quantity realized live on the same points.
3. **It removes pool competition.** Adaptation and evaluation no longer bid for
   the same scarce high-target class, so the fatal "adaptation set exhausts
   class X" guard, `max_distinct_eval`, and the dirichlet-mode
   with-replacement evaluation draw all disappear (no duplicated evaluation
   examples anywhere).

# Context (current behaviour, for the implementer)

`split_adapt_eval` (`rejopt_eval.py`) draws `n_adapt = max(--sizes)` indices at
the target prior from the whole pool — per class, so stratified — gives the
disjoint remainder to evaluation, and draws `--n-eval` evaluation indices from
it. `--n-eval` defaults to the largest all-distinct evaluation set the
remainder supports (`max_distinct_eval`), or to a fixed 1000 with replacement in
dirichlet mode. `run_sweep` then draws the adaptation pool once per trial, uses
nested prefixes `pool_idx[:n]` for the swept sizes, and scores every size on the
one fixed evaluation set — which is why the true-prior and training-prior
plugins are computed once per trial, outside the size loop, and are flat in `n`.

**Note.** The one-line description of the current code as a "random 50/50
split" is stale — that was the behaviour *before*
`tasks/adaptation_eval_split.md`.

# Design

Per trial, from the whole pool of `N` labeled examples:

1. Draw the adaptation pool exactly as today:
   `pool_idx = resample_to_prior(all_idx, y, target_prior, n_max, rng, replace_short=adapt_replace)`.
2. For each swept size `n`, take the nested prefix `idx = pool_idx[:n]` and use
   it **as both the adaptation set and the evaluation set**: `P[idx]` feeds the
   MCMC, and every predictor is scored on `P[idx]` / `y[idx]`.

No `setdiff1d`, no second `resample_to_prior` call, no `n_eval`. The prefix
nesting is kept, so neighbouring sizes share examples and the sweep still
reflects `n` rather than resampling noise.

## What moves inside the size loop

With one evaluation set per trial, the reference and the two non-adaptive
plugins were loop-invariant. They no longer are — **the evaluation set now
changes with `n`** — so `post_ev`, `y_ev`, `h_true`, `losses_ref`, `acc_true`,
`h_train` and `acc_train` all move inside `for i, n in enumerate(sizes)`. The
comments and the `base_accuracy_panel` docstring claiming the true- and
training-prior plugins are "constant in n (flat curve)" become wrong and must
go: those curves now vary with `n` through the evaluation sample, which is a
result to show, not a bug to hide.

## Regret reference: keep the nominal target prior

`losses_ref` stays the plugin at the nominal target prior `α*` (the resampling
target, or the per-draw Dirichlet sample). Rationale: the reference must not
depend on the adaptation draw, or the methods would be scored against a moving
yardstick they partly control, and the numbers stay comparable with the
disjoint-split runs already in the paper.

**Consequence to document, not to suppress: regret can be negative.** A finite
batch of `n` examples has an empirical label distribution that differs from
`α*`, so the plugin at `α*` is *not* the loss-minimising prior-corrected
classifier on that batch. A method that infers the batch's own prior can beat
it. This is a real property of the transductive setting and arguably the
headline finding of the change; it is not a sign of a broken reference. Two
reporting requirements follow:

- Add a **transductive-floor diagnostic**: the plugin at the *empirical* prior
  of the evaluated batch (`np.bincount(y[idx]) / n`), reported at full coverage
  as mean regret vs. the `α*` reference. It quantifies how much of the observed
  (possibly negative) regret is finite-batch prior mismatch rather than method
  quality, and it bounds what any prior-adaptation method can achieve on that
  batch.
- Note in the report and the README that "coverage @ regret ≤ ε" now admits
  negative regret, so the metric is a budget against the *population* oracle,
  not against an attainable optimum.

`PREDICTOR_LABELS["plugin_supervised_prior"]` (documented in the module
docstring, currently **not computed** in the sweep path) must not be reinstated
naively: with merged sets the supervised prior counted from the adaptation
labels *is* the empirical prior of the evaluation batch, i.e. exactly the
diagnostic above. Reinstate it under that name or not at all.

## Ragged curves

`selective_curves` returns one point per evaluation example, so a curve at size
`n` has length `n`, and in dirichlet mode (`replace_short=False`) the realized
length can be shorter than `n` and differ across trials. The current
`(len(sizes), trials, n_eval)` arrays cannot hold this. Fix in two parts:

1. **Compute every scalar area per trial, inside `run_sweep`, from that trial's
   full realized curve**: AuRC (`.mean()`), AuRC50 (`truncated_area`) and AuGRC
   (`generalize_curve(...).mean(-1)`) join the existing
   `coverage_at_target`. They keep their `(len(sizes), trials)` shape and stay
   exact regardless of ragged lengths. `_sweep_outputs` then *receives* these
   dicts instead of deriving them from the stored curves.
2. **Keep the stored curves only for the per-size coverage-curve figures**, as
   a `dict[name] -> list over sizes` of `(trials, n_i)` arrays. Within a size,
   truncate all trials to the common minimum realized length before stacking,
   and report that length. (Only dirichlet-mode truncation can make it differ
   from `n`, and then only slightly.)

`make_curves_at_n_figure` / `make_gen_curves_at_n_figure` already take one size
at a time, so they need no change beyond the new indexing. The dirichlet
per-prior collapse (`risk_curves_d[n][:, j] = risk_curves[n].mean(axis=1)`)
becomes the same per-size list assignment.

## `--n-eval` and the pre-flight block

- **Retire `--n-eval`.** The evaluation size is `--sizes`. Make the flag an
  explicit error ("the evaluation set is the adaptation set; use --sizes")
  rather than silently ignored, so the pinned `run_rejopt_eval.sh`
  configurations and any old script fail loudly.
- Delete `max_distinct_eval` and the `eval_avail` / "exhausted class"
  feasibility block; the only remaining pool requirement is that
  `max(--sizes)` is drawable at the target prior. Keep the existing "class
  absent from the pool" fatal check, and keep the `short_adapt` "resampled WITH
  replacement" note (now the only shortfall channel — the `short_eval` set and
  the dirichlet "evaluation drawn WITH replacement" note both disappear).
- Replace the `n_eval_resolved` entry in the saved-args file with
  `eval_set : same as adaptation set (sizes)`, and drop `n_eval` from the two
  report header lines.

## Scope: a mode flag, not an outright replacement

Implement as `--eval-on-adapt`, **default off** (current disjoint behaviour), and
add the mode to `run_rejopt_eval.sh` alongside `nocalib` / `beta`, writing to
`runs/<ds>/transductive/`. Reason: every number in the paper configuration
changes under the merge, and the disjoint-vs-merged comparison is itself a
result worth having in the same run set. Flipping the default (or deleting the
disjoint path) is a separate decision to take after that comparison, not part of
this change. The ragged-curve refactor above is written so both modes share one
code path — the disjoint mode is simply the case where every size has the same
curve length.

# Consequences to expect and document

- **Error bars grow, and their meaning changes back.**
  `tasks/adaptation_eval_split.md` noted that with a near-pool-sized evaluation
  set the trial-to-trial spread is adaptation variance alone, so
  adaptation-independent baselines show near-zero spread. That reverses: the
  evaluation set is now `n` examples, so evaluation-sampling noise returns and
  dominates at small `n`, and the true-/training-prior plugins acquire visible
  spread. Update the README note rather than leaving the old interpretation
  standing.
- **The sweep confounds two effects.** AuRC vs. `m` now mixes "adaptation
  improves with `m`" and "evaluation gets less noisy with `m`". This is
  intrinsic to the transductive design (a deployed batch of 50 *is* estimated
  from and scored on 50 points), but it must be stated wherever the sweep
  figures are interpreted. Raising `--trials` averages the noise down without
  removing the confound.
- **Coverage granularity is `1/n`.** At `n = 50` the selective curves have 50
  ranks and the AuRC50 window ~25; the low-coverage tail is 0/1-grained. AuRC50
  and AuGRC are the robust readings at small `n`; say so.
- **Small `n` cannot cover many-class datasets.** At `n = 50` on CIFAR-100 most
  classes get `target_counts` zero, so the evaluation batch simply lacks them.
  That is the honest transductive answer, not a defect; report the realized
  number of represented classes per size instead of guarding against it.

# Edge cases

- **Dirichlet mode.** The adaptation draw keeps `replace_short=False`
  (truncation, no duplicated inputs double-counting evidence in the MCMC
  likelihood); the evaluation set inherits that truncation, so realized sizes
  are the existing `realized_n` and the existing "adaptation sets truncated to
  pool availability" note now covers evaluation too — reword it accordingly.
  The per-draw prior `α` is the reference for that draw, so the negative-regret
  discussion applies per draw.
- **Target prior with exact zeros.** Unchanged: those classes are drawn zero
  times and are absent from the (single) set.
- **`max(--sizes)` exceeding pool availability for a class.** Unchanged
  `short_adapt` handling: with-replacement in fixed-prior mode (now duplicating
  *evaluation* examples too, which inflates the apparent coverage — warn
  explicitly), truncation in dirichlet mode.

# Tasks

1. `run_sweep`: draw `pool_idx` once per trial; per size use `idx =
   pool_idx[:n]` for both the MCMC and the scoring; move the reference and the
   two plugin baselines inside the size loop; drop the "constant in n" comments.
   Retire `split_adapt_eval` (or reduce it to the single draw) and the
   `short_eval` return.
2. Move AuRC50 / AuGRC computation from `_sweep_outputs` into the trial loop and
   change the stored curves to per-size `(trials, n_i)` arrays truncated to the
   common realized length; update both callers (`run_sweep_report`,
   `run_dirichlet_sweep_report`).
3. Add the empirical-prior (transductive-floor) diagnostic row to the report.
4. Add `--eval-on-adapt`; error on `--n-eval` under it; delete
   `max_distinct_eval` and the eval-feasibility pre-flight block on that path;
   update the printed sizes line, the report headers and the saved-args entry.
5. Add the mode to `run_rejopt_eval.sh` (`transductive` -> `runs/<ds>/transductive/`)
   and to `run_all_rejopt_eval.sh`.
6. Verify: (a) adaptation and evaluation indices are identical per size; (b) the
   sweep runs on a many-class dataset (CIFAR-100) at the smallest size without
   shape or empty-class errors; (c) AuRC computed per trial equals the old
   path's value in disjoint mode (regression check on the refactor of task 2);
   (d) `epi_vs_regret_calibration.png` points sit closer to `y = x` than in the
   disjoint run, which is the claim in Motivation 2.
7. Update the module docstring (the split paragraph and the
   supervised-baseline bullet) and the README: the new mode, the retired
   `--n-eval` on that path, negative regret and the transductive floor, and the
   error-bar reinterpretation.
