# Sampled target priors: evaluate the reject-option experiment over a Dirichlet prior distribution

## Goal

Extend `run_real_reject_option_exp.py` so that, instead of a single fixed
target test prior, the experiment is repeated over target priors **sampled
from a Dirichlet distribution centered on the configured target prior**. With
the model given the same Dirichlet, the Bayesian prior-learning machinery is
*well specified*, which makes this the definitive test of whether the
epistemic uncertainty is calibrated (its average should match the average
realized regret — the property that fails under the fixed, misspecified
prior).

## Command-line interface

- `--dirichlet SUM_PARAMS` (float, > 0) — enables *dirichlet mode*. The
  target prior of each repetition is sampled from

  $$p(\alpha) \propto \prod_{y=1}^{Y} \alpha_y^{\beta_y - 1},
  \qquad \beta_y = s \, p_y,$$

  where `s = SUM_PARAMS` is the total concentration and `p` is the central
  target prior built by the existing arguments (`--pair-ratio`,
  `--confusable-pair`, `--pair-rest-ratio`, or `--test-prior`). Larger `s`
  concentrates the draws around `p`; `s -> inf` recovers the current
  fixed-prior experiment.
- `--trials-prior N` (int, default 5) — number of sampled priors. Each
  sampled prior runs the full existing trial loop (`--trials`, default 10),
  so the total number of runs is `N * T`. Passing `--trials-prior` without
  `--dirichlet` is an error (exit with a message).
- `--n-eval` — unchanged semantics; **in dirichlet mode only**, its default
  changes from the auto-max to a fixed `1000` (the auto-max is undefined
  when the prior varies per draw). The evaluation set is drawn **with
  replacement** where a class's pool is too small.
- `--beta` — unchanged: overrides the **model** prior (the Dirichlet used by
  the methods) with a symmetric per-class concentration. In dirichlet mode
  the data are still generated from `Dir(s p)`, so passing `--beta` makes
  the model prior *deliberately misspecified* — this is the intended use
  (misspecification control), and the report must state it (see Reporting).
- Without `--dirichlet`, the script behaves exactly as it does now.

Validation: in dirichlet mode, every class must have positive central mass
(`p_y > 0` for all y), otherwise `Dir(s p)` is improper — exit with an error
pointing at the offending arguments (`--pair-rest-ratio A 0`, zeros in
`--test-prior`).

## Generative protocol (dirichlet mode)

For each prior repetition `i = 1..N`:

1. Sample the target prior `alpha_i ~ Dir(s p)`.
2. Run the existing trial loop (`t = 1..T`) unchanged, with `alpha_i` in the
   role the fixed target prior plays today. In particular, **everywhere the
   current implementation uses the true test prior — the reference plugin,
   the regret definition, the oracle plugin accuracy — dirichlet mode uses
   the per-draw `alpha_i`.**

Pool split per trial (adaptation-first, as now), with two dirichlet-mode
amendments:

- **Evaluation set**: fixed size `--n-eval`, sampled at `alpha_i` from the
  remainder, with replacement where a class falls short. Replacement here is
  expected, not an anomaly: the current "resampled WITH replacement" warning
  becomes a plain report note in dirichlet mode.
- **Adaptation set**: replacement is **forbidden**. Per-class requests are
  truncated to the class's pool availability, so the realized adaptation set
  may be smaller than requested (`n_realized <= n`). In sweep mode the
  nested-prefix pool is built from the truncated draw, and a requested size
  exceeding the pool length is capped to it: the figures keep the nominal
  `n` on the x-axis, and the report lists the mean realized `n` per size
  whenever any truncation occurred. (Consequence to document in the report,
  not to "fix": truncation censors over-demanded classes, so for small `s`
  the adaptation sample no longer follows `alpha_i` exactly and
  well-specifiedness is approximate — visible as honest deviations in the
  calibration figure, not a bug.)

The startup feasibility checks that currently `sys.exit` when the fixed
prior exhausts a class become per-draw and non-fatal in dirichlet mode
(truncation and replacement are the defined handling).

## Methods under test

Unchanged set. The prior-dependent pieces are aligned as follows:

- **Bayesian methods** (learned-prior base predictor; total- and
  epistemic-uncertainty reject options): the MCMC prior is `beta = s p` —
  identical to the generator, hence well specified — unless `--beta`
  overrides it (misspecification control).
- **Supervised-prior baseline**: uses the Dirichlet **posterior mean** given
  the observed adaptation labels,
  $\hat p_y = (\text{count}_y + \beta_y) / (n + \sum_y \beta_y)$,
  with the *same* `beta` the Bayesian methods use (including a `--beta`
  override) — both method families always exploit identical prior
  information.
- Plugin with the training prior and plugin with the true (per-draw) prior:
  unchanged.

## Aggregation and error bars

- **Means**: pooled over all `N * T` runs (equivalently: mean of the N
  per-prior means, since T is constant), at every sweep size.
- **Error bars**: the **standard deviation of the N per-prior means** — they
  show variability across sampled priors, which dominates the total
  variance. Figure legends must say so explicitly, e.g.
  "mean over N x T runs; band = +-1 std over N sampled priors" — not
  "s.e.m.".

## Figures and reporting

- All existing figures are produced as now, with the aggregation above and
  corrected titles/legends (`N` priors x `T` trials).
- **New figure, new file** `epi_vs_regret_calibration.png`: average
  epistemic uncertainty (x-axis) versus average realized regret (y-axis) of
  the Bayesian predictor at full coverage. **One point per (prior draw,
  sweep size)**: the pair of means over that draw's T trials, colored/marked
  by the sweep size `n`, with the identity line `y = x` drawn. Under a well
  specified prior the points should lie on the diagonal; under `--beta`
  misspecification or heavy truncation they will depart from it. (In
  non-sweep mode: one point per prior draw.) This is distinct from the
  existing `epistemic_metrics_vs_n_test.png`, which plots both quantities
  against `n` but never against each other.
- **Report**: record `sum_params`, the central prior `p`, the model prior
  actually used (and a prominent "MODEL PRIOR MISSPECIFIED via --beta" line
  when overridden), and per-draw lines with the sampled `alpha_i` (at least
  its values on the confusable pair plus min/max elsewhere). Write the full
  sampled priors to `sampled_priors.txt` in the output directory — without
  them an unlucky draw is indistinguishable from a bug.

## Reproducibility

Seeding hierarchy from the single `--seed`: the master RNG draws N
per-prior seeds; each per-prior RNG samples `alpha_i` and then draws its T
trial seeds. Any single (draw, trial) is thus replayable from the recorded
seeds without rerunning the rest.

## Acceptance checks

1. Without `--dirichlet`: byte-identical behavior to the current script
   (same seeds, same outputs).
2. `--dirichlet` with very large `s` (e.g. 1e6): results statistically
   indistinguishable from the fixed-prior run at the same `p`.
3. Default well-specified run: points of `epi_vs_regret_calibration.png`
   scatter around the diagonal, tightening as `n` grows.
4. Same run with `--beta 1`: points shift systematically below the diagonal
   (epistemic underestimates regret), reproducing the known misspecified
   behavior.
5. `--trials-prior` without `--dirichlet`, `s <= 0`, or a zero-mass class in
   `p` with `--dirichlet`: clean error exits.
