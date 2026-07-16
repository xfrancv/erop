# Plan

Extend `run_synth_reject_option_exp.py` and `run_real_reject_option_exp.py` with
generalized risk-coverage and generalized regret-coverage curves, their areas,
and the corresponding figures and report tables.

# Context

The reject-option predictors are currently evaluated with risk-coverage and
regret-coverage curves. In sweep mode these are written per adaptation-set size
to the `coverage_curves/` subdirectory; in non-sweep mode they are written as
`risk_coverage.png` and `regret_coverage.png` to the run root.

## Definitions

Rank the `n` evaluation examples by ascending uncertainty `u` and let $\pi$ be
that order. At rank `k` the predictor accepts the `k` least uncertain examples.

$$coverage(k)=\frac{k}{n},\qquad genrisk(k)=\frac{1}{n}\sum_{i=1}^k \ell(y_{\pi(i)}, h(x_{\pi(i)}))$$

$$genregret(k)=\frac{1}{n}\sum_{i=1}^k\Big(\ell(y_{\pi(i)}, h(x_{\pi(i)})) - \ell(y_{\pi(i)}, \hat h_{true\text{-}prior}(x_{\pi(i)}))\Big)$$

The generalized risk-coverage curve is the set of points
$((coverage(1),genrisk(1)),\ldots,(coverage(n),genrisk(n)))$; the generalized
regret-coverage curve is
$((coverage(1),genregret(1)),\ldots,(coverage(n),genregret(n)))$.

The existing curves normalize by `k` (the accepted count); the generalized ones
normalize by `n` (all evaluation examples). Equivalently,
$genrisk(k) = coverage(k)\cdot risk(k)$.

## Motivation

The generalized curves carry the same information as the existing ones: at a
fixed coverage the two differ by the factor `k/n`, which is common to all
predictors, so they rank the predictors identically. The payload is the area.

$$AuGRC = \operatorname{mean}_k genrisk(k) = \frac{1}{n^2}\sum_i (n-i+1)\,\ell_{(i)}$$

$$AuRC = \operatorname{mean}_k risk(k) = \frac{1}{n}\sum_i \Big(\sum_{k\ge i}\tfrac{1}{k}\Big)\ell_{(i)} \approx \frac{1}{n}\sum_i \ln(n/i)\,\ell_{(i)}$$

AuRC weights the top-ranked example by $H_n/n \approx \ln(n)/n$, inflating its
fair $1/n$ share by a factor of $\ln n$ (~8 at n~3000), with the inflation
decaying as $\ln(n/i)$ down the ranking. That is what makes AuRC sensitive to
the noisy low-coverage tail. AuGRC replaces it with bounded linear rank weights
$(n-i+1)/n^2$, making it an ordinary linear rank statistic.

**Scale constraint.** AuGRC's weights sum to $(n+1)/2n \approx 1/2$, so under
0/1 loss AuGRC lies in [0, ~0.5] while AuRC lies in [0, 1]. They are not on a
common scale and must never share an axis or a table column.

# Tasks

1. **Computation.** Compute `genrisk` / `genregret` alongside the existing
   curves for every predictor in `REJECT_LABELS`, in both scripts. Since
   `genrisk(k) = coverage(k) * risk(k)` exactly, this is a post-hoc transform of
   the arrays `selective_curves()` already returns -- no change to the trial
   loops, the MCMC, or the ranking. The shared machinery lives in
   `run_synth_reject_option_exp.py` and is imported by the real script; add it
   there.

2. **Sweep-mode curve figures.** One two-panel figure (generalized risk |
   generalized regret) per adaptation-set size, written to the existing
   `coverage_curves/` subdirectory as `gen_coverage_curves_n{n_test}.png`,
   mirroring `make_curves_at_n_figure()`.

3. **Non-sweep curve figures.** Generalized curves for the single configuration,
   written to the run root as `gen_risk_coverage.png` and
   `gen_regret_coverage.png`, mirroring `make_curve_figures()`'s existing
   one-file-per-metric layout.

4. **Areas vs. adaptation-set size.** AuGRC (risk) and AuGRC (regret) vs. `n` in
   their own two-panel figure, `gen_aurc_vs_n_test.png`. Do not add them to
   `aurc_vs_n_test.png`.

5. **Report tables.** Add AuGRC to both the sweep and non-sweep reports, in
   tables separate from the existing AuRC blocks (see the scale constraint).

**Out of scope.** `coverage_at_target` and the `--risk-target` /
`--regret-target` budgets keep operating on the selective curves only: those
budgets are per-accepted-example and would be a unit error against the
generalized curves.

**Note on the oracle.** The oracle baseline (`--optimal-rejection`, off by
default) uses a metric-specific ranking -- realized loss for the risk curve,
realized regret for the regret curve -- so $\pi$ above is per-curve for that
entry, not single. Its lower-envelope property is unaffected: dividing by fixed
`n` preserves prefix-sum minimality, so `oracle_curves()` needs no
re-derivation.
