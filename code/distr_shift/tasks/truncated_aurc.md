# Plan

Extend `run_synth_reject_option_exp.py` and `run_real_reject_option_exp.py` with
the area under the risk-coverage and regret-coverage curves restricted to the
coverage range `[0.5, 1]`, its report tables, and a new sweep figure
`aurc50_vs_n_test.png`.

# Context

The reject-option predictors are summarized by the area under their
risk-coverage and regret-coverage curves, `AuRC = (1/n) sum_{k=1..n} metric(k)`,
computed as `risk.mean()` / `regret.mean()` over all ranks. In sweep mode the
areas are plotted against the adaptation-set size in `aurc_vs_n_test.png` (left
panel risk, right panel regret) and tabulated in the `AuRC (risk)` / `AuRC
(regret)` report blocks; in non-sweep mode they appear in the AuRC table and in
the curve-figure legends.

## Definition

Rank the `n` evaluation examples by ascending uncertainty `u`; at rank `k` the
predictor accepts the `k` least uncertain examples and `coverage(k) = k / n`.
Let `c_min = 0.5` and `k0 = ceil(c_min * n)` be the first rank with
`coverage(k) >= c_min`. Define

$$\text{AuRC50} = \frac{1}{n - k_0 + 1}\sum_{k=k_0}^{n} risk(k)$$

and likewise for the regret. This is the **mean of the curve over the retained
ranks**, not the integral $\int_{0.5}^{1} risk\,dc$ — the two differ by the
width of the window, i.e. the integral is half as large.

**Scale.** The mean is the right choice precisely because it keeps AuRC50 on the
existing AuRC scale ([0, 1] under 0/1 loss, and equal to the full AuRC when
`c_min -> 0`), so it is directly comparable with the AuRC it is meant to replace
and may share a table with it. Contrast the AuGRC, which is normalized by `n`
and must never share an axis or a column with the AuRC. The `50` in the name is
`100 * c_min`; "area" in `AuRC50` is inherited from `AuRC` and is loose — the
quantity is an average of the curve over a coverage window.

## Motivation

Two reasons, in order of strength:

1. **Coverage in `[0.5, 1]` is the practically useful operating regime.** A
   reject-option predictor that throws away more than half its inputs is not a
   deployment a reader cares about, so the summary should not average over it.

2. **The discarded region is where the estimates are noisy.** `risk(k)` at small
   `k` is an average of few examples and is 0/1-grained at the very top (one
   unlucky example makes `risk(1) = 1`).

Writing AuRC50 as a rank statistic shows exactly what the restriction buys:

$$\text{AuRC50} = \frac{1}{n-k_0+1}\sum_i \Big(\sum_{k=\max(i,k_0)}^{n}\tfrac{1}{k}\Big)\,\ell_{(i)}$$

For `i <= k0` the inner sum is $\sum_{k=k_0}^{n} 1/k \approx \ln(1/c_{min}) =
\ln 2$, **constant in `i`**; for `i > k0` it decays as $\ln(n/i)$ to 0. So
AuRC50 replaces the full AuRC's $\ln(n/i)$ weights — which inflate the
top-ranked example's fair `1/n` share by a factor of $\ln n$ (a measured 8.4x
between rank 1 and rank n/2 at n = 200) — with flat weights over the top half.
This is the same defect the AuGRC addresses with linear rank weights; AuRC50
addresses it by truncation instead, and unlike the AuGRC it stays on the AuRC
scale.

## Caveat to carry into the docstrings and the report

The flat weighting above is exactly an **invariance: AuRC50 does not change
under any re-ordering of the ranking within the top `k0` ranks.** Every retained
`risk(k)`, `k >= k0`, is a prefix mean containing all of the top half, so
permuting inside it changes no retained point. (Verified against
`selective_curves`: shuffling the scores within the top half moves the full AuRC
0.284 -> 0.258 and leaves AuRC50 bit-identical at 0.2685, for risk and regret
alike.)

This matters because every entry in `REJECT_LABELS` shares the same base
predictor `h_bayes` and differs *only* in the ranking score `u`, so all curves
meet at the same value at coverage 1. AuRC50 can therefore separate
`bayes_total` from `bayes_epistemic` only through which examples fall in the
top-half *set* and how the bottom half is ordered — it discards signal along
with noise, and the between-predictor gaps will compress relative to the full
AuRC. **A smaller AuRC50 gap does not mean the ranking rules became more
similar.** Say so where the numbers are read, so the compression is not
misinterpreted.

# Tasks

1. **Computation.** Add the shared machinery to `run_synth_reject_option_exp.py`
   (the real script imports from it), mirroring `generalize_curve`:

   - a module constant `MIN_COVERAGE = 0.5`;
   - `truncated_area(curve, min_coverage=MIN_COVERAGE)` returning the mean over
     ranks `k0 = ceil(min_coverage * n)` .. `n`, i.e. `curve[..., k0-1:].mean(-1)`.
     Ranks run along the **last** axis, as in `generalize_curve`, so one call
     handles a single curve, the non-sweep `(trials, n_eval)` stack and the
     sweep `(len(sizes), trials, n_eval)` stack alike. Retain at least one rank.

   Both scripts already keep the full per-rank curves (`risk_curves` /
   `regret_curves`), so this is a slice of arrays already in memory: no change to
   the trial loops, the MCMC, the ranking, or the runtime.

2. **Sweep figure.** `aurc50_vs_n_test.png` — two panels, left AuRC50 (risk),
   right AuRC50 (regret), vs. the adaptation-set size, in both scripts. Add a
   thin `make_trunc_sweep_figure()` wrapper mirroring `make_gen_sweep_figure()`;
   `make_sweep_figure()` is already parameterized by `metrics` and `fname`.
   Derive the filename from the constant — `f"aurc{round(100 * MIN_COVERAGE)}_vs_n_test"`
   — so the name cannot go stale if `MIN_COVERAGE` is edited. Axis labels /
   titles must name the window, e.g. `AuRC (selective risk, coverage >= 0.5)`.
   Do not add these areas to `aurc_vs_n_test.png`.

3. **Sweep report tables.** Two blocks, `AuRC50 (risk)` and `AuRC50 (regret)`,
   placed directly after the existing `AuRC (risk)` / `AuRC (regret)` blocks and
   before the AuGRC blocks, same `n_test` x predictor layout. Header note: same
   scale as the AuRC above; averaged over ranks with coverage >= 0.5 only. In
   the real script these go into `real_reject_option_sweep_report.txt` (and the
   printed copy); the synth script only prints its tables.

4. **Non-sweep report tables.** An `AuRC50 risk` / `AuRC50 regret` block in both
   scripts, mirroring the existing AuGRC block's placement and its
   mean ± std-over-trials formatting. Keep it a separate block rather than two
   more columns on the AuRC table — the rows are already at the 76-column ruler.

5. **Figure inventory.** Add `aurc50_vs_n_test.png` to the trailing
   "figures written to ..." prints in both scripts, and to the figure list in
   `README.md`.

**Out of scope.**

- **The AuGRC gets no truncated variant.** It already addresses the same
  low-coverage weighting defect via linear rank weights; a second variant of it
  would double the table columns for one purpose. `aurc_vs_n_test.png` and
  `gen_aurc_vs_n_test.png` stay exactly as they are.
- `coverage_at_target` and the `--risk-target` / `--regret-target` budgets are
  untouched.
- The coverage-curve figures (`risk_coverage.png`, `regret_coverage.png`,
  `coverage_curves/`) are untouched — no `c_min` marker line, no truncated axes.
  The curves keep showing the full coverage range; only the summary is
  restricted.
- No new CLI flag. `MIN_COVERAGE` is a module constant, which keeps the
  requested `aurc50_vs_n_test.png` filename truthful. If it should become a
  sweepable flag later, the filename derivation in task 2 already follows it.

**Note on the oracle.** The oracle baseline (`--optimal-rejection`, off by
default) minimizes the selective metric at *every* rank `k`, so it minimizes any
non-negative weighted average of those ranks — including the truncated one. Its
lower-envelope property carries over unchanged and `oracle_curves()` needs no
re-derivation, despite its metric-specific per-curve ranking.
