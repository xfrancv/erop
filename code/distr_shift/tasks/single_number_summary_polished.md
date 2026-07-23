# Task: summarize each sweep curve by a single number

## Goal

In `--sweep` mode, `run_synth_reject_option_exp.py` and
`run_real_reject_option_exp.py` report every performance measure as a *curve
over the adaptation-set size* `n` (one table row per size in `--sizes`).
Augment both scripts so that each such curve is also summarized by a **single
scalar**: the mean of the measure over the swept sizes. The summaries go **only
into the text tables** — no new figures, no change to any curve or figure.

## Scope

- **`--sweep` mode only.** The single-size (non-sweep) mode adapts from one
  `n_test`, so there is no size axis to average over; leave it unchanged.
- **Both reject-option scripts**, kept consistent with each other.
- The summary is a pure *addition* to the existing output. No existing number,
  table cell, ranking, figure, or `*_args.txt` record changes.

## Which tables get a summary

Apply the summary **uniformly to every per-size table** the sweep prints — i.e.
every table whose rows are indexed by `n_test`. Concretely this is, for each
predictor column:

- AuRC — selective **risk** and selective **regret**;
- AuRC50 (high-coverage-half average) — risk and regret;
- AuGRC (generalized-curve area) — risk and regret;
- coverage-at-target — every risk budget and every regret budget;
- the epistemic-uncertainty metrics (avg epistemic, avg regret, portion
  negligible).

Treating all per-size tables the same way avoids the question of which curves
"count" and keeps the two scripts mechanically identical. If a future table is
added with `n_test` rows, it should get the same summary row by the same
mechanism.

## Definition of the summary

For a fixed table cell `M[name][i]` — the measure for predictor `name` at the
`i`-th swept size, already averaged over trials as the tables do today — the
summary is the unweighted arithmetic mean over the `len(sizes)` swept sizes:

    summary[name] = mean_i  M[name][i]

Report it in the **same numeric format as that table's cells** (bare mean, same
decimals; no new `± std`). Do not weight by `n`, and do not integrate — this is
a mean over the sampled sizes, matching the plain reading of "average over the
adaptation-set sizes."

## Output format

Append one row to the bottom of each per-size table, in the same column layout
as the size rows, with the row label `avg` in the `n_test` column, e.g.:

```
AuRC (risk)
  n_test    warn   Bayesian, total unc.   Bayesian, epistemic    Plugin, sup. prior
      50    0.00                 0.0036                0.0226                0.0036
     ...
    2000    0.00                 0.0034                0.0602                0.0034
     avg    0.00                 0.0034                0.0405                0.0034
```

For tables that carry a `warn` column (the AuRC risk/regret tables), the `avg`
row also shows the mean of `warn` over sizes, so the reader can see how much of
the average was accumulated in weakly-identifiable regimes (see Caveats).

## Caveats to record (in the spec and, briefly, in-script/README)

These do not block the implementation but must be stated so the number is not
over-read:

1. **Grid-dependent.** The mean is over the *sampled* sizes, and `--sizes` is
   typically log-spaced, so the small-`n` region is weighted more heavily than
   large-`n`. The summary is therefore only comparable across runs that share
   the same `--sizes`. It is a convenience scalar, not a grid-invariant
   functional of the underlying curve.
2. **Hides the `n`-dependence.** Several of these curves change qualitatively
   with `n` (e.g. epistemic AuRC *degrades* as `n` grows; regret decays only
   past an identifiability transition). A single mean collapses that structure
   and must not be read as a substitute for the curve.
3. **Mixes trustworthy and untrustworthy sizes.** The average spans sizes on
   both sides of the identifiability warning; the reported mean `warn` is the
   flag for how much of it comes from the untrustworthy regime.
4. **Distinct from the single-size AuRC.** This averaged-over-sweep number is
   not the AuRC that the non-sweep mode prints at one size; the `avg` label and
   its position under the sweep table keep the two from being conflated.

## Non-goals

- No grand number across metrics of different scales (AuGRC is normalized by
  `n_eval`, AuRC50 is on the AuRC scale). Each curve is summarized on its own.
- No new figures and no annotation of existing figures.
- No change to single-size mode, to the accuracy tables' existing content, or
  to any ranking/decision logic.

## Acceptance

- Running either script with `--sweep` prints an `avg` row at the foot of every
  per-size table, matching that table's columns and formatting.
- The `avg` value for a predictor equals the plain mean of that column's
  per-size cells (verifiable by eye against the rows above).
- Non-sweep runs are byte-for-byte unchanged.
- Both scripts share the same helper / code path for producing the row.
- README's sweep sections note the summary row and its grid-dependence caveat.
