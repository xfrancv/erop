# Task: percentile uncertainty bands in the figures

## Goal

Add an optional `--percentile-band X` flag to `run_synth_reject_option_exp.py`
and `run_real_reject_option_exp.py`. When given, every figure that currently
shades a `mean ± s.e.m.` (or `mean ± std`) uncertainty band instead draws a
**central percentile band** and centers the solid curve on the **pointwise
median**. When the flag is absent the scripts behave exactly as today.

## Scope

- **Both reject-option scripts**, kept consistent (they share the figure
  builders and the band helper, so the change lives in one place).
- **Every band-carrying figure, in both single-size and `--sweep` modes**:
  risk/regret-coverage, generalized risk/regret-coverage, AuRC-vs-`n`,
  AuRC50-vs-`n`, AuGRC-vs-`n`, epistemic-metrics-vs-`n`, coverage-at-target,
  base-accuracy-vs-`n`, and the per-size coverage-curve figures.
- **Figures only.** The text tables and reports (`mean ± std`, the `avg` sweep
  row, etc.) are left on the mean and unchanged (see Caveats for why this split
  is deliberate and how the figures are labeled to avoid confusion).
- The flag is a pure *addition*. With it absent, the produced figures are
  numerically identical to today's.

## CLI

Add to both scripts' argument parsers:

```
--percentile-band X    float in [0, 100], optional (default: unset).
                       X is the central PERCENTAGE of replicate values the band
                       contains: e.g. 80 -> the 10th-90th percentile interval.
                       When set, the solid line is the pointwise median and the
                       band is that central interval. Omit for the default
                       mean +/- s.e.m. band.
```

- Validate `0 <= X <= 100`; error clearly otherwise.
- The setting is recorded in each script's saved `*_args.txt` like every other
  argument the mode reads.

## Definition

All quantities are computed **pointwise** along the same replicate axis the
current bands use (see "Interaction with existing aggregation"), exactly as the
mean/std are computed today — one value per x-position.

Let `v` be the replicate values at a given x. Write `X` for the flag value and
`lo = (100 - X) / 2`, `hi = (100 + X) / 2`.

- **Default (flag unset)** — unchanged:
  - center = `mean(v)`
  - band = `[mean(v) - h, mean(v) + h]`, where `h` is `std(v)` or
    `std(v)/sqrt(reps)` per the existing `sem`/`std` setting.
- **Percentile mode (flag set)**:
  - center = `median(v)` = `percentile(v, 50)`
  - band = `[percentile(v, lo), percentile(v, hi)]`

Use NumPy's default linear interpolation for `percentile`. The band is in
general **asymmetric** about the median, and always contains it by construction.

## Implementation approach (single shared code path)

The current call sites each compute `mean = arr.mean(axis)` and a half-width
`h = _band(arr, axis, reps)`, then `fill_between(x, mean - h, mean + h)`. Because
the percentile band is asymmetric and the center changes, replace that pair with
one helper that returns the center line and the **absolute** lower/upper edges:

```python
def _series(arr, axis, reps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(center, lower, upper) along the replicate ``axis``.
    Default: mean and mean-+/-(std or std/sqrt(reps)) per the sem/std setting.
    Percentile mode: median and the central [lo, hi] percentile interval."""
```

Each figure builder then does `center, lo, hi = _series(...)`, plots `center` as
the solid line, and `fill_between(x, lo, hi)`. This is the one code path both
scripts use, so no builder keeps its own mean/`_band` logic. `_band` may be kept
as the internal default-branch helper or folded into `_series`; either way there
must be a single implementation.

The percentile level is module state set once per run (like the existing
`_AGG_BAND`/`_AGG_DESC`), e.g. a `configure_percentile_band(x)` setter called
from each script's `main` after argument parsing. Absent the flag it stays
`None` and `_series` takes the default branch.

## Title / label text

Titles come from `_agg_desc` (today `"mean ± s.e.m., {reps} trials"`). In
percentile mode it must describe the actual content, e.g.
`"median, central {X:g}% band, {reps} trials"` (or, for the dirichlet mode, the
matching per-prior wording). No figure may keep the `"mean ± s.e.m."` caption
while drawing a median + percentile band.

## Interaction with existing aggregation

- The replicate axis and `reps` are unchanged: whatever axis the current bands
  reduce (trials in the fixed-prior mode, sampled priors in the real-data
  dirichlet mode, via `configure_aggregation`) is the axis the median and
  percentiles reduce.
- `--percentile-band` **overrides the band type** for that axis: it takes
  precedence over both the `sem` default and the dirichlet mode's `std` choice,
  replacing `mean +/- <spread>` with `median + central interval`. It does not
  change which axis is reduced or the `reps` count reported in titles.

## Edge cases and validation

- `X = 100` -> `[min, max]`; `X = 0` -> a zero-width band collapsed onto the
  median line (degenerate but valid). Both follow from the definition; no
  special-casing needed beyond the range check.
- Reject `X < 0` or `X > 100` at argument-parse time with a clear message.
- Few replicates (defaults: 20 trials synth, 10 real) make the median and the
  band coarse; this is inherent to percentiles of small samples, not a bug (see
  Caveats).

## Non-goals

- No change to any text table, report, or the `avg` sweep row — those stay
  mean-based.
- No change to default (flag-unset) behavior or output.
- No new figures, and no change to which axis is reduced or to `reps`.

## Caveats to record (spec + brief in-script/README note)

1. **Figures switch to the median; tables stay on the mean.** A
   `--percentile-band` run shows median-centered curves while the adjacent
   tables remain mean-centered. This is intentional (the tables are a separate,
   mean-based summary), and the figure titles name the center as the median so
   the two are not conflated.
2. **A percentile band answers a different question than s.e.m.** s.e.m. is the
   uncertainty *of the mean estimate* and shrinks as `reps` grows; a percentile
   band is the *dispersion of the replicate values* and does not shrink. The two
   band styles are not directly comparable; pick one per figure set.
3. **Small-sample coarseness.** With 10-20 replicates the "contains X% of
   values" property is only nominal and the band edges are noisy.

## Acceptance

- Running either script **without** `--percentile-band` produces figures
  numerically identical to today's (same center line, same band).
- With `--percentile-band 80`: every band figure's solid line is the pointwise
  median and its band is the pointwise 10th-90th percentile interval, in both
  single-size and `--sweep` modes and in both scripts.
- Figure titles in percentile mode read as median + central X%, never
  "mean ± s.e.m.".
- `X` outside `[0, 100]` is rejected at parse time; the chosen `X` appears in the
  saved `*_args.txt`.
- Both scripts share one `_series`/band code path; no builder reimplements it.
- Text tables and reports are unchanged in both modes.
