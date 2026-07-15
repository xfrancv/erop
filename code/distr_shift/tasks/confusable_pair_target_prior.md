# Plan

Extend `run_real_reject_option_exp.py` with two new command-line arguments that
give finer control over the simulated label shift (the target test prior): one
to choose *which* pair of classes is treated as confusable, and one to set how
much total probability mass the pair holds versus the remaining classes. The
existing `--pair-ratio` (the split *within* the pair) is kept unchanged.

# Context

The script simulates label shift by resampling the real labeled pool to a
*target prior*. Today that target is built by `default_target_prior`, which:

- takes the confusable pair from the dataset registry (`spec.confusable_pair`,
  mapped to class indices),
- keeps the pair's **combined** training mass `p_tr[i] + p_tr[j]` fixed,
- splits that mass between the two classes according to `--pair-ratio`,
- leaves every other class at its training-prior probability.

`--test-prior` (an explicit full Y-vector) bypasses this entirely and takes
precedence.

Two limitations motivate the change: the confusable pair is fixed to the
registry's hand-picked choice, and the pair's total mass is fixed to whatever
it was in training (so for datasets whose training pair ratio is already close
to the default split, the shift is nearly a no-op).

# New arguments

Recommended names (parallel to the existing `--pair-ratio`):

1. **`--confusable-pair I J`** — two integer class indices (0-based) naming the
   confusable pair, overriding the registry default. Example:
   `--confusable-pair 0 4` uses classes 0 and 4.
   - Default: the registry pair `spec.confusable_pair` mapped to indices, i.e.
     current behaviour.
   - Validation: exactly two integers, both in `[0, Y)`, and `I != J`; error
     with a clear message otherwise (e.g. reject 1-based labels or `I == J`).

2. **`--pair-rest-ratio A B`** — two non-negative floats setting the split
   between the pair's total mass and the remaining classes' total mass:
   `pair_total = A/(A+B)` (= `p[i] + p[j]`) and `rest_total = B/(A+B)`
   (= sum of the other `Y-2` classes). Example: `--pair-rest-ratio 1 0`
   puts all mass on the pair.
   - Default: **sentinel (unset)** meaning "keep the pair's *training* combined
     mass" (`pair_total = p_tr[i] + p_tr[j]`), i.e. current behaviour. This is
     dataset-dependent, so it cannot be a fixed numeric default — use `None`.
   - Validation: two non-negative floats with `A + B > 0`. `A == 0` or `B == 0`
     is allowed (see edge cases) but should emit a warning.

# Target-prior construction

Given training prior `p_tr` (length `Y`), pair indices `(i, j)`, the
within-pair ratio `--pair-ratio = (c, d)`, and the pair/rest ratio
`--pair-rest-ratio = (a, b)` or `None`:

1. **Pair total mass**
   - if `--pair-rest-ratio` is `None`: `pair_total = p_tr[i] + p_tr[j]`
   - else: `pair_total = a / (a + b)`
2. `rest_total = 1 - pair_total`
3. **Within the pair** (via `--pair-ratio`, normalised `r = (c, d)/(c+d)`):
   `p[i] = pair_total * r[0]`, `p[j] = pair_total * r[1]`
4. **Within the rest**, distribute `rest_total` **proportionally to the
   training prior** of the non-pair classes: for every `k` not in `{i, j}`,
   `p[k] = rest_total * p_tr[k] / Q_rest`, where `Q_rest = sum_{m != i,j} p_tr[m]`.
   (If `rest_total == 0`, all non-pair classes get 0.)

**Why proportional (not uniform):** it is the minimal-surprise shift — only the
pair-vs-rest balance and the within-pair split change; the relative shape of
the non-confusable classes is preserved. It also makes the new code a strict
generalisation of the current one: with `--pair-rest-ratio` unset,
`rest_total = 1 - (p_tr[i]+p_tr[j]) = Q_rest`, so `p[k] = p_tr[k]` exactly — the
non-pair classes keep their training probabilities, reproducing today's output
bit-for-bit.

The result sums to 1 by construction (`pair_total + rest_total = 1`, and the
rest terms sum to `rest_total`).

`default_target_prior` is restructured to take `pair_rest_ratio` in addition to
its current arguments; `--pair-ratio` continues to control step 3 only.

# Precedence and interaction

- `--test-prior` still overrides everything: when given, the `--confusable-pair`,
  `--pair-rest-ratio`, and `--pair-ratio` arguments are ignored (optionally warn
  if both `--test-prior` and any `--confuse*`/`--pair*` argument are supplied).
- The three knobs are orthogonal: `--confusable-pair` selects *which* two
  classes, `--pair-rest-ratio` sets *pair-total vs rest-total*, `--pair-ratio`
  sets the *within-pair* split.
- If the dataset has no registry confusable pair **and** `--confusable-pair` is
  not given, keep current behaviour (no shift / target = training prior). In
  that case supplying `--pair-rest-ratio` is contradictory (no pair to assign
  mass to) — error with a clear message.

# Edge cases

- `--pair-rest-ratio 1 0` → `rest_total = 0`: every non-pair class gets
  probability 0. Numerically safe (the target prior is only used to resample
  the eval/adapt sets and to build the true-prior oracle and regret reference;
  it is never logged or fed to the MCMC). **But** it removes all decoy classes,
  so the eval set contains only the two confusable classes — a degenerate case
  for the epistemic-vs-total reject-option contrast (no regret-free hard
  decoys). Emit a warning.
- `--pair-rest-ratio 0 1` → `pair_total = 0`: the confusable pair vanishes from
  the eval set. Also degenerate; warn.
- Concentrating mass on classes the pool has few of forces heavy
  with-replacement resampling; this is already detected and reported via the
  existing `short` mechanism ("classes resampled WITH replacement"), so no new
  handling is needed beyond keeping that note.

# Reporting and saved-args fixes (required)

`--confusable-pair` overriding the registry pair makes the current reporting
stale, because the report line and the args file print `spec.confusable_pair`
(the registry names) while the actual pair may be different:

- The report's `confusable : ...` line
  ([run_real_reject_option_exp.py:669](../run_real_reject_option_exp.py)) must
  print the class **names of the actually-used indices** (`class_names[i]`,
  `class_names[j]`) plus the indices, and indicate the source (registry default
  vs. user override).
- The saved-args `extra` dict
  ([run_real_reject_option_exp.py:599](../run_real_reject_option_exp.py)) must
  record the actual pair (names + indices), the resolved `pair_total` (or the
  `--pair-rest-ratio`), and `--pair-ratio`, so a run is reproducible from its
  args file.

# Tasks

1. Add `--confusable-pair` and `--pair-rest-ratio` argparse arguments with the
   defaults and validation above; keep `--pair-ratio`.
2. Restructure `default_target_prior` to implement the construction in
   "Target-prior construction", accepting the pair indices, `pair_ratio`, and
   `pair_rest_ratio`; verify it reproduces the current output when
   `--pair-rest-ratio` is unset.
3. Resolve the pair indices in `main()`: `--confusable-pair` if given, else the
   registry pair; validate; handle the "no pair available" case.
4. Wire precedence with `--test-prior`.
5. Fix the `confusable` report line and the saved-args `extra` to reflect the
   actual pair and mass settings.
6. Add the edge-case warnings (`rest_total == 0`, `pair_total == 0`).
7. Update the module docstring and the README's real-data section to document
   the new arguments.

# Naming note

The names above (`--confusable-pair`, `--pair-rest-ratio`) are chosen for
consistency with the existing `--pair-ratio` and the registry field
`confusable_pair`. The originally-proposed names were `--confuse-pair` and
`--confuse-rest-ratio`; if preferred, keep those, but `--pair-rest-ratio`
reads more clearly as "pair : rest". A single-number alternative for the mass
knob (`--pair-mass F`, the pair's total probability in `[0,1]`, rest gets
`1-F`) was considered and rejected for inconsistency with the `A B` ratio style
of the other two knobs.
