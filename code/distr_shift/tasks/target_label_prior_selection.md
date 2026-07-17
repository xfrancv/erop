# Plan

Invent a strategy to select target test label prior for real datasets.

# Context

The current code provides several options to set the target label prior; they
are described in README.md. 

All these methods have arguments which are not easy to set up. It is possible
to tune the arguments manually to obtain desired results. However, the setting
is different for each dataset.

Common problems are as follows:
- For some target prior settings, the number of evaluation examples is small (< 300)
leading to noisy results.
- For some settings, the epistemic and Bayesian reject option work almost identically.

The goal is to invent a method which finds a suitable target lable prior autonomously
for all datasets.

Requirements:
- The target lable prior guarantees, that the number of evaluation examples will not be less
then a given number. 
- The generated test data will contain both examples with a large epistemic uncertainty and
large aleatoric uncertainty, so that the epistemic and the Baysian reject option predictors
perform differnetly.

# Tasks

Propose the target label prior selection stategy that is applicable for all datasets.

# Proposal

## What the selection must deliver

The two observed problems map to two formal requirements on the target prior
`α` (given the pool class counts `N_c`, the adaptation size `n_test`, and a
user floor `N_min`, default 300):

- **R1 (eval size).** The auto evaluation size
  `n_eval = floor(min_c eval_avail[c] / α[c])` must satisfy `n_eval >= N_min`,
  with no with-replacement resampling.
- **R2 (separation).** At `α`, the evaluation set must contain *both* a
  non-trivial mass of **prior-sensitive** examples (decision flips when the
  confusable-pair split moves within its plausible range → epistemic
  uncertainty, and regret) *and* a non-trivial mass of **prior-insensitive
  hard** examples (high conditional risk that no pair split changes →
  aleatoric decoys). Only on their union do the total and epistemic rankings
  disagree (see the `epistemic_showcase` analysis in README).

The strategy below searches the *existing* three-knob family
(`confusable pair`, `pair_ratio`, `pair_rest_ratio` →
`default_target_prior`) rather than the full simplex: the family already
spans "how much eval mass is prior-sensitive" (pair mass × skew) vs. "how
much is decoy" (rest mass), which are exactly the quantities R2 constrains,
and it keeps the chosen prior interpretable and reproducible from three
numbers.

## Step 0 — gate: is separation achievable at all?

Epistemic uncertainty exists only if some class pair is **weakly
identifiable** from unlabeled data under the base model. This is checkable
per pair *without MCMC* from the pool posteriors `P` alone, via the Fisher
information of the mixture likelihood along the pair-split direction:
with `R_y(x) = p_tr(y|x)/p_tr(y)`,

    I_ij(α) = mean_x [ (R_i(x) − R_j(x))² / (Σ_y R_y(x) α_y)² ]

(the pool mean importance-weighted to `α`). The predicted posterior std of
the split from `n` unlabeled examples is `~1/sqrt(n · I_ij)`; dividing by the
label-counting std `sqrt(α_i α_j / (α_i+α_j) / n)` gives a **predicted
ident_ratio** — the same statistic the MCMC diagnostic reports, computable
in closed form for every pair and candidate prior. `n` cancels in the ratio.

Because the predicted ratio is strongly `α`-dependent (a skewed feasible
prior can be weakly identified where the reference prior is not — observed
on DermaMNIST: 1.8 → 3.3 across the grid), the threshold is enforced **per
candidate**, at the candidate's own prior; the reference-prior ranking only
*screens* which pairs enter the grid (top 3 plus the registry pair).

- If no feasible candidate reaches predicted ratio ≥ 3, report honestly
  that this dataset + base model cannot produce an epistemic-vs-total
  contrast by prior choice alone (the base model resolves every pair), and
  suggest the other levers: smaller `n_test`, or a lower-capacity /
  less-well-fit base model.

## Step 1 — feasible set for R1 (closed form)

The adaptation set is drawn first (`n_adapt = n_test`, or `max(sizes)` in a
sweep), so class `c` supports at most
`N_c ≈ n_adapt·α[c] + n_eval·α[c]` draws. R1 therefore reduces to per-class
caps:

    α[c] <= cap[c] := (N_c − 1) / (n_adapt + N_min)        for all c

(the −1 absorbs rounding in `target_counts`). Any candidate prior is
**projected** onto the box: clamp violating classes to `cap[c]`, distribute
the freed mass proportionally over the unclamped classes, repeat until
fixed-point (water-filling; converges in ≤ Y passes). After projection,
verify exactly with `target_counts` + `max_distinct_eval` and shrink the
pair skew one grid step if the check still fails. This *guarantees*
`n_eval >= N_min` by construction, for every dataset, with no manual tuning.

## Step 2 — candidate grid

Candidates are the cross product (order ~20 points, all projected via
Step 1):

- pair = the registry pair plus the top-3 pairs of the Step-0 screening
  (each candidate's own predicted ident ratio then decides, see above);
- `pair_ratio ∈ {(1,3), (1,5), (1,7), (1,9)}`, rare side chosen as the pair
  class with the *smaller* cap (so the skew never fights the caps);
- `pair_rest_ratio ∈ {None, (3,7), (1,1), (7,3)}` — this is the knob that
  trades epistemic mass (pair) against aleatoric decoy mass (rest).

Discard candidates whose projection destroyed the intent (achieved pair
skew < 2:1, or pair mass < 5% of the target).

## Step 3 — score candidates without MCMC (the flip test)

For a candidate `α` with pair `(i,j)`, pair mass `M` and skew `ρ = c/(c+d)`,
the MCMC posterior of the split spreads (when weakly identifiable) roughly
between the mirrored splits `(ρM, (1−ρ)M)` and `((1−ρ)M, ρM)`. Under the
label-shift correction the decision between `i` and `j` depends only on
`log(R_i/R_j)` vs. the split, so each pool example is classified in closed
form:

- **flip set** `F`: `|log(R_i(x)/R_j(x))| < log(ρ/(1−ρ))` *and* the pair
  wins the decision (`max(R_i α_i, R_j α_j) > max_{k∉{i,j}} R_k α_k`) —
  decisions the plausible splits disagree on;
- **decoy set** `D`: not in `F`, and conditional risk of the committed
  decision at `α` above a threshold (e.g. > 0.25).

Every pool statistic is importance-weighted to `α` by
`w(x) = α[y(x)] / pool_freq[y(x)]` (labels are legitimate here — this is
benchmark *construction*, not the method under test). Report per candidate:

    E(α) = weighted fraction of F            (epistemic mass)
    A(α) = weighted fraction of D            (aleatoric decoy mass)
    G(α) = weighted expectation (over the whole eval distribution) of the
           loss gap between the wrong-split and true-split decisions on F
                                             (expected full-coverage regret)
    S(α) = weighted fraction of D whose conditional risk exceeds the median
           conditional risk in F             (decoy strength: do decoys
                                              outrank pair examples in total
                                              uncertainty, so the two
                                              rankings actually cross?)

**Selection rule:** among candidates with `E ≥ 0.05`, `A ≥ 0.05` and
`S ≥ 0.5`, maximise `G`. If the constrained set is empty, relax `S`, then
`A`, and report which requirement could not be met.

## Step 4 — validation gate with the real pipeline

Run the winner through 2–3 cheap trials of `run_real_reject_option_exp.py`
(small `n_test`, real MCMC) and check the operational symptoms directly:
the identifiability warning fires on the chosen pair; mean epistemic
uncertainty and the portion-non-negligible are above threshold; and
`AuRC_regret(total) − AuRC_regret(epistemic)` is nonzero beyond trial std.
On failure fall back to the next-best candidate (at most k=3 attempts).
This catches what the proxies can miss (calibration bias steering the MCMC,
posterior width narrower than the mirrored-split assumption).

## Interface

- New module `prior_shift/target_prior_search.py`: the gate, caps,
  projection, flip-test scoring, and the grid search — pure NumPy on `(P,
  y_pool, train_prior)`, no torch.
- New flag in `run_real_reject_option_exp.py` (and inherited by the planned
  `run_real_synth_labels_reject_option_exp.py`):
  `--auto-target-prior [N_MIN]` (default 300). Mutually exclusive with
  `--test-prior`; overrides the pair/ratio knobs. Prints a selection report
  (per-candidate `E/A/G/S`, predicted ident_ratio, chosen prior) and stores
  the chosen prior + knob equivalents in the saved args file so the run is
  reproducible without re-searching.

## Notes and edge cases

- **`n_test` interplay.** Epistemic signal decays as the split posterior
  concentrates: its width is `~1/sqrt(n·I_ij)` while the ident *ratio* is
  `n`-free (both stds scale as `1/sqrt(n)`), so the ratio cannot recommend a
  size. What can: requiring the mirrored split (at distance
  `Δ = |α_i − α_j|`) to stay within ~2 posterior stds gives
  `n_test ≤ 4/(I_ij·Δ²)`. Report it; do not silently override the user's
  `--n-test`.
- **Synthetic-labels variant** (`real_data_synth_labels.md`): the same
  search applies unchanged, but R1 becomes easy (eval labels are generated
  from the model, so only input availability binds) and the flip test
  becomes exact rather than a proxy, because the label model *is* the
  corrected posterior the flip test analyses.
- **Ordinal datasets** (RetinaMNIST): restrict candidate pairs to adjacent
  grades — non-adjacent flips are clinically meaningless and the registry
  already encodes this.
- **Calibration warning**: if the bundle's marginal-consistency check fired,
  abort the search — the flip test inherits the same biased posteriors the
  MCMC does, so no prior choice is trustworthy (same rule as the README's
  calibration caveat).




