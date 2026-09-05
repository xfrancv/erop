# Implementation notes

`README.md` is the specification; this file records how it is implemented, and
the handful of places where the code deliberately does something other than the
literal text. Every deviation is listed here with its reason.

## Layout

```
download_datasets.py         fetch the eight datasets of S7 into data/
analyze_datasets.py          self-contained HTML report per dataset
base_predictor_training.py   train + calibrate + emit Theta and the eval posterior
rejopt_eval.py               the adaptation + reject-option experiment
figures.py                   the S6.5 panels (also runnable on a results.json)
selftest.py                  brute-force checks of the S2 inference and S6.4 budgets

exact/          the library
  splits.py       S6.1 split policy
  calibration.py  S6.1 BCTS calibration, NLL, equal-mass ECE
  priors.py       S7 generation and serialisation of Theta
  inference.py    S2/S5 exact log-space inference + the identifiability diagnostic
  protocol.py     S6.2-S6.4 grid, trial sampling, budgets
  sweep.py        the inner loop: trials -> pooled triplets
  metrics.py      S4 curves, summary scalars, trial-level bootstrap
  textutil.py     report formatting
data_tools/     dataset registry, download, loaders, HTML report
```

Pipeline:

```bash
python download_datasets.py
python analyze_datasets.py
./run_base_pred_training.sh all      # or: python base_predictor_training.py <ds> runs/<ds>
./run_rejopt_eval.sh all             # or: python rejopt_eval.py runs/<ds>
python selftest.py
```

`base_predictor_training.py` is the only script that needs torch. It writes
`eval_log_post.npz` — the calibrated `log p_tr(y|x)` of the whole evaluation
split — so `rejopt_eval.py` is pure NumPy/SciPy and runs no forward passes. The
inference is closed-form; there is nothing to put on a GPU.

## What the two scripts emit

`runs/<ds>/`: `model.pt`, `eval_log_post.npz`, `priors.txt` (Theta with the S7
TV column), `report.txt`, `learning_curves.png`.

`runs/<ds>/rejopt/`: `results.json` (the machine interface; `RESULTS_SCHEMA = 1`),
`report.txt`, and one `panels_<stratum>.png` per stratum plus
`risk_coverage_shifted.png` and `theta_breakdown.png`.

## Correctness

`selftest.py` checks `exact/inference.py` against a literal transcription of the
S2 formulas (products over `theta` and `y`, no logsumexp, no leave-one-out) on
small random problems. Agreement is to 2e-16 on `T` and `A`, with exact
agreement on `H` and `theta_map`. It also checks the two S2.4 properties
(`A(x,D) <= T(x,D,yhat)` for *every* `yhat`, and `argmin_yhat E = argmin_yhat T
= H`), log-space stability at `m = 500`, the S6.4 budget bookkeeping, the S7
guards, and the sign of the identifiability diagnostic.

Two identities in `inference.py` are worth knowing while reading it:

* `p(theta | x_i, D^(i))` is **the same for every triplet of a trial** — the
  leave-one-out factor the query contributes cancels the one the adaptation set
  is missing. So a trial needs one normalisation over `theta`, not `m + 1`.
* `p(y | x, D)` is the plugin rule at the posterior-mean prior
  `thetabar = E[theta | D]`. The code uses the logsumexp form of S5 (which keeps
  `log p(y, theta | x, D)` around for the aleatoric term), but the identity is
  asserted in `selftest.py` and is the cleanest way to describe the method.

## Deviations from the README, and why

**1. `m_max` is capped per dataset instead of globally (S6.2).**
The assert `m_max <= |eval| / 10` fails for DermaMNIST, whose val+test is 3,008
examples and so caps `m` at 300. `resolve_grid` truncates the grid and both
reports say so; `--strict-grid` restores the hard failure. Every other dataset
takes the full grid.

**2. The duplicate-rate bound in S6.3 is stated per-class, and measured.**
S6.3 gives the expected recurrences of a query inside its own `D^(i)` as
`m / |eval_y|` and says the S6.2 assert keeps it near 5%. The correct expression
carries the prior, `m * theta_y / |eval_y|`, and the S6.2 assert bounds the
*total* eval size, not the per-class counts — so it does not control this. Under
the `tau = 0.2` spike of S7 the rate on a rare class is `Y * tau` times the
uniform-prior value: on DermaMNIST's rarest class (35 examples) it exceeds 1.0
at `m = 200`. The code therefore **measures** the realised rate per `m`, prints
it in the diagnostics table, and flags any stratum that exceeds 5%.
`analyze_datasets.py` shows the per-class table before any training is run.

**3. "Shifted trials only" is drawn conditionally, not filtered post-hoc (S6.4).**
S6.4 defines the main panels by conditioning the pool on `theta_* != theta_1`.
Filtering after the fact would leave roughly `B (C-1)/C` triplets and
`N(m)(C-1)/C` trials, so the "constant curve budget" would not be constant.
Drawing `theta_*` uniformly from `Theta \ {theta_1}` is the same distribution and
keeps `B` and `N(m)` at their nominal values. Two consequences are stated rather
than hidden: the main-panel population no longer matches the model's `p(theta)`
(Appendix A.5 assumes they agree), and on that population the Bayesian rule is
no longer the risk-minimising one — a rule with `p(theta)` uniform over
`Theta \ {theta_1}` would beat it. The `marginal` stratum is the like-for-like
comparison and is always computed.

**4. Constant `B` does not equalise the confidence intervals (S6.4).**
S6.4 fixes `B = 2000` so that bands do not narrow left-to-right. But the
bootstrap unit is the trial, so the effective sample size is `N(m)`, which falls
from 2,000 at `m = 0` to 50 at `m = 500`. The bands therefore **widen** to the
right; the artifact is inverted, not removed. Nothing in the protocol is changed
— `B` is held at 2000 as specified — but `N(m)` is printed in every report table
and in every figure subtitle so the effect is legible. Equalising band width
would need constant `N(m)` instead.

**5. The bootstrap replicates are paired across rejectors.** S6.4 fixes the
resampling unit (the trial) but not whether each rejector gets its own
resample. Every rejector is evaluated on the *same* resampled trials, so two
bands are directly comparable and a per-replicate difference between rejectors
is meaningful; independent resamples would leave each interval valid on its own
but make comparisons look noisier than they are.

**6. The even split of `B` across trials needs a remainder rule (S6.4).**
`B / N(m)` is not an integer at several grid points (at `m = 2`, 2000/667). Each
trial contributes `floor(B / N)` triplets and `B mod N` trials contribute one
more; the bootstrap re-draws which trials those are on every replicate. A trial's
`m + 1` triplets are already in uniformly random order (S6.3 draws the examples
i.i.d.), so taking a prefix is itself a uniform subsample.

**7. Tie-breaks (S3).** S3 says ties in `theta_map`, `h(x, theta)` and `H(x, D)`
are all resolved "by lowest index in `Theta`". That is right for `theta_map`;
the other two are argmaxes over *labels*, so the code breaks them by lowest
class index. The `m = 0` degeneracy is exact: the leave-one-out subtraction is
done before adding `log p(theta)`, so at `m = 0` every prior ties bit-for-bit
and `theta_map` falls back to `theta_1` — `selftest.py` checks it.

**8. The `A(x, D)` rejector (S3, optional row 6) is implemented.** It is always
in `results.json` and in the report tables; `figures.py` draws it only on
request, so the default panels match S6.5 exactly.

**9. The misspecified arm of Appendix A.1 is implemented** as
`--dirichlet-scale s` (`./run_rejopt_eval.sh <ds> misspec`): `theta_* ~
Dir(s theta_tr)`, so `theta_*` is not in `Theta` almost surely while the model
still uses `Theta`. Not part of the default run.

## Diagnostics added beyond the README

**Identifiability of `Theta` from unlabeled data.** S6.2 justifies the log-spaced
grid by saying the posterior over `Theta` "concentrates exponentially in `m`",
but the *rate* is set by the KL between the induced marginals over `x`, which
the S7 constructions do not control. `identifiability_table` computes, exactly,

    K[a,b] = E_{x ~ p_te(.|theta_a)} [ log w(x, theta_a) - log w(x, theta_b) ]

under the actual S6.3 sampling distribution. Separating the closest pair takes
about `1 / min K` adaptation examples, and `rejopt_eval.py` warns when that
exceeds `m_max`: flat curves are then a property of `Theta`, not of the method.
This matters most on CIFAR-100, where the doubling priors sit at
`TV = 0.01` from `theta_tr` — exactly the S7 de-duplication threshold.

The quantity is a KL, hence non-negative, only when `p_tr(y|x)` is exactly
calibrated. A **negative** entry says the likelihood drifts towards the wrong
prior, i.e. the base model's calibration error outweighs the prior shift for
that pair and `theta_map` converges to the wrong element of `Theta`. The report
flags this. It is a sharp, cheap version of S6.1's "label-shift correction is
highly sensitive to calibration".

**Near-degeneracy warning in `priors.txt`.** The S7 guard drops priors closer
than `TV = 0.01`; `write_prior_set` additionally warns when the closest
surviving pair is within a factor of 2 of that threshold.

**Per-cell diagnostics** in `report.txt`: realised duplicate rate,
`P(theta_map = theta_*)`, `E[p(theta_* | D)]`, and the means of `T`, `A`, `E`.
These are what show the posterior concentrating, and they make a flat panel
interpretable.

## Things to expect in the figures

* **`theta_2` (uniform) is dropped on the balanced datasets.** Fashion-MNIST,
  CIFAR-10 and CIFAR-100 have `theta_tr` exactly uniform, so `theta_1 = theta_2`
  and the S7 guard removes one of them; `C` is 7, not 8. The generator's tie
  breaks (hardest classes, most frequent classes) are stable sorts, so `Theta`
  is still a deterministic function of the training run even when every class
  ties for "most frequent".
* **Rows 1 and 2 coincide over much of the low-coverage region.** `E(x, D)` is
  exactly zero whenever every prior in `Theta` votes for the same label, which
  is common; within that mass the S3 tie-break by `T` governs the ranking. The
  epistemic rejector's AuRC is therefore *worse* than the total rejector's by
  construction — it is not ordering by expected loss. Panels 2 and 3 are where
  it is supposed to pay off.
* **Panel 4's "ceiling" is only a ceiling in expectation.** The true-prior plugin
  and the adapted predictor are built on the same imperfect `p_tr(y|x)`, which
  is exactly why S4.1 notes selective regret can go negative. The caption should
  say so.
* **`Reg@c` compares different accepted subsets** across rejectors, since each
  ranks by its own score. That is what a selective metric means, but it does let
  a score that correlates with "the oracle also errs" look good.
