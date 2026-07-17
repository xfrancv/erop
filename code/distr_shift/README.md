# Bayesian label-prior adaptation under label shift

Research code that learns the **test label prior from unlabeled test data** by
Bayesian inference, and uses it to correct a discriminative classifier under
*label shift* (a.k.a. prior / target shift).

## Problem

A base model is trained on supervised data drawn from `p_tr(x, y)`, giving an
estimate of the training posterior `p_tr(y | x)` and prior `p_tr(y)`. At test
time we only observe unlabeled inputs `D = (x_1, ..., x_n)` from a distribution
whose **class conditionals are unchanged** (`p_tr(x | y) = p_te(x | y)`) but
whose **prior `p_te(y)` is unknown and different**. The goal is to predict the
labels of `D` and quantify the uncertainty of the prior.

## Method

Under label shift the test posterior is a re-weighting of the training
posterior by the unknown test prior $\alpha(y) = p_{te}(y)$:

$$p(y | x, \theta) \propto p_{tr}(y | x)  \alpha(y) / p_{tr}(y)\qquad      \theta = (\alpha(1),\ldots,Y))
$$

The marginal test density of one input is, up to the unparameterised constant
`p_tr(x)`,

$$p(x | \theta) \propto \sum_{y}  R(x, y) \alpha(y)\,,\quad     R(x, y) = p_{tr}(y | x) / p_{tr}(y)$$


so with a $Dirichlet(\beta)$ prior on $\theta$ the log posterior over the prior is

$$
\log p(\theta | D) = \sum_{y} (\beta_y − 1) \log \alpha(y)  +  \sum_i \log \sum_y R(x_i, y) \alpha(y)  +  const.
$$

> **Note on the likelihood.** In the task write-up the per-sample term is
> written as $\sum_y p(y | x_i, \theta)$. Because the *normalised* posterior sums to 1,
> that literal reading is constant in $\theta$; the quantity that carries the
> information about the prior is the **un-normalised** marginal
> $\sum_y R(x_i, y) \alpha(y)$ used above. This is the standard label-shift
> (Saerens et al., 2002) likelihood.

We draw $\theta$ from $p(\theta | D)$ with a **random-walk Metropolis–Hastings**
chain. To respect the simplex constraint the sampler works in an unconstrained
$z \in {\mathbb R}^{Y−1}$ via an additive-logistic (softmax) reparameterisation and adds the change-of-variables Jacobian $\sum_y \log \alpha(y)$, so the chain targets the correct density on the simplex.

The Bayesian label posterior averages the normalised re-weighting over the
posterior draws,

$$
\hat p(y | x, D) = (1/N) \sum_i  normalise_y[ R(x, y) \alpha_i(y) ],
$$

and predictions minimise the expected user loss
$h(x) = \arg\min_{\hat y} \sum_y \hat p(y| x, D) \ell( \hat y, y) $ (argmax for 0/1 loss).

## Experiment

Synthetic data from a mixture of 2-D Gaussian class conditionals. The generator
setting — the number of Gaussians, their means and covariances, and the
train/test label priors — is read from a JSON file passed via `--config`. The
default, [configs/default.json](configs/default.json), has `Y = 4` classes with
means/covariances tuned so the Bayes error under a uniform prior is ≈ 10%, a
uniform training prior, and a strongly imbalanced test prior
`[0.60, 0.20, 0.15, 0.05]`. The base model is multinomial logistic regression.

### Configuration

The generator setting is a JSON file (see
[configs/default.json](configs/default.json)) with four fields: `means`, `covs`,
`train_prior`, `test_prior`. The number of classes `Y` is implied by the length
of `means`, and `covs` and both priors must have the same length. Priors must be
strictly positive and sum to 1.

**Covariance parametrization.** Each covariance can be given in either of two
forms, and the two may be mixed within one config:

1. **Rotated-axes (recommended)** — a dict `{"sx": .., "sy": .., "theta": ..}`.
   The matrix is built as

   $$\Sigma = R(\theta)\begin{bmatrix} s_x^2 & 0 \\ 0 & s_y^2 \end{bmatrix}R(\theta)^{\top},\qquad R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta\\\sin\theta & \cos\theta\end{bmatrix}$$

   i.e. the eigendecomposition read geometrically: `sx`, `sy` are the standard
   deviations along the ellipse's two principal axes (so `sx²`, `sy²` are the
   eigenvalues of `Σ`), and `theta` is the angle in radians by which those axes
   are rotated from the coordinate axes. Because `sx², sy² > 0`, the result is
   always symmetric positive definite, so this form cannot specify an invalid
   covariance.
2. **Full matrix** — a nested list `[[a, b], [b, c]]`, used verbatim.

Whichever form is used, the loader validates that each covariance is 2×2,
symmetric, and positive definite (via a Cholesky factorization), that means are
2-D, and that the priors are well-formed — raising a clear error otherwise. See
[configs/three_gaussians.json](configs/three_gaussians.json) for a 3-class
example mixing both covariance forms.

Predictors compared:

| Predictor | Uses |
| --- | --- |
| Optimal Bayes, test prior | true conditionals + true test prior (**upper bound**) |
| Plugin, training prior | logistic posterior, no adaptation |
| Plugin, true test prior | logistic posterior + oracle prior |
| Plugin, supervised prior estimate | logistic posterior + prior from **labeled** test frequencies |
| **Bayesian, learned prior** | logistic posterior + prior learned from unlabeled `D` (**proposed**) |

The *supervised prior estimate* is a reference baseline that estimates the test
prior as the empirical class frequencies of the **labeled** test data (available
only because the generator is synthetic),
$\hat p_{te}(y) = \frac1n \sum_i [\,y_i = y\,]$, and plugs it into the same
label-shift correction $p(y|x)\propto p_{tr}(y|x)\,\hat p_{te}(y)/p_{tr}(y)$.
It is the supervised counterpart of the proposed method: both adapt the prior
from the same `n` test examples, but only this one is allowed to see their
labels — so it bounds how well the unsupervised learner could hope to do at a
given `n`.

One caveat on how it is scored in the single-size experiment: there the prior is
counted from `y_te`, the same labels the accuracy is then measured against, so
the estimate is in-sample and the baseline is mildly optimistic. At `n_test =
2000` with four classes the bias is small, but it is not zero. `--sweep` does
not share this defect — the supervised prior is counted from a pool of `n` test
labels while every predictor is scored on a disjoint fixed evaluation set.

### Results (20 trials, m_train = n_test = 2000)

```
learned prior   : [0.595 0.197 0.153 0.056]   (true: [0.60 0.20 0.15 0.05])
prior L1 error  : 0.034 ± 0.016   (learned vs true)

predictor                                       test acc      std
Optimal Bayes, true test prior (upper bound)      0.9294    0.0058
Plugin, true test prior (oracle)                  0.9249    0.0049
Plugin, supervised prior estimate                 0.9255    0.0052
Bayesian, learned prior (proposed)                0.9253    0.0050
Plugin, training prior (no adaptation)            0.9040    0.0075
```

The learned prior recovers the true test prior closely, and the proposed
predictor matches the oracle that is *given* the true test prior — recovering
essentially all of the ~2 accuracy points lost to the unadapted classifier. It
also matches the supervised prior estimate, i.e. at this sample size learning
the prior from *unlabeled* data costs nothing relative to counting labels.

### How much unlabeled data does the adaptation need?

`--sweep` varies the number of test examples `n` the prior is adapted from and
scores every predictor on a **fixed** labeled evaluation set. The three oracle /
no-adaptation baselines never use an adapted prior, so they are constant in `n`;
the two adaptation curves move with it — the proposed predictor from the same
`n` *unlabeled* inputs, and the supervised estimate from those same `n` inputs'
*labels*. As `n` grows the learned prior converges toward the truth (roughly
`~1/√n` until a floor set by the base model's posterior error), and the proposed
predictor climbs from the unadapted plugin up to the oracle, tracking the
supervised estimate the whole way:

### Results (20 trials, m_train = 2000, n_eval = 2000)

```
n_test   prior L1   opt(test)  plugin(test)  plugin(sup)  bayes(learn)  plugin(train)
    50      0.157       0.9225      0.9197       0.9156        0.9179       0.9009
   100      0.110       0.9225      0.9197       0.9167        0.9174       0.9009
   200      0.083       0.9225      0.9197       0.9194        0.9187       0.9009
   500      0.064       0.9225      0.9197       0.9192        0.9184       0.9009
  1000      0.045       0.9225      0.9197       0.9194        0.9188       0.9009
  2000      0.034       0.9225      0.9197       0.9199        0.9190       0.9009
  5000      0.030       0.9225      0.9197       0.9201        0.9190       0.9009
```

A few hundred unlabeled examples already recover most of the achievable gain:
by `n ≈ 200` the learned-prior predictor sits within ~0.001 accuracy of the
oracle that is handed the true test prior, having closed ≈95% of the ~1.9-point
gap between the unadapted plugin (0.9009) and that oracle (0.9197). Beyond a
few thousand examples the prior error plateaus — the residual is dominated by
the imperfect logistic posterior, not by the amount of unlabeled data. The
unsupervised learner tracks the supervised frequency estimate throughout (it
is even slightly ahead at `n ≤ 100`, where empirical counts of a rare class
are noisy), so at every `n` learning the prior without labels costs
essentially nothing relative to counting them.

### When can this fail? The identifiability warning

Estimating the prior from **unlabeled** data works only if the class
conditionals are distinguishable. The method fits the unlabeled marginal
$p(x\,|\,\alpha) = \sum_y p(x|y)\,\alpha(y)$; if two conditionals are (nearly)
identical, $p(x|i) \approx p(x|j)$, then moving mass between $\alpha(i)$ and
$\alpha(j)$ leaves the marginal (nearly) unchanged — the likelihood is flat
along that direction and the split is **not identifiable** from unlabeled data.
The posterior of $\alpha$ then spreads into a ridge, and the posterior mean
along the ridge is determined by the Dirichlet prior rather than the data. The
resulting misallocated prior can make the adapted predictor *worse* than no
adaptation, because with overlapping conditionals the Bayes decision between
the confusable classes is prior-dominated. (Only labels can resolve such a
split, which is why the supervised frequency baseline is unaffected.)

[configs/model1.json](configs/model1.json) demonstrates the failure: four
isotropic unit-variance Gaussians with means on a line at 0.0, 1.0, 1.5, 1.1 —
classes 1 and 3 are 96% overlapping (total-variation distance 0.04). The
Fisher information of the mixture likelihood is near-singular (condition
number ~9000 vs ~4.5 for the default config), so even at `n = 2000` the prior
L1 error stays ~0.36 and the proposed predictor falls below the unadapted
plugin, while the supervised baseline still matches the oracle.

**The built-in diagnostic.** The sampler compares, per class, the posterior
std of $\alpha(y)$ against the std of the best *supervised* estimator at the
same sample size — counting $n$ labels, $\sqrt{\alpha(y)(1-\alpha(y))/n}$
(`MCMCResult.ident_ratio`). When the problem is well conditioned the ratio is
~1–1.5 (unlabeled data is nearly as informative as labels, see the tables
above); a ratio **above 3×** for any class means the unlabeled data is far
less informative than labels would be, i.e. the prior is only weakly
identifiable — because of near-identical class conditionals or simply too
little data. This benchmark is self-calibrating: no absolute threshold on the
posterior width is needed, so the check works at any `n`.

`MCMCResult.identifiability_warning()` returns the warning message (or `None`
if healthy) naming the affected classes and their ratios. The experiment
script surfaces it in both modes:

- single mode prints it in the report with the count of affected trials, e.g.

  ```
  !!! IDENTIFIABILITY WARNING (fired in 3/3 trials) !!!
      test prior only weakly identifiable from the unlabeled data: posterior
      std of alpha(y=1, y=2, y=3) is [11.9  7.  10.4]x the std of counting the
      same number of labels (threshold 3x). ...
  ```

- `--sweep` adds a `warn` column with the fraction of trials flagged at each
  `n` (for `model1` it is 1.00 for `n ≥ 200`; for the default config 0.00
  everywhere).

If the warning fires, the learned prior — and anything downstream of it —
should not be trusted; the posterior std reported in `mcmc_diagnostics.png`
shows which class splits the data cannot resolve.

Figures are written to `figures/`:
`data_and_prior.png`, `mcmc_diagnostics.png`, `accuracy_comparison.png`, and
(from `--sweep`) `accuracy_vs_n_test.png`. The single-size (non-`--sweep`) mode
also writes `report.txt`: the printed accuracy table plus, per predictor, the
confusion matrix on the trial-0 test examples (rows = true class, columns =
predicted class).

Every run also records its argument setting next to the figures, as
`run_synth_bayesian_learning_exp_args.txt` or
`run_synth_bayesian_learning_exp_sweep_args.txt`. The file holds the
command line, a timestamp, the arguments the mode actually reads (the one it
ignores — `--sizes` in single mode, `--n-test` in `--sweep` — is left out rather
than shown at an unused default), and the config's name and priors.

## Reject-option predictors

`run_synth_reject_option_exp.py` extends the same setting with **selective
prediction**: the predictor may abstain on inputs it is unsure about. A
reject-option predictor is a pair of a base predictor `h(x)` and an uncertainty
score `u(x)`; it emits `h(x)` when `u(x)` is below a threshold and rejects
otherwise. Sweeping the threshold traces out a curve, so no threshold has to be
fixed in advance.

Three predictors are compared. All are scored under the 0/1 loss, so the
conditional risk of a decision is one minus its posterior probability.

| Reject-option predictor | Base predictor `h(x)` | Uncertainty `u(x)` |
| --- | --- | --- |
| Bayesian, total uncertainty | Bayesian, learned prior | $\hat T(x,D)$ |
| Bayesian, epistemic uncertainty | Bayesian, learned prior | $\hat T(x,D)-\hat A(x,D)$ |
| Oracle (best attainable) | Bayesian, learned prior | actual per-example loss (risk curve) / actual per-example regret (regret curve) |

The **oracle** is a label-aware reference, not a deployable predictor: it ranks
the evaluation examples by their *realised* loss for the risk-coverage curve and
by their *realised* regret for the regret-coverage curve (two metric-specific
orderings), so it is the lower envelope — the best selective risk / regret any
rejection rule could reach for the Bayesian predictor on that sample. (The
supervised-prior plugin is no longer a reject-option curve; it remains an
accuracy reference in `base_accuracy_vs_n_test.png` / `accuracy_vs_n_test.png`.)

The **total** uncertainty is the conditional risk of the committed decision
under the posterior-averaged label distribution,

$$\hat T(x,D) = \frac1N\sum_{i=1}^N \sum_y p(y\mid x,\theta_i)\,\ell(y, h(x,D)),$$

and the **aleatoric** part is the risk that would remain if each posterior draw
$\theta_i$ were the true prior, i.e. the per-draw *minimal* conditional risk,

$$\hat A(x,D) = \frac1N\sum_{i=1}^N \min_{\hat y}\sum_y p(y\mid x,\theta_i)\,\ell(y, \hat y).$$

Their difference $\hat T - \hat A \ge 0$ is the **epistemic** uncertainty: the
excess risk incurred by having to commit to one decision before the prior is
known. It is large exactly where the posterior draws of $\alpha$ *disagree*
about the label, and it vanishes where they agree — even if that agreed-upon
label is itself uncertain. This distinction is what the two Bayesian scores
measure, and it is why they rank inputs very differently (see below).

### Evaluation: risk-coverage and regret-coverage

Rank the evaluation examples by ascending `u` and let $\pi$ be that order. At
rank `k` the predictor accepts the `k` least uncertain examples:

$$coverage(k)=\frac kn,\qquad risk(k)=\frac1k\sum_{i=1}^k \ell(y_{\pi(i)}, h(x_{\pi(i)}))$$

$$regret(k)=\frac1k\sum_{i=1}^k\Big(\ell(y_{\pi(i)}, h(x_{\pi(i)})) - \ell(y_{\pi(i)}, \hat h_{true\text{-}prior}(x_{\pi(i)}))\Big)$$

**Selective risk** is the error rate on the accepted examples; a good
uncertainty score makes it fall as coverage shrinks. **Selective regret**
measures the same examples against the plugin predictor *given the true test
prior* — it isolates the cost of not knowing the prior, and unlike the risk it
can be **negative** (the adapted predictor sometimes beats the true-prior
plugin, since both share the same imperfect logistic posterior). Each curve is
summarised by its area, $\text{AuRC}=\frac1n\sum_{k=1}^n metric(k)$, and both
curves are averaged over trials.

Because AuRC integrates uniformly over all coverages, a large gap confined to
one coverage regime gets diluted. Both scripts therefore also report the
**coverage at target** — the dual statistic: the largest coverage at which the
selective metric stays within a budget for every accepted rank,
$\text{cov@}\varepsilon = \frac1n\max\{k : metric(j)\le\varepsilon\ \forall
j\le k\}$ (the first few ranks are a grace region, since selective metrics at
tiny $k$ are 0/1-grained). Both budgets accept **one or more values** and the
metric is reported for each: `--regret-target` (default 0.002) and
`--risk-target`, defaulting to a single budget — the per-trial full-coverage
risk of the true-prior reference, i.e. *"how much coverage while staying no
worse than the oracle-prior plugin's average error."* It is
computed per trial and then averaged (threshold crossings are nonlinear, so
the order matters). In `--sweep` mode it adds a third figure,
`cov_at_target_vs_n_test.png`.

**AuRC50.** Both scripts also report $\text{AuRC50}$: the same areas averaged
over the ranks with $coverage\ge0.5$ only. Rejecting more than half the inputs
is not an operating point anyone deploys, and it is exactly where the estimates
are noisiest ($risk(1)$ is a single example). Averaging over the window rather
than integrating over it keeps AuRC50 on the AuRC scale, so the two may be read
side by side — unlike AuGRC. **Caveat:** truncation makes the statistic
invariant to the ranking *within* the accepted half, and since the
reject-option predictors share one base predictor and differ only in their
ranking score, their gaps compress. A smaller AuRC50 gap is not evidence the
rankings agree more. In `--sweep` mode the areas get their own two-panel
figure, `aurc50_vs_n_test.png`.

**No in-sample bias.** Unlike `run_synth_bayesian_learning_exp.py`'s single-size mode, *both*
modes here score on a **fixed labeled evaluation set** that is disjoint from the
`n_test` examples used to adapt the prior. The supervised plugin reference
counts its prior from the *labels of the adaptation set*, never from the
evaluation set, so the caveat noted above does not apply to this script.

### Results (20 trials, m_train = n_test = n_eval = 2000)

```
configs/three_gaussians.json            AuRC risk        AuRC regret
Bayesian, total uncertainty          0.0033 ± 0.0007   0.0000 ± 0.0000
Bayesian, epistemic uncertainty      0.0601 ± 0.0097   0.0000 ± 0.0000
Plugin, supervised prior (reference) 0.0033 ± 0.0007   0.0000 ± 0.0000

configs/model1.json                     AuRC risk        AuRC regret
Bayesian, total uncertainty          0.0639 ± 0.0089   0.0000 ± 0.0002
Bayesian, epistemic uncertainty      0.1464 ± 0.0151   0.0001 ± 0.0001
Plugin, supervised prior (reference) 0.0631 ± 0.0085  -0.0000 ± 0.0001
```

**Total uncertainty ranks for risk; epistemic uncertainty does not.** Under 0/1
loss $\hat T(x,D)$ is one minus the posterior probability of the predicted
label, i.e. the estimated probability of erring on `x` — the optimal quantity to
reject by. Ranking by it drives selective risk to ~0 at low coverage, and it
matches the supervised plugin reference to within noise (0.0033 vs 0.0033 on
`three_gaussians`), so learning the prior from unlabeled data costs nothing for
selective prediction either.

Ranking by **epistemic** uncertainty *inverts* the curve: selective risk
**rises** as coverage shrinks — to ≈0.45 on `three_gaussians` and ≈0.40 on
`model1`, against full-coverage error rates of 0.041 and 0.153. Rejecting at
random would give a flat curve at exactly those error rates, so at low coverage
the epistemic score is worse than useless. (Integrated over all coverages it
comes out worse than random on `three_gaussians`, 0.0601 vs ≈0.041, and roughly
on par on `model1`, 0.1464 vs ≈0.153, where its mid-coverage dip compensates.)

This is not a bug — it is what the score measures. Inputs deep inside a
class-overlap region draw the same label distribution from *every* posterior
draw of $\alpha$, so their epistemic uncertainty is ≈0 even though they are
exactly the inputs the classifier gets wrong. Epistemic uncertainty flags
inputs whose label is *sensitive to the prior*, not inputs that are hard. It is
the wrong score for minimizing selective risk, and the right one for asking
where the residual prior uncertainty still matters.

**The regret curve is degenerate on both shipped configs.** The adapted
predictor makes the *same decisions* as the plugin given the true prior on all
but the ~15% most uncertain inputs, so `regret(k) = 0` exactly for coverage
below ≈0.85 and the AuRC is zero to four decimals. This is not only an
artifact of large `n`: the sweep below (on `three_gaussians`) gives
`|AuRC regret| < 5·10⁻⁵` with error bars straddling zero at **every** `n` from
50 up — the generator is well-conditioned enough that even a prior learned from
50 unlabeled examples induces the same argmin as the true prior almost
everywhere. Regret would
become informative only where the learned prior is bad enough to flip
decisions — the weakly-identifiable regime described above, which neither
`three_gaussians.json` nor the current `model1.json` exhibits.

### When epistemic rejection wins: `epistemic_showcase.json`

For the epistemic score to beat the total score on *regret*, the data must
satisfy two conditions at once — and on their intersection the two scores must
disagree:

1. **regret must exist**: a class pair with *identical* conditionals (split
   unidentifiable from unlabeled data) and a strongly asymmetric true split,
   so the learned split is prior-dominated and flips decisions on the whole
   pair region;
2. **total uncertainty must be misled**: a *decoy* region of regret-free
   examples with **higher** conditional risk than the pair region. Two-class
   ambiguity caps total uncertainty at ~0.5, so the decoy must be a
   **three-way overlap** of well-identified classes: total ≈ 2/3, epistemic
   ≈ 0.

[configs/epistemic_showcase.json](configs/epistemic_showcase.json) builds
exactly this: five classes — a tight triangle of three identifiable Gaussians
(the decoy) far from two *coincident* Gaussians with true split 0.05 / 0.35.
Run with little unlabeled data and a well-fit base model so the pair split
stays genuinely flat (a logistic posterior fit on few training samples breaks
the tie between the coincident classes by noise, and enough unlabeled data
lets MCMC latch onto that spurious signal):

```bash
python run_synth_reject_option_exp.py --config configs/epistemic_showcase.json \
    --m-train 10000 --n-test 500
```

Results (20 trials): the identifiability warning fires in 20/20 trials
(ident ratio ≈ 5x for the coincident pair), and the regret-coverage curves
finally separate:

```
reject-option predictor                    AuRC risk         AuRC regret
Bayesian, total uncertainty            0.2789 ± 0.0881   0.0801 ± 0.0825
Bayesian, epistemic uncertainty        0.3172 ± 0.0393   0.0372 ± 0.0366
Plugin, supervised prior (reference)   0.1471 ± 0.0079   0.0000 ± 0.0002
```

Epistemic ranking holds `regret(k) = 0` up to coverage ≈ 0.60 — exactly
`1 − (α₃+α₄)`, i.e. it defers the *entire* unidentifiable region to the last
ranks — while total ranking starts paying regret from coverage ≈ 0.28 because
its top ranks are spent rejecting the three-way-overlap decoys, which carry no
regret at all. Both curves meet at the same full-coverage regret (same base
predictor); the supervised reference has none, since labels resolve the split.
The price is unchanged: on selective *risk* the epistemic ranking remains the
worst of the three. The two curves ask different questions, and this generator
is one where they give opposite answers.

**A consequence of *exactly* identical conditionals**: under `--sweep` the
AuRC-regret of the two Bayesian reject-option predictors never decays to zero
— only the supervised plugin's does. No amount of unlabeled data adds
information along an exactly flat likelihood ridge (in fact the posterior
concentrates on the *spurious* split direction contributed by the base
model's finite-training noise), so the base predictor keeps paying regret at
full coverage no matter how well the epistemic score ranks.
[configs/epistemic_showcase_near.json](configs/epistemic_showcase_near.json)
repairs this by separating the pair means by 0.3σ: the split becomes
identifiable *in the limit* but stays weakly identified at small `n`
(posterior width ≈ 8–10× label counting). Sweeping `n` (10 trials,
`--m-train 10000`):

```
n_test    AuRC regret:  total   epistemic   plugin(sup)
    50                 0.0621      0.0320        0.0006
   100                 0.0774      0.0341        0.0003
   200                 0.0162      0.0083        0.0000
   500                 0.0147      0.0078        0.0000
  1000                 0.0001      0.0001        0.0001
```

The epistemic predictor now dominates the total-uncertainty predictor by ~2×
at every `n` where regret exists, *and* both decay to zero once the split
posterior concentrates on the truth (`n ≥ 1000`) — the regime transition is
visible in the `warn` column, which switches on at `n = 200` and stays on
while the posterior remains wide relative to label counting.

### Sweep: AuRC vs. number of unlabeled examples

`--sweep` varies the `n` unlabeled examples the prior is adapted from, scoring
every point on the same fixed labeled evaluation set (10 trials,
`three_gaussians.json`, `n_eval = 2000`):

```
n_test   AuRC risk: total   epistemic   plugin(sup)      AuRC regret (all three)
    50              0.0036      0.0226       0.0036              < 5e-5, ~0
   100              0.0034      0.0333       0.0034              < 5e-5, ~0
   200              0.0033      0.0391       0.0033              < 5e-5, ~0
   500              0.0034      0.0516       0.0034              < 5e-5, ~0
  1000              0.0034      0.0533       0.0034              < 5e-5, ~0
  2000              0.0034      0.0602       0.0034              < 5e-5, ~0
```

The total-uncertainty score is already at its floor by `n = 50` and flat
thereafter: selective prediction needs far less unlabeled data than the
accuracy curve above, because rejecting by conditional risk depends on the
*posterior*, which the prior barely perturbs once it is roughly right.

Counter-intuitively the epistemic AuRC gets **worse** as `n` grows (0.023 →
0.060). As the posterior of $\alpha$ concentrates, $\hat T - \hat A \to 0$ for
every input, so the score degenerates toward numerical noise and its ranking
toward arbitrary. At small `n` it retains enough signal to correlate weakly
with genuine ambiguity. The lesson is the same as before: this quantity is not
a proxy for error probability.

Figures are written to the `--out-dir`: `risk_coverage.png`,
`regret_coverage.png`, and (from `--sweep`) `aurc_vs_n_test.png` and
`aurc50_vs_n_test.png`. The argument setting is saved alongside them as
`run_synth_reject_option_exp_args.txt` or
`run_synth_reject_option_exp_sweep_args.txt` — named so that the two scripts
can share one `--out-dir` without overwriting each other's record.

## Setup

**conda (recommended):**

```bash
conda create -n distr_shift python=3.11
conda activate distr_shift
python -m pip install -r requirements.txt
```

> Use `python -m pip` (not bare `pip`) so packages install into the active
> conda environment rather than the user-level site-packages.

**venv:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python run_synth_bayesian_learning_exp.py                 # 20 trials, writes figures + table
python run_synth_bayesian_learning_exp.py --trials 5 --n-test 1000 --seed 1
python run_synth_bayesian_learning_exp.py --sweep         # accuracy vs. #unlabeled examples
python run_synth_bayesian_learning_exp.py --sweep --sizes 100 1000 10000 --n-eval 4000
python run_synth_bayesian_learning_exp.py --config configs/three_gaussians.json   # other generator
```

Reject-option predictors (same flags, plus `--n-eval` in both modes):

```bash
python run_synth_reject_option_exp.py            # risk/regret-coverage curves + AuRC
python run_synth_reject_option_exp.py --sweep    # AuRC vs. #unlabeled examples
python run_synth_reject_option_exp.py --trials 5 --n-test 500 --n-eval 1000
```

## Real datasets

The synthetic experiments above are self-contained. To validate the
epistemic reject-option story on real data (see
[`tasks/datataset_proposal.md`](tasks/datataset_proposal.md)), two scripts
download and inspect the candidate datasets. They use only the standard
library plus NumPy/matplotlib/tqdm — no torch/torchvision/medmnist.

| Dataset | key | shape | classes | source | confusable pair |
|---------|-----|-------|---------|--------|-----------------|
| Fashion-MNIST | `fashion_mnist` | 28×28 grayscale | 10 | Zalando IDX files | Shirt / T-shirt |
| CIFAR-10 | `cifar10` | 32×32 RGB | 10 | fast.ai PNG-folder mirror | cat / dog |
| CIFAR-100 | `cifar100` | 32×32 RGB | 100 | fast.ai PNG-folder mirror (nested by superclass) | boy / girl |
| DermaMNIST | `dermamnist` | 28×28 RGB | 7 | MedMNIST v2 `.npz` (Zenodo) | melanoma / nevus |
| BloodMNIST | `bloodmnist` | 28×28 RGB | 8 | MedMNIST v2 `.npz` (Zenodo) | neutrophil / immature granulocyte |
| RetinaMNIST | `retinamnist` | 28×28 RGB | 5 | MedMNIST v2 `.npz` (Zenodo) | grade 1 / grade 2 (adjacent DR severity) |

CIFAR-100's fast.ai mirror is laid out two levels deep
(`<split>/<superclass>/<fine-class>/*.png`); the image-folder loader labels by
the leaf (fine-class) folder, so it yields the 100 fine classes. RetinaMNIST is
ordinal (5 diabetic-retinopathy severity grades), so its "confusable pair" is
the adjacent mild/moderate boundary rather than two arbitrary classes.

```bash
python download_datasets.py                 # fetch all into data/
python download_datasets.py cifar10 bloodmnist   # or a subset
python download_datasets.py --list          # list dataset keys + confusable pairs

python analyze_datasets.py                   # download (if needed) + build reports
python analyze_datasets.py fashion_mnist     # one dataset
python analyze_datasets.py --per-class 12    # examples per class in the montage
```

`analyze_datasets.py` writes a **self-contained HTML report** per dataset to
`data/reports/<key>.html` (plus an `index.html`), with all images embedded as
base64. Each report gives the split sizes (train/val/test), the input
dimensionality and flattened feature count, the number of classes and their
per-split balance, and a grid of example images for every class. The class
rows are tagged with each dataset's proposed *confusable pair* (e.g. cat/dog,
melanoma/nevus), linking the data back to the reject-option motivation.

Downloaded data and generated reports live under `data/` (gitignored). CIFAR-10
is decoded once from PNGs and cached as an `.npz`, so re-analysis is a few
seconds rather than ~100 s.

### Running the adaptation experiment on real data

Two further scripts carry the reject-option story onto the real datasets. These
require **torch/torchvision** (unlike the download/analysis tools above).

1. **`run_base_predictor_exp.py`** trains and calibrates a neural-network base
   predictor. It splits the training subset (class-stratified) into a fit part
   and a model-selection part, selects the best epoch by validation error, and
   calibrates on the model-selection part — by default **bias-corrected
   temperature scaling** (scalar `T` + per-class bias; `--calibration
   temperature` for plain scaling). BCTS matters downstream: a single
   temperature flattens overconfident logits globally, which inflates the mean
   posterior of rare classes, and the label-shift MCMC misreads that bias as
   prior shift (on DermaMNIST this put 60% of the learned prior on a 1% class).
   The script reports a **calibration-consistency check** — mean calibrated
   posterior over the held-out split divided by class frequency, ~1 per class
   when healthy, stored in the bundle and re-printed with a warning by
   `run_real_reject_option_exp.py` (it is the bias-detecting complement of the
   variance-detecting `ident_ratio`). The saved `model.pt` bundle holds the
   best-epoch weights + `T` + bias + the check + estimated training prior +
   normalization.
   Architectures default per dataset: LeNet for Fashion-MNIST, a 32×32/28×28
   small-input-adapted ResNet-18 (3×3 stem, no max-pool, trained from scratch)
   for CIFAR-10 and the MedMNIST sets.

   ```bash
   python run_base_predictor_exp.py fashion_mnist runs/fashion
   python run_base_predictor_exp.py bloodmnist runs/blood --epochs 30 --device cuda
   ```

2. **`run_real_reject_option_exp.py`** is the real-data counterpart of
   `run_synth_reject_option_exp.py`: it loads a `model.pt` base predictor,
   computes calibrated posteriors on the (val+test) pool, **simulates label
   shift** by resampling that labeled pool to a target prior, and runs the same
   predictors and reject-option curves. The target prior defaults to the
   training prior with the dataset's confusable pair skewed asymmetrically (a
   genuine, pair-targeted shift). Three knobs shape it:

   - `--confusable-pair I J` — the two class indices to treat as the pair
     (default: the dataset registry's, e.g. cat/dog);
   - `--pair-rest-ratio A B` — the pair's total mass `A/(A+B)` vs. the rest
     `B/(A+B)`, the rest spread proportionally to the training prior (default:
     keep the pair's training mass — so with `--pair-rest-ratio` unset the
     target is exactly today's);
   - `--pair-ratio A B` — the split of the pair's mass between its two classes.

   `--test-prior` overrides all three with an explicit `Y`-vector.

   **Autonomous selection: `--auto-target-prior [N_MIN]`.** Tuning the three
   knobs by hand is dataset-specific and easy to get wrong (too little
   evaluation data, or a prior at which the epistemic and total reject-option
   predictors behave identically). The flag replaces all of them with the
   search of [`tasks/target_label_prior_selection.md`](tasks/target_label_prior_selection.md)
   (implemented in `prior_shift/target_prior_search.py`):

   1. **Gate** — for every class pair it predicts, in closed form from the
      pool posteriors (Fisher information of the mixture likelihood along the
      pair-split direction), the `ident_ratio` the MCMC diagnostic would
      report. The prediction is `α`-dependent, so the threshold (ratio ≥ 3)
      is enforced per candidate prior; the reference-prior ranking only
      screens which pairs enter the grid. If no feasible candidate is weakly
      identifiable, no target prior can separate the epistemic and total
      scores and the script says so instead of searching.
   2. **Eval-size guarantee** — candidate priors are projected onto the box
      `α[c] ≤ (N_c − 1)/(n_adapt + N_MIN)`, which makes the auto `n_eval` at
      least `N_MIN` (default 300) by construction, with no with-replacement
      resampling.
   3. **Flip-test scoring** — each candidate (a grid over the pair, its skew
      and its mass vs. the rest) is scored without MCMC by comparing plugin
      decisions under the candidate split and its mirror: `E` (prior-sensitive
      eval mass → epistemic uncertainty), `A` (prior-insensitive high-risk
      mass → aleatoric decoys), `G` (expected full-coverage regret) and `S`
      (do decoys out-rank pair examples in total uncertainty). Among
      candidates with `E ≥ 0.05`, `A ≥ 0.05`, `S ≥ 0.5` the search maximises
      `G`, relaxing `S`, then `A` if the set is empty (and reporting it).
   4. **Validation** — the winner is confirmed by 2 short MCMC trials (ident
      warning fires on the pair, epistemic mass non-negligible, regret-AuRC
      gap between the total and epistemic rankings beyond trial noise),
      falling back to the runners-up (3 attempts) on failure.

   The selection report (gate table, candidate scores, chosen prior, and a
   recommended `n_test` ceiling above which the split posterior concentrates
   and the epistemic signal fades) is printed and recorded in the saved args
   file. `--auto-target-prior` is mutually exclusive with `--test-prior`; if
   the base model failed its calibration-consistency check the search aborts,
   since the flip test would inherit the same biased posteriors the MCMC
   does. For ordinal datasets (RetinaMNIST, tagged `ordinal` in the registry)
   only adjacent-grade pairs are considered.

   ```bash
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood --auto-target-prior
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood --auto-target-prior 500 --sweep
   ```

   **How the target prior `p` is built** from the base model's training prior
   `p_tr`, the confusable pair `(i, j)`, `--pair-rest-ratio (a, b)` and
   `--pair-ratio (c, d)`:

   1. **Pair total mass** — `pair_total = a/(a+b)` if `--pair-rest-ratio` is
      given, else `p_tr[i] + p_tr[j]` (keep the training mass); and
      `rest_total = 1 − pair_total`.
   2. **Within the pair** — split `pair_total` by `--pair-ratio`:
      `p[i] = pair_total·c/(c+d)`, `p[j] = pair_total·d/(c+d)`.
   3. **The remaining classes** — spread `rest_total` *proportionally to the
      training prior*: `p[k] = rest_total · p_tr[k] / Q_rest` for every `k ∉
      {i,j}`, where `Q_rest = Σ_{m∉{i,j}} p_tr[m]`.

   `p` sums to 1 by construction. Step 3's proportional spread makes the scheme
   a strict generalisation of the default: with `--pair-rest-ratio` unset,
   `rest_total = Q_rest`, so `p[k] = p_tr[k]` — every non-pair class keeps its
   training probability and only the pair is re-weighted. The label shift is
   then *realised* by resampling the real labeled pool to `p`: for each class,
   `round(m·p[k])` examples are drawn (without replacement where the pool is
   large enough, with replacement otherwise, which is flagged in the report).
   Because only the mixing proportions change and every example keeps its true
   class, this is a genuine label shift — the class-conditionals `p(x|y)` are
   untouched.

   ```bash
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood
   # different pair, and concentrate 70% of the mass on it:
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood \
       --confusable-pair 3 6 --pair-rest-ratio 7 3
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood \
       --n-test 300 --test-prior 0.1 0.1 0.1 0.35 0.1 0.1 0.05 0.1
   python run_real_reject_option_exp.py runs/blood/model.pt runs/blood --sweep
   ```

   `--sweep` mirrors the synthetic sweep: it varies the adaptation-set size
   over `--sizes` (nested prefixes of one resampled pool per trial, scored on
   a fixed evaluation set) and writes the AuRC-vs-n, AuRC50-vs-n,
   epistemic-metrics (two panels: regret/epistemic-uncertainty overlaid, and
   the negligible portion), and coverage-at-target figures plus the per-size
   coverage-curve figures in a `coverage_curves/` subfolder, plus a sweep
   report. It also writes `base_accuracy_vs_n_test.png`: the test accuracy of
   the Bayesian learned-prior predictor and the supervised-prior plugin as
   they adapt from the `n` examples, against the (flat) true-prior plugin as
   the oracle ceiling.

   It reports the four plugin/Bayesian predictors' accuracy (no optimal-Bayes
   upper bound — the true conditionals are unknown for real data), the
   reject-option AuRC and epistemic-uncertainty metrics, and writes the
   risk/regret-coverage figures. The **supervised-prior plugin baseline** — the
   prior counted from the adaptation-set labels — appears both as an accuracy
   reference and as a reject-option predictor.

   **Pool split and evaluation size.** Each trial splits the labeled pool
   **adaptation-first**: the adaptation set (`--n-test`, or `max(--sizes)` in a
   sweep) is drawn at the target prior from the whole pool — per class, so it is
   stratified — and the disjoint remainder feeds the evaluation set. `--n-eval`
   then defaults to the **largest all-distinct evaluation set** that remainder
   supports (`floor(min_c eval_avail[c] / target[c])`), printed at startup;
   pass an integer to pin it. This maximises the evaluation set (typically most
   of the pool), which sharply reduces the variance of every reported metric —
   e.g. on Fashion-MNIST the true-prior oracle's trial std fell from 0.013 at
   `n_eval=500` to 0.002 at the auto size (~5500). Two consequences worth
   knowing: (1) because the evaluation set is now most of the pool, consecutive
   trials re-score nearly the same examples, so the error bars mainly reflect
   *adaptation* variance (which adaptation draw you got) rather than
   evaluation-sampling noise — the oracle / no-adaptation baselines therefore
   show near-zero spread, which is expected, not a bug; (2) a very large
   adaptation size competes with evaluation for a scarce high-target class and
   shrinks the auto `n_eval` (the script errors clearly if it would leave a
   wanted class with no evaluation examples). Note the `--n-eval` **default
   changed** from a fixed `2000` to this auto-max.

## Layout

```
prior_shift/
  data.py        Gaussian class-conditional generative model + exact Bayes posterior
  base_model.py  logistic-regression posterior + empirical training prior
  mcmc.py        Metropolis–Hastings prior sampler + Bayesian label posterior
  predictors.py  Bayes decision rule and plugin label-shift correction
  config.py      JSON loader/validator for the generator setting
configs/
  default.json          the original 4-Gaussian setting (used by default)
  three_gaussians.json  3-class example with different priors
data_tools/
  registry.py    per-dataset metadata: download URLs, class names, confusable pair
  download.py    stream files into data/<key>/ (skip-if-present, progress bars)
  loaders.py     load each source into a common uint8-image / int-label Dataset
  report.py      render the self-contained HTML analysis report
run_synth_bayesian_learning_exp.py end-to-end synthetic experiment, metrics, figures
run_synth_reject_option_exp.py     synthetic reject-option predictors + coverage curves
run_base_predictor_exp.py          train + calibrate a NN base predictor on a real dataset
run_real_reject_option_exp.py      real-data adaptation + reject-option experiment
download_datasets.py            download the real candidate datasets into data/
analyze_datasets.py             build a self-contained HTML report per dataset
```

Requires `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
