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
(from `--sweep`) `accuracy_vs_n_test.png`.

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
python run_experiment.py                 # 20 trials, writes figures + table
python run_experiment.py --trials 5 --n-test 1000 --seed 1
python run_experiment.py --sweep         # accuracy vs. #unlabeled examples
python run_experiment.py --sweep --sizes 100 1000 10000 --n-eval 4000
python run_experiment.py --config configs/three_gaussians.json   # other generator
```

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
run_experiment.py  end-to-end experiment, metrics, and figures
```

Requires `numpy`, `scipy`, `scikit-learn`, `matplotlib`.
