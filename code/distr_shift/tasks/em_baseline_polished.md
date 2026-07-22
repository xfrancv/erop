# Plan

Implement a new baseline method based on the EM algorithm and a plugin Bayes
reject-option predictor.

# Context

The current code base evaluates two reject-option predictors based on Bayesian
learning (total and epistemic uncertainty of an MCMC-averaged predictor). The
goal is to compare them against the classical approach for this task: the test
label prior is estimated by the EM algorithm, and the estimated prior is plugged
into a plugin Bayes reject-option predictor. Conceptually, EM is the
MLE/point-estimate counterpart of the MCMC posterior over the *same* likelihood
— so this baseline isolates the effect of point-estimating the test prior versus
averaging over its posterior.

# The new baseline: EM algorithm + plugin Bayes reject-option predictor

## Setting

We work under the label-shift assumption: the class conditionals are unchanged
between train and test, $p(x\mid y)=p_{tr}(x\mid y)$, while the label prior
shifts, $p(y)\neq p_{tr}(y)$. The base predictor supplies the calibrated
training posterior $p_{tr}(y\mid x)$ and the training prior $p_{tr}(y)$; the
unknown parameters are the test prior $\theta=(p(1),\ldots,p(Y))$.

## EM estimate of the test prior

Assume unsupervised (unlabeled) adaptation data $D=(x_1,\ldots,x_m)$ are given.
The EM algorithm estimates the test label prior by maximizing the log-likelihood
of the unsupervised data. Using label shift,
$\sum_y p_{tr}(y\mid x)\frac{p(y)}{p_{tr}(y)}=\frac{p(x)}{p_{tr}(x)}$, so the
log-likelihood decomposes as

$$
\log p(D\mid \theta)
=\sum_{i=1}^m \log \sum_y p_{tr}(y\mid x_i)\frac{p(y)}{p_{tr}(y)}
\;+\;\underbrace{\sum_{i=1}^m \log p_{tr}(x_i)}_{\text{const w.r.t. }\theta},
$$

where the second term does not depend on $\theta$ and can be dropped during
optimization.

The EM algorithm starts from an initial prior estimate — e.g. the uniform prior
$\hat p(y)=(\tfrac{1}{Y},\ldots,\tfrac{1}{Y})$ — and iterates two steps:

**E-step.** Compute the responsibilities (posterior over the label) for each
adaptation example:

$$
\alpha_i(y)\propto p_{tr}(y\mid x_i)\,\frac{\hat p(y)}{p_{tr}(y)},
\qquad i=1,\ldots,m,\;\; y=1,\ldots,Y,
$$

normalized so that $\sum_y \alpha_i(y)=1$.

**M-step.** Update the label prior:

$$
\hat p(y)\gets \frac{1}{m}\sum_{i=1}^m \alpha_i(y),\qquad y=1,\ldots,Y.
$$

The iteration stops when the increase in the log-likelihood between two
consecutive steps falls below a threshold (or a maximum iteration count is
reached). This is the standard Saerens–Latinne–Decaestecker (2002) EM procedure
and monotonically increases the log-likelihood above.

## Plugin Bayes reject-option predictor

Given the learned test prior $\hat p(y)$, form the plugin label-shift posterior

$$
q_{em}(y\mid x)=\frac{p_{tr}(y\mid x)\,\hat p(y)/p_{tr}(y)}
{\sum_{y'}p_{tr}(y'\mid x)\,\hat p(y')/p_{tr}(y')}.
$$

The base predictor is the Bayes-optimal decision under this posterior, and the
uncertainty score is the corresponding conditional Bayes risk (loss convention:
$\ell(y,\hat y)$ is the loss of predicting $\hat y$ when the true label is $y$;
in code this is the loss matrix indexed `loss[yhat, y]`):

$$
\hat h_{em}(x)=\arg\min_{\hat y}\sum_{y} q_{em}(y\mid x)\,\ell(y,\hat y),
\qquad
u_{em}(x)=\min_{\hat y}\sum_{y} q_{em}(y\mid x)\,\ell(y,\hat y).
$$

Note that the normalization of $q_{em}$ is irrelevant for the $\arg\min$ (so the
decision could equivalently use the unnormalized weights), but it is required
for $u_{em}$, which is compared across examples to rank them for rejection.
Higher $u_{em}$ means more uncertain: examples are accepted in ascending order of
$u_{em}$, matching the convention used by the existing reject-option predictors.

Being a point estimate, this baseline has no epistemic component; it contributes
a single reject-option curve (its total uncertainty equals its aleatoric
uncertainty), analogous to the retired supervised-prior plugin.

# Task

Implement the described baseline in the reject-option experiments used by both
`run_real_reject_option_exp.py` and `run_synth_reject_option_exp.py`.
Concretely:

- Add an EM prior estimator (E-/M-step iteration above) that consumes the
  unlabeled adaptation posteriors $p_{tr}(y\mid x_i)$ and the training prior,
  and returns $\hat p(y)$. It mirrors the existing supervised-prior path, but
  the prior comes from EM on the unlabeled inputs instead of from label counts.
- Build the plugin Bayes reject-option predictor from $\hat p(y)$: reuse
  `corrected_posterior` for $q_{em}$, `bayes_decision` for $\hat h_{em}$, and the
  conditional Bayes risk `(post @ loss.T).min(axis=1)` for $u_{em}$.
- Register the new baseline as a reject-option predictor (add an entry to
  `REJECT_LABELS`) so it appears in the risk/regret-coverage comparison figures.
- Augment the accuracy-vs-adaptation-size figure (`make_base_accuracy_figure`)
  with a curve for the EM plugin base predictor, estimating $\hat p(y)$ per
  adaptation size inside the sweep (alongside the supervised-prior curve).
