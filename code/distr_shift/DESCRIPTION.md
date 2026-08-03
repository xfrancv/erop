# Epistemic Reject Option Prediction — details

The method, the metrics, the datasets and the command-line options behind the
minimal recipe in [README.md](README.md).

## Experiment on a single dataset

The `run_all_*.sh` scripts of the README loop over these two steps; run them
directly to reproduce or debug one dataset.

**Train base predictor.**
Two predictors are trained, one with and one without posterior calibration:
```
./run_base_pred_training.sh bloodmnist 
./run_base_pred_training.sh bloodmnist nocalib
```
The calibrated one lands in `runs/bloodmnist/`, the uncalibrated one in
`runs/bloodmnist_nocalib/`.
The script uses CUDA. If not available, change `DEVICE=cuda` to `DEVICE=cpu` in the header.

**Evaluate label-prior adaptation with the reject option.**
Bayesian label-prior adaptation is evaluated with the Bayesian and the epistemic
reject-option strategy. For the ablation the methods are also evaluated i) with
a misspecified model prior (`beta`) and ii) without posterior calibration
(`nocalib`).
```
./run_rejopt_eval.sh bloodmnist
./run_rejopt_eval.sh bloodmnist beta
./run_rejopt_eval.sh bloodmnist nocalib
```
Outputs go to a timestamped directory under `runs/bloodmnist/`,
`runs/bloodmnist/beta/` and `runs/bloodmnist_nocalib/` respectively; the
`nocalib` mode reads the uncalibrated base predictor trained above.


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

We draw $\theta$ from $p(\theta | D)$ with a **latent-variable Gibbs sampler**
(`--sampler gibbs`, the default). Each likelihood term
$\sum_y R(x_i, y)\alpha(y)$ is the marginal of a latent class $z_i$ with
$p(z_i = y \mid \alpha) \propto R(x_i, y)\alpha(y)$; given the assignments the
posterior of $\alpha$ is conjugate, $\alpha \mid z \sim
\text{Dirichlet}(\beta + \text{counts}(z))$. The sampler alternates these two
exact conditionals, so every move is accepted.

A **random-walk Metropolis–Hastings** chain (`--sampler mh`) targets the same
posterior and is kept as an independent cross-check. To respect the simplex
constraint it works in an unconstrained $z \in {\mathbb R}^{Y−1}$ via an
additive-logistic (softmax) reparameterisation and adds the change-of-variables
Jacobian $\sum_y \log \alpha(y)$. It is not the default: random-walk MH needs
$O((Y-1)^2)$ steps to traverse the posterior, so at the built-in chain length it
does not mix on the larger label sets (Gelman–Rubin $\hat R$ up to 2.4 at
$Y = 100$, against $\approx 1.000$ for Gibbs), and it is also 4–6× slower. See
the `prior_shift/mcmc.py` docstring for the measurements.

The Bayesian label posterior averages the normalised re-weighting over the
posterior draws,

$$
\hat p(y | x, D) = (1/N) \sum_i  normalise_y[ R(x, y) \alpha_i(y) ],
$$

and predictions minimise the expected user loss
$h(x) = \arg\min_{\hat y} \sum_y \hat p(y| x, D) \ell( \hat y, y) $ (argmax for 0/1 loss).

## Experiment

The experiment runs on **real datasets** (Fashion-MNIST, CIFAR-10/100 and six
MedMNIST v2 sets -- see [Real datasets](#real-datasets)). A neural base
predictor is trained and calibrated on the training split by
`base_predictor_training.py`; label shift is then *simulated* by resampling the
labeled val+test pool to a target prior, so the class conditionals `p(x|y)` are
untouched and only the mixing proportions change.
`rejopt_eval.py` adapts the prior from the unlabeled adaptation
inputs by MCMC and scores every predictor on a disjoint labeled evaluation set
-- or, with `--eval-on-adapt`, on the adaptation examples themselves (the
transductive deployment setting). The details -- the two target-prior
interfaces and their degenerate default, the adaptation-first split and its
transductive alternative, the dirichlet mode -- are in [Running the adaptation
experiment on real data](#running-the-adaptation-experiment-on-real-data).

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

On real data this is the regime the *confusable pair* lives in: two classes the
base predictor cannot separate well (cat/dog, melanoma / melanocytic nevi,
kidney-left / kidney-right) split the likelihood along a nearly flat direction,
so their prior mass is the part the unlabeled data cannot pin down. It is also
exactly where epistemic uncertainty is informative — the decisions that flip
with the unknown split.

**The built-in diagnostic.** The sampler compares, per class, the posterior
std of $\alpha(y)$ against the std of the best *supervised* estimator at the
same sample size — counting $n$ labels, $\sqrt{\alpha(y)(1-\alpha(y))/n}$
(`MCMCResult.ident_ratio`). When the problem is well conditioned the ratio is
~1–1.5 (unlabeled data is nearly as informative as labels); a ratio
**above 3×** for any class means the unlabeled data is far
less informative than labels would be, i.e. the prior is only weakly
identifiable — because of near-identical class conditionals or simply too
little data. This benchmark is self-calibrating: no absolute threshold on the
posterior width is needed, so the check works at any `n`.

`MCMCResult.identifiability_warning()` returns the warning message (or `None`
if healthy) naming the affected classes and their ratios, e.g.

```
!!! IDENTIFIABILITY WARNING (fired in 3/3 trials) !!!
    test prior only weakly identifiable from the unlabeled data: posterior
    std of alpha(y=1, y=2, y=3) is [11.9  7.  10.4]x the std of counting the
    same number of labels (threshold 3x). ...
```

The sweep report carries it as a `warn` column: the fraction of trials flagged
at each adaptation size `n`. If the warning fires, the learned prior — and
anything downstream of it — should not be trusted.

## Reject-option predictors

The experiment pairs the adaptation with **selective prediction**: the predictor
may abstain on inputs it is unsure about. A
reject-option predictor is a pair of a base predictor `h(x)` and an uncertainty
score `u(x)`; it emits `h(x)` when `u(x)` is below a threshold and rejects
otherwise. Sweeping the threshold traces out a curve, so no threshold has to be
fixed in advance.

Two reject-option predictors are compared. They share one base predictor — the
Bayesian learned-prior rule — and differ only in the score they rank by, so
their curves meet at full coverage. All are scored under the 0/1 loss, so the
conditional risk of a decision is one minus its posterior probability.

| Reject-option predictor | Base predictor `h(x)` | Uncertainty `u(x)` |
| --- | --- | --- |
| Bayesian, total uncertainty | Bayesian, learned prior | $\hat T(x,D)$ |
| Bayesian, epistemic uncertainty | Bayesian, learned prior | $\hat T(x,D)-\hat A(x,D)$ |

The supervised-prior plugin (prior counted from the adaptation-set *labels*) and
the true-prior plugin are kept as **accuracy** references only — see
`base_accuracy_vs_n_test.png` — not as reject-option curves. The label-aware
best-attainable *oracle* envelope was removed along with `--optimal-rejection`.

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
one coverage regime gets diluted. The report therefore also carries the
**coverage at target** — the dual statistic: the largest coverage at which the
selective metric stays within a budget for every accepted rank,
$\text{cov@}\varepsilon = \frac1n\max\{k : metric(j)\le\varepsilon\ \forall
j\le k\}$ (the first few ranks are a grace region, since selective metrics at
tiny $k$ are 0/1-grained). The budget is the **regret** budget
`--regret-target` (default 0.002), which accepts one or more values; the metric
is reported and plotted for each, one panel per budget in
`cov_at_target_vs_n_test.png`. (Coverage at a *risk* budget was dropped with
`--risk-target`: it needed a per-trial reference risk to be meaningful and no
downstream table or figure used it.)

**AuRC50.** The report also carries $\text{AuRC50}$: the same areas averaged
over the ranks with $coverage\ge0.5$ only. Rejecting more than half the inputs
is not an operating point anyone deploys, and it is exactly where the estimates
are noisiest ($risk(1)$ is a single example). Averaging over the window rather
than integrating over it keeps AuRC50 on the AuRC scale, so the two may be read
side by side — unlike AuGRC. **Caveat:** truncation makes the statistic
invariant to the ranking *within* the accepted half, and since the
reject-option predictors share one base predictor and differ only in their
ranking score, their gaps compress. A smaller AuRC50 gap is not evidence the
rankings agree more. The areas get their own two-panel figure,
`aurc50_vs_n_test.png`.

**No in-sample bias (default mode).** By default every predictor is scored on a
**labeled evaluation set disjoint from the adaptation examples** used to learn
the prior. The supervised plugin reference counts its prior from the *labels of
the adaptation set*, never from the evaluation set. `--eval-on-adapt` gives up
that disjointness on purpose — see **Transductive evaluation** below.

### Sweep: the adaptation-set size

Every measure is reported as a curve over `--sizes`, the number `n` of unlabeled
adaptation examples. Per trial the `n` adaptation examples are nested prefixes
of one resampled pool, so neighbouring sizes share draws and the curves reflect
`n` rather than re-sampling noise; by default each point is scored on the same
fixed labeled evaluation set (under `--eval-on-adapt`, on the `n` adaptation
examples themselves). Pass a single size for a one-point run.

**Single-number summary (`avg` row).** Every per-size sweep table above —
AuRC, AuRC50, AuGRC, coverage-at-target and the epistemic-uncertainty metrics —
ends with an `avg` row: each column's mean over the swept sizes (the plain mean
of the cells above it; where the table has a `warn` column its mean is shown
too). It is a convenience scalar for comparing runs at a glance, printed only in
the text tables, not in the figures. Two caveats keep it from being over-read:
the mean is over the *sampled* `--sizes`, which are log-spaced, so it weights
small `n` more heavily and is **only comparable across runs sharing the same
`--sizes`**; and it collapses exactly the `n`-dependence these curves exist to
show, so it is no substitute for the curve.

Figures are written to the run's timestamped output directory:
`aurc_vs_n_test.png` and `aurec_vs_n_test.png` (the AuRC/AuReC sweeps as two
independent single-panel figures), `gen_aurc_vs_n_test.png`,
`aurc50_vs_n_test.png`, `epistemic_metrics_vs_n_test.png`,
`cov_at_target_vs_n_test.png`, `base_accuracy_vs_n_test.png`,
`epi_vs_regret_calibration.png` (dirichlet mode), and the per-size
risk/regret-coverage curves under `coverage_curves/`. The argument setting is
saved alongside them as `rejopt_eval_args.txt`, which also
records the resolved target prior and **how it was obtained**. (Runs made
before the script was renamed carry the same file as
`run_real_reject_option_exp_args.txt`.)

**Uncertainty bands.** By default every figure shades a `mean ± s.e.m.` band
(the real-data dirichlet mode uses `± std` over sampled priors) around the mean
curve. Passing `--percentile-band X` (`X` in
`[0, 100]`) instead draws the **central `X`% percentile interval** — e.g. `80`
→ the 10th–90th percentile — and moves the solid line to the pointwise
**median**. The band is then generally asymmetric (it always contains the
median), and, unlike the s.e.m. band, describes the *spread of the replicate
values* rather than the *uncertainty of their mean*, so it does not shrink as
trials are added; at the default 10–20 replicates its edges are coarse. The
text tables follow suit: under `--percentile-band` their central value is the
**median** over replicates (matching each figure's solid line) instead of the
mean, while the reported `± std` dispersion is unchanged. The figure titles name
the median so the two are not conflated.

**Restyling figures without re-running (figure specs).** Every `*.png`
`rejopt_eval.py` writes drops a reusable **figure spec** beside it:
`<name>.figspec.json` (the editable text — labels, titles, colours, legend
placement, axis scales) plus `<name>.figspec.npz` (the numeric curve/band
arrays, kept exact rather than stringified into the JSON). The spec fully
describes the figure; `figspec.py` renders the PNG from it, so the two stay in
sync by construction (a re-render reproduces the original PNG). To tune a
figure's look for the paper — a data-free loop, no experiment re-run —
hand-edit its `.figspec.json` and/or apply a matplotlib style sheet, then
re-render with `render_figspecs.py`:

```bash
python render_figspecs.py figures/                      # re-render every spec (no-op check)
python render_figspecs.py figures/ --style styles/paper.mplstyle --format pdf
python render_figspecs.py figures/risk_coverage.figspec.json --format pdf svg
```

`--style` centralises presentation (fonts, sizes, line widths, colour cycle) in
one rcParams file decoupled from the data; `--format` emits vector output (PDF
or SVG) for LaTeX. The specs are gitignored along with the rest of `figures/`.

## Setup

`./setup_environment.sh` does the venv route below. To install by hand instead:

**conda:**

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
# 1. train + calibrate a base predictor on a real dataset
python base_predictor_training.py bloodmnist runs/blood --calibration bcts

# 2. adapt + evaluate the reject option. Always a sweep over --sizes.
#    With no target-prior flag the target IS the training prior: no label
#    shift, the degenerate control (the script warns).
python rejopt_eval.py runs/blood/model.pt runs/blood \
    --sizes 1 10 100 500 --trials 20

# an explicit shifted target prior, repeated over priors drawn around it
python rejopt_eval.py runs/blood/model.pt runs/blood \
    --sizes 1 2 5 10 50 100 200 500 --trials 20 \
    --test-prior 0.17 0.01 0.01 0.25 0.15 0.15 0.25 0.01 \
    --dirichlet 20 --trials-prior 20 --percentile-band 50

# the same, as shipped: the paper configuration per dataset
./run_rejopt_eval.sh bloodmnist            # or: ./run_all_rejopt_eval.sh sbatch

# collect several runs into one table (--reports takes each run's results.json,
# its text report, or its directory; or let make_summary_table.sh /
# make_ablation_table.sh find the latest run of each dataset for you)
python summary_table.py --reports runs/*/*/results.json \
    --sizes 1 10 avg --output summary_table.txt
```

## Real datasets

The experiment runs on the datasets below (see
[`tasks/datataset_proposal.md`](tasks/datataset_proposal.md) for how they were
chosen). Two helper scripts download and inspect them; these use only the
standard library plus NumPy/matplotlib/tqdm — no torch/torchvision/medmnist.

| Dataset | key | shape | classes | source | confusable pair |
|---------|-----|-------|---------|--------|-----------------|
| Fashion-MNIST | `fashion_mnist` | 28×28 grayscale | 10 | Zalando IDX files | Shirt / T-shirt |
| CIFAR-10 | `cifar10` | 32×32 RGB | 10 | fast.ai PNG-folder mirror | cat / dog |
| CIFAR-100 | `cifar100` | 32×32 RGB | 100 | fast.ai PNG-folder mirror (nested by superclass) | boy / girl |
| DermaMNIST | `dermamnist` | 28×28 RGB | 7 | MedMNIST v2 `.npz` (Zenodo) | melanoma / nevus |
| BloodMNIST | `bloodmnist` | 28×28 RGB | 8 | MedMNIST v2 `.npz` (Zenodo) | neutrophil / immature granulocyte |
| TissueMNIST | `tissuemnist` | 28×28 grayscale | 8 | MedMNIST v2 `.npz` (Zenodo) | glomerular / interstitial endothelial cells |
| OrganAMNIST | `organamnist` | 28×28 grayscale | 11 | MedMNIST v2 `.npz` (Zenodo) | kidney-left / kidney-right |
| OrganSMNIST | `organsmnist` | 28×28 grayscale | 11 | MedMNIST v2 `.npz` (Zenodo) | kidney-left / kidney-right |

CIFAR-100's fast.ai mirror is laid out two levels deep
(`<split>/<superclass>/<fine-class>/*.png`); the image-folder loader labels by
the leaf (fine-class) folder, so it yields the 100 fine classes. The
confusable pairs for TissueMNIST, OrganAMNIST and
OrganSMNIST are initial proposals (e.g. the near-identical left/right kidney in
the two Organ datasets); validate them before relying on them. The pair is
**reporting only** — it is named in the reports and in the per-draw dirichlet
lines, and it is where epistemic uncertainty is expected to concentrate, but it
never builds the target prior. OrganSMNIST is the sagittal-view counterpart of the
axial-view OrganAMNIST — the same 11 abdominal-organ labels, sliced along a
different plane.

**Split policy (`val_role`).** Each dataset's registry entry carries a
`val_role` field that controls how `base_predictor_training.py` uses the official
`val` split. The default, `"test"`, merges `val` into the *test* subset
(evaluation only) and carves the model-selection set out of `train` — this suits
the datasets with small test splits (DermaMNIST and BloodMNIST) as well as
Fashion-MNIST and CIFAR, which ship no official `val` split at all and so have
nothing to merge. TissueMNIST, OrganAMNIST and OrganSMNIST instead use
`val_role="train"`: they train on the *whole* official `train` split, use the
official `val` split as the model-selection set, and evaluate on the official
`test` split alone — their test splits are large enough (8,827–47,280) that no
merge is needed.

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

1. **`base_predictor_training.py`** trains and calibrates a neural-network base
   predictor. It splits the training subset (class-stratified) into a fit part
   and a model-selection part, selects the best epoch by validation error, and
   optionally calibrates on the model-selection part. `--calibration` selects
   the scheme: **`none`** (default) leaves the raw softmax posterior
   uncalibrated — an ablation for measuring the method's sensitivity to
   miscalibration; **`bcts`** is bias-corrected temperature scaling (scalar `T`
   + per-class bias); **`temperature`** is plain scaling. BCTS matters
   downstream: a single temperature flattens overconfident logits globally,
   which inflates the mean posterior of rare classes, and the label-shift MCMC
   misreads that bias as prior shift (on DermaMNIST this put 60% of the learned
   prior on a 1% class) — so pass `--calibration bcts` for the best-calibrated
   run; `none` isolates the uncalibrated baseline.
   The script reports a **calibration-consistency check** — mean calibrated
   posterior over the held-out split divided by class frequency, ~1 per class
   when healthy, stored in the bundle and re-printed with a warning by
   `rejopt_eval.py` (it is the bias-detecting complement of the
   variance-detecting `ident_ratio`). The saved `model.pt` bundle holds the
   best-epoch weights + `T` + bias + the check + estimated training prior +
   normalization.
   Architectures default per dataset: LeNet for Fashion-MNIST, a 32×32/28×28
   small-input-adapted ResNet-18 (3×3 stem, no max-pool, trained from scratch)
   for CIFAR-10 and the MedMNIST sets.

   ```bash
   python base_predictor_training.py fashion_mnist runs/fashion
   python base_predictor_training.py bloodmnist runs/blood --epochs 30 --device cuda
   ```

2. **`rejopt_eval.py`** loads a `model.pt` base predictor,
   computes calibrated posteriors on the (val+test) pool, **simulates label
   shift** by resampling that labeled pool to a target prior, and runs the
   adaptation, the predictors and the reject-option curves. It always sweeps
   `--sizes`.

   **Target prior: two interfaces.**

   - `--test-prior P ... P` — an explicit `Y`-vector summing to 1.
   - `--prior-classes Y ... Y` with `--prior-weights W ... W` and
     `--prior-rest-weight R` — *relative* per-class weights: the named classes
     get the aligned weights, every unnamed class gets `R` individually (so the
     unnamed classes stay equiprobable among themselves), and the whole vector
     is normalised. Convenient with many classes: on CIFAR-100 the paper run is
     `--prior-classes 11 35 --prior-weights 1 1 --prior-rest-weight 5`, i.e.
     down-weight the boy/girl pair by 5x against everything else.

   The two are mutually exclusive. **With neither, the target prior is the
   training prior** — no label shift at all. That default is deliberate: it is
   the degenerate control that measures what the adaptation costs when nothing
   needs adapting (every predictor then coincides up to posterior noise, and
   the regret is ~0 by construction). Because that is invisible in the numbers,
   the script says so on stdout and in the report:

   ```
   confusable   : neutrophil / immature granulocytes  (classes 6, 3; registry default)
   target prior : training prior (DEFAULT: no label shift -- degenerate control)
   warning: the target prior defaults to the TRAINING prior, i.e. no label shift;
            pass --test-prior or --prior-classes/--prior-weights/--prior-rest-weight
            for a shifted target.
   ```

   The two lines are independent by design: the first names a *dataset*
   property (the registry's confusable pair, carried into the reports and the
   per-draw dirichlet lines), the second says how *this run* obtained its
   target prior — `training prior (DEFAULT...)`, `explicit (--test-prior)`, or
   `class weights ...`, prefixed `central prior for Dir(s * p)` in dirichlet
   mode. Under `--dirichlet` the default is **not** degenerate: the draws
   around the training prior are genuinely shifted.

   The shift is *realised* by resampling the real labeled pool to the target
   prior: for each class, `round(m·p[k])` examples are drawn (without
   replacement where the pool is large enough, with replacement otherwise,
   which the report flags). Because only the mixing proportions change and
   every example keeps its true class, this is a genuine label shift — the
   class conditionals `p(x|y)` are untouched.

   ```bash
   python rejopt_eval.py runs/blood/model.pt runs/blood \
       --sizes 1 10 100 500
   python rejopt_eval.py runs/blood/model.pt runs/blood \
       --sizes 1 10 100 500 --test-prior 0.1 0.1 0.1 0.35 0.1 0.1 0.05 0.1
   ```

   **Many target priors: `--dirichlet SUM_PARAMS`** (see
   [`tasks/multiple_priors_polished.md`](tasks/multiple_priors_polished.md)).
   One fixed target prior makes every number conditional on that one choice.
   With `--dirichlet s` the whole sweep is repeated over `--trials-prior`
   target priors drawn from `Dir(s·p)`, `p` being the configured prior in its
   *central* role; larger `s` concentrates the draws around `p`. The methods
   use the matching Dirichlet as their model prior (well specified) — passing
   `--beta B` there replaces it with the symmetric `Dir(B)` instead, i.e.
   deliberately misspecifies the model while the data keep coming from
   `Dir(s·p)`. The regret reference is each sampled prior; adaptation draws
   truncate to pool availability (duplicated inputs would double-count evidence
   in the MCMC likelihood) while evaluation draws may use replacement. Figures
   then aggregate the per-prior means with `±1 std over priors` bands, the
   report adds a **paired win-rate block** per area table (on what percentage
   of the sampled priors the total-uncertainty ranking beats each competitor)
   and the per-draw prior summary, and two extra outputs appear:
   `epi_vs_regret_calibration.png` (per-draw average epistemic uncertainty vs.
   realized regret against the `y = x` line — the calibration check) and
   `sampled_priors.txt` (every draw, as an audit trail).

   The sweep varies the adaptation-set size over `--sizes` (nested prefixes of
   one resampled pool per trial, scored on a fixed evaluation set — or, with
   `--eval-on-adapt`, on the prefix itself) and writes
   the AuRC-vs-n and AuReC-vs-n figures (`aurc_vs_n_test.png`,
   `aurec_vs_n_test.png` — two independent single-panel figures, one per curve,
   so each drops straight into a paper), AuRC50-vs-n, AuGRC-vs-n, the
   epistemic-metrics figure (two panels: regret/epistemic-uncertainty overlaid,
   and the negligible portion), the coverage-at-target figure (one panel per
   `--regret-target`), the per-size coverage-curve figures in a
   `coverage_curves/` subfolder, the sweep report
   `real_reject_option_sweep_report.txt` and its machine-readable twin
   `results.json` (the same numbers, keyed by name — this is what
   `summary_table.py` reads, so the text report stays free to be reworded). It
   also writes
   `base_accuracy_vs_n_test.png`: the test accuracy of the Bayesian
   learned-prior predictor as it adapts from the `n` examples, against the
   training-prior plugin (no adaptation) and the true-prior plugin as the oracle
   ceiling — both flat in `n`, since neither uses the adaptation examples
   (under `--eval-on-adapt` they do vary, through the evaluation sample).

   The report tables cover the four plugin/Bayesian predictors' accuracy (no
   optimal-Bayes upper bound — the true conditionals are unknown for real
   data), the reject-option areas (AuRC, AuRC50, AuGRC — risk and regret), the
   coverage-at-target rows, and the epistemic-uncertainty metrics. The
   **supervised-prior plugin baseline** — the prior counted from the
   adaptation-set labels — is an accuracy reference only.

   **Pool split and evaluation size.** Each trial splits the labeled pool
   **adaptation-first**: the adaptation pool (`max(--sizes)` examples) is drawn
   at the target prior from the whole pool — per class, so it is
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

   **Transductive evaluation (`--eval-on-adapt`).** With this flag there is no
   second draw: the `n` adaptation examples *are* the evaluation set
   (`merge_eval_adapt_sets_polished.md`). This is the deployment setting — a
   batch of unlabeled inputs arrives, its prior is learned from it, and that
   same batch has to be classified — and the setting in which the epistemic
   term is exactly the posterior-expected regret *on the scored points* rather
   than on a separate sample. `--n-eval` is rejected under it (the evaluation
   size is `--sizes`), and the pool-competition guards disappear. What to expect:

   - **Regret may go negative.** The reference stays the plugin at the true
     target prior, which on a finite batch is not the loss-minimising prior;
     a method that infers the batch's own prior can beat it. The report's
     **transductive floor** table (the plugin at the batch's *empirical* prior:
     mean regret against the oracle, its accuracy, the number of classes the
     batch actually contains, and the realized evaluation size) says how much
     of the observed regret is finite-batch prior mismatch rather than method
     quality. `coverage @ regret <= eps` is then a budget against the
     *population* oracle, not against an attainable optimum.
   - The floor is usually **small**, because the batch is *resampled* to the
     target prior class-by-class rather than drawn i.i.d. from it: its
     empirical prior differs from the target only by the integer rounding of
     `target_counts`. It grows as `n` shrinks (coarser rounding).
   - **Error bars grow and change meaning.** The disjoint default scores nearly
     the whole pool every trial, so its error bars are almost pure adaptation
     variance and the non-adaptive baselines have near-zero spread. Here the
     evaluation set is `n` examples, so evaluation-sampling noise returns and
     dominates at small `n`, and the two plugin baselines acquire visible
     spread.
   - **The sweep confounds two effects**: AuRC vs. `n` mixes "adaptation
     improves with `n`" with "evaluation gets less noisy with `n`". That is
     intrinsic to the design (a deployed batch of 50 *is* both estimated from
     and scored on 50 points); raising `--trials` averages the noise down but
     does not remove the confound.
   - **Coverage granularity is `1/n`**, so at small `n` the low-coverage tail is
     0/1-grained; AuRC50 and AuGRC are the robust readings there. At `n` below
     the class count (e.g. 50 on CIFAR-100) the batch simply cannot contain
     every class — the honest transductive answer, reported as `classes seen`.
   - The supervised-prior baseline is *not* computed here; under
     `--eval-on-adapt` it would coincide exactly with the transductive floor.

## Layout

```
prior_shift/
  mcmc.py          Metropolis-Hastings / Gibbs prior sampler + Bayesian label posterior
  predictors.py    Bayes decision rule and plugin label-shift correction
  reject_option.py selective risk/regret curves, their areas, coverage-at-target,
                   the uncertainty decomposition, and the replicate aggregation
  sampling.py      target priors from weights; stratified adaptation/evaluation draws
  sweep.py         the inner loop: every trial x size for one target prior
data_tools/
  registry.py    per-dataset metadata: download URLs, class names, confusable pair
  download.py    stream files into data/<key>/ (skip-if-present, progress bars)
  loaders.py     load each source into a common uint8-image / int-label Dataset
  report.py      render the self-contained HTML analysis report
base_predictor_training.py        train + calibrate a NN base predictor on a real dataset
rejopt_eval.py                    adaptation + reject-option experiment (always a sweep):
                                  the command line and the two mode drivers
reporting.py                      the text report and the machine-readable results.json
reject_figures.py                 figure builders for the experiment's outputs
figspec.py / render_figspecs.py   declarative figure specs + offline re-rendering
summary_table.py                  one LaTeX-ish table across datasets and sizes
download_datasets.py              download the real datasets into data/
analyze_datasets.py               build a self-contained HTML report per dataset
run_base_pred_training.sh         train one dataset (or 'all'); [nocalib] variant
run_rejopt_eval.sh                the paper configuration for one dataset (or 'all');
                                  [nocalib | beta] ablation modes
run_all_base_pred_training.sh     train every dataset, both variants; [sbatch]
run_all_rejopt_eval.sh            every dataset, all three modes; [sbatch]
setup_environment.sh              create .venv/ and install requirements.txt
download_datasets.sh              download every dataset + build the HTML reports
make_summary_table.sh             summary_table.txt from the latest run per dataset
make_ablation_table.sh            summary_table_ablation.txt (default/beta/nocalib)
make_figures.sh                   copy the latest RC-curve figures into figures/
                                  and re-render them as styled PDFs
```

Requires `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm` and -- for the
two experiment scripts -- `torch`/`torchvision`.
