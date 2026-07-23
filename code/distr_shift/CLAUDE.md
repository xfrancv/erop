# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

* Claude never modifies TODO.md unless explicitely asked by the user.

## What this is

Research code for **Bayesian label-prior adaptation under label shift**: learn the
*test* label prior `p_te(y)` from *unlabeled* test data by Bayesian inference, then
use it to correct a discriminative classifier when only the prior shifts between
train and test (class conditionals `p(x|y)` are assumed unchanged). The headline
contribution is a reject-option (selective-prediction) story that separates
**epistemic** uncertainty (label is sensitive to the unknown prior) from
**total/aleatoric** uncertainty. See [README.md](README.md) — it is long and
authoritative on the math, the metrics (AuRC / AuRC50 / coverage-at-target /
regret-coverage), and the experimental findings.

## Repo layout note

This project lives at `code/distr_shift/` inside a larger git repo whose root is
`/home/xfrancv/Work/erop` (siblings: `arxiv/`, `icml2026/`, `aaai2027/`, paper
drafts). Run git from wherever, but treat `code/distr_shift/` as the project root.

## Environment & commands

There is **no test suite, no linter, and no build step** — everything is run as
plain scripts. Two dependency tiers:

- **Synthetic experiments + data download/analysis**: `numpy scipy scikit-learn
  matplotlib tqdm` only (see [requirements.txt](requirements.txt)).
- **Real-data NN scripts** (`run_base_predictor_exp.py`,
  `run_real_reject_option_exp.py`): additionally need **torch/torchvision**,
  pinned to the `cu126` wheels — this machine's NVIDIA driver caps at CUDA 12.9
  and the default PyPI (cu130) wheel fails `torch.cuda.is_available()`.

Setup (conda recommended, Python 3.11; use `python -m pip` so installs land in the
active env):

```bash
conda create -n distr_shift python=3.11 && conda activate distr_shift
python -m pip install -r requirements.txt
```

Common runs:

```bash
# synthetic accuracy experiment (default config, 20 trials, writes figures + table)
python run_synth_bayesian_learning_exp.py
python run_synth_bayesian_learning_exp.py --sweep --sizes 100 1000 10000   # vs. #unlabeled
python run_synth_bayesian_learning_exp.py --config configs/three_gaussians.json

# synthetic reject-option (risk/regret-coverage curves + AuRC)
python run_synth_reject_option_exp.py
python run_synth_reject_option_exp.py --sweep --trials 10 --regret-target 0.005 0.01

# regenerate all shipped synthetic figures (writes into figures/<config>/)
./run_all_synth_exp.sh

# real data: download + inspect (no torch needed)
python download_datasets.py --list
python analyze_datasets.py fashion_mnist        # writes data/reports/<key>.html

# real data: train a base predictor, then adapt (torch needed)
python run_base_predictor_exp.py fashion_mnist runs/fashion
python run_real_reject_option_exp.py runs/fashion/model.pt runs/fashion --auto-target-prior
```

Every experiment run also drops a `*_args.txt` (command line + timestamp + the
args that mode actually reads + config priors) next to its figures, so past runs
are self-documenting — read those to reproduce a figure.

## Architecture

The reusable library is the `prior_shift/` package; the six top-level
`run_*.py` / `*_datasets.py` scripts are thin experiment drivers around it. Data
outputs (`data/`, `figures/`, `runs/`) are gitignored and reproducible.

Core method pipeline (synthetic): `data.py` generates 2-D Gaussian
class-conditional data → `base_model.py` fits logistic regression for the
training posterior `p_tr(y|x)` and empirical `p_tr(y)` → `mcmc.py` samples the
test prior `α` from the label-shift posterior over unlabeled data → `predictors.py`
applies the plugin label-shift correction and Bayes decision rule.

- **`prior_shift/data.py`** — `GaussianClassConditionalModel`; closed-form exact
  Bayes posterior for any prior (the oracle upper bound). `cov_from_axes(sx,sy,theta)`
  builds SPD covariances from the readable rotated-axes form.
- **`prior_shift/config.py`** — `ExperimentConfig`, the JSON loader/validator for a
  generator setting (`means`, `covs`, `train_prior`, `test_prior`). Covariances are
  either a full 2×2 list or `{"sx","sy","theta"}`; the loader validates SPD via
  Cholesky and that priors are positive and sum to 1. Number of classes is implied
  by `len(means)`.
- **`prior_shift/mcmc.py`** — `sample_prior_posterior(..., sampler=...)` with **two
  interchangeable chains targeting the same posterior**: `"mh"` (default,
  random-walk Metropolis–Hastings in a softmax-reparameterised unconstrained space
  with the change-of-variables Jacobian) and `"gibbs"` (latent-variable sampler).
  Returns an `MCMCResult` carrying the built-in **identifiability diagnostic**
  (`ident_ratio`, `identifiability_warning()`): posterior std of `α(y)` vs. the std
  of counting the same number of labels; ratio > 3× ⇒ the prior is only weakly
  identifiable (near-identical class conditionals or too little data) and the
  learned prior should not be trusted.
- **`prior_shift/predictors.py`** — `zero_one_loss_matrix`, `bayes_decision` (argmin
  expected loss), `corrected_posterior` (the plugin `p(y|x) ∝ p_tr(y|x)·α(y)/p_tr(y)`).
- **`prior_shift/target_prior_search.py`** — closed-form (no-MCMC) autonomous
  selection of a *benchmark* target prior for real data (`--auto-target-prior`):
  predicts pair identifiability from the mixture-likelihood Fisher information and
  scores candidates by a plugin "flip test". Implements
  [tasks/target_label_prior_selection.md](tasks/target_label_prior_selection.md).

`data_tools/` (separate from `prior_shift/`) handles real datasets:
`registry.py` (per-dataset URLs, class names, the designated *confusable pair*),
`download.py`, `loaders.py` (each source → common uint8-image/int-label dataset),
`report.py` (self-contained base64 HTML report).

## Working conventions specific to this repo

- **`tasks/*.md` are design specs, not TODOs.** Each is a written proposal that a
  script/module implements; source docstrings cite them by filename (e.g.
  `target_prior_search.py` ↔ `target_label_prior_selection.md`). When changing a
  feature, read its task spec first and keep the two consistent. `TODO.md` is the
  actual short task list.
- **README.md is a living lab notebook**, not just install docs — it records the
  quantitative results and their interpretation. If you change behavior that
  affects the reported numbers, figures, or metric definitions, update the relevant
  README section in the same change.
- The `mh` and `gibbs` samplers must stay statistically equivalent — they are two
  routes to the same posterior; a change to one usually needs the matching change
  (or a deliberate note) for the other.
- `figures/`, `runs/`, and `data/` are gitignored; don't commit generated outputs.
