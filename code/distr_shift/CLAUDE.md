# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

* Claude never modifies TODO.md unless explicitely asked by the user.
* Claude never uses git without explicitely asking the user for permission. 
* Claude never modifies files in 'tasks/' directory without explicitely asking the user for permission.


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
plain scripts. Dependencies:

- **Data download/analysis** (`download_datasets.py`, `analyze_datasets.py`) and
  the plotting helpers: `numpy scipy scikit-learn matplotlib tqdm` (see
  [requirements.txt](requirements.txt)).
- **The experiment scripts** (`base_predictor_training.py`,
  `rejopt_eval.py`): additionally need **torch/torchvision**,
  pinned to the `cu126` wheels — this machine's NVIDIA driver caps at CUDA 12.9
  and the default PyPI (cu130) wheel fails `torch.cuda.is_available()`. Since
  the synthetic experiment was removed, every experiment run needs torch.

Setup (conda recommended, Python 3.11; use `python -m pip` so installs land in the
active env):

```bash
conda create -n distr_shift python=3.11 && conda activate distr_shift
python -m pip install -r requirements.txt
```

Common runs:

```bash
# data: download + inspect (no torch needed)
python download_datasets.py --list
python analyze_datasets.py fashion_mnist        # writes data/reports/<key>.html

# train a base predictor, then adapt (torch needed)
python base_predictor_training.py fashion_mnist runs/fashion
# always sweeps --sizes; the target prior defaults to the TRAINING prior (no
# shift — the degenerate control), so a real experiment names one:
python rejopt_eval.py runs/fashion/model.pt runs/fashion \
    --test-prior 0.25 0.01 0.43 0.01 0.01 0.01 0.25 0.01 0.01 0.01 --dirichlet 20

# the paper configuration: per dataset (or 'all'), then every dataset at once
./run_base_pred_training.sh bloodmnist [nocalib]
./run_rejopt_eval.sh bloodmnist [nocalib | beta]
./run_all_base_pred_training.sh [sbatch]
./run_all_rejopt_eval.sh [sbatch]

# LaTeX-ish summary table across datasets
python summary_table.py --reports runs/*/*/real_reject_option_sweep_report.txt \
    --sizes 1 10 avg --output summary_table.txt
```

Every experiment run also drops a `*_args.txt` (command line + timestamp + the
args + the resolved target prior and how it was obtained) next to its figures,
so past runs are self-documenting — read those to reproduce a figure.

## Architecture

The reusable library is the `prior_shift/` package; the top-level
`base_predictor_training.py` / `rejopt_eval.py` / `*_datasets.py` scripts are
thin experiment drivers around it, the `run_*.sh` wrappers pin the paper
configuration, and
`reject_figures.py` / `figspec.py` / `render_figspecs.py` are the plotting layer
(kept out of the package so the library never imports matplotlib). Data outputs
(`data/`, `figures/`, `runs/`) are gitignored and reproducible.

Core method pipeline: `base_predictor_training.py` trains and
temperature/BCTS-calibrates a CNN for `p_tr(y|x)` and records the empirical
`p_tr(y)` in its `model.pt` bundle → `rejopt_eval.py` resamples
the labeled val+test pool to a chosen target prior → `mcmc.py` samples the test
prior `α` from the label-shift posterior over the unlabeled adaptation inputs →
`predictors.py` applies the plugin label-shift correction and Bayes decision
rule → `reject_option.py` turns the result into selective risk/regret curves and
their areas.

- **`prior_shift/reject_option.py`** — the reject-option computation layer
  (`REJECT_LABELS`, `selective_curves`, `coverage_at_target`,
  `bayesian_posterior_and_aleatoric`, `epistemic_metrics`, `truncated_area`,
  `generalize_curve`) plus the replicate-axis aggregation (`configure_*`,
  `_series`, `_center`) that keeps the report tables and the figures reporting
  the same central value. Pure numpy; the matching figure builders live in
  top-level `reject_figures.py`.
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

`data_tools/` (separate from `prior_shift/`) handles real datasets:
`registry.py` (per-dataset URLs, class names, the designated *confusable pair* —
reporting only: it is named in the reports and the per-draw dirichlet lines, but
never builds the target prior), `download.py`, `loaders.py` (each source →
common uint8-image/int-label dataset), `report.py` (self-contained base64 HTML
report).

**The shell wrapper layer** (four scripts, each usable directly or via `sbatch`;
their headers carry the Slurm directives):

- `run_base_pred_training.sh <dataset|all> [nocalib]` → `base_predictor_training.py`.
  Default trains with `--calibration bcts` into `runs/<ds>/`; `nocalib` omits the
  flag (raw softmax) and writes `runs/<ds>_nocalib/`.
- `run_rejopt_eval.sh <dataset|all> [nocalib|beta]` → `rejopt_eval.py`, holding the
  per-dataset paper configuration (target prior, `--sizes`, `--dirichlet`,
  `--percentile-band`) in one `run_<dataset>` function each. `nocalib` reads/writes
  `runs/<ds>_nocalib/` (the uncalibrated ablation); `beta` adds `--beta $BETA_SUM`
  (misspecified model prior) and writes `runs/<ds>/beta/`.
- `run_all_base_pred_training.sh [sbatch]` / `run_all_rejopt_eval.sh [sbatch]` — loop
  the single-dataset wrappers over every dataset and every mode, either in the
  current shell or as one Slurm job per run.

Keep the two `nocalib` spellings in sync: it is both the wrappers' mode keyword
and the `_nocalib` run-directory suffix that `summary_table_ablation.sh` reads.

Target prior: two interfaces, `--test-prior` (explicit vector) and
`--prior-classes`/`--prior-weights`/`--prior-rest-weight` (relative weights).
With neither, the target **is** the training prior — no label shift, the
deliberate degenerate control; the script says so in the report and warns on
stdout, and `--dirichlet` still makes the individual draws shifted.

## Working conventions specific to this repo

- **`tasks/*.md` are design specs, not TODOs.** Each is a written proposal that a
  script/module implements; source docstrings cite them by filename (e.g. the
  dirichlet mode ↔ `multiple_priors_polished.md`). When changing a feature, read
  its task spec first and keep the two consistent. `tasks/historical/` holds the
  specs of features removed with the synthetic experiment — read as history, not
  as a description of the code. `TODO.md` is the actual short task list.
- **README.md is a living lab notebook**, not just install docs — it records the
  quantitative results and their interpretation. If you change behavior that
  affects the reported numbers, figures, or metric definitions, update the relevant
  README section in the same change.
- The `mh` and `gibbs` samplers must stay statistically equivalent — they are two
  routes to the same posterior; a change to one usually needs the matching change
  (or a deliberate note) for the other.
- **Do not rename what `summary_table.py` parses**: the report table titles
  `AuRC (regret)` / `win% AuRC (regret)`, the truncated column headers
  `Bayesian, epistemic un` / `Bayesian, total uncert`, and the report filename
  `real_reject_option_sweep_report.txt` (also hard-wired in `summary_table*.sh`).
  The `n_test` spelling in figure filenames is likewise load-bearing for
  `figures.sh`.
- `figures/`, `runs/`, and `data/` are gitignored; don't commit generated outputs.
