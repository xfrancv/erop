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

The reusable library is the `prior_shift/` package (method, sampling, and the
sweep engine); the top-level `base_predictor_training.py` / `rejopt_eval.py` /
`*_datasets.py` scripts are thin experiment drivers around it, the `run_*.sh`
wrappers pin the paper configuration, `reporting.py` is the text/JSON output
layer, and `reject_figures.py` / `figspec.py` / `render_figspecs.py` are the
plotting layer (both kept out of the package so the library never imports
matplotlib). Data outputs (`data/`, `figures/`, `runs/`) are gitignored and
reproducible.

`rejopt_eval.py` itself is now only the command line and the two mode drivers:
`build_parser` / `validate_args` / `resolve_target_prior` /
`resolve_model_prior` / `resolve_eval_size` turn flags into a configured run,
then `run_sweep_report` or `run_dirichlet_sweep_report` calls
`prior_shift.sweep.run_sweep` and hands the result to `reporting` and
`reject_figures`. Keep new work on that split: computation in `prior_shift/`,
presentation in `reporting.py` / `reject_figures.py`, flags in the driver.

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
  `generalize_curve`) plus the `Aggregation` value (`series`, `center`,
  `describe`) that keeps the report tables and the figures reporting the same
  central value. Pure numpy; the matching figure builders live in top-level
  `reject_figures.py`.
- **`prior_shift/sampling.py`** — drawing label-shifted samples from the labeled
  pool: `target_prior_from_weights`, `target_counts`, `resample_to_prior`,
  `split_adapt_eval`, `max_distinct_eval`, `sample_target_prior`. Raises rather
  than exiting, so the driver owns every user-facing message.
- **`prior_shift/sweep.py`** — the experiment's inner loop: one `run_sweep` call
  covers every trial at every adaptation size for *one* target prior and returns
  a `SweepResult` on the common `(len(sizes), replicates)` layout that the
  reporting and figure layers consume.
- **`prior_shift/mcmc.py`** — `sample_prior_posterior(..., sampler=...)` with **two
  chains targeting the same posterior**: `"gibbs"` (default, latent-variable
  sampler) and `"mh"` (random-walk Metropolis–Hastings in a
  softmax-reparameterised unconstrained space with the change-of-variables
  Jacobian). Gibbs is the default because at the built-in chain length it is the
  only one that converges — R-hat ≈1.000 vs. up to 2.4 at 100 classes, ~100× the
  effective sample size, and 4–6× faster; the module docstring carries the
  measurements. `mh` is the independent cross-check, not a run-of-record option.
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
  (or a deliberate note) for the other. *Equivalent in the limit*: at the default
  chain length only `gibbs` has actually converged, which is why it is the
  default. Comparing the two is the way to check the model rather than the
  sampler, but give `mh` a much longer chain before believing a disagreement.
- **`results.json` is the machine interface, the text report is for humans.**
  `rejopt_eval.py` writes both; `summary_table.py` reads the JSON, so the text
  report's wording and column widths are free to change. What *is* load-bearing:
  the JSON key names (bump `RESULTS_SCHEMA` when removing or renaming one), the
  filename `real_reject_option_sweep_report.txt` (the `make_*.sh` scripts glob
  for it as the marker of a completed run), and the `n_test` spelling in figure
  filenames (`make_figures.sh`). `summary_table.py` still carries a legacy text
  parser for run directories written before `results.json`; it can go once every
  run of record has been regenerated.
- **The replicate-axis aggregation is an explicit `Aggregation` value**
  (`prior_shift/reject_option.py`), constructed once per run in the driver and
  passed to both the report tables and every figure builder. Keep it that way:
  one object for both is what guarantees a figure's solid line and its table
  cell are the same statistic. Do not reintroduce module-level configuration.
- `figures/`, `runs/`, and `data/` are gitignored; don't commit generated outputs.
