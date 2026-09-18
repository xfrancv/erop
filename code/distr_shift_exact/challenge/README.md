# Challenge: *Know what you don't know under a new prior*

Self-contained pipeline for the TissueMNIST label-prior-shift competition.
Copying this directory elsewhere is enough to run it — nothing here imports from
the parent project.

The specification is `../tasks/challenge_polish.md`. Section references of the
form **C4**, **C9.2** point there; **S2.1**, **S6.3** point at the parent
`../README.md`, which remains the authoritative description of the method.

## The task

A CNN is trained on TissueMNIST and calibrated to emit `p_tr(y | x)`. At test
time a competitor receives **batches of unlabeled images**, each batch drawn
i.i.d. under one unknown label prior `θ*` taken from a known finite set `Θ` of
8 admissible priors. For every image they submit a label and a **confidence**;
the organiser ranks by confidence, keeps the most confident 80 %, and scores the
**regret against a reference predictor that was given the true prior**,
averaged over the seven batch sizes `m ∈ {1, 2, 5, 10, 20, 50, 100}`. Lower is
better, and negative scores are possible (the intended optimum reaches −0.0016
at `m = 20`).

`m_max` is 100 rather than the parent project's 500: measured on the trained
model, the two best rejectors are separated by *exactly* zero at `m ≥ 100`, so
`m = 200` and `m = 500` cost 77 % of all rows and return no ranking information.
See C9.8 for the numbers, including why the grid stops at 100 rather than 50.

The training prior is deliberately **not** a member of `Θ`, so every batch is
shifted and the non-adapted predictor is beatable on all of them.

## Kaggle 

Install Kaggle:
```
pip install kaggle
```

Sandbox submittion to test the evaluation engine:
```
kaggle competitions submit know-what-you-dont-know-under-a-new-prior -f sample_submission.csv -m "test submission" --sandbox

```

Create solution:
```
kaggle competitions solution create know-what-you-dont-know-under-a-new-prior -p solution.csv
```

Get solution status:
```
kaggle competitions solution status know-what-you-dont-know-under-a-new-prior     --json
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python selftest.py                      # no data, no network, ~10 s
```

Only `train_base_model.py` needs torch; everything downstream reads the
calibrated log-posterior it writes and is pure NumPy/SciPy/pandas.

## Pipeline

```bash
python download_data.py                                             # ~60 MB
python train_base_model.py out/model --epochs 30                    # hours on CPU
python make_priors.py out/model priors_tissuemnist_challenge.txt
python prepare_kaggle_data.py out/model priors_tissuemnist_challenge.txt out/kaggle
python make_dev_benchmark.py out/model priors_tissuemnist_challenge.txt \
    out/kaggle out/kaggle
python make_metric_notebook.py metric-template.ipynb
```

Then the pre-launch checks of C7 — compute every baseline and score it:

```bash
python baseline_solutions.py out/kaggle out/submissions
./run_baselines.sh                      # the same, plus the paired comparison
```

Finally the bundle that ships to students, assembled from `student/`,
`chal/metric.py` and the trained model:

```bash
python make_student_bundle.py out/kaggle_code --zip
```

`student/` holds those sources. The builder asserts two things about what it
produces: **nothing from `chal/inference.py`, `chal/predictors.py`,
`optimal_solution.py` or `batch_meta.csv`** reaches it, because the intended
solution has to stay discoverable rather than documented; and **no model weights
and no pixel data** reach it, because that is what keeps the released data from
being matched against the public TissueMNIST archive (C9.5). It rebuilds from an
empty directory, so a file removed from `student/` cannot linger in the bundle.

`out/` is gitignored. A capped smoke run of the whole pipeline, for checking
the plumbing rather than producing anything usable:

```bash
python train_base_model.py out/smoke --epochs 2 --max-fit 6000
python prepare_kaggle_data.py out/smoke priors.txt out/smoke_kaggle \
    --n-min 5 --batch-scale 20
```

## Training on a GPU machine

Training is the only step that wants a GPU; everything downstream is NumPy and
takes about a minute. So train remotely, bring back the model directory, and
generate the competition data wherever you like.

**1. Copy the code** — the directory is ~180 KB without `out/`:

```bash
tar --exclude=out --exclude=__pycache__ -czf challenge.tgz challenge/
scp challenge.tgz gpubox:~/           # or git push / rsync
```

**2. Install with a CUDA wheel** matching the driver — check
<https://pytorch.org/get-started/locally/> for the current index URL:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python selftest.py                    # 10 s, needs no data; catches a broken install
```

**3. Fetch the data and train** (125 MB from Zenodo, so no need to copy it):

```bash
python download_data.py
python train_base_model.py out/model --epochs 30 --device cuda
```

**4. Bring back the model directory** (~94 MB):

```bash
scp -r gpubox:~/challenge/out/model out/
```

Then continue locally — `prepare_kaggle_data.py`, `make_dev_benchmark.py`,
`baseline_solutions.py`, `run_baselines.sh` all work unchanged.

**Why there is no reproducibility risk in splitting the work.** The downstream
scripts never re-derive the splits: `log_post.npz` carries `y_eval`,
`pool_of_eval`, `y_dev` and the calibrated posteriors, and `images.npz` carries
the pixels. Nothing calls `make_splits` after training, so the split that the
GPU machine made is the split the competition data is built from, whatever
sklearn version the local machine has.

**Of the four outputs**, only `log_post.npz` (3 MB) and `images.npz` (46 MB) are
read by the rest of the pipeline; `model.pt` (45 MB) is the artifact of record,
and `report.txt` holds the NLL/ECE numbers that belong in the paper.

Raising `--batch-size` (256 or 512) is the usual GPU speed-up and changes the
optimisation, not the protocol; scale `--lr` with it if you do. Keep
`--seed 0` so the splits stay comparable between runs.

## What lands where

| Uploaded to Kaggle | Given to Kaggle, hidden from students | Never uploaded |
| :-- | :-- | :-- |
| `test_priors.csv`, `train_prior.csv`, `predictions.csv`, `test.csv`, `test_batches.csv`, `sample_submission.csv`, `dev.csv`, and the `dev_*` benchmark | `solution.csv` | `batch_meta.csv`, `manifest.json`, `out/model/` |

**No images and no model are published** (C9.5). The released images are
byte-identical to TissueMNIST, whose labels are public and which the licence
obliges us to credit; shipping the pixels let all 49 670 test labels be recovered
by hashing, in 28 seconds. A competitor now gets the calibrated posterior per
image and nothing else.

`batch_meta.csv` holds `θ*` per batch. It exists only so `baseline_solutions.py`
can build the true-prior oracle, which is the metric's own reference line rather
than a competitor.

## Scripts

| Script | Does |
| :-- | :-- |
| `download_data.py` | fetch and verify the TissueMNIST archive |
| `train_base_model.py` | train the CNN on the fit split, BCTS-calibrate on the calibration split, report NLL/ECE before and after |
| `make_priors.py` | build `Θ` — 8 sliding-pair priors, training prior excluded (C3.3) |
| `prepare_kaggle_data.py` | draw every test batch and write the whole upload (C4, C5) |
| `make_dev_benchmark.py` | the same protocol over the student development split, so competitors can score themselves offline |
| `make_metric_notebook.py` | generate `metric-template.ipynb` from `chal/metric.py` |
| `baseline_solutions.py` | the six baselines of C7, written as submissions |
| `compare_baselines.py` | score submissions on **shared** bootstrap resamples and report paired differences — the C8 separation criterion |
| `make_student_bundle.py` | assemble `out/kaggle_code/` — the starter code students get — and audit it for leaks of the intended solution |
| `optimal_solution.py` | the intended optimum alone — **organisers only** |
| `evaluate.py` | score a submission, print the per-size table with bootstrap CIs, plot |
| `selftest.py` | brute-force checks of the inference, the protocol and the metric |

## The library

| Module | Holds |
| :-- | :-- |
| `chal/data.py` | download and load TissueMNIST |
| `chal/splits.py` | the 0.80/0.10/0.10 development split and the three disjoint eval pools (C3.1) |
| `chal/priors.py` | `Θ`, its guards and its file format (C3.3) |
| `chal/protocol.py` | the grid, `N(m)`, the batch sampler (C4) |
| `chal/generate.py` | the one generation path, shared by the competition data and the dev benchmark |
| `chal/inference.py` | the exact S2 inference: `H`, `T`, `A`, `E`, the MAP plugin |
| `chal/predictors.py` | the S3 reject-option predictors, built from the released files |
| `chal/calibration.py` | BCTS, NLL, equal-mass ECE |
| `chal/metric.py` | `score()` — the single source for Kaggle and `evaluate.py` |
| `chal/ids.py` | random image ids — used only for the local-only `row_image.csv`, since no image key is published (C9.7) |

## Three things that are easy to get wrong

**The metric cannot be debugged after launch.** `chal/metric.py` is the only
copy: `evaluate.py` imports it and `metric-template.ipynb` is generated from it
by `make_metric_notebook.py`. Never hand-edit the notebook.

**`true_plugin` must score exactly 0.000000.** It *is* the reference the metric
measures against, so any other number means the generation and the scoring have
diverged. It is the first thing to check after regenerating data.

**The bootstrap unit is the batch, not the row.** Rows inside a batch share `θ*`
and the same adaptation evidence. `N(m)` — not `B_m` — is what sets the width of
every interval (C9.2).

**Overlapping marginal intervals do not mean two methods are tied.**
`bayes_total` and `bayes_epistemic` share a base predictor and differ only in
their ranking, so most of their variance is common and cancels in the
difference. Use `compare_baselines.py`, which bootstraps them on the same
resampled batches and reports the paired difference; `evaluate.py`'s per-method
bands are marginal and will overlap even when one method wins every replicate.
