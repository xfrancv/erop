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

## The hard variant

`tasks/hard_variant.md` specifies a second competition on the same data in
which competitors must discover most of the steps themselves. They get
**images**, not posteriors, and no network; the shift is described only as
"9 locations with different class frequencies"; the reference predictor is
described in words; and the uniform choice of location is stated only as "every
location contributed the same number of test batches". Everything organiser-side
is shared with the easy variant — the sampler, the metric, the inference, the
baselines — so the two cannot drift apart.

```bash
python hard_make_data.py out/hard/data --priors-out priors_tissuemnist_hard.txt
python train_base_model.py out/hard/model --hard-data out/hard/data --device cuda
python hard_package.py out/hard/data out/hard/model out/hard/kaggle
python make_metric_notebook.py metric-template-hard.ipynb --source chal/metric_hard.py
python make_student_bundle.py out/hard/kaggle_code --variant hard --zip

./run_baselines.sh out/hard/kaggle/organiser/test out/hard/submissions
python hard_audit_matching.py out/hard/data --sweep 0 2 8 32
```

**Locations.** The easy variant's 8 priors cannot be used as they are: each
class has total probability 1.0 across them, so every location would be capped
at the size of the rarest class. `chal/locations.py` solves the linear program
of the task — the smallest TV move of the 8 priors that places every training
image, with a floor `eps` on every prior and at least `n_min` images per
location — and location 8 takes the remainder. With the defaults
(`eps = 0.01`, `n_min = 5000`) the priors move by at most 0.021 in TV, locations
0–7 hold 5,006 images each and location 8 holds 73 % of the training data; the
closest two locations are 0.30 apart. The result is committed as
`priors_tissuemnist_hard.txt`. A location's class frequency in `train.csv` *is*
its prior, exactly — `hard_package.py` asserts it.

**Batches.** Same sampler and grid as the easy variant, under the 9 location
priors, but `balanced=True`: `N(m)` is rounded up to a multiple of 9 and every
location gets exactly `N(m)/9` batches of every size in every usage, which is
what the description promises. That makes 12,744 test batches (129,924 rows)
and 1,080 development batches.

**Images.** Every released image is rotated by 90/180/270° and given Gaussian
noise of std 2 grey levels (`chal/transform.py`), **per row**: an image drawn
into two batches is released as two different arrays, so identical pixels
cannot link rows. This defeats hashing only. `hard_audit_matching.py` recovers
the label of 100 % of sampled test rows by nearest-neighbour search against the
public TissueMNIST in about 30 s, and still 84 % at noise std 32. The rules (no
label recovery, code re-run on a differently drawn test set) are what protect
the labels.

**Reference predictor.** One ResNet-18, trained by `train_base_model.py
--hard-data` on the released training images with locations ignored, BCTS on a
stratified validation part, then the plug-in rule under the location's prior.
Since every released row is its own array, the network scores all ~141 000
development and test rows; the packaging is torch-free.

| Uploaded to Kaggle | Given to Kaggle, hidden from students | Never uploaded |
| :-- | :-- | :-- |
| `train.csv`, `train_images.npy`, `test_batches.csv`, `test_images.npy`, `sample_submission.csv`, `dev_test_batches.csv`, `dev_images.npy`, `dev_solution.csv`, `dev_sample_submission.csv` | `solution.csv` | `organiser/`, `manifest.json`, `out/hard/data/`, `out/hard/model/` |

There is no `test.csv` / `dev_test.csv`: the batch size `m` is a column of
`test_batches.csv` and `dev_test_batches.csv`, repeated on every row, so
competitors need no join to learn a row's batch size.

`organiser/test/` and `organiser/dev/` are laid out like an easy-variant upload
(`predictions.csv` from the reference network, `test_priors.csv` with the 9
location priors, `batch_meta.csv` with the location per batch), which is what
lets `baseline_solutions.py`, `evaluate.py`, `compare_baselines.py` and
`run_baselines.sh` run on the hard variant unchanged. `true_plugin` must again
score exactly 0.000000. The sample submission predicts the majority class with a
constant confidence: a baseline built on the organisers' network would hand
competitors its output on every test image.

| Script / module | Does |
| :-- | :-- |
| `hard_make_data.py` | split, locations, priors file, batches, transformed images |
| `hard_package.py` | the reference predictor; the upload, `solution.csv` and `organiser/` |
| `hard_audit_matching.py` | how well released images match back to TissueMNIST |
| `chal/locations.py` | the location linear program and the assignment |
| `chal/transform.py` | rotation and noise |
| `chal/metric_hard.py` | **generated** by `make_metric_notebook.py --hard-module`: the metric with neutral docstrings |
| `student_hard/`, `description_hard/` | the starter kit and the Kaggle pages |
