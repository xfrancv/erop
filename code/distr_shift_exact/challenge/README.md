# Challenge: *Know what you don't know: nine locations* (v3)

Self-contained pipeline for the TissueMNIST location-shift competition.
Copying this directory elsewhere is enough to run it — nothing here imports from
the parent project.

The specification is `tasks/even_harder_variant.md`; `tasks/hard_variant.md`
(v2) specifies everything v3 leaves unchanged. Section references of the form
**C4**, **C9.2** point at the original challenge specification; **S2.1**,
**S6.3** point at the parent `../README.md`. This branch supports v3 only.

## The task

Competitors get **images**, not posteriors. The training images each carry a
class and one of 9 **locations**; the locations differ only in their class
frequencies. Test images arrive in **batches** of `m ∈ {1, 2, 5, 10, 20, 50,
100}`, each batch from one location they are not told. For every image they
submit a label and a confidence; the organiser keeps the most confident 80 %
per batch size and scores the **regret against the Bayes predictor that knows
the location**, averaged over the seven sizes. Lower is better.

What v3 changes against v2:

1. **Synthetic labels.** Every label — training, development and test — is
   drawn from a secret label model `q(y | x)`, a calibrated CNN fitted to a
   *secret* 30 % of the TissueMNIST training split that is never released. The
   reference predictor is therefore the true Bayes predictor given the location,
   not an estimate of it: the expected score of any submission is `>= 0`, and
   the best achievable one is strictly positive at small `m`.
2. **Secret location prior.** The location of every development and test batch
   is drawn i.i.d. from a secret, non-uniform `w`. Competitors are told only
   that the development batches follow the same process; the intended solution
   estimates `w` by EM on the development batches.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python selftest.py                      # no data, no network, ~15 s
```

Only `train_base_model.py` needs torch; everything downstream reads the
calibrated log-posterior it writes and is pure NumPy/SciPy/pandas.
This installs the CPU build of torch; for a GPU machine, where the CUDA build
must match both the driver and the GPU, see **Training on a GPU machine** below.

## Pipeline

```bash
python download_data.py                                            # ~125 MB
python make_split.py out/v3/split                                  # rotate + split
python train_base_model.py out/v3/model --split out/v3/split --device cuda
python hard_make_data.py out/v3/data --split out/v3/split --model out/v3/model \
    --priors-out priors_tissuemnist_v3.txt
python hard_package.py out/v3/data out/v3/kaggle
python make_metric_notebook.py metric-template.ipynb
python make_student_bundle.py out/v3/kaggle_code --zip

./run_baselines.sh out/v3/kaggle/organiser out/v3/submissions     # pre-launch checks
```

`out/` is gitignored. A capped smoke run of the whole pipeline, for checking
the plumbing rather than producing anything usable (the scoring of ~187 000
pool images still takes ~20 minutes on a CPU):

```bash
python make_split.py out/smoke/split
python train_base_model.py out/smoke/model --split out/smoke/split \
    --epochs 1 --max-fit 3000
python hard_make_data.py out/smoke/data --split out/smoke/split \
    --model out/smoke/model --n-min 20 --batch-scale 100 --dev-n-min 20 \
    --dev-batch-scale 100
python hard_package.py out/smoke/data out/smoke/kaggle
./run_baselines.sh out/smoke/kaggle/organiser out/smoke/submissions
```

### Training on a GPU machine

Training the label model is the only step that wants a GPU; everything else is
NumPy and takes minutes. So split and train remotely, bring back the two
directories, and generate the competition data wherever you like.

**1. Get the code** on the GPU machine:

```bash
git fetch origin && git checkout challenge-v3      # in an existing clone
```

**2. Install with a CUDA wheel** that fits both the driver and the GPU. Two
constraints, and `torch.cuda.is_available()` checks only the first:

* **Driver.** `nvidia-smi` prints the highest CUDA version the driver supports
  (top right); the wheel's CUDA must not exceed it, or torch reports *"The
  NVIDIA driver on your system is too old"* and `is_available()` is `False`.
* **GPU architecture.** The wheel must contain kernels for the GPU's compute
  capability (`nvidia-smi --query-gpu=name,compute_cap --format=csv`). If it
  does not, `is_available()` still prints `True` but the first convolution
  fails with *"no kernel image is available for execution on the device"*
  (plus a warning that `sm_XX` is not compatible). This is what breaks the
  `cu126` wheel on Blackwell GPUs (RTX 50xx, RTX PRO Blackwell, B200:
  compute capability 10.0 / 12.0), e.g. on `gauss`.

| GPU (compute capability) | index | needs driver CUDA |
| :-- | :-- | :-- |
| Blackwell (10.0, 12.0) | `cu130` (or `cu128`) | ≥ 13.0 (≥ 12.8) |
| Turing – Hopper (7.5 – 9.0) | `cu130`, `cu128` or `cu126` | ≥ 13.0, 12.8, 12.6 |
| Volta, V100 (7.0) | `cu126` (dropped from `cu128`+) | ≥ 12.6 |

Pick the index from **both** columns: the row of your GPU, and among its
indexes one whose CUDA does not exceed the driver's. Copying the `cu130`
command blindly fails on the older machines: a V100 node with driver CUDA 12.9
(e.g. `n26`) gets *"driver too old"* from `cu130` and no `sm_70` kernels from
`cu128`; only `cu126` works there.

| Machine | GPU | driver CUDA | `CU=` |
| :-- | :-- | :-- | :-- |
| `gauss` | RTX PRO 4000 Blackwell (12.0) | 13.2 | `cu130` |
| `n26` (cluster) | Tesla V100-SXM2 (7.0) | 12.9 | `cu126` |

The plain PyPI torch wheel is the newest CUDA build and works only when the
driver is new enough, so always install torch **first**, from an explicit index
with `--index-url` (not `--extra-index-url`, which lets pip pick the PyPI wheel
anyway), and only then the rest: `requirements.txt` is then satisfied by the
torch already installed and leaves it alone. See
<https://pytorch.org/get-started/locally/> for the current indexes.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # the GPU row of the table
nvidia-smi | grep "CUDA Version"                        # the driver limit
CU=cu126                                                # from the table above
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
python -c "import torch; c = torch.nn.Conv2d(3, 8, 3).cuda(); print(c(torch.randn(2, 3, 28, 28, device='cuda')).shape)"
python selftest.py                    # 15 s, needs no data; catches a broken install
```

The first check must print `True` and list the GPU's `sm_XY` (compute
capability X.Y); the convolution must print `torch.Size([2, 8, 26, 26])`. On
`n26` the first line reads `2.14.0+cu126 12.6 True [..., 'sm_70', ...]`. If
either fails, replace the wheel with one from the right index; pip will not do
it by itself, since the installed version already satisfies the requirement:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU
```

**3. Fetch the data, split and train** (125 MB from Zenodo, so no need to copy
it):

```bash
python download_data.py
python make_split.py out/v3/split
python train_base_model.py out/v3/model --split out/v3/split --device cuda
```

**4. Bring back both directories** — the split (~185 MB, the rotated images)
and the model (~50 MB):

```bash
scp -r gauss:~/Work/erop/code/distr_shift_exact/challenge/out/v3/split out/v3/
scp -r gauss:~/Work/erop/code/distr_shift_exact/challenge/out/v3/model out/v3/
```

Then continue locally with `hard_make_data.py`. Copy the split rather than
re-running `make_split.py` locally: the stratified split is drawn by sklearn,
whose shuffling may differ between versions. `hard_make_data.py` refuses a
model whose `split_id` does not match the split it is given, so a model and a
split from different runs cannot be combined by accident.

Raising `--batch-size` (256 or 512) is the usual GPU speed-up and changes the
optimisation, not the protocol; scale `--lr` with it if you do.

## How the data are generated

**Rotation and split** (`make_split.py`). Every image is rotated once by
90/180/270° and never again; no noise. The original training split is divided,
stratified by the original class, into the secret set (30 %), the training
data (70 % × 90 % ≈ 104 000) and the development pool (70 % × 10 % ≈ 11 600).
The test pool is the original val + test (70 920), partitioned into the Kaggle
`Public` / `Private` / `Ignored` pools. The original labels are used only for
the stratification and for fitting the label model.

**Label model** (`train_base_model.py`). ResNet-18 on the secret set, split
into a weight-fitting and a validation part; epoch selection and BCTS on the
validation part. It scores every training, development-pool and test-pool
image. `hard_make_data.py --temperature T` tempers it (`q^(1/T)`); the result
*is* the ground truth either way, so `T` only sets the Bayes error.

**Labels, locations, batches** (`hard_make_data.py`).

* Each training image gets one label, `y ~ q(y | x)`. The location LP of v2
  (`chal/locations.py`) then runs on these **generated** labels: locations
  0–7 are the 8 priors of `priors_tissuemnist_challenge.txt` moved as little
  as possible, location 8 takes the remainder. A location's class frequency in
  `train.csv` *is* its prior `pi_l`, exactly.
* Every development and test row is drawn independently: the batch's location
  `l ~ w`, then `y ~ pi_l`, then an image of the pool with probability
  `∝ q(y | x)` (`chal/protocol.py`). No pool image carries a fixed label; the
  same image can appear in several rows with different labels, so linking
  copies across batches reveals nothing.
* The reference prediction of a row from pool `P` is
  `argmax_y q(y | x) pi_l(y) / pibar_P(y)`, with `pibar_P` the mean of `q`
  over the pool. With the sampler above this is the Bayes rule given the
  location, exactly — `selftest.py` checks the sampler against it.

**The location prior** `w` defaults to `chal/locprior.py:W_DEFAULT` — TV 0.28
from uniform, every weight `>= 0.03`, far from the training location shares.
`hard_make_data.py` refuses a `w` that violates those guards and reports how
well EM on the development labels recovers it. The development set has 1 500
batches (`--dev-n-min 150`) rather than v2's 1 080, to make that estimate
usable; reusing development images costs nothing since every row has a fresh
label.

## What lands where

| Uploaded to Kaggle | Given to Kaggle, hidden from students | Never uploaded |
| :-- | :-- | :-- |
| `train.csv`, `train_images.npy`, `test_batches.csv`, `test_images.npy`, `sample_submission.csv`, `dev_test_batches.csv`, `dev_images.npy`, `dev_solution.csv`, `dev_sample_submission.csv` | `solution.csv` | `organiser/`, `manifest.json`, `out/v3/split/`, `out/v3/model/`, `out/v3/data/` |

`organiser/test/` and `organiser/dev/` are laid out for `chal/predictors.py`:
`predictions.csv` holds the **true** label model, rescaled per pool so that the
plug-in rule under `train_prior.csv` reproduces `pred_ref` exactly
(`chal.generate.debiased_posterior`); `test_priors.csv` the 9 location priors;
`batch_meta.csv` the location of every batch; `location_prior.csv` the secret
`w`. The baselines built on it are what a competitor with a perfect model of
`q` achieves.

## Pre-launch checks

`run_baselines.sh` estimates `w` by EM on `organiser/dev` exactly as a
competitor would (`estimate_location_prior.py`: development labels plus the
location priors), then runs every baseline under a uniform, the EM-estimated
and the true location prior, and compares them on shared bootstrap resamples.
Read, in order:

1. **`true_plugin` must score exactly 0.000000.** It *is* the reference; any
   other number means generation and scoring have diverged.
2. **No score may be clearly negative.** The reference is the Bayes predictor
   given the location; a clearly negative score is a bug.
3. **`bayes_epistemic_em` must beat `bayes_epistemic_uniform`** with a paired
   interval excluding 0. Otherwise the secret prior is not worth discovering;
   move `w` further from uniform (`--location-prior`) and regenerate.
4. **`bayes_epistemic` must beat `bayes_total`** (same prior): the C8
   separation criterion.

**The metric cannot be debugged after launch.** `chal/metric.py` is the only
copy: `evaluate.py` imports it, `metric-template.ipynb` is generated from it by
`make_metric_notebook.py`, and `make_student_bundle.py` copies it. Never
hand-edit the notebook. Its `score()` docstring is rendered to competitors, so
the bundle audit also applies to it.

**The bootstrap unit is the batch, not the row.** Rows inside a batch share
the location and the same adaptation evidence; `N(m)`, not `B_m`, sets the
width of every interval (C9.2).

## Scripts

| Script | Does |
| :-- | :-- |
| `download_data.py` | fetch and verify the TissueMNIST archive |
| `make_split.py` | rotate every image once; secret / training / development / test split |
| `train_base_model.py` | fit and BCTS-calibrate the label model on the secret set; score every pool |
| `hard_make_data.py` | generated labels, locations, priors file, batches, reference predictions |
| `hard_package.py` | the Kaggle upload, `solution.csv` and `organiser/` |
| `estimate_location_prior.py` | EM estimate of `w` from the development labels, with a bootstrap |
| `baseline_solutions.py` | the C7 baselines under a chosen location prior |
| `optimal_solution.py` | the intended optimum alone — **organisers only** |
| `evaluate.py` | score a submission, per-size table with bootstrap CIs, plot |
| `compare_baselines.py` | paired bootstrap comparison of submissions |
| `run_baselines.sh` | all of the above, in the order the pre-launch checks need |
| `make_metric_notebook.py` | generate `metric-template.ipynb` from `chal/metric.py` |
| `make_student_bundle.py` | assemble `student_hard/` + the metric, audited for leaks |
| `selftest.py` | brute-force checks of the inference, the sampler, the metric, the generation |

## The library

| Module | Holds |
| :-- | :-- |
| `chal/data.py` | download and load TissueMNIST |
| `chal/transform.py` | the one-off rotation |
| `chal/splits.py` | the Kaggle `Usage` pools of the test pool |
| `chal/locations.py` | the location linear program and the assignment |
| `chal/locprior.py` | the secret location prior, its guards, EM |
| `chal/protocol.py` | the grid, `N(m)`, the sampler `x ~ p_P(x | y)` |
| `chal/generate.py` | tempering, label draws, the reference predictor, the organiser posterior |
| `chal/priors.py` | the priors file format and guards |
| `chal/inference.py` | the exact S2 inference: `H`, `T`, `A`, `E`, the MAP plugin |
| `chal/predictors.py` | the S3 reject-option predictors, built from `organiser/` |
| `chal/calibration.py` | BCTS, NLL, equal-mass ECE |
| `chal/bootstrap.py` | batch-level bootstrap of the metric |
| `chal/metric.py` | `score()` — the single source for Kaggle and `evaluate.py` |

`student_hard/` and `description_hard/` hold the starter kit and the Kaggle
pages.
