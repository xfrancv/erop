The underlying images are 28×28 greyscale pictures of human kidney cortex cells in 8 tissue classes, derived from TissueMNIST (MedMNIST v2).

**The images are not published, and no image identifier is published either.** Each test row appears only as the trained network's eight calibrated class probabilities, in `predictions.csv`, keyed by the same `row_id` your submission uses. Every file below is a CSV.

For **every row** of `test_batches.csv` you predict a class `0`–`7` and a
confidence. See **Evaluation** for the submission format, and **Description**
for how the batches were generated.

## Files

### Test set

*   **`test.csv`** — one row per test batch; 12,600 batches.
*   **`test_batches.csv`** — one row per test image; 126,000 rows. Says which
    batch each image belongs to.
*   **`sample_submission.csv`** — a valid submission, produced by the network
    alone with no use of the batch. The baseline to beat.

| batch size `m` | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| batches | 6,000 | 3,000 | 1,200 | 600 | 600 | 600 | 600 |
| images | 6,000 | 6,000 | 6,000 | 6,000 | 12,000 | 30,000 | 60,000 |

### What the network gives you

*   **`predictions.csv`** — the network's calibrated output for every test row,
    keyed by `row_id`; 126,000 rows. This is the only representation of a test
    image you get.
*   **`train_prior.csv`** — the class frequencies the network was trained on.
*   **`test_priors.csv`** — the 8 admissible test priors. Every batch used
    exactly one of them, drawn uniformly at random. The training prior is **not**
    among them.

### Development set, for scoring yourself offline

Built by the identical procedure, from images whose labels you have. The test
labels are not in the data, so this is the only way to measure a predictor
before submitting.

*   **`dev.csv`** — 16,547 labeled development images: the true `label` and
    the same eight probabilities. A plain labeled sample, so you can fit or
    recalibrate on it directly. None of these images appears in any test batch.
*   **`dev_test.csv`** — 1,050 development batches, same size grid as the test
    set.
*   **`dev_test_batches.csv`** — one row per development test image; 10,500 rows.
*   **`dev_predictions.csv`** — the network's output for those 10,500 rows.
*   **`dev_solution.csv`** — the answers for those 10,500 rows.
*   **`dev_sample_submission.csv`** — the baseline, on the development batches.

## Columns

### `test_batches.csv`, `dev_test_batches.csv`

*   `row_id` — identifies one test image; globally unique. **This is the key
    your submission must use.**
*   `id_test` — which batch the image belongs to. All images sharing an
    `id_test` were drawn under the same prior.
*   `slot` — position within the batch, `0` to `m-1`. The order is arbitrary and
    carries no information.

### `test.csv`, `dev_test.csv`

*   `id_test` — the batch.
*   `m` — how many images it contains.

### `sample_submission.csv`, `dev_sample_submission.csv`

*   `row_id` — one row for every test image.
*   `pred` — predicted class, integer `0`–`7`.
*   `confidence` — any finite real number; larger means "keep this prediction".

### `predictions.csv`, `dev_predictions.csv`

*   `row_id` — matches `test_batches.csv` / `dev_test_batches.csv`.
*   `p0` … `p7` — the network's calibrated probability of each class **under the
    training mix**, i.e. \(p\_{tr}(y \mid x)\). Each row sums to 1.

### `test_priors.csv`

*   `id` — prior id, `0`–`7`.
*   `p0` … `p7` — the probability of each class under that prior. Each row sums
    to 1.

### `train_prior.csv`

*   `p0` … `p7` — the class frequencies the network was trained on,
    \(p\_{tr}(y)\). One row.

### `dev.csv`

*   `label` — the true class, `0`–`7`.
*   `p0` … `p7` — the network's calibrated probability of each class.

### `dev_solution.csv`

*   `row_id`, `id_test`, `slot` — as in `dev_test_batches.csv`.
*   `m` — the batch size, repeated on every row for convenience.
*   `label` — the true class.
*   `pred_ref` — the class predicted by the **reference predictor**, which was
    told that batch's true prior. The metric measures how much more often your
    `pred` is wrong than this column is.

## Classes

| label | tissue |
| --: | :-- |
| 0 | Collecting Duct, Connecting Tubule |
| 1 | Distal Convoluted Tubule |
| 2 | Glomerular endothelial cells |
| 3 | Interstitial endothelial cells |
| 4 | Leukocytes |
| 5 | Podocytes |
| 6 | Proximal Tubule Segments |
| 7 | Thick Ascending Limb |

Classes 2 and 3 are close to indistinguishable at 28×28. Some error is
irreducible, which is why the metric compares you against a reference predictor
working from the same network output rather than against perfection.

## A note on independence

Batches are drawn from a finite pool **with replacement**, so the same
underlying image occasionally turns up in more than one batch. That is an
artefact of the sampling, not part of the problem: rows in different batches are
to be treated as independent, and the rules forbid linking them by image
identity. No image identifier is published, so nothing here invites it.

### Dataset attribution

This challenge uses the TissueMNIST dataset from MedMNIST v2.

TissueMNIST is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

TissueMNIST is derived from BBBC051 from the Broad Bioimage Benchmark
Collection (BBBC), originally released under CC BY 3.0.

References:

J. Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for
2D and 3D Biomedical Image Classification," Scientific Data, 2023.

A. Woloshuk et al., "In Situ Classification of Cell Types in Human Kidney
Tissue Using 3D Nuclear Staining," Cytometry, 2020.