The images are 28×28 greyscale pictures of human kidney cortex cells in 8
tissue classes, derived from TissueMNIST (MedMNIST v2). They were collected at
9 locations; see **Description**.

Every image file is a NumPy array `(n, 28, 28)` of `uint8`, and **row *k* of the
matching CSV describes image *k***. For every row of `test_batches.csv` you
predict a class `0`–`7` and a confidence; see **Evaluation** for the submission
format.

## Files

### Training data

*   **`train_images.npy`** — 148,919 labeled images.
*   **`train.csv`** — one row per training image: its `label` and `location`.

### Test set

*   **`test.csv`** — one row per test batch; 12,744 batches.
*   **`test_batches.csv`** — one row per test image; 129,924 rows. Says which
    batch each image belongs to.
*   **`test_images.npy`** — the test images, in the row order of
    `test_batches.csv`.
*   **`sample_submission.csv`** — a valid submission that predicts the most
    frequent class for everything. It shows the format; it is not a serious
    baseline.

| batch size `m` | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| batches | 6,021 | 3,024 | 1,215 | 621 | 621 | 621 | 621 |
| images | 6,021 | 6,048 | 6,075 | 6,210 | 12,420 | 31,050 | 62,100 |

### Development set, for scoring yourself offline

Built by the identical procedure, from images that appear neither in the
training data nor in any test batch.

*   **`dev_test.csv`** — one row per development batch; 1,080 batches, same
    size grid as the test set.
*   **`dev_test_batches.csv`** — one row per development image; 11,268 rows.
*   **`dev_images.npy`** — the development images, in the row order of
    `dev_test_batches.csv`.
*   **`dev_solution.csv`** — the answers for those rows: the true class and the
    reference predictor's prediction.
*   **`dev_sample_submission.csv`** — the sample submission, on the development
    batches.

## Columns

### `train.csv`

*   `label` — the true class, `0`–`7`.
*   `location` — where the image was collected, `0`–`8`.

### `test_batches.csv`, `dev_test_batches.csv`

*   `row_id` — identifies one image; globally unique. **This is the key your
    submission must use.**
*   `id_test` — which batch the image belongs to. All images sharing an
    `id_test` come from the same location.
*   `slot` — position within the batch, `0` to `m-1`. The order is arbitrary and
    carries no information.

### `test.csv`, `dev_test.csv`

*   `id_test` — the batch.
*   `m` — how many images it contains.

### `sample_submission.csv`, `dev_sample_submission.csv`

*   `row_id` — one row for every image.
*   `pred` — predicted class, integer `0`–`7`.
*   `confidence` — any finite real number; larger means "keep this prediction".

### `dev_solution.csv`

*   `row_id`, `id_test`, `slot` — as in `dev_test_batches.csv`.
*   `m` — the batch size, repeated on every row for convenience.
*   `label` — the true class.
*   `pred_ref` — the class predicted by the **reference predictor** for the
    batch's location. The metric measures how much more often your `pred` is
    wrong than this column is.

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

Some classes are close to indistinguishable at 28×28. Some error is
irreducible, which is why the metric compares you against a reference predictor
rather than against perfection.

## A note on the images

Every released image — training, development and test alike — was rotated by a
random multiple of 90° and carries light pixel noise. The same underlying image
can occur in more than one batch, each time with its own rotation and noise.
That is an artefact of how the batches were drawn, not part of the problem:
rows of different batches are to be treated as independent, and the rules
forbid linking them.

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
