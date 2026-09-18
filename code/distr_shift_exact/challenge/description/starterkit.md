The starter kit is optional. `predictions.csv` and `sample_submission.csv` are already in the competition data, so you can work straight from the CSV files. The kit exists so you can see how the baseline was built and — more usefully — **score yourself offline** before spending a submission.

## Contents

| File | What it is |
| :-- | :-- |
| `make_sample_submission.py` | writes a baseline submission; documents the format |
| `metric.py` | the competition metric, the same code the leaderboard runs |
| `evaluate.py` | scores a submission against `dev_solution.csv` |

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

numpy and pandas are all you need; matplotlib only for `evaluate.py --plot`. There are no images to decode and no network to run, so there is no deep-learning dependency. On a Kaggle notebook everything is preinstalled — install nothing.

## Use

Point `DATA` at the competition data:

```
DATA=/kaggle/input/know-what-you-dont-know-under-a-new-prior
```

**Build a baseline and score it.** Start here — it takes seconds and proves the whole loop works:

```
python make_sample_submission.py --data-dir $DATA \
    --batches dev_test_batches.csv --out dev_sample_submission.csv
python evaluate.py dev_sample_submission.csv $DATA/dev_solution.csv
```

You get a per-batch-size table and one number. That number is what the leaderboard would give this baseline on those rows. Beating it is the competition.

**Score your own predictor.** Write a CSV with columns `row_id`, `pred`, `confidence` covering every row of `dev_test_batches.csv`, then:

```
python evaluate.py my_submission.csv $DATA/dev_solution.csv --plot
```

`--plot` saves a regret-coverage curve: selective regret against the fraction of predictions kept, one line per batch size. The score reads each curve at coverage 0.8, and the shape to the left of that tells you whether your confidence ordering is doing anything useful.

**Produce a submission.** Same predictor, run on the test batches instead:

```
python make_sample_submission.py --data-dir $DATA \
    --batches test_batches.csv --out submission.csv
```

## How the files join up

Everything is keyed on `row_id`. `test_batches.csv` says which batch a row belongs to (`id_test`); `predictions.csv` gives that row's eight class probabilities. The development benchmark is the same pair, `dev_test_batches.csv` and `dev_predictions.csv`. `dev.csv` is the labeled pool — a `label` and the same eight probabilities — for fitting or recalibrating anything you like.

Rows sharing an `id_test` were drawn under the same unknown prior. Rows in different batches are independent; see the rules for what you may not do with them.
