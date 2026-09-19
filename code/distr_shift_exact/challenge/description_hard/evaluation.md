Submissions are scored by **average regret at 80 % coverage**: how much more
often your kept predictions are wrong than those of a reference predictor.
**Lower is better.**

## The metric

Fix a batch size $m$. All $B_m$ test images of that size — pooled **across all
batches of that size** — are sorted by *descending* `confidence`, giving an
order $\pi$; ties are broken by ascending `row_id`. The most confident
$k = \lceil 0.8\, B_m \rceil$ are kept and the rest discarded. On the kept
predictions we compute

$$\text{Reg}(m) = \frac{1}{k} \sum_{i=1}^{k} \Big(
  \ell\big(\hat y_{\pi(i)}, y_{\pi(i)}\big) \; - \;
  \ell\big(\hat y^{\text{ref}}_{\pi(i)}, y_{\pi(i)}\big) \Big)$$

where $\hat y$ is your `pred`, $y$ the true class, $\hat y^{\text{ref}}$ the
reference predictor's prediction, and $\ell$ the 0/1 loss. The final score
averages the seven batch sizes with equal weight:

$$\text{AvgRegAtCoverage} \; = \; \frac{1}{7}
  \sum_{m \in \{1,2,5,10,20,50,100\}} \text{Reg}(m).$$

## The reference predictor

For each location there is a reference predictor. It was trained on the same
training data as provided to you and then fine-tuned to maximise performance at
that location. Each test image is compared against the reference predictor of
the location its batch came from.

The reference predictors themselves are not released. You get their
predictions on the development batches, in the `pred_ref` column of
`dev_solution.csv`.

The reference predictors make mistakes on ambiguous images, and you are not
charged for the mistakes they also make. **A negative score is therefore
possible and is not an error.**

## Submission file

One row for every `row_id` in `test_batches.csv` — 129,924 rows plus a header.
`pred` is an integer in `0`–`7`; `confidence` is any finite real number, larger
meaning "keep this one".

    row_id,pred,confidence
    77469,1,0.41251432860164206
    107339,7,0.48841011716486
    2276,4,0.27580701738501207
    etc.

## Public and private leaderboard

Public: one set of test batches, shown during the competition.
Private: a different, disjoint set of batches; decides the final ranking.

## Scoring yourself before you submit

The test labels are not in the data, so you cannot evaluate on the test
batches. Instead, `dev_test_batches.csv` and `dev_solution.csv` give you a fully
labeled set of batches built by the identical procedure. `evaluate.py` in the
starter kit scores them with the same function the leaderboard runs, so an
offline number is directly comparable — bearing in mind the development set is
smaller and therefore noisier.
