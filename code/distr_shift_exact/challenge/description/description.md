## The setting

Let $x$ be an image and $y \in \{0,\dots,7\}$ its class. A
convolutional network was trained on images you never see, and its calibrated output
$$p_{tr}(y \mid x)$$
is given to you for every released image in `predictions.csv`. The class frequencies it was trained under,  $p_{tr}(y)$, are in `train_prior.csv`.

**The images themselves are not published**, and neither is any image identifier: each test row appears only as those eight probabilities, keyed by its `row_id`. Everything below is a computation on those vectors; there is nothing to decode and no network to run.

Batches are drawn from a finite pool **with replacement**, so an image occasionally recurs across batches. That is an artefact of the sampling rather than part of the problem: rows of different batches are to be treated as independent, and the rules say so explicitly.

The test images do not follow that mix. Each test **batch** was generated in two steps:

1. a class prior $\theta$, which is a vector of 8 probabilities summing to 1, was drawn uniformly at random from a known set $\Theta$ of **8 candidates**, listed in `test_priors.csv`;
2. the batch's $m$ images were drawn independently, each by picking a class from $\theta$ and then an image of that class.

Different batches get independently drawn priors. Batch sizes are $m \in \{1, 2, 5, 10, 20, 50, 100\}$.

## What does and does not change

This is **label shift**: the mix of classes changes, but the appearance of each class does not. Formally, for every class $y$,
$$p_{te}(x \mid y) = p_{tr}(x \mid y), \qquad p_{te}(y) = \theta_y \neq p_{tr}(y).$$

Under label shift the test posterior for a known prior
$\theta$ is the training posterior re-weighted by the ratio of priors:
$$p_{te}(y \mid x, \theta) \; \propto \; \frac{\theta_y}{p_{tr}(y)} p_{tr}(y \mid x).$$

So *if* you knew a batch's prior, classifying its images would be immediate, however, you do not know it.

## Why the batch matters

The only evidence about $\theta$ is the batch itself: $m$ unlabeled images. You may use all of them, including the image you are currently classifying -- nothing in the format prevents it and nothing in the scoring penalises it.

The amount of evidence varies across the competition. At $m = 100$ the eight candidates are easy to tell apart. At $m = 1$ they are essentially indistinguishable, and the best you can do is hedge across all eight. Most of the difficulty sits in between.

**The training prior is not one of the eight candidates.** Every test batch is shifted relative to training, so ignoring the batch and reporting $\arg\max_y p_{tr}(y \mid x)$ is never the right answer. That is exactly what `sample_submission.csv` does, and it is the baseline to beat.