# Plan

Create an even harder variant (v3) of the hard challenge (v2).

# Context

In v2 the competitors are compared against a reference predictor that is only
an estimate of the Bayes predictor: the plug-in rule built on a CNN posterior
fitted to the real TissueMNIST labels. The v2 description also tells the
competitors that every location contributed the same number of test batches,
i.e. the parameter prior is known.

v3 makes two changes:

1. **Synthetic labels.** All labels are drawn from a secret, known label model.
   The reference predictor is then the true Bayes predictor for the location,
   not an estimate of it.
2. **Secret parameter prior.** The distribution over locations is no longer
   uniform and is not disclosed. It has to be estimated from the development
   data.

Everything not mentioned below stays as in v2 (`tasks/hard_variant.md`): the
9 locations and the linear program assigning training images to them, the
batch-size grid, the metric formula, the public/private split and the file
formats.

The code in this branch will be used only for the new challenge version. That is, the scripts will not support the previous two challenge versions. This will simplify the code.

# Changes

## Images

Every image of the original dataset (train, val and test splits) is rotated
once, by an angle drawn uniformly from {90°, 180°, 270°}. Unlike v2, no pixel
noise is added. The rotated image is the one released and the one the label
model is applied to; the original orientation is never used again.

The noise is no longer needed: matching an image against TissueMNIST reveals
only its original label, and the generated label is independent of the original
one given the image (see *Labels*).

## Data split

| Part | Source | Share | Images | Use |
| :-- | :-- | --: | --: | :-- |
| Secret set | original train | 30 % | ≈ 49,640 | fitting the label model; never released |
| Training data | original train | 70 % × 90 % | ≈ 104,240 | released, with labels and locations |
| Development pool | original train | 70 % × 10 % | ≈ 11,580 | images of the development batches |
| Test pool | original val + test | all | 70,920 | images of the test batches (public and private pools as in v2) |

All splits of the original train split are stratified by the original class.
The original labels are used only for this stratification and for fitting the
label model.

## Label model

The label model is a CNN trained by the existing pipeline
(`train_base_model.py`) on the secret set only, with the original labels:

1. The secret set is split, stratified by class, into a weight-fitting part and
   a validation part.
2. The weights are fitted on the weight-fitting part; the epoch is selected on
   the validation part.
3. The posterior is calibrated (BCTS) on the validation part.

The result is the calibrated posterior $q(y \mid x)$. It defines the ground
truth: by construction, $q$ is the true posterior of the data-generating
process, whether or not it is a good model of the real tissue classes. The
calibration temperature therefore sets the Bayes error of the challenge and can
be treated as a difficulty knob.

For a pool $P$ of images (training data, development pool, public test pool,
private test pool) define the label marginal of the pool

$$\bar\pi_P(y) = \frac{1}{|P|} \sum_{x \in P} q(y \mid x),$$

and the class-conditional distribution over the pool

$$p_P(x \mid y) = \frac{q(y \mid x)}{|P|\,\bar\pi_P(y)}, \qquad x \in P.$$

Note that $\bar\pi_P$, not the class frequencies of the secret set, is the
prior under which $q$ is the posterior on pool $P$.

## Labels

**Training data.** Each training image gets one label, drawn once:
$y \sim q(y \mid x)$. The linear program of v2 then assigns the training images
to the 9 locations using these **generated** labels (class counts $D_c$, the
floor $\varepsilon$, $n_{\min}$ and $\delta$ are all recomputed on them). The
resulting 9 location priors $\pi_1, \dots, \pi_9$ are written to the priors
file and used for the development and test batches as well.

**Development and test batches.** No image carries a fixed label. Every row of
every batch is drawn independently:

1. draw the location of the batch, $\ell \sim w$ (see *Parameter prior*);
2. for each of the $m$ rows: draw $y \sim \pi_\ell$, then draw an image
   $x \sim p_P(x \mid y)$, i.e. pick $x$ from the pool $P$ with probability
   proportional to $q(y \mid x)$.

The same image may therefore occur in several batches, or twice in one batch,
each time with its own, independently drawn label. Linking copies of an image
across batches reveals nothing about its labels.

With this procedure the true posterior of a row from location $\ell$ is, exactly,

$$p_\ell(y \mid x) = \frac{q(y \mid x)\, \pi_\ell(y) / \bar\pi_P(y)}
  {\sum_{y'} q(y' \mid x)\, \pi_\ell(y') / \bar\pi_P(y')}.$$

## Metric

The metric formula (average regret at 80 % coverage) is unchanged. The
reference prediction of a row from location $\ell$ drawn from pool $P$ is the
Bayes prediction given the location:

$$\hat y^{\rm ref} = \arg\max_y \; q(y \mid x)\, \pi_\ell(y) / \bar\pi_P(y).$$

Consequences, to be reflected in the description:

- The expected regret of any honest submission is non-negative, because the
  reference knows the location and competitors do not. A realised score can
  still be slightly negative through sampling noise, mostly at large $m$.
- Zero is not attainable for small $m$: a batch of one or two images says
  little about its location. The best achievable score is strictly positive.

## Parameter prior

The location of each development and test batch is drawn i.i.d. from a secret
distribution $w$ over the 9 locations. The same $w$ is used for every batch
size and for the development, public test and private test batches.

**Choice of $w$.** $w$ must be neither uniform nor proportional to the
training location sizes (location 9 holds most of the training data), since
both are the first guesses competitors will try. Suggested: a fixed vector at
TV distance about 0.2–0.3 from uniform, with every $w_\ell \ge 0.03$, so that
every location occurs in the development data.

**Told to competitors.** The locations of the test batches are drawn
independently from a fixed but undisclosed distribution; the development
batches are generated by exactly the same process, with the same distribution.

**Intended solution.** Estimate $w$ by EM on the development batches, then use
it as the prior over locations in the epistemic reject-option predictor. Rule 3
forbids statistics over the test set, so the development batches are the only
legitimate source.

**Development set size.** v2 has 1,080 development batches, of which only
about 210 have $m \ge 10$ and identify their location reliably; that gives a
standard error of roughly 0.02 on each $w_\ell$. Because rows are drawn with
replacement and with fresh labels, reusing development images is harmless, so
the number of development batches can be increased (e.g. `--dev-n-min 150`)
without enlarging the pool. Before fixing the setting, check by simulation that
EM on the development data gives a clear gain over a uniform prior.

## Implementation notes

- `train_base_model.py`: train and calibrate on the secret set only.
- `hard_make_data.py`: new split, rotation without noise, label generation for
  the training data, LP on the generated labels, $\bar\pi_P$ per pool.
- `chal/protocol.py` / `chal/generate.py`: a sampler that draws $x$ with
  probability proportional to $q(y \mid x)$; location drawn from $w$ instead of
  uniformly or balanced; reference prediction with $\bar\pi_P$ as the base
  prior.
- `selftest.py`: check that the regret of the reference against itself is 0,
  that an oracle knowing $\ell$ matches the reference, and that EM on the
  development batches recovers $w$ within the expected error.

## New description

Update the files in `description_hard/` (or a copy for v3). At least:

- `description.md`: remove "every location contributed the same number of test
  batches"; state that locations are drawn independently from an undisclosed
  distribution shared by the development and test batches.
- `evaluation.md`: the reference predictor is the best possible predictor that
  knows the location of the batch; remove "a negative score is possible and is
  not an error" and explain that the best achievable score is positive.
- `dataset_description.md`: new row and batch counts; remove the note on pixel
  noise; replace "each time with its own rotation and noise" by the fact that
  an image may occur several times, each time with an independently drawn label.
- Dataset attribution: the images are derived from TissueMNIST; the labels are
  synthetic.
