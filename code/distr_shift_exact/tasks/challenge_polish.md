# Challenge plan — *Know what you don't know under a new prior*

Polished specification. Supersedes `challenge.md`. Section numbers of the form
**S2.1**, **S6.3** refer to `README.md`, which remains the authoritative
description of the method; this document fixes only what the competition
changes or adds.

Every open point is collected in [§9](#9-open-issues) with its resolution and
the reasoning behind it, and every one is now **[resolved]** and written into
the body of this document. The two that needed measurement rather than
judgement — §9.7 and §9.8 — were settled against the trained base model; §9.8
records the numbers and the caveat that they must be recomputed if the base
model is retrained.

---

## 1. Purpose

Students receive

* a CNN base predictor trained on TissueMNIST and calibrated to emit
  $\hat p_{tr}(y\mid x)$,
* the training label prior $\hat\theta_{tr}$,
* the finite set $\Theta$ of admissible test priors and the statement that
  $p(\theta)$ is uniform on $\Theta$,
* batches of **unlabeled** test images, each batch drawn i.i.d. under one
  unknown $\theta_*\in\Theta$.

They must, for designated query images, output a **label** and a **confidence**.
Confidence induces a ranking; the organiser sweeps the coverage. The score is
the **regret at 80 % coverage against the true-prior plugin predictor**,
averaged over batch sizes.

**The intended discovery.** Under 0/1 loss the posterior expected regret of
answering $\hat y$ is exactly the *epistemic* uncertainty
$E(x,D,\hat y)=T(x,D,\hat y)-A(x,D)$ of S2.4. So the predictor that minimises
$\text{Reg}@c$ is: base predictor $H(x,D)$ (the Bayesian learned-prior rule,
S2.3) with rejection ordered by $E(x,D)$ — row 2 of the S3 table. Total
uncertainty $T$ (row 1) is the natural-looking but *wrong* score, because it
also charges the predictor for aleatoric noise the reference predictor pays
too. Making that distinction discoverable is the point of the competition, so
neither the rules nor the starter code may name it.

**Consequence of dropping $\hat\theta_{tr}$ from $\Theta$** (see
[§3.3](#33-the-prior-set-theta)): the non-adapted base predictor is no longer a
member of the admissible hypothesis class, so *every* test batch is shifted and
the naive baseline is beatable on every single batch. This is the desired
behaviour — it widens the leaderboard spread and removes the "do nothing" local
optimum — but it invalidates three conventions inherited from the README; see
[§9.1](#91-removing-the-training-prior-from-theta-adopted).

---

## 2. Kaggle

Private competition: <https://www.kaggle.com/competitions/know-what-you-dont-know-under-a-new-prior/>

| Setting | Value |
| :-- | :-- |
| Metric | `AvgRegAtCoverage` (custom notebook metric, [§6](#6-metric)) |
| Direction | **minimise** |
| Evaluation | rows split `Public` / `Private` / `Ignored` via the `Usage` column |
| Submissions/day | 5 (Kaggle default; keep it — it is the main defence against leaderboard probing, see [§9.6](#96-leaderboard-probing-and-the-ignored-split-adopted)) |
| Rules | code submission mandatory at the end; **no data other than the competition data may be used**, no external pretrained weights |

Note the Kaggle spelling of the third usage value is **`Ignored`**, not
`Ignore` — **confirmed against Kaggle's own competition-setup page**. A wrong
literal here would silently drop every row of that pool from scoring rather than
raise an error, so `prepare_kaggle_data.py` asserts it at generation time.

---

## 3. Data

### 3.1 Dataset and splits

TissueMNIST (MedMNIST v2, 28×28 grayscale, $Y=8$ classes), as distributed:
`train` 165 466, `val` 23 640, `test` 47 280.

| Split | Source | Size | Use |
| :-- | :-- | --: | :-- |
| development | MedMNIST `train` | 165 466 | ↓ |
| ↳ training | 0.80 of development | 132 372 | fit the network weights |
| ↳ calibration | 0.10 of development | 16 547 | model selection, BCTS, per-class error rates |
| ↳ **student development** | 0.10 of development | 16 547 | released to students as `dev.csv` |
| evaluation | MedMNIST `val` + `test` merged | 70 920 | generate all test batches |

All splits are **stratified by class** and drawn with a fixed seed. Training
must not reweight classes (no balanced sampler, no class-weighted loss);
`assert` it, because $\hat\theta_{tr}$ is defined as the empirical class
frequency of the *training* split and the whole re-weighting
$\theta_y/\hat p_{tr}(y)$ is only valid if the network was fit under that prior.

The three-way development split is a **change from the README's 0.80/0.20**:
releasing the calibration split to students would hand them data on which the
base predictor's posterior is optimistically calibrated, so their offline
estimates of their own score would be biased. See
[§9.4](#94-calibration-leakage-into-the-student-development-data-adopted).

The evaluation split is partitioned, stratified by class, into **three disjoint
image pools** of 23 640 each, one per `Usage` value. `assert`
$m_{\max}=100\le 23\,640/10$ (S6.2) — satisfied with a factor of 23 to spare.

### 3.2 What "adaptation set" means here (changed)

In the README a trial of size $m$ yields $m+1$ triplets by leave-one-out: the
query is *excluded* from its own adaptation set. **The competition does not use
leave-one-out.** A test batch is $m$ unlabeled images, all of them visible to
the student at once, and the Bayes-optimal decision for the label of image $i$
given the whole batch is

$$p(y_i\mid x_1,\ldots,x_m)=\sum_{\theta\in\Theta}p(\theta\mid x_1,\ldots,x_m)\;p_{te}(y_i\mid x_i,\theta),$$

i.e. the posterior over $\theta$ conditions on **all $m$ images, the query
included**. Nothing in a transductive setting forbids this, it is strictly more
informative than the leave-one-out variant, and it is what students will
naturally do — so it is what the reference optimal solution does too.

Three simplifications follow, and they should be stated in the code:

* the grid is over the **batch size $m$** directly, not $m+1$;
* the leave-one-out trick of S5 is unnecessary — one $p(\theta\mid D)$ per
  batch, not one per query;
* the "known residual leak" of S6.3 (a query recurring inside its own
  adaptation set) is moot, since the query is in the batch by construction.

> **How much this actually changes, on inspection.** For $H$, $T$, $A$ and $E$
> it changes *nothing*: full-batch and leave-one-out are the same predictor, not
> an approximation of one another. Writing S2.2 with $D^{(i)}=D\setminus\{x_i\}$,
> the query enters through $p_{te}(x_i,y\mid\theta)$, whose factor
> $w(x_i,\theta)$ exactly restores the one the leave-one-out set is missing:
> $$p(\theta)\,w(x_i,\theta)\!\!\prod_{j\neq i}\!\! w(x_j,\theta)=p(\theta)\prod_{j} w(x_j,\theta)$$
> which is why the parent code's `pth_query` is documented as "the same for
> every triplet $i$". The **only** rejector the choice moves is the MAP plugin,
> which conditions on the adaptation set alone; here it uses the full-batch
> posterior, which is simpler and strictly better informed. So §9.3 is a
> simplification of the code and a correction to the task statement, not a
> change to the intended optimum. `selftest.py` asserts the identity.

Sampling *with replacement* is still what makes the batch exactly i.i.d. from
$p_{te}(\cdot\mid\theta_*)$ and what keeps spiked priors on rare classes from
exhausting the pool, so it is kept.

### 3.3 The prior set $\Theta$

$\Theta$ is read from `priors_tissuemnist_challenge.txt`: $C=8$ priors, each
placing $\tau=0.35$ on an adjacent pair of classes $(i,i+1 \bmod 8)$ and
$(1-2\tau)/(Y-2)=0.05$ on each of the other six.

$$\Theta=\{\theta_1,\ldots,\theta_8\},\qquad p(\theta)=1/8 \text{ uniform — for the model } \textbf{and} \text{ for drawing } \theta_*.$$

Every element is equidistant from every other by construction, which is what
makes a per-$\theta_*$ breakdown comparable across elements. Their TV distance
to $\hat\theta_{tr}$ ranges over $[0.25,\,0.62]$.

**$\hat\theta_{tr}\notin\Theta$.** The training prior was removed from the
admissible set. It is still shipped (`train_prior.csv`) because the label-shift
re-weighting $\theta_y/\hat p_{tr}(y)$ needs it, but it is not a candidate for
$\theta_*$.

`create_prior.py` currently always emits the training prior as `theta[0]`; the
challenge file was hand-trimmed and re-indexed. Add a `--no-train-prior` flag so
the shipped file is reproducible from code
([§9.1](#91-removing-the-training-prior-from-theta-adopted)).

---

## 4. Generating the test batches

For each usage $u\in\{\texttt{Public},\texttt{Private},\texttt{Ignored}\}$,
independently, drawing only from that usage's image pool:

**Grid.**
$$\mathcal M=\{1,\,2,\,5,\,10,\,20,\,50,\,100\},\qquad |\mathcal M|=7$$

Log-spaced on purpose: the posterior over $\Theta$ concentrates exponentially in
$m$, so the interesting regime is small $m$ — and on the trained base model it
is even smaller than the parent S6.2 grid assumes.

**Why $m_{\max}=100$ and not 500.** Measured on the real model, the two best
rejectors are separated by *exactly* zero at $m\ge100$: the posterior over
$\Theta$ has concentrated on $\theta_*$ and both answer identically on every
row. $m=200$ and $m=500$ therefore cost **77 % of all rows** and return no
ranking information about the top of the leaderboard — dropping them leaves the
separation ratio of the decisive comparison unchanged at 1.04 (Private) / 1.51
(Public), because they scale the gap and its confidence interval by the same
factor. $m=100$ is kept rather than cutting at 50, where the contender gap is
already marginal, because **weaker entries than the reference rejectors remain
separable there**: a non-adapting predictor scores ≈0.099 at $m=100$ against
≈0.000 for anything that adapts. The margin is for the submissions that are not
near-optimal. See [§9.8](#98-the-grid-and-the-weighting-resolved).

**Batch counts.** $N(m)$ is a *batch-count* rule, not a budget on scored rows:

$$N(m)=\max\!\left(N_{\min},\;\left\lceil \frac{2000}{m}\right\rceil\right),\qquad N_{\min}=200$$

Every image of every batch is scored, so the pooled ranking at size $m$ has

$$B_m=N(m)\cdot m$$

rows — from 2000 at $m=1$ to 20 000 at $m=100$. The README's constant curve
budget ($B=2000$ for every $m$, with the surplus subsampled away) is **not**
applied here; see [§9.2](#92-the-constant-curve-budget-is-lifted-adopted) for
why, and for why lifting it changes the metric so little.

$N_{\min}$ is the load-bearing constant. Rows inside one batch share $\theta_*$
and the same adaptation evidence, so they are strongly correlated and the
precision of $\text{Reg}@c(m)$ is governed by the number of *batches*, not the
number of rows: for intra-batch correlation $\rho$, the effective sample size is
$N(m)\,m/(1+(m-1)\rho)$, which is capped at $N(m)/\rho$ however large $m$ gets.
Lowering $N_{\min}$ is therefore expensive and lowering it a lot is fatal —
$\lceil 2000/100\rceil=20$ batches at $m=100$ would make one seventh of the
final score almost pure noise. (The README's $N_{\max}=2000$ clip is inert on this
grid, since $\lceil 2000/m\rceil\le 2000$ for every $m\ge1$; drop it.)

**For each $m\in\mathcal M$ and each batch $j=1,\ldots,N(m)$:**

1. Sample $\theta_*\sim p(\theta)$, uniform on $\Theta$.
2. Sample $m$ labeled examples from the usage's evaluation pool **with
   replacement**, with class frequencies following $\theta_*$, giving the batch
   $D=\{(x_1,y_1),\ldots,(x_m,y_m)\}$. Labels are withheld from students.
3. Assign slots $0,\ldots,m-1$ in draw order — which is uniformly random, since
   the $m$ examples were drawn i.i.d. Every slot is a submission row and every
   slot is scored, so $\max(\texttt{slot})+1=m$ for every batch.

**Resulting numbers, per usage:**

| $m$ | $N(m)$ | rows $B_m=N\cdot m$ | duplicate fraction |
| --: | --: | --: | --: |
| 1 | 2000 | 2 000 | 0.0000 |
| 2 | 1000 | 2 000 | 0.0000 |
| 5 | 400 | 2 000 | 0.0005 |
| 10 | 200 | 2 000 | 0.0007 |
| 20 | 200 | 4 000 | 0.0016 |
| 50 | 200 | 10 000 | 0.0036 |
| 100 | 200 | 20 000 | 0.0067 |
| **total** | **4200** | **42 000** | |

The duplicate-fraction column is the realised rate at which an image recurs
inside its own batch (S6.3 asks for it to be logged rather than ignored); at
$m_{\max}=100$ drawn from a pool of 23 640 it peaks at 0.7 %.

Across the three usages: **12 600 batches** and **126 000** rows in
`test_batches.csv`, in `solution.csv` and in a submission — the three files have
one row per test image each, with identical `row_id` sets in the latter two.

Measured on a generated run rather than estimated:

| File | Size |
| :-- | --: |
| `predictions.csv` | 21.5 MB |
| `sample_submission.csv` (= a submission) | 3.4 MB |
| `solution.csv` | 3.6 MB |
| `dev.csv` | 2.8 MB |
| `test_batches.csv` | 1.8 MB |
| `dev_predictions.csv` | 1.8 MB |
| the remaining `dev*.csv`, `test.csv`, priors | < 1 MB |

**Student-visible total: 32.2 MB**, against 68.4 MB when the PNGs were
shipped. `predictions.csv` grew when it went from one row per image to one row
per test row (C9.7); `test_batches.csv` shrank by the column that change
removed. A submission is ~3.4 MB, comfortably inside Kaggle's 100 MB limit —
which matters, because competitors regenerate one for every attempt. Generation
takes about 5 s end to end, so regenerating after a parameter change is cheap.

Each $m$ has $N_{\min}=200$ batches to bootstrap over, with the **batch as the
resampling unit** — rows within a batch share $\theta_*$ and the adaptation
evidence, so bootstrapping rows directly would badly understate the intervals.

**No $\theta_*$ stratification.** The README's "main panels: shifted trials
only, $\theta_*\neq\theta_1$" convention is vacuous here, since
$\hat\theta_{tr}\notin\Theta$ makes *every* batch shifted. Drop the rule; pool
everything. Keep the per-$\theta_*$ breakdown only as a diagnostic in the
development-set evaluator.

**Seeding.** One master seed; per-usage and per-$m$ streams derived from it by
`np.random.default_rng(...).spawn(...)`, so regenerating one usage does not
perturb the others. Record the seed in a manifest next to the generated files.

---

## 5. Files uploaded to Kaggle

**No images are published, and no image identifier either.** An image reaches a
competitor only as the base predictor's calibrated posterior, published **per
row** and keyed by `row_id`. Withholding the pixels closes the label-recovery
route of [§9.5](#95-no-images-are-published-adopted); withholding the image key
closes the cross-batch channel of [§9.7](#97-cross-batch-image-linkage-adopted).

**Identifier ranges.** Competition `row_id` runs $0\ldots125\,999$ and `id_test`
$0\ldots12\,599$; the development benchmark offsets both by $10^6$, so the two
id spaces cannot collide when a competitor loads both.

`kaggle_code/` — the starter-code bundle (22 kB), built by
`make_student_bundle.py`: a script that writes the baseline submission and
documents the file format, the metric, and an `evaluate.py` that scores a
submission against `dev_solution.csv`. It ships **no model and no pixels**
([§9.5](#95-no-images-are-published-adopted)), so it needs neither torch nor an
image library — numpy and pandas suffice.

The bundle is **built, never hand-maintained**: its `metric.py` is copied from
`chal/metric.py` so a competitor's offline number is the leaderboard's number,
and the builder **asserts** that no file in it names `epistemic`, `aleatoric`,
`true_plugin`, `batch_meta` or the other terms that would hand over the intended
solution ([§1](#1-purpose)).

| File | Rows | Visible to students |
| :-- | --: | :-- |
| `test_priors.csv` | 8 | yes |
| `train_prior.csv` | 1 | yes |
| `predictions.csv` | 126 000 | yes |
| `test.csv` | 12 600 | yes |
| `test_batches.csv` | 126 000 | yes |
| `sample_submission.csv` | 126 000 | yes |
| `dev.csv` | 16 547 | yes |
| `dev_test.csv` | 1 050 | yes |
| `dev_test_batches.csv` | 10 500 | yes |
| `dev_predictions.csv` | 10 500 | yes |
| `dev_solution.csv` | 10 500 | yes |
| `dev_sample_submission.csv` | 10 500 | yes |
| `dev_batch_meta.csv` | 1 050 | yes |
| `solution.csv` | 126 000 | **no** |
| `batch_meta.csv`  `row_image.csv` | — | **never uploaded** |

### Column descriptions

**`test_priors.csv`** — the admissible test priors $\Theta$; $\theta_*$ of every
batch is one of these rows, drawn uniformly.

| column | description |
| :-- | :-- |
| `id` | prior id, $0,\ldots,7$ |
| `p0` … `p7` | prior of label $0,\ldots,7$; each row sums to 1 |

**`train_prior.csv`** — the training label prior $\hat\theta_{tr}$, a single row.

| column | description |
| :-- | :-- |
| `p0` … `p7` | empirical class frequency of the training split |

**`predictions.csv`** — the calibrated base posterior $\hat p_{tr}(y\mid x)$,
**one row per test row**, keyed by `row_id` (C9.7). An image drawn into two
batches therefore appears twice, with identical probabilities.
`dev_predictions.csv` is the same file for the development benchmark.

| column | description |
| :-- | :-- |
| `row_id` | matches `test_batches.csv` / `dev_test_batches.csv` |
| `p0` … `p7` | posterior of label $0,\ldots,7$ |

**`test.csv`** — one row per test batch.

| column | description |
| :-- | :-- |
| `id_test` | batch id |
| `m` | number of images in the batch |

**`test_batches.csv`** — the contents of each batch, one row per test image.

| column | description |
| :-- | :-- |
| `row_id` | identifies the test image; globally unique, matches the submission |
| `id_test` | batch id |
| `slot` | $0,\ldots,m-1$; order is fixed and carries no meaning |

**No image key is published** (C9.7): a row is identified only by `row_id`, and
the posterior is looked up in `predictions.csv` by the same key.

**`sample_submission.csv`** — one row per test image, i.e. one row for every
row of `test_batches.csv`. The baseline it encodes is the non-adapted base
predictor.

| column | description |
| :-- | :-- |
| `row_id` | identifies the test image; globally unique |
| `pred` | predicted label $0,\ldots,7$ |
| `confidence` | real number; **higher = more likely to be accepted** |

**`solution.csv`** — not visible to students.

| column | description |
| :-- | :-- |
| `row_id` | matches `sample_submission.csv` |
| `id_test` | batch id |
| `slot` | $0,\ldots,m-1$ |
| `m` | batch size |
| `label` | ground-truth label $0,\ldots,7$ |
| `pred_ref` | label emitted by the reference predictor |
| `Usage` | `Public`, `Private` or `Ignored`; constant within a batch |

Since every slot of every batch is present, $m$ is recoverable as
$\max(\texttt{slot})+1$ within each `id_test`, exactly as originally intended.
It is nevertheless carried as its own column: `score()` groups by $m$, and a
column costs one integer per row while a group-by-max is one more thing to get
wrong in code that cannot be debugged after launch.

**`dev.csv`** — the labeled development pool, self-contained and with no key:
nothing joins to it, and giving development images an identifier that test rows
lack would only invite the question of why.

| column | description |
| :-- | :-- |
| `label` | ground-truth label |
| `p0` … `p7` | posterior of label $0,\ldots,7$ |

### The reference predictor

`pred_ref` is the plugin Bayes predictor adapted to the **true** test prior of
that batch (S2.1, S3 row 5):

$$h(x,\theta_*)=\operatorname*{arg\,max}_{y}\;\frac{\theta_{*,y}}{\hat p_{tr}(y)}\,\hat p_{tr}(y\mid x)$$

evaluated with the same calibrated $\hat p_{tr}(y\mid x)$ the students get.
Shipping $h(x,\theta_*)$ rather than $\theta_*$ keeps $\theta_*$ out of the
solution file entirely. Ties in the `arg max` resolve to the lowest label index.

---

## 6. Metric

### 6.1 Definition

At a fixed batch size $m$, all rows of that $m$ — **pooled across batches, and
therefore across different $\theta_*$** — form a single ranking.
Let $B_m=N(m)\cdot m$ be the number of such rows ([§4](#4-generating-the-test-batches)), $\pi$ the permutation sorting them by
**descending `confidence`**, $h$ the submitted predictor (`pred`), $h_*$ the
reference (`pred_ref`), and $\ell$ the 0/1 loss. At rank $k$ the predictor
accepts the $k$ most confident rows:

$$coverage(k)=\frac{k}{B_m},\qquad
regret(k)=\frac1k\sum_{i=1}^{k}\Big(\ell\big(y_{\pi(i)},h(x_{\pi(i)})\big)-\ell\big(y_{\pi(i)},h_*(x_{\pi(i)})\big)\Big)$$

$$\text{Reg}@c(m)=regret\big(\lceil c\,B_m\rceil\big),\qquad c=0.8$$

$$\boxed{\;\text{AvgRegAtCoverage}=\frac{1}{|\mathcal M|}\sum_{m\in\mathcal M}\text{Reg}@c(m)\;}$$

Each $m$ contributes equally regardless of how many rows it has — note $B_m$
ranges from 2000 to 20 000, so the seven terms are *not* equally precise; their
precision tracks $N(m)$, which is 2000 at $m=1$ and 200 everywhere from $m=10$
up. Lower is better. **The score can be negative**: both predictors are built on the same
imperfect calibrated posterior, so an adapted predictor occasionally beats the
true-prior plugin. State this in the competition description, otherwise the
first negative leaderboard entry will be reported as a bug.

### 6.2 Conventions the students must be told

* Only the **ranking** induced by `confidence` matters; any strictly increasing
  transformation of it gives the same score.
* Confidences are pooled **across batches** at a fixed $m$, so they must be
  comparable between batches — a per-batch normalisation changes the score.
* **Ties in `confidence` are broken by ascending `row_id`.** Without a stated
  rule the accepted set is ill-defined for constant-confidence submissions.
* Students are not asked to reject anything themselves; the organiser sweeps the
  coverage. A label and a confidence for **every** row of `test_batches.csv`
  are mandatory; there is no way to abstain in the file format.
* The 80 % threshold is global within an $m$, not per batch. A batch whose
  prior the student could not pin down should therefore carry *uniformly lower*
  confidence than a batch they could — that cross-batch calibration is most of
  the problem.

### 6.3 `score()` implementation notes

Written into `metric-template.ipynb`. Kaggle passes all solution columns except
`Usage`, aligned with the submission on `row_id_column_name`.

```
score(solution, submission, row_id_column_name) -> float
```

1. Align `solution` and `submission` on `row_id` (Kaggle aligns them, but sort
   both explicitly and assert equality — do not rely on it).
2. Validate, raising `ParticipantVisibleError` with a message students can act
   on: `pred` present, integer, in $\{0,\ldots,7\}$; `confidence` present,
   numeric, finite (reject NaN and ±inf explicitly); no missing or duplicate
   `row_id`; row count matches.
3. Group by `m`. For each group: sort by `(-confidence, row_id)`, take the first
   $\lceil 0.8\,B_m\rceil$ rows, compute the mean of
   `(pred != label) - (pred_ref != label)`. $B_m$ is the group's own row count
   and differs by group; never hard-code 2000.
4. Return the unweighted mean over the 7 groups as a `float`. Assert all 7
   values of $\mathcal M$ are present in the group keys — a missing $m$ means
   the solution file was generated wrong, and silently averaging over 6 would
   change the scale of the leaderboard.
5. Add doctests; Kaggle renders the docstring on the metric page, so keep it
   under 8 000 characters and write it for a student audience.

Roughly 42 000 rows reach `score()` per usage. Sort once per group with
`numpy.lexsort`; do not do anything per batch in a Python loop.

---

## 7. Code

Everything lives in `challenge/` and is self-contained: copying the directory
elsewhere must be enough to run it, with no imports from the parent tree. That
means the parts of `exact/` the challenge needs (`inference`, `priors`,
`protocol`, `metrics`, `calibration`) are **vendored** into
`challenge/exact/`, not imported from `../exact`.

| Script | Does |
| :-- | :-- |
| `download_data.py` | fetch and preprocess the TissueMNIST `.npz` |
| `train_base_model.py` | train the CNN on the training split, BCTS-calibrate on the calibration split, report val NLL and ECE (15 equal-mass bins) before and after, emit $\hat\theta_{tr}$ |
| `make_priors.py` | build `priors_tissuemnist_challenge.txt` — the 8 pair priors, **without** the training prior |
| `prepare_kaggle_data.py` | generate everything in [§5](#5-files-uploaded-to-kaggle), plus a manifest recording seed, split sizes, per-$m$ counts and the realised class frequencies per $\theta_*$ |
| `metric-template.ipynb` | the `score` function ([§6.3](#63-score-implementation-notes)) |
| `evaluate.py` | `evaluate.py <submission.csv> <solution.csv>` → prints `AvgRegAtCoverage`, plots $\text{Reg}@0.8$ vs $m$ with bootstrap 95 % bands (**batch** as the resampling unit), and a risk–coverage curve per $m$. Optional `--budget B` re-imposes the constant-$B$ pooling for paper-style panels ([§9.2](#92-the-constant-curve-budget-is-lifted-adopted)) |
| `optimal_solution.py` | the intended optimum: base predictor $H(x,D)$, rejection ordered by $E(x,D)=T(x,D)-A(x,D)$ with ties broken by $T$; writes a submission |
| `make_dev_benchmark.py` | run the [§4](#4-generating-the-test-batches) protocol on the **development** split, emitting `dev_test.csv`, `dev_test_batches.csv`, `dev_solution.csv` (no `Usage`) and `dev_sample_submission.csv` |
| `make_student_bundle.py` | assemble `kaggle_code/` from `student/` and `chal/metric.py`, and audit it — for leaks of the intended solution, and for any model or pixel data, which [§9.5](#95-no-images-are-published-adopted) withholds |
| `compare_baselines.py` | score submissions on shared bootstrap resamples and report paired differences — the [§8](#8-pre-launch-checklist) separation criterion |

`make_dev_benchmark.py` output, `evaluate.py` and a starter notebook are shipped
to students so they can score themselves offline before submitting. Reuse one
generation function for both the Kaggle data and the dev benchmark — a divergence
between the two is the single most likely source of a silent scoring mismatch.

**Baselines to compute before launch,** so the leaderboard's range is known and
the intended optimum is verified to be the optimum:

1. non-adapted base predictor (the sample submission),
2. MAP plugin $h(x,\hat\theta_{map})$, uncertainty $1-\hat p_{te}(\hat y\mid x,\hat\theta_{map})$,
3. Bayesian base predictor with **total** uncertainty $T$ — the plausible wrong answer,
4. Bayesian base predictor with **epistemic** uncertainty $E$ — the intended optimum,
5. true-prior plugin — identically zero regret, a sanity check on the pipeline.

If (3) and (4) are not clearly separated on the private split, the competition
has no discoverable structure and $\tau$, $\mathcal M$ or $c$ need retuning.
Check this first; it is the one result the whole design rests on.

---

## 8. Pre-launch checklist

- [ ] (3) and (4) above separated by a **paired** bootstrap interval excluding 0
      (`compare_baselines.py`; comparing the two *marginal* intervals is the
      wrong test and a conservative one — they share a base predictor, so most
      of their variance cancels in the difference)
- [ ] `score()` on `optimal_solution.py`'s output equals `evaluate.py`'s number
- [ ] `score()` on `sample_submission.csv` reproduces the baseline number
- [ ] every `row_id` in `sample_submission.csv` appears exactly once in `solution.csv`, and `test_batches.csv` has the same `row_id` set
- [ ] `max(slot)+1 == m` for every `id_test`, and no `slot` is missing
- [x] `Usage` literal matches Kaggle's (`Ignored`, not `Ignore`) — confirmed on
      the Kaggle setup page, asserted in `prepare_kaggle_data.py`, and the
      generated `solution.csv` holds exactly `{Public, Private, Ignored}`
- [ ] the three image pools are disjoint; assert it
- [ ] no image in `dev.csv` occurs in any test batch; assert it
- [ ] `m_max` $\le$ pool size / 10, per pool
- [ ] competition configured to **minimise**
- [ ] the description states: score may be negative (it is — see
      [§9.7](#97-negative-scores-and-the-leaderboard-resolved)), ties broken by
      `row_id`, higher confidence = accepted first
- [ ] rules state: no external data, no external pretrained weights, code submission mandatory
- [ ] `make_student_bundle.py` audit passes, and the bundle's `evaluate.py` run on `dev_sample_submission.csv` reproduces the organiser's number for those rows

---

## 9. Open issues

### 9.1 Removing the training prior from $\Theta$ **[adopted]**

Three conventions in the README are conditioned on $\theta_1=\hat\theta_{tr}$
and break silently now that it is gone.

* **S3's MAP tie-break** ("at $m=0$ every $\theta$ ties; resolve by lowest
  index, so the MAP plugin degenerates gracefully to the train-prior plugin").
  With $\hat\theta_{tr}\notin\Theta$ the lowest index is `pair on classes [0,1]`
  — an arbitrary strongly-shifted prior. *Resolution:* the competition has no
  $m=0$ batch, so the exact tie only arises for a measure-zero coincidence of
  likelihoods. Keep lowest-index as the deterministic tie-break, and drop the
  README's claim about graceful degeneration from the challenge documentation.
* **S6.4's "shifted trials only" stratification** is now vacuous — every
  $\theta_*$ is shifted. *Resolution:* drop it; pool everything. Keep the
  per-$\theta_*$ breakdown as a development-set diagnostic only.
* **S7's index-1-is-the-training-prior convention** no longer holds, and
  `create_prior.py` still writes `theta[0] = train` unconditionally, so the
  shipped file is a hand-edit and not reproducible. *Resolution:* add
  `--no-train-prior` to `create_prior.py` and regenerate
  `priors_tissuemnist_challenge.txt` from it, so the file has a command line
  behind it.

The *intended* effect of the removal is good and should be kept: the naive
non-adapted predictor is no longer in the hypothesis class, so it is beatable on
every batch.

### 9.2 The constant curve budget is lifted **[adopted]**

The README fixes $B=2000$ scored rows at every $m$ and subsamples the surplus
away ("stratified so each trial contributes the same number"). Carried into the
competition, that means a student submits 20 000 rows at $m=100$ of which 2000
count, and the subsampling has to live inside `score()` — the one piece of code
that cannot be debugged after launch.

*Resolution:* drop the budget. Score every row; let $B_m=N(m)\cdot m$. Read
$N(m)$ as a batch-count rule and keep $N_{\min}=200$.

**Why this barely moves the metric.** Both designs estimate the same population
quantity — expected regret conditional on confidence above the 80th percentile
— so there is no change of estimand and no bias either way (if anything the
plug-in bias from estimating the threshold on the same sample, $O(1/B_m)$,
shrinks). What changes is variance, and much less than the 50× row count
suggests:

* **Batch-level weighting is identical.** Each batch contributed
  $q/B = 1/N(m)$ of the pool before and contributes $m/B_m = 1/N(m)$ now. The
  subsample was weight-preserving; only within-batch resolution changed.
* **Precision is capped by the batch count.** With intra-batch correlation
  $\rho$, $N_{\text{eff}} = N(m)\,n/(1+(n-1)\rho) \to N(m)/\rho$. Going from
  $n=10$ to $n=100$ at $m=100$ multiplies $N_{\text{eff}}$ by
  $10(1+9\rho)/(1+99\rho)$: **1.02× at $\rho=0.8$, 1.09× at $\rho=0.5$,
  1.32× at $\rho=0.2$** — i.e. a 1–13 % reduction in standard error.
* **$\rho$ is large exactly where the rule bound.** At large $m$ the posterior
  concentrates on $\theta_*$ almost surely, so regret is ~0 for nearly every
  row and the variance is carried by the rare batches where adaptation picks the
  wrong $\theta$ — in which *every* row of that batch suffers together. The
  per-batch outcome is nearly binary, putting $\rho$ near 1. Where $\rho$ would
  be small enough to matter, at $m\le10$, the budget was not binding anyway —
  the per-batch quota $\lfloor B/N(m)\rfloor$ equals $m$ for every $m\le10$, so
  every row was already scored there.

**What lifting it costs:** nothing in images or `test_batches.csv` (the budget
never controlled how many images are *shown*), and nothing in submission size
under the file design of [§5](#5-files-uploaded-to-kaggle), where students
predict every image regardless. `solution.csv` grows to 126 000 rows and
`score()` sorts more rows. Both trivial.

**What it buys:** `slot` runs $0,\ldots,m-1$ again and $\max(\texttt{slot})+1=m$,
as originally intended; `score()` becomes a group-by with no subsampling; and at
large $m$ the pooled ranking now orders 100 rows per batch rather than 10, so a
good *within*-batch uncertainty ordering is rewarded more finely.

**What the rule was for, and why it does not apply.** README S6.4 introduced it
so bootstrap bands would not narrow left-to-right and make a flat curve look
like improvement. A leaderboard is a single scalar with no bands, so the hazard
does not arise; and constant $B$ would not have fixed it anyway, since the
bootstrap unit is the batch and band width tracks $N(m)$, which still spans
2000 down to 200 across the grid. Whatever equalisation the README's formula
achieves comes from the $N_{\min}$ clip, not from $B$.

*Keep the constant-$B$ pooling as an option in `evaluate.py` if you want
paper-style panels with comparable bands — but say in the caption that $N(m)$ is
the knob that actually sets them.*

### 9.3 Leave-one-out vs. the full batch **[adopted]**

The previous draft says "the adaptation set will be $m-1$ if they use Bayesian
learning". That is a *recommendation*, not something the format can enforce —
the student holds the whole batch. It is also suboptimal: conditioning on all
$m$ images is the Bayes rule for the transductive problem actually posed.

*Resolution:* [§3.2](#32-what-adaptation-set-means-here-changed). Condition on
the full batch, define the grid over $m$ directly, drop the LOO trick from the
challenge code, and drop the S6.3 duplicate-leak note as inapplicable.

Note what this does *not* do: as the box in §3.2 shows, $H$, $T$, $A$ and $E$
are identical under both readings, so the intended optimum is unchanged and no
result from the parent code is invalidated. Only the MAP plugin moves.

### 9.4 Calibration leakage into the student development data **[adopted]**

The previous draft releases the validation split as `dev.csv` — the same split
BCTS was fit on. Students' offline estimates of their own score would be
optimistic, and by an amount that differs between methods (label-shift
correction is very sensitive to calibration), so the offline ranking would not
match the leaderboard ranking. That is the worst kind of discrepancy: it looks
like a bug in the competition.

*Resolution:* three-way development split, 0.80 / 0.10 / 0.10
([§3.1](#31-dataset-and-splits)). 16 547 calibration examples is ample for
$2Y=16$ BCTS parameters, and the students get an untouched 16 547.

### 9.5 No images are published **[adopted]**

The CC BY licence of MedMNIST v2 obliges the competition to name TissueMNIST as
its source. TissueMNIST is public and **its labels are public**, so naming it
tells every competitor where the answer key lives.

Measured, not assumed: hashing the released 28×28 PNGs against the public
`tissuemnist.npz` recovered **49 670 / 49 670 test labels — 100 %, zero
ambiguity, in 28 seconds.** The released images were byte-identical to the
originals; random filenames hid nothing, because the pixels *are* the identifier.

*Resolution:* **publish no images at all.** A competitor receives the base
predictor's calibrated posterior per row (`predictions.csv`) keyed by `row_id`,
and nothing else. There is no pixel data to match against the archive,
and the starter kit no longer ships `model.pt`, so the posteriors cannot be
recomputed for the archive either. Both halves are needed: the model without the
images, or the images without the model, would each re-open the route.

Perturbing the pixels instead was considered and rejected: at 28×28, any noise
small enough to leave the posteriors intact leaves nearest-neighbour matching
trivial.

**What this costs.** Competitors can no longer fine-tune the network, engineer
features, or look at an image. The competition becomes purely a problem of
reasoning about a classifier's output under an unknown prior — which is the
intended problem, and the intended solution never touched a pixel. It also
removes a class of off-topic work and shrinks the upload from 68.4 MB to 24.1 MB.

**What it does not fix.** Two rows of the same image still carry identical
probability vectors, so the pool-clustering observation of
[§9.6](#96-leaderboard-probing-and-the-ignored-split-adopted) is unchanged; and nothing *technically* stops a competitor training on
TissueMNIST directly and ignoring the provided posteriors. The defences there
are the rules, the code requirement, and the fact that recovered labels score
about **−0.30** against **+0.019** for the best legitimate method — a gap no
honest submission can reach.

### 9.7 Cross-batch image linkage **[adopted]**

Images are drawn from a finite pool **with replacement**, so one recurs across
batches. That is a channel the intended solution does not use: an image that
keeps appearing in batches enriched on classes $\{i,i+1\}$ is probably class $i$
or $i+1$. It needs **no external data** — the counts were in the released files
and `dev.csv` supplies the labels to calibrate on.

Measured on the Private split, with $\theta$ *estimated per batch* as a
competitor would have to:

| | | |
| :-- | --: | --: |
| intended solution | +0.0189 | accuracy 0.6891 |
| occurrence count only | −0.0312 | accuracy 0.7450 |
| full co-occurrence attack | **−0.0737** | accuracy 0.7833 |
| reference predictor | 0.0000 | accuracy 0.7043 |

Two things were tried and rejected:

* **Balancing the pools by class** (`--balance-pools`, kept as an option) shuts
  the occurrence-*count* channel — spread 4.50× → 1.05×, gain +0.109 → +0.002 —
  but shrinks each pool from 23 640 to 6 704, so repetition rises 2.5 → 6.3 and
  the dominant channel *doubles*, to −0.1429. Do not use it.
* **More priors.** Going from the 8 sliding pairs to all 28 pairs leaves the
  leak untouched (−0.0166 / −0.1355), because its strength is set by
  $\log(\tau_{hi}/\tau_{lo}) = 1.946$ nats and by the fraction of priors
  enriching a class (25 % in both) — neither depends on $C$.

Eliminating it would need roughly one occurrence per image, which the rarest
class forbids: 838 images per usage attract ~2 750 draws under spiked priors.
Sampling without replacement is infeasible for exactly the reason S6.3 chose
replacement.

*Resolution:* **remove the handle and rule the channel out of scope.** No image
identifier is published; `predictions.csv` is keyed by `row_id`, one row per
test row, and `dev.csv` carries its own posteriors. A competitor must now
reconstruct identity by matching identical probability vectors — deliberate
rather than accidental — and [rules §5](../challenge/description/rules.md)
forbids it in terms narrow enough to be checked in submitted code: linking rows
of *different* batches by image identity, while explicitly leaving all other
cross-batch computation allowed, since the metric ranks confidences across
batches and requires it.

`audit_leakage.py` measures the channel on any generated dataset, using the
local-only `row_image.csv`; run it before every launch.

**Residual risk.** This is an honour-system component backed by code review, so
it binds only entrants claiming credit. A clearly negative leaderboard score
remains the live signal — the intended optimum scores +0.019, this attack
−0.074, outright label theft −0.30 — but note a competitor may legitimately
recalibrate the posterior on `dev.csv` and edge slightly below zero, so
investigate below about −0.02 rather than treating any negative as proof.

### 9.6 Leaderboard probing and the `Ignored` split **[adopted]**

Because two rows of the same image carry identical probability vectors
([§9.5](#95-no-images-are-published-adopted))
and the three usage pools are disjoint, a determined student can cluster the
13 800 batches into three groups by which image pool they draw from, then
identify which group is `Public` with two probe submissions (corrupt one group,
see whether the public score moves). **The `Ignored` split is therefore
defeatable and must not be relied on.**

*Resolution:* keep the disjoint pools anyway. Disjointness is what protects the
*private* score, which is the one that decides the competition: labels a student
extracts by probing the public leaderboard are about public-pool images and are
worthless on the private pool. `Ignored` stays because it is free and still
stops casual probing; it is just not a guarantee. The real defences are the 5
submissions/day cap and the mandatory code submission.

The alternative — a single shared pool for all three usages — hides the
partition completely but lets probed labels transfer to the private split. That
trade is the wrong way round; do not take it.

### 9.7 Negative scores and the leaderboard **[resolved]**

**Confirmed on the real model, not just possible in principle:** the intended
optimum scores $-0.0016$ at $m=20$ on the private split. The competition
description must say so, or the first negative entry will be reported as a bug.

*Resolution:* the short version — "scores below zero are possible and are not an
error". Explaining *why* (both predictors are built on the same imperfect
$\hat p_{tr}(y\mid x)$) is honest but hands students a hint that the reference is
a plugin on that same posterior, which is most of the way to the intended
solution. No floor is worth reporting; the baselines of [§7](#7-code) bracket
the interesting range.

### 9.8 The grid and the weighting **[resolved]**

Two questions — whether $m=1$ earns its slot, and whether the sizes should be
weighted equally — turned out to have one answer: **fix the grid, keep equal
weighting.** All numbers below are measured on the trained base model (31.8 %
evaluation error, ECE 0.008), comparing `bayes_total` against `bayes_epistemic`,
which share a base predictor and differ only in their ranking.

Per-size paired gap, 95 % CI, fraction of 1000 replicates won:

| $m$ | Private | Public | |
| --: | :-- | :-- | :-- |
| 1 | +0.0025 [−0.0081, +0.0125] 68 % | +0.0169 [+0.0050, +0.0269] 100 % | weak, pool-inconsistent |
| 2 | +0.0075 [−0.0056, +0.0200] 87 % | +0.0112 [−0.0019, +0.0231] 94 % | weak |
| 5 | +0.0169 [+0.0031, +0.0294] 99 % | +0.0169 [+0.0025, +0.0244] 99 % | **signal** |
| 10 | +0.0125 [+0.0012, +0.0244] 98 % | +0.0138 [+0.0019, +0.0237] 99 % | **signal** |
| 20 | +0.0119 [+0.0031, +0.0197] 100 % | +0.0122 [+0.0059, +0.0209] 100 % | **signal** |
| 50 | +0.0034 [+0.0001, +0.0079] 98 % | +0.0026 [−0.0000, +0.0065] 96 % | marginal |
| 100 | −0.0001 | +0.0006 | ~0 between contenders |
| 200 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **exactly zero** |
| 500 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | **exactly zero** |

Separation ratio of the decisive comparison (paired gap ÷ CI width) under
candidate grids, with the rows each costs per usage:

| grid | rows/usage | Private | Public |
| :-- | --: | --: | --: |
| [1,2,5,10,20,50,100,200,500] | 182 000 | 1.04 | 1.51 |
| **[1,2,5,10,20,50,100]** | **42 000** | **1.04** | **1.51** |
| [1,2,5,10,20,50] | 22 000 | 1.04 | 1.51 |
| [1,2,5,10,20] | 12 000 | 0.99 | 1.46 |
| [2,5,10,20,50] | 20 000 | 1.04 | 1.30 |
| [5,10,20] | 8 000 | 1.06 | 1.23 |
| [1,2,5,10] | 8 000 | 0.82 | 1.25 |

**Resolution — $\mathcal M=\{1,2,5,10,20,50,100\}$, equal weights.** Three
readings drive it:

* **Dropping $m=200,500$ is free.** Their per-size gap is *identically* zero on
  every replicate, so removing them multiplies the overall gap and its interval
  by the same $9/7$ and the separation ratio is unchanged — 1.04 / 1.51 either
  way. They cost 77 % of all rows for no ranking information at the top of the
  leaderboard.
* **Keep $m=1$.** Its own gap is weak and disagrees between pools, but removing
  it leaves Private unchanged (1.04) and *costs* Public 1.51 → 1.30. It is the
  honest "no information" anchor and it earns its slot empirically.
* **Stop at 100 rather than 50.** Between the two best rejectors $m=100$ adds
  nothing, and cutting to 50 would be free by the same argument. It is kept as
  margin for **weaker entries than the reference rejectors**, which stay clearly
  separable there: a non-adapting predictor scores ≈0.099 at $m=100$ while
  anything that adapts scores ≈0.000. The metric has to rank the whole field,
  not only its top two.

With the dead sizes gone, **equal weighting needs no correction** — the
reweighting idea existed only to suppress them, and dropping a rule beats adding
one on a competition page.

*Caveat on transferring these numbers.* They are specific to this base model. A
better $\hat p_{tr}(y\mid x)$ identifies $\theta$ faster and moves saturation to
*smaller* $m$: on a deliberately under-trained model (45.6 % error) saturation
began at $m\approx200$, on the real one at $m\approx50$. Retrain the base model
and this table must be recomputed — `compare_baselines.py` against a filtered
`solution.csv` is the whole procedure, and it takes minutes.

### 9.9 Minor fixes folded in **[adopted]**

* `test_bacthes.csv` → `test_batches.csv`; `train_priors.csv` → `train_prior.csv`
  (the file list and the column-description heading disagreed).
* The regret formula wrote both predictors as $h$; the reference is
  $h_*(x)=h(x,\theta_*)$, shipped as `pred_ref`.
* "Ranked by descending `confidence`" is now stated with the acceptance
  direction and the tie-break rule, and matches the README's ascending-$u$
  convention under $u=-\text{confidence}$.
* $m$ is carried explicitly in `solution.csv` as well as being recoverable from
  `max(slot)+1`, so `score()` needs no derivation step.
* The grid is written $\{1,2,5,10,20,50,100\}$ everywhere; the original draft
  twice elided it as $\{1,2,5,10,\ldots,500\}$, which reads as a different
  sequence.
