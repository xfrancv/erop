# Exact Bayesian label-prior adaptation with a finite prior set


## Notation

Fixed once, used consistently throughout. **No symbol is reused.**

| Symbol | Meaning |
| --- | --- |
| $\mathcal Y=\{1,\ldots,Y\}$ | label space, $Y$ classes |
| $\Theta=\{\theta_1,\ldots,\theta_C\}$ | finite parameter set, $C=\lvert\Theta\rvert$ |
| $p(\theta)$ | prior over $\Theta$ (known; uniform by default) |
| $\theta_*$ | the true test prior of a given trial |
| $D=(x_1,\ldots,x_m)$ | unlabeled **adaptation set**, $m=\lvert D\rvert$ |
| $D'=\{(x_i,y_i)\}_{i=1}^{m+1}$ | one labeled **test set** (a trial); yields $m+1$ triplets |
| $\mathcal M$ | grid of adaptation sizes $m$ |
| $N(m)$ | number of trials at size $m$ |
| $B$ | pooled ranking budget (points per curve), constant across $m$ |
| $k$ | rank in the ranking, $1\le k\le B$ |
| $\ell(y,y')$ | 0/1 loss, $=1$ iff $y\neq y'$ |
| $T,A,E$ | total / aleatoric / epistemic uncertainty |
| $H(x,D)$ | Bayesian learned-prior predictor |
| $h(x,\theta)$ | plugin predictor for a fixed $\theta$ |

$\operatorname*{arg\,min}$ / $\operatorname*{arg\,max}$ are written with
`\operatorname*{arg\,min}` rather than `\argmin`, which GitHub's MathJax does
not define.

---

## 1. Problem

A base model is trained on supervised data drawn from $p_{tr}(x,y)$, giving an
estimate of the training posterior $p_{tr}(y\mid x)$ and of the training prior
$p_{tr}(y)$.

At test time we receive a batch of **unlabeled** inputs $D=(x_1,\ldots,x_m)$
drawn from a distribution whose **class conditionals are unchanged**
($p_{tr}(x\mid y)=p_{te}(x\mid y)$) but whose **prior $p_{te}(y)$ is unknown and
different**.

**Goal.** Given a query input $x$ together with the unlabeled batch $D$, predict
the label of $x$ and quantify the uncertainty induced by not knowing the test
prior. The batch $D$ carries information about the test prior; the query $x$ is
*not* a member of $D$ (see §6.3).

---

## 2. Method

### 2.1 Label-shift identities

Under label shift the test posterior is a re-weighting of the training
posterior by the unknown test prior $p_{te}(y\mid\theta)$:

$$p_{te}(y\mid x,\theta) = p_{tr}(y\mid x)\,\frac{p_{te}(y\mid\theta)}{p_{tr}(y)}\,\frac{p_{tr}(x)}{p_{te}(x\mid\theta)}$$

The joint test distribution of $(x,y)$:

$$p_{te}(x,y\mid\theta) = p_{tr}(y\mid x)\,\frac{p_{te}(y\mid\theta)}{p_{tr}(y)}\,p_{tr}(x)$$

and the marginal density of one input:

$$p_{te}(x\mid\theta) = p_{tr}(x)\sum_{y}\frac{p_{te}(y\mid\theta)}{p_{tr}(y)}\,p_{tr}(y\mid x)$$

The parameter space is assumed **finite**, $\theta\in\Theta=\{\theta_1,\ldots,\theta_C\}$,
and the parameter prior $p(\theta)$ is assumed **known**.

> **$p_{tr}(x)$ never has to be estimated.** It appears as a common factor in
> every numerator and denominator below and cancels exactly. Define the
> tractable **weight**
> $$w(x,\theta)\;\triangleq\;\frac{p_{te}(x\mid\theta)}{p_{tr}(x)}=\sum_{y}\frac{\theta_y}{p_{tr}(y)}\,\hat p_{tr}(y\mid x)$$
> and work with $w$ throughout. See §5 for the log-space form.

### 2.2 Posterior over labels

Let $D=(x_1,\ldots,x_m)$ be an i.i.d. sample from the test distribution. The
data likelihood is

$$p_{te}(D\mid\theta)=\prod_{i=1}^{m}p_{te}(x_i\mid\theta)$$

The label posterior given the query and the batch is

$$p(y\mid x,D)=\frac{\sum_{\theta}p(\theta)\,p_{te}(x,y\mid\theta)\,p_{te}(D\mid\theta)}{\sum_{\theta}p(\theta)\,p_{te}(x\mid\theta)\,p_{te}(D\mid\theta)}$$

### 2.3 Total uncertainty and the Bayesian rule

$$T(x,D,\hat y)={\mathbb E}_{y\sim p(y\mid x,D)}\big[\ell(y,\hat y)\big]=\frac{\sum_{y}\sum_{\theta}p(\theta)\,p_{te}(x,y\mid\theta)\,p_{te}(D\mid\theta)\,\ell(y,\hat y)}{\sum_{\theta}p(\theta)\,p_{te}(x\mid\theta)\,p_{te}(D\mid\theta)}$$

The **Bayesian learned-prior rule**:

$$H(x,D)=\operatorname*{arg\,min}_{\hat y}\,T(x,D,\hat y)$$

and the total uncertainty of that predictor is $T(x,D)=T(x,D,H(x,D))$.

> **Under 0/1 loss** these collapse to
> $T(x,D,\hat y)=1-p(\hat y\mid x,D)$, so
> $H(x,D)=\operatorname*{arg\,max}_y p(y\mid x,D)$ and
> $T(x,D)=1-\max_y p(y\mid x,D)$.
> Implement it this way; it is cheaper and better conditioned.

### 2.4 Aleatoric and epistemic uncertainty

$$A(x,D)={\mathbb E}_{\theta,y\sim p(\theta,y\mid x,D)}\big[\ell\big(y,h(x,\theta)\big)\big]=\frac{\sum_{\theta}\sum_{y}p(\theta)\,p_{te}(x,y\mid\theta)\,p_{te}(D\mid\theta)\,\ell\big(y,h(x,\theta)\big)}{\sum_{\theta}p(\theta)\,p_{te}(x\mid\theta)\,p_{te}(D\mid\theta)}$$

where the per-$\theta$ plugin predictor is

$$h(x,\theta)=\operatorname*{arg\,min}_{\hat y}\sum_{y}p_{te}(y\mid x,\theta)\,\ell(y,\hat y)\;\overset{0/1}{=}\;\operatorname*{arg\,max}_{y}\;\frac{\theta_y}{p_{tr}(y)}\,\hat p_{tr}(y\mid x)$$

The epistemic uncertainty of a decision $\hat y$:

$$E(x,D,\hat y)={\mathbb E}_{\theta,y\sim p(\theta,y\mid x,D)}\big[\ell(y,\hat y)-\ell\big(y,h(x,\theta)\big)\big]=T(x,D,\hat y)-A(x,D)$$

and of the Bayesian predictor: $E(x,D)=T(x,D)-A(x,D)$.

**Two properties worth stating in the paper, both immediate:**

1. $E(x,D)\ge 0$. Because $h(x,\theta)$ minimises the per-$\theta$ conditional
   risk, $A(x,D)\le T(x,D,\hat y)$ for *every* $\hat y$.
2. $A(x,D)$ does not depend on $\hat y$, so
   $\operatorname*{arg\,min}_{\hat y}E=\operatorname*{arg\,min}_{\hat y}T=H(x,D)$.
   The two Bayesian reject-option predictors therefore share one base
   predictor and their risk-coverage curves **meet at full coverage**.

---

## 3. Reject-option predictors

A reject-option predictor is a pair (base predictor $h$, uncertainty score
$u$): it emits $h(x)$ when $u(x)$ is below a threshold and abstains otherwise.
Sweeping the threshold traces a curve, so no threshold is fixed in advance.

All predictors are scored under the 0/1 loss.

| # | Reject-option predictor | Base predictor | Uncertainty $u$ |
| --- | --- | --- | --- |
| 1 | Bayesian, total | $H(x,D)$ | $T(x,D)$ |
| 2 | Bayesian, epistemic | $H(x,D)$ | $E(x,D)=T(x,D)-A(x,D)$, ties broken by $T(x,D)$ |
| 3 | MAP-plugin | $h(x,\hat\theta_{map})$ | $1-p_{te}\big(h(x,\hat\theta_{map})\mid x,\hat\theta_{map}\big)$ |
| 4 | train-prior plugin | $h(x,\hat\theta_{tr})$ | $1-p_{te}\big(h(x,\hat\theta_{tr})\mid x,\hat\theta_{tr}\big)$ |
| 5 | true-prior plugin *(oracle)* | $h(x,\theta_*)$ | $1-p_{te}\big(h(x,\theta_*)\mid x,\theta_*\big)$ |

where

$$\hat\theta_{map}=\operatorname*{arg\,max}_{\theta\in\Theta}\;p(\theta\mid D),\qquad p(\theta\mid D)\propto p(\theta)\,p_{te}(D\mid\theta)$$

$\hat\theta_{tr}$ is the training label prior (empirical class frequency of the
training split, §6.1), and $\theta_*$ is the true test prior of the trial.

**Notes that prevent misreading the figures:**

- **Each rejector has its own ordering $\pi$ and its own base predictor.**
  Rows 3–5 therefore do *not* meet rows 1–2 at full coverage; only rows 1 and 2
  do.
- **Row 5's regret is identically zero** by construction — its base predictor
  *is* the regret reference. It is a reference line, not a competitor. Plot it
  in the risk panel and the accuracy panel; omit or grey it in the regret panel.
- **$\hat\theta_{map}$ tie-break.** At $m=0$ the posterior equals $p(\theta)$,
  which is uniform, so *every* $\theta$ ties. Resolve `arg max` ties by
  **lowest index in $\Theta$**; since $\theta_1=\hat\theta_{tr}$ (§7), the MAP
  plugin degenerates gracefully to the train-prior plugin at $m=0$. The same
  rule applies to ties in $h(x,\theta)$ and $H(x,D)$.
- *(Optional, not required.)* A sixth row using $A(x,D)$ as the score would
  complete the $T=A+E$ decomposition. Cheap to add — every quantity is already
  computed.

---

## 4. Metrics

### 4.1 The ranking

All triplets at a fixed $m$ are **pooled into a single ranking** (§6.4). Let
$B$ be the number of pooled triplets, ranked by ascending $u$ under the
permutation $\pi$, with $h$ denoting the rejector's own base predictor. At rank
$k$ the predictor accepts the $k$ least uncertain triplets:

$$coverage(k)=\frac{k}{B},\qquad risk(k)=\frac1k\sum_{i=1}^{k}\ell\big(y_{\pi(i)},h(x_{\pi(i)})\big)$$

$$regret(k)=\frac1k\sum_{i=1}^{k}\Big(\ell\big(y_{\pi(i)},h(x_{\pi(i)})\big)-\ell\big(y_{\pi(i)},h(x_{\pi(i)},\theta_*^{(i)})\big)\Big)$$

where $\theta_*^{(i)}$ is the true prior of the trial that triplet $i$ came
from (it varies across the pool — see §6.4).

**Selective risk** is the error rate on accepted triplets; a good uncertainty
score makes it fall as coverage shrinks.

**Selective regret** measures the same triplets against the plugin predictor
*given the true test prior*. It isolates the cost of not knowing the prior, and
unlike the risk it can be **negative** — the adapted predictor sometimes beats
the true-prior plugin, since both are built on the same imperfect calibrated
posterior $\hat p_{tr}(y\mid x)$.

### 4.2 Summary scalars

$$\text{AuRC}=\frac{1}{B}\sum_{k=1}^{B}risk(k),\qquad \text{AuRegC}=\frac{1}{B}\sum_{k=1}^{B}regret(k)$$

**Regret at coverage $c$** is the selective regret at a *fixed* coverage
budget:

$$\text{Reg}@c = regret\big(\lceil cB\rceil\big),\qquad c=0.8\ \text{by default}$$

*(This replaces the previous phrasing "the minimal regret at which the coverage
is within a budget", which inverted the two quantities. With pooling, $B\ge
2000$, so any $c$ is attainable to within $1/B$.)*

---

## 5. Numerically stable computation

Everything is computed in log space. For a query or batch element $x$ and
$\theta\in\Theta$:

$$\log w(x,\theta)=\operatorname{logsumexp}_{y}\Big(\log\theta_y-\log p_{tr}(y)+\log\hat p_{tr}(y\mid x)\Big)$$

Since $p_{te}(x\mid\theta)=p_{tr}(x)\,w(x,\theta)$ and every expression in §2 is
a ratio with $m+1$ factors of $p_{tr}(\cdot)$ above and below, **all $p_{tr}$
factors cancel** and $w$ may be used in place of $p_{te}$ everywhere.

**Leave-one-out trick (this is where the claimed efficiency comes from).** For
one test set $D'$ of size $m+1$, precompute the matrix
$r_{i\theta}=\log w(x_i,\theta)$ once, and its column sums
$S_\theta=\sum_{i=1}^{m+1}r_{i\theta}$. Then the log-likelihood of the
adaptation set for triplet $i$ is a subtraction:

$$\log p_{te}(D^{(i)}\mid\theta)\;-\;\text{const}\;=\;S_\theta-r_{i\theta}$$

Cost is $O\big((m+1)\,C\,Y\big)$ for the precompute plus $O\big((m+1)\,C\big)$
for all $m+1$ triplets — **not** $O(m^2 C)$.

Normalise $p(\theta\mid x,D)$ and $p(y\mid x,D)$ with a single `logsumexp` over
$\theta$ (and over $y$). Never exponentiate a raw product of $m$ densities:
for $m=500$ it underflows to zero in float64.

---

## 6. Evaluation protocol

### 6.1 Splits and the base model


| Split | Fraction | Use |
| --- | --- | --- |
| development |  | ↓ |
| ↳ training | 0.80 of development | fit the network weights |
| ↳ validation | 0.20 of development | model selection, calibration, per-class error rates |
| evaluation |  | generate all test sets $D'$ |

Datasets with defined train/test splits use the whole test subset for evaluation; datasets with train/val/test splits use **val + test merged** for evaluation only. Training split is used for development, i.e. it is randomly split into training and validation part. 

All splits are **stratified by class** and drawn with a fixed seed.

**Training must not reweight classes.** No class-balanced sampler, no
class-weighted loss. `assert` this in the training script, because
$\hat\theta_{tr}$ is defined as the empirical class frequency of the training
split and the whole re-weighting $\theta_y/p_{tr}(y)$ is only valid if the
network was actually fit under that prior.

**Calibration.** Bias-corrected temperature scaling with a **per-class
temperature** (BCTS): the logits are mapped by
$$\hat p_{tr}(y = k \mid x) \;\propto\; \exp\!\big(z_k(x)/T_k + b_k\big),$$
with all $2Y$ parameters fit by minimising validation NLL (LBFGS), over
$\log(1/T_k)$ so each $T_k > 0$, and $b$ reported mean-centred since softmax is
invariant to a constant shift. The per-class bias is what a single shared
temperature cannot supply: label-shift correction re-weights each class
separately, so a per-class calibration error does not cancel. The training
script emits the calibrated posterior $\hat p_{tr}(y\mid x)$, used throughout as
a proxy for the unknown $p_{tr}(y\mid x)$, and **reports validation NLL and ECE
(15 equal-mass bins) before and after calibration**. Label-shift correction is
highly sensitive to calibration, so these two numbers belong in the paper.
*(Scalar temperature scaling, $T_k \equiv T$, is kept as the ablation; BCTS is
the default.)*

**Order of operations.** The $\Theta$ file (§7) is an **output of the training
run**, not a hand-authored config: two of its entries depend on the training
class frequencies and on validation per-class error rates.

```
train → calibrate → per-class validation errors → emit priors file → evaluate
```

### 6.2 Adaptation-size grid

$$\mathcal M=\{0,\,1,\,2,\,5,\,10,\,20,\,50,\,100,\,200,\,500\},\qquad m_{\max}=500$$

Log-spaced on purpose: the posterior over $\Theta$ concentrates exponentially
in $m$, so the interesting regime is small $m$ and a linear grid would spend
its budget where nothing changes. $m=0$ (empty $D$, posterior $=p(\theta)$) is
the free anchor point.

`assert` $m_{\max}\le\lvert\text{eval set}\rvert/10$ (see §6.3).

### 6.3 Generating one test set

For each $m\in\mathcal M$ and each trial $j=1,\ldots,N(m)$:

1. Sample $\theta_*\sim p(\theta)$.
2. Sample $m+1$ labeled examples from the evaluation set **with replacement**,
   with class frequencies following $\theta_*$, giving
   $D'=\{(x_1,y_1),\ldots,(x_{m+1},y_{m+1})\}$.
3. Form $m+1$ triplets $(x,D,y)$: example $i$ supplies the query $x=x_i$ and
   target $y=y_i$, and the inputs of the **remaining $m$ examples** form the
   adaptation set $D^{(i)}$.

Sampling with replacement is what makes $D'$ exactly i.i.d. from
$p_{te}(\cdot\mid\theta_*)$ and removes the pool-exhaustion problem for spiked
priors on rare classes.

> **Known residual leak.** With replacement, the query $x_i$ may reappear as a
> duplicate inside its own $D^{(i)}$ (leave-one-out removes only index $i$).
> The expected number of duplicates of any given example is
> $\approx m/\lvert\text{eval}_{y}\rvert$, which the assert in §6.2 keeps below
> ~5%. **Log the realised duplicate rate per $m$** and report it once; do not
> silently ignore it.

### 6.4 Pooling, trial counts, and confidence intervals

Curves are computed **per $m$, pooled across trials** — not per trial and then
averaged. (A per-trial curve at $m=0$ has one point and carries no ranking
information at all.)

**Constant curve budget.** The pooled count $N(m)\times(m+1)$ would otherwise
grow 500× across the x-axis, so confidence intervals would narrow left-to-right
and panels 1–3 would show apparent improvement that is really just shrinking
error bars. Fix the budget instead:

$$B=2000,\qquad N(m)=\mathrm{clip}\!\left(\left\lceil \frac{B}{m+1}\right\rceil,\;N_{\min},\;N_{\max}\right),\quad N_{\min}=50,\;N_{\max}=2000$$

Pool all $N(m)\,(m+1)$ triplets; if that exceeds $B$, subsample exactly $B$ of
them, **stratified so each trial contributes the same number**. Every curve
then has exactly $B$ points, and every $m$ has at least $N_{\min}$ trials to
bootstrap over.

**Confidence intervals.** Bootstrap with the **trial as the resampling unit**:
resample $N(m)$ trials with replacement, re-pool, re-subsample, recompute the
curve and its scalars. 1000 replicates, 95% percentile intervals. Bootstrapping
the $B$ triplets directly is wrong — triplets within a trial share $m-1$
adaptation points and are strongly dependent.

**Stratification by $\theta_*$.** Pooling mixes trials with different $\theta_*$.
Since epistemic uncertainty is systematically near zero when $\theta_*$ happens
to equal $\hat\theta_{tr}$, an unstratified pool lets *whole trials* — rather
than hard-vs-easy examples — populate the low-coverage region.

- **Main panels: shifted trials only**, i.e. conditioned on $\theta_*\neq\theta_1$.
- **Supplementary panel:** the same curves broken down per $\theta_*\in\Theta$,
  plus the unstratified marginal.

State the convention in every caption.

### 6.5 Figures

For each dataset, four panels, all against adaptation set size $m$ (log x-axis),
each with bootstrap 95% bands:

1. **AuRC** vs $m$ — rejectors 1–5.
2. **AuRegC** vs $m$ — rejectors 1–4 (rejector 5 is identically 0).
3. **Regret at coverage $c=0.8$** vs $m$ — rejectors 1–4.
4. **Accuracy at full coverage** vs $m$ — **four** predictors: train-prior
   plugin, true-prior plugin, MAP plugin, Bayesian learned-prior. The
   train-prior and true-prior lines are **flat in $m$** (they never see $D$)
   and act as floor and ceiling.

*(Panel 4 is $1-risk(B)$ of the corresponding rejector, i.e. the full-coverage
endpoint of panel 1. It is kept because the floor/ceiling reading is much
easier there — but note the redundancy in the caption rather than presenting it
as independent evidence.)*

---

## 7. Datasets and the prior set $\Theta$

| Dataset | key | shape | classes $Y$ | source |
|---------|-----|-------|---------|--------|
| Fashion-MNIST | `fashion_mnist` | 28×28 grayscale | 10 | Zalando IDX files |
| CIFAR-10 | `cifar10` | 32×32 RGB | 10 | fast.ai PNG-folder mirror |
| CIFAR-100 | `cifar100` | 32×32 RGB | 100 | fast.ai PNG-folder mirror (nested by superclass) |
| DermaMNIST | `dermamnist` | 28×28 RGB | 7 | MedMNIST v2 `.npz` (Zenodo) |
| BloodMNIST | `bloodmnist` | 28×28 RGB | 8 | MedMNIST v2 `.npz` (Zenodo) |
| TissueMNIST | `tissuemnist` | 28×28 grayscale | 8 | MedMNIST v2 `.npz` (Zenodo) |
| OrganAMNIST | `organamnist` | 28×28 grayscale | 11 | MedMNIST v2 `.npz` (Zenodo) |
| OrganSMNIST | `organsmnist` | 28×28 grayscale | 11 | MedMNIST v2 `.npz` (Zenodo) |

$\Theta$ is read from a per-dataset text file, **generated** by the training run
(§6.1). The default generator produces $C=2+1+S$ priors, where
$S=\min\big(5,\lceil Y/2\rceil\big)$:

**$\theta_1$ — training prior.** $\theta_1=\hat\theta_{tr}$, the empirical class
frequency of the training split. *(Index 1 by convention, so that the
lowest-index tie-break in §3 degenerates to the train-prior plugin.)*

**$\theta_2$ — uniform prior.** $\theta_{2,y}=1/Y$.

**$\theta_3$ — hard-class spike.** Let $c_1,c_2$ be the two classes with the
highest validation error rate. Set
$$\theta_{3,c_1}=\theta_{3,c_2}=\tau,\qquad \theta_{3,y}=(1-2\tau)\,\frac{\hat\theta_{tr,y}}{\sum_{y'\notin\{c_1,c_2\}}\hat\theta_{tr,y'}}\ \ \text{otherwise}$$
with $\tau=0.2$ by default — so **$0.2$ each, $0.4$ total**, and the remaining
$0.6$ is redistributed **in proportion to the training prior**. Requires
$Y\ge 3$ and $2\tau<1$.

**$\theta_4,\ldots,\theta_{3+S}$ — frequent-class doubling, one prior per class.**
For each of the $S$ **most frequent** classes $c$ (a *proper subset*, since
$S\le\lceil Y/2\rceil<Y$ for $Y\ge 2$):
$$\theta_c = 2\,\hat\theta_{tr,c},\qquad \theta_y = \hat\theta_{tr,y}\,\frac{1-2\hat\theta_{tr,c}}{1-\hat\theta_{tr,c}}\ \ (y\neq c)$$
Skip any $c$ with $2\hat\theta_{tr,c}\ge 1$.

> **Why a proper subset.** The previous rule doubled the top
> $\min(10,Y)$ classes *simultaneously*. For $Y\le 10$ that is **every** class,
> and doubling every class then renormalising returns the training prior
> exactly — a duplicate of $\theta_1$ on Fashion-MNIST, CIFAR-10, DermaMNIST,
> BloodMNIST and TissueMNIST (5 of 8 datasets). Looping over a proper subset
> fixes this and makes $C$ unambiguous.

**Guards, asserted at generation time:**

- Every $\theta$ sums to 1 and has all entries $>0$.
- **No two elements of $\Theta$ coincide:** $\mathrm{TV}(\theta_a,\theta_b)\ge10^{-2}$
  for all $a\neq b$. Drop duplicates and re-index. *(This also catches the case
  where the training set happens to be balanced, which would make
  $\theta_1=\theta_2$.)*
- The file records, for each $\theta$, its
  $\mathrm{TV}(\theta,\hat\theta_{tr})=\tfrac12\sum_y\lvert\theta_y-\hat\theta_{tr,y}\rvert$,
  so the shift magnitude is legible per dataset.

**Parameter prior.** $p(\theta)=1/C$, uniform over $\Theta$ — for both the
model *and* the generation of $\theta_*$.

> **On comparability across datasets.** A $\tau=0.2$ spike on 2 of 7 classes
> and a $\times2$ nudge on 1 of 100 classes are very different shifts. The TV
> column makes this visible. If cross-dataset comparison becomes a claim in the
> paper, replace the constructions above with priors built at **target TV
> distances** (e.g. $\mathrm{TV}\in\{0.1,0.25,0.5\}$) instead of fixed $\tau$
> and $\times2$.

---

## Appendix A — Assumptions and limitations to state in the paper

1. **The setup is well-specified by construction.** $\Theta$ is finite,
   $\theta_*\in\Theta$, and the model is handed both $\Theta$ and $p(\theta)$.
   This is the best case for the Bayesian method and should be presented as
   such. **Recommended second arm:** a misspecified mode in which $\theta_*$ is
   drawn from $\mathrm{Dir}(s\,\hat\theta_{tr})$ (so $\theta_*\notin\Theta$
   almost surely) while the model still uses $\Theta$. Without it, the
   epistemic-calibration claim is not falsifiable.
2. **$\hat p_{tr}(y\mid x)$ is a proxy** for the true training posterior. This
   is why selective regret can go negative, and why ECE is reported.
3. **The validation split is used three ways** — model selection, calibration,
   and the per-class error rates that define $\theta_3$. Mild, disclosed.
4. **Triplets within one $D'$ are dependent** (they share $m-1$ adaptation
   points); this is why the bootstrap unit is the trial.
5. **$\theta_*$ and $p(\theta)$ come from the same distribution.** If you want
   to measure sensitivity to a mismatched parameter prior, that is a separate
   knob from (1).

