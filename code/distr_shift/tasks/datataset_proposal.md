# Real-data proposal: showing the advantage of epistemic reject-option

Goal: find a real-life dataset where the epistemic reject-option predictor
significantly outperforms the (total-uncertainty) Bayesian reject-option
predictor on the area under the regret-coverage curve, mirroring what
[configs/epistemic_showcase.json](../configs/epistemic_showcase.json)
demonstrates synthetically.

## What the dataset must offer

The synthetic experiment isolated two conditions that must hold
*simultaneously*, on the same data, for the two rankings to diverge:

1. **A confusable class group** — two (or more) classes whose class-conditionals
   nearly coincide as seen by the base model, so the prior split within the
   group is weakly identifiable from unlabeled data (`ident_ratio > 3` for
   them, per `MCMCResult.identifiability_warning`). This is where epistemic
   uncertainty and regret live.
2. **A plausible, strongly asymmetric prior shift within that group** — the
   true split must be very uneven (e.g. 0.05 / 0.35) while the training data is
   more balanced, so adapting the prior actually matters and getting it wrong
   is costly.
3. **Plenty of regret-free hard examples elsewhere** — the decoys. In
   synthetic data these had to be engineered as a three-way overlap; real data
   is kinder here, since naturally hard/noisy examples of well-identified
   classes have high conditional risk and zero regret. This decoy pool is what
   splits the two rankings apart: total uncertainty spends its rejection
   budget on the decoys, epistemic uncertainty ignores them.

## Ranked suggestions

### 1. Medical prevalence shift — DermaMNIST or BloodMNIST (MedMNIST)

The strongest real-life story. Label shift *is* the canonical epidemiological
model (symptoms-given-disease `p(x|y)` stable, prevalence `p(y)` varies across
clinics/seasons), and the reject option has a native interpretation: defer to
a specialist.

- **DermaMNIST**: melanoma vs. benign nevus are notoriously near-identical
  visually — a weakly identifiable pair — and their ratio genuinely varies
  enormously between a screening population and a referral clinic. The other
  five lesion classes supply aleatoric decoys.
- **BloodMNIST**: band vs. segmented neutrophils (an infection shifts this
  split dramatically), with other cell types as decoys.

This is the option to lead with in a write-up.

### 2. CIFAR-10 with a resampled test prior

The recognizable vision benchmark. Cat/dog is the classic weakly identifiable
pair under a moderate base model. Design the test prior as, say, 5% cat / 35%
dog with the rest spread over the remaining classes, several of which
(deer/horse/bird) supply naturally hard, regret-free decoy examples. Use
pretrained frozen features + a logistic head as the base model — keeps the
pipeline close to the current one and keeps the posterior reasonably
calibrated.

### 3. Fashion-MNIST — cheap first step

The {t-shirt, pullover, shirt, coat} group is heavily confusable; pick shirt
vs. t-shirt as the asymmetric pair. Runs in minutes with the existing
logistic-regression base model, so it's the right place to debug the pipeline
before touching images that need feature extractors. Risk: the confusable
group and the decoy pool largely coincide here, which is the `model1.json`
failure mode (regret rankings tie) — check the identifiability diagnostic
before investing further.

### 4. iNaturalist / CUB birds — ambitious

Near-identical species whose relative abundance shifts with region and
season — the most *genuinely* naturally occurring label shift, with no
resampling story needed. But it's long-tailed, needs deep features, and
covariate shift creeps in across regions, which violates the
`p_tr(x|y) = p_te(x|y)` invariance assumption. Save for a follow-up.

## Protocol notes

- **The shift will be simulated by resampling the test split to a target
  prior** (standard protocol, e.g. Lipton et al. 2018 and follow-ups). This is
  unavoidable — labels are needed to compute regret anyway — and it also gives
  control over the asymmetric split within the confusable pair. The realism
  comes from the features and the class confusability, not from the shift
  itself.
- **Keep `n_test` small enough that the split posterior stays wide.** The
  synthetic finding transfers directly: with a large unlabeled set, MCMC
  latches onto small spurious differences in the fitted posteriors and the
  epistemic signal collapses. A few hundred unlabeled examples is the
  interesting regime — and also the realistic one (a new clinic does not hand
  you 50k unlabeled samples).
- **Calibrate the base model** (temperature scaling on a held-out set). A
  miscalibrated deep posterior plays the same role as an under-trained
  logistic-regression base model did in the synthetic experiment — it
  manufactures fake identifiability for the confusable pair and quietly kills
  the effect. This is the real-data analog of the `--m-train 10000` fix used
  in `configs/epistemic_showcase.json`.
- **Let the identifiability diagnostic pick the pair, not intuition.** Before
  committing to a dataset: fit the base model, resample a shifted test set,
  run the MCMC, and check `ident_ratio` — the intended pair should read `>3`,
  everything else `~1`, with the identifiability warning firing consistently.
  If it doesn't fire, the regret curves will tie no matter how confusable the
  classes look to a human.

## Suggested next step

Validate the criteria cheaply on Fashion-MNIST first — it drops into the
existing `run_reject_option_experiment.py` structure with minimal changes (a
data loader plus the resampling step) — before committing to the medical
dataset for the final write-up.
