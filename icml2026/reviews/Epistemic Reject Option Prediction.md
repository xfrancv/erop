---
title: "Epistemic Reject Option Prediction"
source: "https://openreview.net/forum?id=JJZ2I8qQeO&noteId=UAmcyDvMCl&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2026%2FConference%2FAuthors%23your-submissions)"
author:
  - "[[Vojtech Franc]]"
  - "[[Jakub Paplhám]]"
published:
created: 2026-05-26
description: "In high-stakes applications, predictive models must not only produce accurate predictions but also quantify and communicate their uncertainty. Reject-option prediction addresses this by allowing the model to abstain when prediction uncertainty is high. Traditional reject-option approaches focus solely on aleatoric uncertainty, an assumption valid only when large training data makes the epistemic uncertainty negligible. However, in many practical scenarios, limited data makes this assumption unrealistic. This paper introduces the epistemic reject-option predictor, which abstains in regions of high epistemic uncertainty caused by insufficient data. Building on Bayesian learning, we redefine the optimal predictor as the one that minimizes expected regret -- the performance gap between the learned model and the Bayes-optimal predictor with full knowledge of the data distribution. The model abstains when the regret for a given input exceeds a specified rejection cost. To our knowledge, this is the first principled framework that enables learning predictors capable of identifying inputs for which the available training data is insufficient to support well-informed predictions."
tags:
  - "clippings"
---
### Vojtech Franc, Jakub Paplhám

Submitted to ICML 2026 Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors [Revisions](https://openreview.net/revisions?id=JJZ2I8qQeO) [BibTeX](#) [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Verify Author List:** I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.

**TL;DR:** The paper introduces a reject-option predictor that abstains in regions of high epistemic uncertainty caused by limited data by rejecting inputs whose expected regret relative to a Bayes-optimal predictor exceeds a specified cost.

**Abstract:**

In high-stakes applications, predictive models must not only produce accurate predictions but also quantify and communicate their uncertainty. Reject-option prediction addresses this by allowing the model to abstain when prediction uncertainty is high. Traditional reject-option approaches focus solely on aleatoric uncertainty, an assumption valid only when large training data makes the epistemic uncertainty negligible. However, in many practical scenarios, limited data makes this assumption unrealistic. This paper introduces the epistemic reject-option predictor, which abstains in regions of high epistemic uncertainty caused by insufficient data. Building on Bayesian learning, we redefine the optimal predictor as the one that minimizes expected regret -- the performance gap between the learned model and the Bayes-optimal predictor with full knowledge of the data distribution. The model abstains when the regret for a given input exceeds a specified rejection cost. To our knowledge, this is the first principled framework that enables learning predictors capable of identifying inputs for which the available training data is insufficient to support well-informed predictions.

**Supplementary Material:** [zip](https://openreview.net/attachment?id=JJZ2I8qQeO&name=supplementary_material "Download Supplementary Material")

**Primary Area:** Probabilistic Methods->Bayesian Models and Methods

**Keywords:** reject-option prediction, epistemic uncertainty, Bayesian learning

**Ethics Agreement:** I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.

**LLM Policy:** This submission allows Policy B.

**Proceedings-only Option:** If this paper is accepted, the authors tentatively plan to present it in person at the conference (as a poster and, if selected, as an oral).

**Reciprocal Reviewing Status:** This submission is NOT exempt from the Reciprocal Reviewing requirement. (We expect most submissions to fall in this category.)

**Reciprocal Reviewing Author:** [Vojtech Franc](https://openreview.net/profile?id=~Vojtech_Franc1)

**Ready For LLM Feedback:** The submitted PDF is ready for LLM feedback.

**LLM Feedback Request:** [Vojtech Franc](https://openreview.net/profile?id=~Vojtech_Franc1)

**Submission Number:** 22311

Add:

#### Paper Decision

**Decision:** Reject

**Comment:**

This paper proposes a novel method for reject-option prediction. Unlike existing approaches that rely on either aleatoric uncertainty or total predictive uncertainty, the proposed method utilizes only epistemic uncertainty. To quantify epistemic uncertainty, this paper proposes using the expected conditional regret as a penalty on the prediction, defined as the expected performance gap between the predictor and the Bayesian optimal predictor. This paper also demonstrates empirical comparisons between the proposed epistemic reject-option method and existing approaches based on aleatoric or total uncertainty, using synthetic data where the ground-truth data-generating process is known.

This paper is borderline and has received mixed reviews. Reject option prediction and uncertainty-aware decision-making are clearly important topics, and this paper is technically well grounded and provides a very elegant decision-theoretic formulation of epistemic uncertainty-focused reject option prediction. However, this paper has some limitations:

- As Reviewer ARiq has pointed out, the contribution seems to be mainly conceptual, especially for researchers who are familiar with the Bayesian decision theory. Though elegant, the derivations in Section 3 are relatively straightforward. This limits the novelty of this paper.
- More importantly, the empirical evaluations in this paper are limited to synthetic data. I think more experiment results, especially those on real datasets, are needed to further strengthen the paper.

Thus, unfortunately, I have to recommend rejecting this paper. This is a tough decision.

#### Official Review of Submission22311 by Reviewer 2imE

**Summary:**

This paper studies reject option prediction under epistemic uncertainty through the lens of Bayesian models. They provided mathematical frameworks to study when a predictive model should abstain when the available training data is not enough to support reliable predictions.

They put emphasis on rejection based on epistemic uncertainty, in contrast to prevoius approaches that focus soley on (estimation of) aleatoric uncertainties.

Their approach reformulated the standard Bayesian reject option prediction using expected regret, and subsequently allow them to decompose the total uncertainty into epistemic and aleatoric components.

## Soundness

(Stength) The paper is technically well grounded and provides a very nice decision-theoretic formulation of epistemic uncertainty-focused reject option prediction. The main theoretical insights characterises the optimal epistemic reject option predictor as the minimiser of some intuitively understandable expected regret. Theoretical claims are supported with formal statements and proofs, and the empirical evaluation in controlled synthetic settings provides evidence consistent with the theoretical predictions.

(Weakeness) Perhaps there could be more empirical evaluations because right now things in the main text are based on synthetic settings. While I understand computing regret requires the full data population, but perhaps would also be nice to perform some empirical experiments where you modify the rejection cost and look at the prediction performance at each cost, to give a more holistic view of how your set up differs across different cost setting.

## Presentation

(Strength) The paper is clearly written, easy to understand, and well structured. Various levels of concepts are introduced carefully and progressively.

(Weakness) The strength in presentation can become a weakness too, given how conference papers have limited pages, quite a lot of spaces are used to explain the low level ideas.. which leaves the innovation part a bit short and under-explained. If I were you, I would leave some of the basics to the appendix. But this is only a personal choice really.

## Signifiance

Reject option prediction and uncertainty-aware decision-making are important topics, especially in high-stakes applications where abstention mechanisms can improve reliability:)

## Originality

The main novelty lies in formulating reject option prediction explicitly in terms of expected regret relative to the Bayes-optimal predictor and deriving a corresponding epistemic rejection rule. It is, to me, a new conceptual framework well positioned in the growing work of epistemic uncertainty centric machine learning community.

**Soundness:** 3: good

**Presentation:** 4: excellent

**Significance:** 3: good

**Originality:** 3: good

**Key Questions For Authors:**

The appendix briefly discusses the case of model misspecification, where epistemic uncertainty vanishes while regret remains non-zero due to approximation error. Could the authors elaborate on how the proposed epistemic reject option predictor should behave in such settings? In particular, are there principled extensions that could account for approximation error in addition to estimation error?

**Limitations:**

Yes.

**Overall Recommendation:** 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Compliance With LLM Reviewing Policy:** Affirmed.

**Code Of Conduct Acknowledgement:** Affirmed.

**Final Justification:**

I didn't have too many questions in the first place, and the author's reply on model mispecification cases makes sense to me. I will keep my scores as it is.

#### Rebuttal by Authors

**Rebuttal:**

**Reviewer:** The appendix briefly discusses the case of model misspecification, where epistemic uncertainty vanishes while regret remains non-zero due to approximation error. Could the authors elaborate on how the proposed epistemic reject option predictor should behave in such settings? In particular, are there principled extensions that could account for approximation error in addition to estimation error?

**Authors:** Our current framework assumes a well-specified model, i.e., that the data-generating distribution lies within the model class. We do not yet have theoretical results for the misspecified setting, where the approximation error remains non-zero. This issue is not specific to our approach: standard Bayesian inference, and more generally generative learning approaches, face the same difficulty. At present, we do not have a principled extension that accounts for approximation error, but we are actively investigating this setting empirically in order to understand the sensitivity of the approach to different forms of model misspecification.

##### Replying to Rebuttal by Authors

#### Rebuttal Acknowledgement by Reviewer 2imE

**Acknowledgement:** (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.

**Reasons:**

I didn't have too many questions in the first place, and the author's reply on model mispecification cases makes sense to me. I will keep my scores as it is.

#### Official Review of Submission22311 by Reviewer EcZt

**Summary:**

This paper proposes a novel method for reject-option prediction, in which a model can abstain from making a prediction when uncertainty is deemed too high. Unlike existing approaches that rely on either aleatoric uncertainty or total predictive uncertainty, the proposed method utilises only epistemic uncertainty. The idea is to enable the model to abstain when it lacks sufficient evidence to make a reliable prediction, rather than in situations where the predictive variance is inherently high.

To quantify epistemic uncertainty, the paper proposes using the expected conditional regret as a penalty on the prediction, defined as the expected performance gap between the predictor and the Bayes-optimal predictor. Finally, the paper presents empirical comparisons between the proposed epistemic reject-option method and existing approaches based on aleatoric or total uncertainty, using synthetic data where the ground-truth data-generating process is known.

**Strengths And Weaknesses:**

**Strengths**

- The paper is clearly written and easy to follow. The overall presentation is reasonably well structured.
- The topic is timely and relevant to the broader ML community. In particular, it highlights the importance of clearly distinguishing between aleatoric and epistemic uncertainty.

**Weaknesses**

- The justification for an epistemic-only reject-option prediction remains unclear (see discussion below).
- The empirical evaluation relies on the unrealistic assumption that the ground-truth data-generating process is known, which limits the practical significance of the proposed method (see discussion below).

**Soundness:** 3: good

**Presentation:** 3: good

**Significance:** 2: fair

**Originality:** 3: good

**Key Questions For Authors:**

- Can the authors provide examples of applications where epistemic-only reject-option prediction is necessary and can actually be implemented, i.e., when the aleatoric uncertainty is completely absent?
- How does the reject-option prediction relate to the **learning-to-defer** literature (see, e.g., \[1\] and references therein)? The problem formulation of this problem in Section 2 looks similar to that of the learning-to-defer problem.
- How does one set a rejection cost in practice? Any theory to support the right choice of this parameter?
- What if we have no access to the ground-truth data-generating process? How realistic is it to reject the prediction based solely on the epistemic uncertainty, given that in most practical applications one must often deal with both aleatoric and epistemic uncertainty?

\[1\] **Calibrated Learning to Defer with One-vs-All Classifiers**. Proceedings of the 39th International Conference on Machine Learning, PMLR 162:22184-22202, 2022.

**Limitations:**

From my point of view, there are *two* major limitations:

1. **Weak justification on epistemic-only reject option**: It remains unclear why an *epistemic-only* reject-option prediction is necessary, especially when the ground-truth data-generating process is not directly accessible. I can understand the motivation that rejecting based solely on the epistemic uncertainty focuses on situations where the predictor lacks sufficient evidence, rather than where the predictive variance is intrinsically high. However, in practice, it seems that both sources of uncertainty may be relevant. For example, when the predictive variance is high, a model might reasonably abstain from making a prediction in order to defer the decision to a human expert. In practice, a model is typically subject to both types of risk: one arising from high predictive variance and the other from a lack of sufficient evidence. It therefore remains unclear in which realistic settings an epistemic-only reject option would be necessary, and whether it can be meaningfully implemented when only finite data are available.
2. **Lack of thorough theoretical and empirical analyses**: The authors devote a substantial portion of the paper to preliminaries (Section 2), which spans nearly three pages. In contrast, the main contribution presented in Section 3 occupies less than a page, with only one theoretical result. This imbalance seems to reflect the paper's contribution relative to the existing literature. The empirical evaluation requires knowledge of the ground-truth data-generating process and assumes that all predictors admit closed-form solutions, making the setting highly unrealistic. Moreover, the advantages of the epistemic reject-option prediction over its Bayesian counterpart remain unclear. For instance, the results in Figure 3 suggest that the epistemic and Bayesian approaches are largely comparable in terms of both the area under the regret–coverage curve and the area under the risk–coverage curve, particularly in the small-sample regime, precisely the regime in which the paper claims the epistemic reject option should matter most.

Taken together, these two limitations diminish the significance of the proposed method.

**Overall Recommendation:** 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Compliance With LLM Reviewing Policy:** Affirmed.

**Code Of Conduct Acknowledgement:** Affirmed.

**Final Justification:**

The rebuttal partially addresses my concerns in general. I chose to maintain my score at **weak reject** because I still have some doubts about the justification and the relative contributions of this work in comparison to the existing literature, as noted in my review.

#### Rebuttal by Authors

**Rebuttal:**

**Q1:** Can the authors provide examples of applications where epistemic-only reject-option prediction is necessary and can actually be implemented, i.e., when the aleatoric uncertainty is completely absent?

**A1:** Our framework does not require aleatoric uncertainty to be absent; such an assumption would indeed be overly restrictive. The epistemic reject option predictor is useful whenever one wants to determine whether the available training data are sufficient to guarantee nearly optimal prediction for a given input, which we believe is valuable in many applications.

As a prototypical example, consider **tumor grading from brain MRI**. If the epistemic reject option predictor (EROP) abstains on a given MRI, this indicates that the input is not sufficiently covered by the current training data, and that annotating such cases could be valuable for extending the training set. If, on the other hand, EROP outputs a prediction (e.g., low-grade versus high-grade glioma), it guarantees that, within the considered statistical framework, no better prediction can be obtained from that MRI alone.

Importantly, this guarantee does not require aleatoric uncertainty to be low. The MRI may still be noisy or ambiguous, for example because of limited image quality. In such cases, aleatoric uncertainty may remain high, yet EROP still certifies that no predictor (neither human expert) using only the same MRI can substantially improve upon the output. Improving the prediction would then require additional information, such as acquiring further MRI sequences or using another imaging modality. By contrast, if a standard Bayesian reject-option predictor abstains, it is generally unclear whether this is due to insufficient training data or to irreducible ambiguity in the input. As a result, when a Bayesian reject-option predictor abstains, the appropriate course of action is often unclear.

---

**Q2:** How does the reject-option prediction relate to the learning-to-defer literature (see, e.g., \[1\] and references therein)? The problem formulation of this problem in Section 2 looks similar to that of the learning-to-defer problem.

**A2:** In reject-option prediction, including our framework, the fallback after rejection is typically abstracted as either a fixed rejection cost or a coverage constraint. In learning to defer, by contrast, the fallback decision-maker (e.g., a human expert) is modeled explicitly. The goal is then to allocate cases between the predictor and the expert so as to optimize the performance of the combined system. Thus, while the formulations are related, the emphasis is different: reject-option prediction focuses on when the model should abstain, whereas learning to defer focuses on when the model or the expert should handle a given case.

---

**Q3:** How does one set a rejection cost in practice? Any theory to support the right choice of this parameter?

**A3:** This issue is common to all reject-option predictors, including aleatoric, Bayesian, and epistemic variants. In practice, there are two main possibilities. First, in some applications both the prediction loss and the rejection cost can be expressed in the same physical or operational units, which allows the rejection cost to be specified directly. Second, one can train predictors for a range of rejection costs and then select the most suitable operating point based on the resulting risk-coverage tradeoff (or regret-coverage tradeoff in our setting).

---

**Q4:** What if we have no access to the ground-truth data-generating process? How realistic is it to reject the prediction based solely on the epistemic uncertainty, given that in most practical applications one must often deal with both aleatoric and epistemic uncertainty?

**A4:** As noted above, our framework does not assume zero aleatoric uncertainty. It also does not require knowledge of the true data-generating process. Indeed, if the true data-generating process were known, there would be no need for learning, since the Bayes predictor could be constructed directly.

Please note that we use the ground-truth data-generating process only for the purpose of fair evaluation of the reject-option predictors in our experiments; it is not required in practical deployment.

Our only assumption is that the true distribution belongs to the considered model class. This is the same standard well-specifiedness assumption that underlies Bayesian learning and, more generally, other generative learning approaches.

##### Replying to Rebuttal by Authors

#### Rebuttal Acknowledgement by Reviewer EcZt

**Acknowledgement:** (b) Partially resolved - I have follow-up questions for the authors.

**Reasons:**

I thank the authors for answering some of my questions. Some of my concerns have been addressed, except **Q1** that may require a follow-up discussion. My reservation about the paper's contribution relative to the existing literature still remain.

#### Official Review of Submission22311 by Reviewer DNAi

**Summary:**

The authors propose a reject-option predictor that abstains based on epistemic uncertainty alone, rather than total or aleatoric uncertainty. The idea is to replace the standard expected loss of a Bayesian learner with an expected regret objective, where the regret measures the performance gap relative to the Bayes-optimal predictor. They also show how different epistemic uncertainty measures can be found as instantiations of their framework under specific loss functions. They provide simple experiments with synthetic data to validate their approach.

**Strengths And Weaknesses:**

Overall, I think this is a well-written and technically sound paper. The presentation can be followed along nicely and the figures distinguish well between aleatoric, total and epistemic uncertainty. It is also satisfying how different epistemic uncertainty measures follow from minimizing the regret and plugging in different loss function.

However, I am unsure why we would want to reject based on epistemic uncertainty alone rather than total uncertainty. If I am concerned about prediction errors, I may not necessarily care whether the uncertainty is aleatoric or epistemic.

The experiments are also quite limited. They use a synthetic dataset and the results (as the authors note) simply show what the theory already guarantees. It would be interesting to see more realistic evaluations that also deal with posterior inference, misspecification and choosing a good prior.

The related work section is narrow and I would like to see more discussion of other lines of work on uncertainty quantification and reject prediction. For example, conformal prediction and the learning to defer literature.

**Soundness:** 3: good

**Presentation:** 3: good

**Significance:** 2: fair

**Originality:** 2: fair

**Key Questions For Authors:**

- Why the explicit focus on epistemic rather than total uncertainty? Do you have a specific setting in mind?
- How do you see this framework situated against non-Bayesian methods for rejection, for example conformal prediction?
- How sensitive is the reject-predictor to the prior choice?

**Limitations:**

yes

**Overall Recommendation:** 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Compliance With LLM Reviewing Policy:** Affirmed.

**Code Of Conduct Acknowledgement:** Affirmed.

**Final Justification:**

The authors rebuttal was well-structured and addressed my main concerns, which were primarily conceptual. I found their responses convincing. I am only concered about the limited scope, e.g. the evaluations are limited to synthetic settings, and the paper does not consider misspecification. However, I believe the merits outweigh these limitations. The paper maintains a clear focus, does not overclaim, and presents its contributions clearly and honestly.

#### Rebuttal by Authors

**Rebuttal:**

**Q1:** Why the explicit focus on epistemic rather than total uncertainty? Do you have a specific setting in mind?

**A1:** To clarify why we base the reject option solely on epistemic uncertainty, and how this differs from a Bayesian reject-option predictor based on total uncertainty, let us describe a prototypical application of the epistemic reject-option predictor (EROP).

Consider **tumor grading from brain MRI**. If EROP abstains on a given MRI, this indicates that the input is not sufficiently covered by the current training data, and that annotating such cases could be valuable for extending the training set. If, on the other hand, EROP outputs a prediction (e.g., low-grade versus high-grade glioma), it guarantees that, within the considered statistical framework, no better prediction can be obtained from that MRI alone.

Importantly, this guarantee does not require aleatoric uncertainty to be low. The MRI may still be noisy or ambiguous, for example because of limited image quality. In such cases, aleatoric uncertainty may remain high, yet EROP still certifies that no predictor (neither human expert) using only the same MRI can substantially improve upon the output. Improving the prediction would then require additional information, such as acquiring further MRI sequences or using another imaging modality. By contrast, if a standard Bayesian reject-option predictor abstains, it is generally unclear whether this is due to insufficient training data or to irreducible ambiguity in the input. As a result, when a Bayesian reject-option predictor abstains, the appropriate course of action is often unclear.

---

**Q2:** How do you see this framework situated against non-Bayesian methods for rejection, for example conformal prediction?

**A2:** In reject-option prediction, the predictor either outputs a prediction or abstains. In conformal prediction, by contrast, the predictor outputs a prediction set whose size reflects uncertainty. Standard conformal prediction is designed to provide distribution-free coverage guarantees for prediction sets or intervals. These guarantees concern coverage, rather than the decomposition of uncertainty into aleatoric and epistemic components. Therefore, standard conformal prediction does not explicitly account for epistemic uncertainty.

---

**Q3:** How sensitive is the reject-predictor to the prior choice?

**A3:** Our current framework assumes a well-specified model, i.e., that the data-generating distribution lies within the model class. We do not yet have theoretical results for the misspecified setting, where the approximation error remains non-zero. This issue is not specific to our approach: standard Bayesian inference, and more generally generative learning approaches, face the same difficulty. At present, we do not have a principled extension that accounts for approximation error, but we are actively investigating this setting empirically in order to understand the sensitivity of the approach to different forms of model misspecification.

##### Replying to Rebuttal by Authors

#### Rebuttal Acknowledgement by Reviewer DNAi

**Acknowledgement:** (a) Fully resolved - My concerns have been adequately addressed. If you select this option, please consider adjusting your score accordingly.

**Reasons:**

Thank you for your response, I don't have any open major questions and I will adjust my score.

#### Official Review of Submission22311 by Reviewer ARiq

**Summary:**

The paper proposes an epistemic reject-option predictor that abstains when the expected regret relative to a Bayes-optimal predictor exceeds a threshold. It proposes to separate epistemic abstention from aleatoric abstention. Particularly, the main difference from aleatoric reject-option predictor is that, aleatoric predictor abstains in cases where the prediction has a high variance, which could come from both unavoidable variance and lack of data observations; while the proposed predictor only abstains in cases where there is a lack of data.

**Strengths And Weaknesses:**

The main contribution is the conceptual proposal of a regret-based objective for abstention. The paper connects to a decision-theoretic framework. The connection to standard uncertainty measures is useful, e.g. to squared loss and KL divergence.

However, the novelty is somewhat limited above the standard Bayesian uncertainty decompositions. The empirical section mainly focuses on synthetic data, where the ground truth distribution is known. From a decision-theoretic perspective, in cases where even the Bayesian predictor is uncertain, it seems still helpful if the model could abstain from outputting an answer or tell the decision maker that it is uncertain. The paper would benefit from a broader discussion on why this regret notion is helpful in the context of uncertainty quantification. I'm not fully convinced that this regret notion is useful. It might be helpful if the paper includes real human experiments to show that this benchmark is useful in improving real decision making.

**Soundness:** 2: fair

**Presentation:** 4: excellent

**Significance:** 2: fair

**Originality:** 2: fair

**Key Questions For Authors:**

Since the paper proposes to separate epistemic abstention from aleatoric abstention, why not allow a fourth output option that the model says "even a Bayesian predictor is unsure"?

**Limitations:**

yes

**Overall Recommendation:** 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.

**Confidence:** 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

**Compliance With LLM Reviewing Policy:** Affirmed.

**Code Of Conduct Acknowledgement:** Affirmed.

**Final Justification:**

I keep my assessment of the paper due to the reasons in the acknowledgement.

#### Rebuttal by Authors

**Rebuttal:**

**Q1:** Since the paper proposes to separate epistemic abstention from aleatoric abstention, why not allow a fourth output option that the model says "even a Bayesian predictor is unsure"?

**A1:** Authors: We interpret the reviewer’s suggestion as proposing a predictor with four possible decisions: (1) predict, (2) aleatoric abstain, (3) epistemic abstain, and (4) abstain due to total uncertainty. At an intuitive level, this indeed appears to be a very appealing framework. The main difficulty, however, is how to formulate an objective function that meaningfully captures all four decisions while remaining interpretable and operationally well motivated. At present, we do not have a satisfactory formulation of such a decision problem. Nevertheless, we agree that this is an interesting direction for future research.

---

Please allow us to comment on the issues raised in the review:

**Reviewer:** The empirical section mainly focuses on synthetic data, where the ground truth distribution is known.

**Authors:** We use synthetic data in order to compare the performance of the learned predictor directly with that of the optimal predictor, since the gap between the learned and optimal predictors is precisely the quantity our approach aims to minimize. In other words, direct evaluation of the proposed objective requires access to the optimal solution. We do have in mind alternative, more indirect ways of evaluating the proposed epistemic reject-option predictor (EROP). For example, EROP could be used as an example-selection strategy in active learning. However, we chose to keep the paper focused on a single topic that is already somewhat nonstandard.

---

**Reviewer:** From a decision-theoretic perspective, in cases where even the Bayesian predictor is uncertain, it seems still helpful if the model could abstain from outputting an answer or tell the decision maker that it is uncertain.

**Authors:** All reject-option predictors considered in our paper (aleatoric, Bayesian, and epistemic) separate the base predictor from the selector (i.e., the rejection strategy). Therefore, even when any of these predictors abstains, the output of the underlying base predictor remains available as the reviewer suggests.

---

**Reviewer:** The paper would benefit from a broader discussion on why this regret notion is helpful in the context of uncertainty quantification. I'm not fully convinced that this regret notion is useful. It might be helpful if the paper includes real human experiments to show that this benchmark is useful in improving real decision making.

**Authors:** To clarify why we base the reject option solely on epistemic uncertainty, and how this differs from a Bayesian reject-option predictor based on total uncertainty, let us describe a prototypical application of the epistemic reject-option predictor (EROP).

Consider **tumor grading from brain MRI**. If EROP abstains on a given MRI, this indicates that the input is not sufficiently covered by the current training data, and that annotating such cases could be valuable for extending the training set. If, on the other hand, EROP outputs a prediction (e.g., low-grade versus high-grade glioma), it guarantees that, within the considered statistical framework, no better prediction can be obtained from that MRI alone.

Importantly, this guarantee does not require aleatoric uncertainty to be low. The MRI may still be noisy or ambiguous, for example because of limited image quality. In such cases, aleatoric uncertainty may remain high, yet EROP still certifies that no predictor (neither human expert) using only the same MRI can substantially improve upon the output. Improving the prediction would then require additional information, such as acquiring further MRI sequences or using another imaging modality. By contrast, if a standard Bayesian reject-option predictor abstains, it is generally unclear whether this is due to insufficient training data or to irreducible ambiguity in the input. As a result, when a Bayesian reject-option predictor abstains, the appropriate course of action is often unclear.

##### Replying to Rebuttal by Authors

#### Rebuttal Acknowledgement by Reviewer ARiq

**Acknowledgement:** (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.

**Reasons:**

I appreciate the author's response. It helps clarify the paper's intended application, but I'm still not convinced that the proposed objective is the right one for reject-option prediction. The rebuttal reinforces my opinion that the contribution is mainly conceptual. The experiments are only in controlled settings where true regret can be computed. It is unclear whether this approach leads to better decision support. For example, the authors note that the Bayesian optimal predictor remains available even when the model abstains, but this makes it less clear what additional support the proposed reject-option mechanism provides beyond the Bayesian predictor itself. More generally, it is unclear how this framework should be used in settings with multiple competing predictors, where the relevant question is not only whether the current predictor has good prediction quality, but also how its output should be combined with other available sources of information.

##### Replying to Rebuttal Acknowledgement by Reviewer ARiq

#### Reply Rebuttal Comment by Authors

**Comment:**

**Reviewer:** "For example, the authors note that the Bayesian optimal predictor remains available even when the model abstains, but this makes it less clear what additional support the proposed reject-option mechanism provides beyond the Bayesian predictor itself. "

**Authors:** As illustrated in the example application above, the added value of the proposed EROP beyond the Bayesian predictor is that it helps distinguish whether the Bayesian predictor’s uncertainty is due to insufficient training data (reducible uncertainty) or to uncertainty inherent in the task itself (irreducible uncertainty). Thus, the reject-option mechanism provides practical support by indicating whether abstention reflects a potentially improvable lack of knowledge or unavoidable ambiguity.

#### LLM Feedback by Program Chairs

**Feedback:**

Dear Author,

We hope you found the PAT automated pre-submission feedback helpful for your ICML 2026 submission.

We are currently gathering feedback to gauge the quality and usefulness of the PAT Program. Could you spare a few minutes to share your experience in the link below? Your insights will directly help us improve the tool's accuracy and utility for future authors.

[https://docs.google.com/forms/d/e/1FAIpQLSdZ\_mSU0bBXa82HiWlfTuFIx3UhJiLeiOuEh4gS1ZQ1CKT65A/viewform?pageHistory=0,1&entry.1908739883=JJZ2I8qQeO&entry.1347704798=A](https://docs.google.com/forms/d/e/1FAIpQLSdZ_mSU0bBXa82HiWlfTuFIx3UhJiLeiOuEh4gS1ZQ1CKT65A/viewform?pageHistory=0,1&entry.1908739883=JJZ2I8qQeO&entry.1347704798=A)

Thank you for your time, Paper Assistant Tool (PAT) Team, Google Research.

#### LLM Feedback by Program Chairs

**Feedback:**

Hello!

You requested a review of your paper submitted to ICML using the Google Paper Assistant Tool (PAT). The resulting AI Feedback can be found below. Note that this feedback is posted automatically, and is only visible to authors. Importantly, the feedback will **not** be used in the review process. Reviewers, area chairs, and program committee members will **not** have access to the PAT feedback.

Disclaimer: Please note that the models used by the PAT pipeline are not infallible; they may hallucinate and make mistakes. Authors should treat the generated feedback with the same critical eye they would apply to a human review.

PAT Feedback Model: MODEL\_A

PATLibraryRunPipeline:

## HIGH LEVEL SUMMARY

---

## Paper Summary

The paper "Epistemic Reject Option Prediction" introduces a framework for constructing predictors that abstain from making predictions specifically when epistemic uncertainty (uncertainty due to limited data) is high. This contrasts with traditional methods that focus on aleatoric uncertainty or Bayesian methods that focus on total uncertainty. The authors formalize this approach by extending the Bayesian learning framework to minimize expected regret, defined as the performance gap between the learned predictor and the Bayes-optimal predictor. The model abstains when this conditional regret exceeds a predefined cost. A key theoretical contribution is demonstrating that this conditional regret aligns with widely used entropy-based and variance-based measures of epistemic uncertainty under specific loss functions. Empirical results on a synthetic regression task demonstrate that the proposed method achieves a lower Area under the Regret-Coverage curve (AuReC) compared to baselines.

## Key Issues Roadmap

- **\[1. Introduction and Background\]**: A potential conceptual error in the definition of the Maximum Likelihood (ML) objective function (L138), which appears to define the Maximum A Posteriori (MAP) objective instead. Additionally, address the ambiguous notation in Example 3 (L175-176) where the input variable $x$ is overloaded.
- **\[3. Theoretical Analysis and Justification\]**: Potential mathematical inaccuracies in the intermediate steps of the derivations in Appendix B, specifically a sign error in the squared loss derivation (L638) and a missing expectation operator in the 0/1 loss derivation (L673-674). Also, address the inconsistency in the logarithm base used for Cross-Entropy loss and Shannon entropy definitions.
- **\[4. Empirical Evaluation and Related Work\]**: Concerns regarding the empirical evaluation methodology, including the use of an unconventional dispersion metric ("central 20% interval" in Figure 3) which may underrepresent variability, and the omission of Risk-Coverage analysis (AuRC), which is necessary to understand trade-offs with risk-minimizing baselines. Furthermore, the novelty claim regarding the application of epistemic uncertainty measures for rejection (L415-416) may require refinement.

## DETAILED SEGMENT REVIEWS

---

## \[1\] SEGMENT: 1. Introduction and Background

## PAGES: \[\[1, 4\]\]

**Summary:** The reviewed segment (Pages 1-4) comprises the Introduction (Section 1) and Preliminaries (Section 2). Section 1 establishes the motivation by distinguishing between aleatoric and epistemic uncertainty, arguing that traditional reject-option methods often neglect epistemic uncertainty, which is critical when data is limited. The paper proposes an "epistemic reject-option predictor" that abstains specifically when epistemic uncertainty is high, formalized by minimizing expected regret. Section 2 reviews the necessary background, formally defining Aleatoric Reject-Option Prediction (Sec 2.1-2.2) and Bayesian Reject-Option Prediction (Sec 2.3-2.4). These frameworks are illustrated using a running example of linear regression, setting the stage for the proposed method in Section 3.

**Potential Mistakes and Improvements:**

1. **Correctness (Sec 2.2): Definition of the ML objective function.** On Page 3, Line 138, the text states that ML learning maximizes the log-likelihood defined as $l \left(\right. \theta \left.\right) = log ⁡ p \left(\right. \theta \left|\right. D \left.\right)$. This expression represents the log posterior, which is maximized in Maximum A Posteriori (MAP) estimation, not Maximum Likelihood (ML) estimation. ML estimation maximizes the log-likelihood $l \left(\right. \theta \left.\right) = log ⁡ p \left(\right. D \left|\right. \theta \left.\right)$.
2. **Clarity (Sec 2.3, Example 3): Ambiguous Notation for Input Variable.** In Example 3 (Page 3, L175-176), the notation $x$ is overloaded. The predictive distribution is defined as $p \left(\right. y \left|\right. x , D \left.\right) = N \left(\right. y \left|\right. x^{T} \mu_{m} , x^{T} \Sigma_{m} x + v \left(\right. x \left.\right) \left.\right)$. In this expression, $x$ is used simultaneously as the scalar input in the variance term $v \left(\right. x \left.\right)$ (as $X = R$ from Example 1) and as the augmented feature vector $x = \left[\right. x , 1 \left]\right.^{T}$ (L176) in the other terms. This ambiguity hinders clarity regarding the dimensionality and application of the terms. It is recommended to use distinct notation for the feature vector (e.g., $\phi \left(\right. x \left.\right)$).
3. **Clarity (Sec 2.1 and Figure 1): Inconsistent Notation for Conditional Risk.** There is an inconsistency in the notation for the true aleatoric uncertainty (the conditional risk of the Bayes-optimal predictor). In Section 2.1 (Page 2, Eq 2), this quantity is defined as $a^ *(x)$. H o w e v e r , F i g u r e 1 a n d i t s c a p t i o n \left(\right. P a g e 3 , L 126 \left.\right) u s e t h e n o t a t i o n$ r^* (x)$ for the same concept.
4. **Clarity (Sec 2.3, Example 3): Missing References for Variables.** The formulas for the posterior parameters $\Sigma_{m}$ and $\mu_{m}$ in Example 3 (Page 3, L178) utilize the variables $X$, $\Sigma$, and $y$. These variables were defined previously in Example 2 (Eqs 7 and 8), but this is not referenced within Example 3. Although a clarification is provided later on Page 4 (L195), referencing the definitions locally within Example 3 would improve readability.

**Minor Corrections and Typos:**

- Page 3, L159: "negligable" should be "negligible".
- Page 3, L161: In the context of statistical consistency, the standard term is "regularity conditions" rather than "regulatory conditions".
- Page 4, L191: "is computes as" should be "is computed as".

---

## \[2\] SEGMENT: 2. Proposed Epistemic Reject-Option Framework

## PAGES: \[\[4, 6\]\]

### 1\. Summary

The reviewed segment (Sections 3 and 4, Pages 5-6) introduces the paper's primary contribution: the framework for Epistemic Reject-Option Prediction.

Section 3 formalizes the approach by modifying the standard Bayesian Learning objective. Instead of minimizing expected loss, the authors propose minimizing expected regret. Regret is defined as the performance gap between the learned predictor $H \left(\right. x , D \left.\right)$ and the Bayes-optimal predictor $h \left(\right. x , \theta \left.\right)$ (which assumes knowledge of the true parameters $\theta$). An objective function $R_{\delta} \left(\right. H , C \left.\right)$ (Eq. 16) is defined, where prediction is penalized by the expected conditional regret $E \left(\right. x , D \left.\right)$, and rejection is penalized by a cost $\delta$. Theorem 3.1 characterizes the optimal epistemic reject-option predictor $Q_{E}$. It shows that the optimal predictor component $H^ *$i s t h e s t a n d a r d B a y e s i a n o p t i m a l p r e d i c t o r \left(\right. E q . 10 \left.\right) , a n d t h e o p t i m a l s e l e c t o r$ C\_E $r e j e c t s w h e n t h e e x p e c t e d c o n d i t i o n a l r e g r e t$ E^* (x, D) $\left(\right. E q . 17 \left.\right) e x c e e d s$ \\delta$.$E^\*(x, D)$ is subsequently used as the measure of epistemic uncertainty.

Section 4 compares the proposed epistemic predictor ($Q_{E}$) with the Bayesian reject-option predictor ($Q_{B}$). It introduces a decomposition of the total uncertainty $T^ *(x, D) $\left(\right. u s e d b y$ Q\_B$\left.\right) i n t o t h e e x p e c t e d a l e a t o r i c u n c e r t a i n t y$ A^* (x, D) $\left(\right. E q . 18 \left.\right) a n d t h e e p i s t e m i c u n c e r t a i n t y$ E^ *(x, D) $\left(\right. u s e d b y$ Q\_E$\left.\right) . T h i s d e c o m p o s i t i o n \left(\right.$T^* = A^\* + E^\*$\left.\right) c l a r i f i e s t h a t$ Q\_E$ bases rejection decisions solely on the epistemic component. This is illustrated via Example 5 and a conceptual summary of the rejection regions for Aleatoric, Bayesian, and Epistemic predictors.

### 2\. Potential Mistakes and Improvements

The methodological development in Sections 3 and 4 appears sound. The definition of the regret-based objective, the derivation of the optimal predictor (Theorem 3.1, verified via Appendix A), and the uncertainty decomposition in Section 4 are mathematically correct within the defined Bayesian framework. No significant issues regarding correctness, clarity, or validity were identified.

### 3\. Minor Corrections and Typos

- Lines 246-247: There is an error in the definition of the predictor/selector pair. It reads: "...represent Q by a pair (H, C), where H: X ×(X × Y) $$ → Y is a predictor and H: X × (X × Y) $$ → Y is a selector." The second clause incorrectly uses H to denote the selector and defines the codomain as Y. It should likely read "C: X × (X × Y) $$ → {0, 1} is a selector", consistent with the definition in Lines 208-209.

---

## \[3\] SEGMENT: 3. Theoretical Analysis and Justification

## PAGES: \[\[5, 5\], \[7, 8\], \[10, 13\]\]

**1\. Summary** The segment under review (Section 3, relevant parts of Section 6 on Pages 7-8, and Appendices A and B) introduces the theoretical foundation of the proposed Epistemic Reject-Option Predictor. Section 3 defines the objective as minimizing expected regret (Eq. 16) and presents Theorem 3.1, which characterizes the optimal predictor $Q_{E}$. This predictor utilizes the standard Bayesian predictor $H^ *$a n d a b s t a i n s i f t h e e x p e c t e d c o n d i t i o n a l r e g r e t$ E\_* (x, D) $\left(\right. d e f i n e d a s t h e e p i s t e m i c u n c e r t a i n t y \left.\right) e x c e e d s a t h r e s h o l d$ \\delta$. A p p e n d i x A p r o v i d e s a p r o o f f o r T h e o r e m 3.1 , w h i c h w a s v e r i f i e d a n d f o u n d t o b e r i g o r o u s . S e c t i o n 6 a n d A p p e n d i x B d e m o n s t r a t e t h a t$ E\_\*(x, D)$ coincides with established uncertainty measures under specific loss functions (Squared, 0/1, and Cross-Entropy).

**2\. Potential Mistakes and Improvements** While the main theorem is sound and the final results of the derivations in Appendix B are correct, there are mathematical errors in the intermediate steps of the derivations and clarity issues that require attention for full mathematical rigor.

- **\[Correctness\] Appendix B.1: Sign Error in Squared Loss Derivation.** On Page 12, Line 638, the algebraic expansion of the expression in Line 637 contains a sign error. When expanding $\left(\right. H_{B} \left(\right. x , D \left.\right) - y \left.\right)^{2} - \left(\right. h \left(\right. x , \theta \left.\right) - y \left.\right)^{2}$, the expansion should yield $- \mu_{y \left|\right. x , \theta}^{2}$ as the final term, but it is written as `+ µ²_{y|x,θ}`. Although subsequent steps utilize the correct expression, this intermediate step is mathematically incorrect as written.
- **\[Correctness\] Appendix B.2: Missing Expectation Operator in 0/1 Loss Derivation.** On Page 13, the derivation of the epistemic uncertainty for the 0/1 loss omits an expectation operator in an intermediate step. Lines 673-674 state: `= p(y = h∗(x, θ) | x, θ) − p(y = H_B(x, D) | x, D)`. This expression is mathematically ill-formed because the first term depends on the random variable θ, which should be marginalized. The expectation $E_{\theta sim p \left(\right. \theta \left|\right. D \left.\right)}$ must be maintained over the first term.
- **\[Clarity\] Inconsistency in Logarithm Base.** There is an inconsistency regarding the base of the logarithm used for Cross-Entropy (CE) loss and Shannon entropy. The CE loss is defined as `− log py` (e.g., L336, L682). The Total Uncertainty T\*(x,D) is equated to the Shannon entropy `H(p(y|x, D))` (L692). However, the captions for Table 1 (L341-342) and Table 2 (L564) explicitly define Shannon entropy H using `log2`. The base of the logarithm must be consistent between the loss definition and the entropy definition for the derivations to hold.
- **\[Clarity\] Definition Error in Section 3.** On Page 5, Lines 246-247, the definition of the predictor components contains an error: "...represent Q by a pair (H, C), where H: X ×(X × Y)m → Y is a predictor and H: X × (X × Y)m → Y is a selector." The second clause incorrectly uses H to define the selector; it should define the selector C (e.g., C: X × (X × Y)m → {0, 1}).

**3\. Minor Corrections and Typos**

- Inconsistent Notation: The notation H\_B(x, D) is used for the Bayesian predictor in the Appendix (e.g., L502, Table 2), while the main text (e.g., Theorem 3.1, Table 1) uses H\*(x, D). Consistent notation should be used throughout the paper.

---

## \[4\] SEGMENT: 4. Empirical Evaluation and Related Work

## PAGES: \[\[6, 9\]\]

**1\. Summary** The reviewed segment (Pages 6-9) encompasses the empirical evaluation (Section 5), related work (Section 6), conclusions (Section 7), and references.

Section 5 details an empirical validation using a synthetic polynomial regression task, chosen for its analytical tractability. The proposed epistemic reject-option predictor is compared against a Bayesian predictor (based on total uncertainty) and an ML plug-in predictor. The evaluation metric is the Area under the Regret-Coverage curve (AuReC). The results (Figure 3) demonstrate that the epistemic predictor achieves the lowest AuReC, supporting the theoretical framework.

Section 6 reviews related work on reject-option prediction and uncertainty decomposition. It connects the proposed conditional regret metric to existing measures of epistemic uncertainty, arguing that this work provides a decision-theoretic justification for these measures based on regret minimization.

Section 7 summarizes the contributions and identifies the need for scalable approximations in complex models as future work.

**2\. Potential Mistakes and Improvements**

- **\[Validity/Rigor\] Section 5, Figure 3: Unconventional and Potentially Misleading Reporting of Statistical Dispersion.** The caption for Figure 3 (L367) states: "Shaded areas indicate the central 20% interval of outcomes across 3000 independent trials." This is a highly unconventional metric for reporting dispersion. By excluding 80% of the outcomes, this visualization significantly underrepresents the variability of the results, making it difficult to assess the statistical significance of the performance differences. Standard measures such as standard deviation, standard error, or a 95% confidence interval should be used.
- **\[Clarity/Completeness\] Section 5: Completeness of Evaluation Metrics (Regret vs. Risk).** The evaluation exclusively uses the Area under the Regret-Coverage curve (AuReC). While the proposed method is designed to minimize expected regret (Eq. 16), the baselines (Bayesian and Aleatoric predictors) are designed to minimize expected loss/risk (Eq. 13 and Sec 2.1). The paper notes (L303-305) that the epistemic predictor may accept inputs with high aleatoric uncertainty (high risk). To provide a comprehensive empirical understanding of the trade-offs between optimizing for regret versus risk, the evaluation should also report the standard Area under the Risk-Coverage curve (AuRC).
- **\[Correctness/Clarity\] Section 6: Precision of Novelty Claim Regarding Uncertainty Measures.** Lines 415-416 state: "However, to our knowledge, these measures \[Eq. 19 and 20\] have not yet been employed for constructing reject-option predictors." This claim may be too strong. These measures (e.g., mutual information in Eq. 19 or variance of the conditional mean in Eq. 20) and their approximations (via ensembles or MC-Dropout) are commonly used in Bayesian Deep Learning as heuristics for rejection tasks (e.g., selective classification, OOD detection). The claim should be refined to emphasize that the novelty lies in the *principled derivation* of these measures as the optimal rejection criteria within the regret-minimization framework, rather than their first application for rejection generally.
- **\[Clarity\] Section 5: Inconsistent Terminology for Baseline.** There is an inconsistency in the naming of a baseline between the text and Figure 3. The text (L278-280) describes a baseline as a "plug-in reject-option predictor" based on the maximum-likelihood estimate. However, the legend of Figure 3 labels this curve "Aleatoric". This may cause confusion with the theoretical optimal Aleatoric predictor (Section 2.1), which assumes full knowledge of the data distribution. Consistent terminology (e.g., "ML Plug-in") should be used.

**3\. Minor Corrections and Typos**

- L277: Missing reference placeholder: "i) the Bayesian reject-option predictor (??)".
- L370: Apparent missing text at the start of the line: "hensive overview..." (likely "For a comprehensive overview...").
- L399-L404: The text refers to definitions in the Appendix (Eq. 23, Table 2, Eq. 24) that duplicate definitions already provided in the main text (Eq. 17, Table 1, Eq. 16). It is recommended to refer to the main text definitions for better readability.
- L439, L443, L466: Rendering issues with the umlaut in the name "Hüllermeier".

---