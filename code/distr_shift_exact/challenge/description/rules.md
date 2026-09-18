## Competition Rules

*This is a course competition. There is no monetary prize; the reward is extra
course credit. Entering constitutes acceptance of these rules. Dates are on the
**Overview → Timeline** page.*

1. **Competition data only.** You may access and use the competition data solely
   for participating in this competition and on the competition's Kaggle forums.
   This is the central rule. You must not use any data other than what is
   published on this competition's Data page — in particular, and without
   limitation, not the TissueMNIST or MedMNIST datasets in any form, no other
   external dataset, and no model whose weights were trained on anything other
   than the competition data.

2. **Each test batch is for predicting its own rows.** A test batch is the set
   of rows sharing an `id_test`, together with their entries in
   `predictions.csv`. You may use the whole of a batch to predict the rows in
   it — that is the task. You may not carry anything from one test batch to
   another, and you may not compute anything over the test set as a whole: no
   training, fine-tuning, calibration, statistics, pseudo-labelling or model
   selection across batches, in any form. Anything you want to fit, tune or
   calibrate, fit on the development data, which is provided for exactly that.

3. **Work independently.** Discuss freely; write and submit your own solution.

4. **Team size is 1.**

5. **Code and a short description are part of your entry.** With your final
   submissions you hand in the code that produced them, the trained weights if
   used, and a short description of what you did (a paragraph to a page, as a
   PDF). The code must run on the competition data as published and reproduce
   your submission. The winning entries and a random sample of the others are
   checked by hand: the code is read, and re-run on a differently drawn test
   set. An entry without code and description is not ranked.

6. **Breaking a rule disqualifies the entry.** If you are unsure whether
   something is allowed, ask on the forum before you rely on it. Asking is never
   penalised.

7. **Submissions.** At most 5 per day; you choose 2 to be scored finally.
