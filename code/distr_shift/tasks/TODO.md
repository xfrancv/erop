## High Priority
- [ ] EM prior shift compensation

## Low Priority
- [ ] confidence bars in performance curvers use STD; change them to show 80% quantil

- [ ] allow to use different loss optimized by the model and different loss for evaluation;

- [ ] add a figure which shows how is the epistemic uncertainty calibraed w.r.t. the true regret. The epistemic uncertainty of the test examples is binned and for each bin an average regret is computed.


- [ ] (needs rethinking). Implement a new script which evaluates the reject option predictors on real data with synthetic labels obtained from the model fitted on training data. This experiment will show the effect of model mispecification.


## Done
- [x] test the latent variable based sampler

- [x] unified target label prior selection strategy (`--auto-target-prior`,
  `prior_shift/target_prior_search.py`; see 'target_label_prior_selection.md')

- [x] compute area under the curve for 0.5-1 coverage (AuRC50; see 'truncated_aurc.md')
- [x] adding additinal baseline using the perfect uncertainty measure, i.e. the actual regret
- [x] adding generalized regret-coverage curve
- [x] put risk-coverage curves to a subdirectory
- [x] remove plugin Bayes with the supervised prior from all curve but the curve which shows the test accuracy vs number of unlabeled test examples n, e.g. 'base_accuracy_vs_n_test.png'
- [x] in the curve showing the epistimic metrics, i.e. 'epistemic_metrics_vs_n_test.png', merge the average regret and average epistemic uncertainty curves into a single figure with curves.
- [x] remove the optimal reject option baseline
