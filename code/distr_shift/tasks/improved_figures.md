# Plan

Change reporting of results produced by 'run_real_reject_option_exp.py',
'run_synth_reject_option_exp.py' and 'run_synth_bayesian_learning_exp.py'.

# Task

The following changes will be implemented:

1. Remove the baseline reject-option predictor plugin Bayes with the supervised prior from all curves and text tables related to reject-option (AuRC and coverage-at-target). However, retain the plugin Bayes with the supervised adaptation base predictor in the curve showing the test accuracy vs number of adaptation examples (i.e. curves 'base_accuracy_vs_n_test.png' and 'accuracy_vs_n_test.png').

2. The figure 'epistemic_metrics_vs_n_test.png' has three panels. The first and
the second panel show the average regret vs number of adaptation examples and average epistemic uncertainty vs number of adaptation examples, respectively.  Overlay the curves in the two panels. After the change the figure will have two panels, not three.

3. Now, the risk-coverage and regret coverage curves are created for each
number of adapation examples separately. Put all these curves into a subfolder
called 'coverage_curves/'.

4. In all figures which has the x-axis labeled as 'number of unlabeled test examples n' rename the label to 'number of unlabeled adaptation examples n'.

5.  Add a new baseline oracle rject option predictor to evaluation. The base predictor is the Bayesian leargnin predictor. The unertainty is the optimal uncerainty for given metric and data sample. That is, in case when the evaluation metric is the selective risk, the uncertainty is the actual risk of the Bayesian predictor on the evaluation sample. In case when the evaluation metric is the selective regret, the uncertainty is the actual selective regret of the Bayesian learning predictor on evaluation sample. This baseline marks the best attainable  solution for give metric and evaluation sample. The oracle is a special case that computes its risk curve and its regret curve from separate, metric-specific rankings, and it must be threaded through AuRC, the coverage curves, the per-size curves, and cov@target accordingly.