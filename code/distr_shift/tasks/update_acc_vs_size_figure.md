# Plan

Remove the supervised baseline and add the plugin-bayes
with unadaptade prior to the figure showing base predictor
accuracy as a function of the adaptation set size.

# Context

The script 'run_real_reject_option_exp.py' produces
figure 'accuracy_and_aurc_vs_n_test.png' which
shows accuracy of base predictors. 

# Tasks

Change the figure such that:
1. plugin bayes with supervised learned label prior is removed.
2. plugin bayes with the unadapted traning label prior is shown.