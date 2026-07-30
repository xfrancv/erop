# Plan 

Refactor the 'run_real_reject_option_exp.py' by removing
input arguments, and outputs that are not finaly exploited.

# Context

During development, the script 'run_real_reject_option_exp.py' has accumulated large number of input arguments and figures. Most of them are useful, however, some are not necessary. The goal is to create a version which will be used by others. To this end, we want to decrease complexity of the script and cut out unnecesary functionalities. 

The synthetic experiment will be removed totaly, i.e. the scripts
'run_synth_bayesian_learning_exp.py', 'run_synth_reject_option_exp.py', 'and run_all_synth_exp.sh'  and directory 'configs/' 
will be removed. 

# Tasks

The task is to remove the following:

- Input argument '--risk-target' will  be removed and
the associated curve in 'cov_at_target_vs_n_test.png' (top panel) will be removed as well.
- Input argument '--optimal-rejection'. The optimal rejection oracle will be never calculaed and evaluated. 
- Input argument '--pair-ratio' will be removed, and the associated way to enter the target prior.
- Input argument '--confusable-pair I J' will be removed, and the associated way to enter the target prior.
- Input argument '--pair-rest-ratio A B' will be removed, and the associated way to enter the target prior.
- Input arguments '--sweep' and '--n-test' will be removoed. The script will always operate in sweep mode, i.e. it will evaluate the predictor for a set of adaptation set sizes (--size). Update the 'run_real_exp.sh' by removing the --sweep aregument. 

By default, the target prior will be set to the label prior
estimated on the training data. This should be available anyway as it is used in the script for other calculations. It is important to
known how the methods will behave under this degenerate setting, i.e.
when no adaptation is needed. 


