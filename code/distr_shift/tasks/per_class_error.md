# Plan

Extend the scripts run_synth_bayesian_learning_exp.py and run_base_predictor_exp.py to compute per class error of the base predictor learned on training examples.

# Context

The scripts learn a base predictors on the training examples. The predictor is evaluated in terms of the classification accuracy.

The goal is to compute, besides the accuracy, also a per-class classification error
$$
Err(k)=\frac{1}{|I_k|}\sum_{i \in I_k} \ell(h(x_i),y_i)
$$
where $I_k$ are indices of test examples with label $y_i=k$. The $Err(k)$ will be computed for all classes $k=1,\ldots, Y$. Use only the trial 0 in case of the synthetic data.

The per-class error is re-presentation of existing information, not new measurement. 

# Tasks

Implement computation of the per-class classification errors described above.

In case of 'run_synth_bayesian_learning_exp.py', the base predictor is the plugin Bayes predictor learned on the training examples using the training prior. The results will be appended to 'report.txt'.

In case of 'run_base_predictor_exp.py', the base predictor is the plugin Bayes predictor obtained from the calibrated class poseriors. Using the calibrated posterior will change the results slightly and it is expected. Compute the per-class error on the calibrated argmax as written, and also switch the reported accuracy + confusion matrix to the calibrated predictor. The results will be appended to 'report.txt'. 
 
