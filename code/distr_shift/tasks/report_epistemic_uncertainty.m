# Plan

Update the `run_reject_option_experiment.py` to report epistemic uncertainty metrics. 
This will involve modifying the experiment script to calculate and log these metrics.

# Context

The `run_reject_option_experiment.py` script currently runs experiments 
related to the reject option in classification tasks. 
We want to enhance this script to also report metrics summarizing the emount
of epistemic uncertainty in the model`s predictions. This will help in understanding 
how uncertain the model is about its predictions.

The followin metrics will be reported:
- **Average epistemic uncertainty**: The epistemic uncertainty over the evaluation examples calucated as
$\frac{1}{N}\sum_i \big(\hat T(x_i,D) - \hat A(x_i,D)\big)$.

- **Average regret**: The average regret over the evaluation examples calculated as
$\frac{1}{N}\sum_i \big(\ell(y_i, h(x_i)) - \ell(y_i, \hat h_{true\text{-}prior}(x_i))\big)$ 

- **Portion of predictions with negligible epistemic uncertainty**: It is calculated as the portion
of predictions for which the epistemic uncertainty is below a certain threshold. By default,
the threshold will be set to 0.001, however, it can be adjusted via input argument.

The metrics will be evaluated the Baysian predictor, i.e. $h$ is the Baysian predictor in the definition.

Read README.md to understand notation. 

# Tasks

The script will create figures which show the average regret, the average
epistemic uncertainty and the portion of predictions with negligible epistemic uncertainty 
as a function of the number oftest examples. In non-sweep mode, the script will also
print the metrics to the console.












