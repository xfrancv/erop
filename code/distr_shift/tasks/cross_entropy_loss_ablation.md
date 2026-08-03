# Plan

Augment 'rejopt_eval.py' with an option to use cross-entropy loss
for ablation.

# Context

The epistemic reject option predictor relies on a measure of
epistemic uncetainyt. Cyurrently, the code calculates the total and aleatoric uncertainy under zero-one loss. Then, the epistemic uncertainty is calculated as a difference of the total and the aleatoric uncertainty. The zero-one loss is the target loss
used to evaluate the performance.

However, it is common to evaluate the epistemic uncertainty using
the cross-entropy loss. It is interesting to see how the performance
changes when the cross-entropy loss is used to compute the epistemic
uncertainty. 

# Task

Augment the 'rejopt_eval.py' with and option --cross-entorpy-loss
that leads to using the cross-entropy loss to compute the 
epistemic uncertainty. The performane evaluation metrics remain
using zero-one loss. 