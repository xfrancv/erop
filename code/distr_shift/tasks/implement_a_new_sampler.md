# Plan

Implement a new sampler described in 'tasks/latent_variable_sampling_dirichlet_posterior_dollar_math.md'

# Context

The current code based uses a generic  random-walk Metropolis–Hastings. The sampler described in
'tasks/latent_variable_sampling_dirichlet_posterior_dollar_math.md'
exploit the specific structure of the problem. 

The goal is to implement the new sampler into
scripts that use it:
- 'run_real_reject_option_exp.py'
- 'run_synth_bayesian_learning_exp.py'
- 'run_synth_reject_option_exp.py'

# Task

Implement the new sampler described above. The scripts
will have an input argument which select the sampler used. By default the existing sampler is used.