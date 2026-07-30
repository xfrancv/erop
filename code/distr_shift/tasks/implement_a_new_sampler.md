> **Note.** Written when the repo still had the synthetic experiment
> (`run_synth_*.py`, `configs/`) and the flags `--sweep` / `--n-test` /
> `--risk-target` / `--optimal-rejection` / the pair-prior knobs. Those
> are gone (see `tasks/refactor_run_real_reject_opt_exp_polished.md`);
> the feature this spec describes lives in
> `run_real_reject_option_exp.py`, `prior_shift/reject_option.py` and
> `reject_figures.py`, which always sweep `--sizes`.

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