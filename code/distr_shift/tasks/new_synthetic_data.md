# Plan

The goal is to extend the experiment to enable different settings of the synthetic data generator.

# Context

Currently, the data are generated from a mixture of 2D Gaussians with fixed parameters.

# Goal

The goal is to allow different settings of the generator parameters. The underlying model will
be always a mixture of 2D Gaussians, however, the parameter setting can be different. The paramters
that can best are:
- the number of Gaussians
- the mean and the covariance matrix of each Gaussian
- the training label prior
- the test label prior

A particular parameters setting will be defined in a simple JSON file. The `run_experiment.py` will accept
the JSON file as an optiona argument. There will be JSON loaded by default with the current parameter setting.

