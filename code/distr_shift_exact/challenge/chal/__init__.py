"""Self-contained library for the TissueMNIST prior-shift challenge.

Nothing here imports from outside ``challenge/``: the directory can be copied
anywhere and still run. Where a module restates something from the parent
project's ``exact/`` package it is a deliberate copy, kept in sync by hand --
the challenge must not break when the research code is refactored.

Section numbers ``S2.1``, ``S6.3`` refer to the parent ``README.md``; section
numbers ``C3.2``, ``C4`` refer to ``tasks/challenge_polish.md``, which is the
specification this code implements.
"""

__all__ = [
    "calibration", "data", "ids", "inference", "metric", "priors",
    "protocol", "splits",
]
