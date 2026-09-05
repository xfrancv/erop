"""Exact Bayesian label-prior adaptation over a finite Theta (README S2, S5).

Everything here is closed-form: the parameter set is finite, so the integrals of
S2 are sums over ``C`` terms and no sampler is involved. All of it runs in log
space (S5).

The one identity worth keeping in mind while reading the code, because it makes
the whole trial cheap:

    p(theta | x_i, D^(i))  ~  p(theta) * prod_{j=1..m+1} w(x_j, theta)

is **the same for every triplet i of a trial** -- the leave-one-out factor
``w(x_i, theta)`` that the query contributes cancels the one the adaptation set
is missing. So a trial needs exactly one normalisation over theta for the
aleatoric term (``pth_query``), plus one cheap subtraction per triplet for the
adaptation-only posterior ``p(theta | D^(i))`` (``pth_adapt``) that the MAP
plugin and the label posterior use.

A second identity is useful for interpreting the results: the label posterior

    p(y | x, D)  ~  (thetabar_y / p_tr(y)) * p_tr(y | x),
    thetabar = E_{theta ~ p(theta | D)}[theta],

i.e. the Bayesian learned-prior rule *is* the plugin rule at the posterior-mean
prior. The code below computes it in the logsumexp form of S5 rather than via
``thetabar`` -- same number, and it keeps the intermediate
``log p(y, theta | x, D)`` available for the aleatoric term.

Shapes throughout: ``n`` triplets in a trial, ``C`` priors, ``Y`` classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def log_weights(log_post: np.ndarray, log_train_prior: np.ndarray,
                log_theta: np.ndarray) -> np.ndarray:
    """``r[i, c] = log w(x_i, theta_c)``, the S2.1 weight in log space.

    ``w(x, theta) = p_te(x | theta) / p_tr(x) = sum_y (theta_y / p_tr(y)) p_tr(y | x)``
    -- the ratio in which the intractable ``p_tr(x)`` has already cancelled.

    Parameters
    ----------
    log_post : (n, Y) calibrated ``log p_tr(y | x_i)``.
    log_train_prior : (Y,) ``log p_tr(y)``.
    log_theta : (C, Y) ``log theta_{c,y}``.
    """
    # (n, 1, Y) + (1, C, Y) -> logsumexp over y -> (n, C)
    terms = log_post[:, None, :] + (log_theta - log_train_prior)[None, :, :]
    return logsumexp(terms, axis=2)


def log_plugin_posterior(log_post: np.ndarray, log_train_prior: np.ndarray,
                         log_theta: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``log p_te(y | x_i, theta_c)`` as an (n, C, Y) array.

    ``p_te(y | x, theta) = (theta_y / p_tr(y)) p_tr(y | x) / w(x, theta)`` --
    the same numerator as :func:`log_weights` before its sum over ``y``.
    """
    return (log_post[:, None, :] + (log_theta - log_train_prior)[None, :, :]
            - r[:, :, None])


@dataclass
class TrialInference:
    """Every per-triplet quantity S2-S3 asks for, for one test set ``D'``.

    All arrays are indexed by the triplet ``i`` (i.e. by which example of the
    trial plays the query), and are the *exact* posterior quantities -- no
    Monte Carlo error, only the float64 arithmetic of the logsumexps.
    """

    bayes_pred: np.ndarray     # (n,) H(x, D)
    total: np.ndarray          # (n,) T(x, D)
    aleatoric: np.ndarray      # (n,) A(x, D)
    epistemic: np.ndarray      # (n,) E(x, D) = T - A
    map_index: np.ndarray      # (n,) argmax_c p(theta_c | D^(i)), lowest c on ties
    map_pred: np.ndarray       # (n,) h(x, theta_map)
    map_unc: np.ndarray        # (n,) 1 - p_te(h | x, theta_map)
    plugin_pred: np.ndarray    # (n, C) h(x, theta_c)
    plugin_unc: np.ndarray     # (n, C) 1 - max_y p_te(y | x, theta_c)
    pth_adapt: np.ndarray      # (n, C) p(theta | D^(i))
    pth_query: np.ndarray      # (C,)   p(theta | x_i, D^(i)); same for all i


def infer_trial(log_post: np.ndarray, log_train_prior: np.ndarray,
                log_theta: np.ndarray, log_p_theta: np.ndarray) -> TrialInference:
    """Run the exact inference of S2 for every triplet of one test set.

    ``log_post`` holds the ``n = m + 1`` rows of the trial; triplet ``i`` uses
    row ``i`` as the query and the other ``n - 1`` rows as the adaptation set
    ``D^(i)``, which is the leave-one-out construction of S6.3.

    Cost is ``O(n C Y)`` for the weight matrix plus ``O(n C)`` for all ``n``
    triplets -- the S5 leave-one-out trick, not ``O(n^2 C)``.
    """
    n = log_post.shape[0]

    r = log_weights(log_post, log_train_prior, log_theta)     # (n, C)
    S = r.sum(axis=0)                                         # (C,)

    # log p(theta) + log p_te(D^(i) | theta), up to an i-dependent constant that
    # is free of theta and so cancels in every normalisation below. The
    # leave-one-out subtraction is done *before* adding log p(theta) so that at
    # m = 0 it is exactly zero (S = r) and every prior ties exactly, which is
    # what lets the lowest-index tie-break of S3 fire.
    g = log_p_theta[None, :] + (S[None, :] - r)               # (n, C)
    pth_adapt = np.exp(g - logsumexp(g, axis=1, keepdims=True))

    # p(theta | x_i, D^(i)) ~ p(theta) prod_j w(x_j, theta): free of i.
    gq = log_p_theta + S                                      # (C,)
    pth_query = np.exp(gq - logsumexp(gq))

    # --- S2.2 label posterior, S2.3 total uncertainty ---------------------
    # log p(y, theta | x, D) up to a constant: g + log theta_y - log p_tr(y)
    #                                            + log p_tr(y | x).
    log_joint = (g[:, :, None] + (log_theta - log_train_prior)[None, :, :]
                 + log_post[:, None, :])                      # (n, C, Y)
    log_lab = logsumexp(log_joint, axis=1)                    # (n, Y)
    log_lab -= logsumexp(log_lab, axis=1, keepdims=True)
    label_post = np.exp(log_lab)

    bayes_pred = label_post.argmax(axis=1)                    # ties -> lowest y
    total = 1.0 - label_post[np.arange(n), bayes_pred]

    # --- S2.4 per-theta plugin predictor and the aleatoric term -----------
    log_plug = log_plugin_posterior(log_post, log_train_prior, log_theta, r)
    plugin_pred = log_plug.argmax(axis=2)                     # (n, C)
    plugin_max = np.exp(np.take_along_axis(
        log_plug, plugin_pred[:, :, None], axis=2)[:, :, 0])  # (n, C)
    plugin_unc = 1.0 - plugin_max

    # A(x, D) = E_{theta ~ p(theta | x, D)} [ 1 - p_te(h(x, theta) | x, theta) ].
    aleatoric = plugin_unc @ pth_query                        # (n,)
    epistemic = total - aleatoric

    # --- S3 row 3: the MAP plugin -----------------------------------------
    # argmax over the *adaptation-only* posterior p(theta | D). At m = 0 the
    # rows of g are exactly constant, so argmax returns index 0 = theta_1 and
    # the MAP plugin degenerates to the train-prior plugin, as S3 requires.
    map_index = g.argmax(axis=1)
    rows = np.arange(n)
    map_pred = plugin_pred[rows, map_index]
    map_unc = plugin_unc[rows, map_index]

    return TrialInference(
        bayes_pred=bayes_pred, total=total, aleatoric=aleatoric,
        epistemic=epistemic, map_index=map_index, map_pred=map_pred,
        map_unc=map_unc, plugin_pred=plugin_pred, plugin_unc=plugin_unc,
        pth_adapt=pth_adapt, pth_query=pth_query)


def plugin_for_prior(log_post: np.ndarray, log_train_prior: np.ndarray,
                     theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(h(x, theta), 1 - max_y p_te(y | x, theta))`` for a single prior.

    Used for the two rejectors whose prior does not come from ``Theta``: the
    train-prior plugin (row 4) and the true-prior oracle (row 5, whose
    ``theta_*`` is a continuous draw in the misspecified arm).
    """
    log_theta = np.log(np.asarray(theta, float))[None, :]
    r = log_weights(log_post, log_train_prior, log_theta)
    log_plug = log_plugin_posterior(log_post, log_train_prior, log_theta, r)[:, 0, :]
    pred = log_plug.argmax(axis=1)
    unc = 1.0 - np.exp(log_plug[np.arange(len(pred)), pred])
    return pred, unc


# --- identifiability of Theta ---------------------------------------------

def identifiability_table(log_post_eval: np.ndarray, y_eval: np.ndarray,
                          log_train_prior: np.ndarray, theta: np.ndarray,
                          num_classes: int) -> np.ndarray:
    """Per-example drift of the log-likelihood ratio between two priors.

    ``K[a, b] = E_{x ~ p_te(. | theta_a)} [ log w(x, theta_a) - log w(x, theta_b) ]``

    is the rate at which the posterior over ``Theta`` separates ``theta_a`` from
    ``theta_b`` as the adaptation set grows: after ``m`` unlabeled examples the
    log posterior odds have drifted by ``m K[a, b]``, so telling the pair apart
    needs on the order of ``1 / K[a, b]`` examples.

    Worth computing because the S7 constructions do not control it. Doubling one
    class out of 100 moves the prior by TV = 0.01 and the *marginal over x* by
    far less, so on a 100-class dataset the whole S6.2 grid can sit below the
    size at which any of ``Theta`` becomes identifiable -- which would show up
    as flat curves for a structural reason, not a methodological one.

    The expectation is taken under the distribution the experiment actually
    samples from (S6.3: class drawn from ``theta_a``, then an example of that
    class uniformly from the evaluation split), so it is computed exactly rather
    than estimated -- no importance-sampling variance.

    If the model were exactly calibrated, ``w`` would be the true density ratio
    and ``K[a, b]`` would be a KL divergence, hence non-negative. A **negative**
    entry is therefore a finding rather than a numerical accident: it says the
    likelihood drifts towards ``theta_b`` when ``theta_a`` is true, i.e. the
    calibration error of ``p_tr(y | x)`` outweighs the prior shift and the MAP
    estimate converges to the wrong element of ``Theta``.
    """
    log_theta = np.log(np.asarray(theta, float))
    r = log_weights(log_post_eval, log_train_prior, log_theta)   # (n, C)
    C = r.shape[1]

    # p_a(x_i) = theta_{a, y_i} / |eval_{y_i}|: the exact S6.3 sampling density.
    counts = np.bincount(y_eval, minlength=num_classes).astype(float)
    per_example = np.asarray(theta, float)[:, y_eval] / counts[y_eval]  # (C, n)

    K = np.zeros((C, C))
    for a in range(C):
        pa = per_example[a] / per_example[a].sum()
        K[a] = pa @ (r[:, [a]] - r)
    np.fill_diagonal(K, 0.0)
    return K
