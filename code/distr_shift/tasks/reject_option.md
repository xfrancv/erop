# Plan

Implement reject-option predictors and evaluate them using risk-coverage and the regret-coverage curve.

# Task formulation

The `run_experiment.py` implements Bayesian learning for test prior adaptation where the predictor
and the baselines are evaluated based on the test accuracy.

The goal is to implement and evaluate reject-option predictor for the exact same setting (i.e. unsupervised test prior adaptation) which allows to reject prediction in uncertain cases. Read 'README.md' and 'tasks/unsuper_prior_learning.md' to understand the setting. 

# Approach 

The reject option predictor will be represented by a base predictor $h(x)$ which outputs a label and a function $u(x)$ which outputs a real-valued uncertainty score. The reject option predictor outputs the label $h(x)$ if the uncertainty score $u(x)$ is below a decision threshold $\theta$.

Three different reject-option predictors will be implemented and evaluated. Assume $D=(x_1,\ldots,x_n)$ are unsupervised test examples. Assume $(\theta_1,\ldots,\theta_N)$ is a sequence sampled from the parameter posterior $p(\theta\mid D)$ via Metropolis–Hastings. Let the parameterized conditional label posterior will be modeled by $$p(y\mid x,\theta)\propto p_{tr}(y\mid x) \frac{\alpha(y)}{p_{tr}(y)}$$where the parameters $\theta=(\alpha(1),\ldots,\alpha(Y))$ represent the unknown prior of the test data, i.e. $\alpha(y)=p_{te}(y)$.
The three reject-option predictors are defined as follows:

**Bayesian reject-option predictor** 
* The base predictor is the Bayesian learning predictor, i.e. $h(x) = h(x,D)$, where 
$$h(x,D)= \argmin_{\hat y}\sum_{y} \hat{p}(y\mid x, D)\ell(y,\hat{y}) $$
and
$$\hat{p}(y\mid x, D)=\frac{1}{N}\sum_{i=1}^N p(y\mid x,\theta_i)$$
is the label posterior estimated by Metropolis-Hastings sampling.
* The uncertainty score is the total prediction uncertainty, i.e. $u(x) = \hat{T}(x,D)$, where 
the (approximate) total uncertainty is computed by:
$$\hat{T}(x,D) = \frac{1}{N}\sum_{i=1}^N\sum_{y}p(y\mid x,\theta_i)\ell(y, h(x,D))$$

**Epistemic reject-option predictor**

* The base predictor is the Bayesian learning predictor, i.e. $h(x) = h(x,D)$, where $(x,D)$ is defined above.
* The uncertainty score is the (approximate) epistemic uncertainty, i.e. $u(x) = \hat{T}(x,D)-\hat{A}(x,D)$, where the total uncertainty $\hat{T}(x,D)$ is defined above and the aleatoric uncertainty is defined by:
$$\hat{A}(x,D) = \frac{1}{N}\sum_{i=1}^N \sum_y p(y\mid x,\theta_i) \ell(y,h(x,\theta_i))$$
where
$$h(x,\theta)=\arg\min_{\hat y}\sum_y p(y\mid x,\theta)\ell(y,\hat{y})$$

**Plugin Bayes reject-option predictor**
This predictor is a supervised reference baseline and uses the supervised test examples.

* The base predictor is the plugin Bayes predictor, i.e., $h(x) = \hat{h}_{learned-prior}(x)$ where
$$ \hat{h}_{learned-prior}(x)=\arg\min_{\hat y}\sum_{y}{\hat p}_{te}(y\mid x)\ell(y,\hat{y})$$
and
$${\hat p}_{te}(y\mid x) \propto {\hat p}_{tr}(y\mid x)\frac{\hat{p}(y)}{\hat{p}_{tr}(y)}$$
and $\hat{p}(y)$ is the test label prior estimated from supervised test examples.
* The uncertainty is the estimated conditional risk
$$u(x) = \min_{\hat{y}} \sum_{y} \hat{p}_{te}(y\mid x) \ell(y,\hat{y})$$

# Evaluation metrics

Let $h(x)$ be a predictor and $u(x)$ be its measure of the prediction uncertainty. Let $\pi\colon \{1,\ldots,n\}\rightarrow \{1,\ldots,n\}$ represent the ranking of the evaluation examples $((x_1,y_1),\ldots,(x_n,y_n))$ according to the uncertainty $u$ such that $$i<j \Rightarrow u(x_{\pi(i)})\leq u(x_{\pi(j)})$$

Let us define three values:  
* The coverage at rank $k$ be defined as
$$coverage(k)=\frac{k}{n}$$
which represent the portion of evaluation examples on which the reject-option predictor outputs a label.  
* The selective regret at $k$ as
$$regret(k)=\frac{1}{k}\sum_{i=1}^k \left( \ell(y_{\pi(i)}, h(x_{\pi(i)})) - \ell(y_{\pi(i)}, \hat{h}_{true-prior}(x_{\pi(i)})) \right)$$
uses the accepted evaluation examples to evaluate the predictor's performance relative to the performance of the plug-in predictor using the true test label prior.
* The selective risk at $k$ as
$$risk(k)=\frac{1}{k}\sum_{i=1}^k \ell(y_{\pi(i)}, h(x_{\pi(i)}) )$$



A reject-option predictor will be evaluated by:
1. The regret-coverage curve is given by a sequence of points $((coverage(1),regret(1)),\ldots,(coverage(n),regret(n)))$.  
2. The area under the regret-coverage curve, $\text{AuRC}_{regret} = \frac{1}{n}\sum_{k=1}^n regret(k)$.
3. The risk-coverage curve is given by a sequence of points $((coverage(1),risk(1)),\ldots,(coverage(n),risk(n)))$.
4. The area under the risk-coverage curve, $\text{AuRC}_{risk} = \frac{1}{n}\sum_{k=1}^n risk(k)$.


# Tasks

Implement a new script called `run_reject_option_experiment.py` which will do the
same things as the `run_experiment.py`. In contrast to `run_experiment.py`, the new script will use a fixed labeled evaluation set in the non-sweep run for evaluation in order to remove the optimistic bias for the supervised plugin predictor baseline. The supervised plugin predictor will use the test examples to estimate the prior, not the evaluation examples. 

In addition, the new script will do the following:

**Non-sweep run (single test set)**
It evaluates the three reject-option predictors by creating: 
1. The risk-coverage curve for all the predictors. For each predictor, it also computed the area under the risk-coverage curve.
2. The regret-coverage curve for all the predictors. For each predictor, it also computed the area under the regret-coverage curve. 

**Sweep run (sweeping along the number of test set examples)** 
It evaluates the three reject-option predictors such that for each test set size it computes the area under the risk-coverage curve and the area under the regret-coverage curve. Then, it creates a figure showing the areas as a function of the number of test examples.

The curves computed will be averages accross the trials.


