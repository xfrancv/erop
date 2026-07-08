# Plan 

Implement research code which will evaluate an algorithm based on Bayesian inference to update a label prior from unsupervised examples.


# Task formulation

Assume we train a model of the class posterior and class prior from supervised training examples $T=((x_1,y_1),\ldots,(x_m,y_m))$ i.i.d. sampled from a distribution $p_{tr}(x,y)$, i.e. after training we an estimate of  $p_{tr}(y\mid x)$ and prior $p_{te}(y)$ at our disposal.  It is assumed that the labels are finite, i.e. $y$ is from $\{1,\ldots,Y \}$. The test data D= $(x_1,\ldots,x_n)$ are i.i.d. generated from $$p_{te}(x)=\sum_y p_{te}(x,y)$$
The training and test distributions are not the same, $p_{tr}(x,y)\neq p_{te}(x,y)$. However, it is assumed that the class conditional distribution of training and testing data is the same, i.e.  $p_{tr}(x\mid y)=p_{te}(x\mid y)$. The test prior $p_{te}(y)$ is unknown. 

The goal is to design a predictor for the test data $D$ and to evaluate the prediction uncertainty. The predictor performance is evaluated by a user specified loss function $\ell\colon \{1,\ldots,Y\}\times\{1,\ldots,Y\}\rightarrow{\mathbb R}$. 
# Approach

We will use Bayesian learning.

### Bayesian learning

The parameterized conditional label posterior will be modeled by $$p(y\mid x,\theta)\propto p_{tr}(y\mid x) \frac{\alpha(y)}{p_{tr}(y)}$$where the parameters $\theta=(\alpha(1),\ldots,\alpha(Y))$ represent the unknown prior of the test data, i.e. $\alpha(y)=p_{te}(y)$. It holds that $\alpha(y)\geq 0$ and $\sum_y \alpha(y)=1$. The models $p_{tr}(y\mid x)$ and $p_{tr}(y)$ are assumed to be known; specifically, their are estimated from supervised training examples $T$.

The prior parameter distribution is set to the Dirichlet distribution, i.e. $$p(\theta)={\rm Dirichlet}(\beta_1,\ldots,\beta_Y)$$ where $(\beta_1,\ldots,\beta_Y)$ are hyper-parameters. By default $\beta_y=1,y=1,\ldots,Y$ will be used.

The data distribution given parameters is $$p(D\mid \theta)=\prod_{i=1}^m \sum_y p(y\mid x_i,\theta) p(x_i)$$
where $p(x)$ is the marginal distribution which is not parameterized. The logarithm of $p(D\mid \theta)$ then reads $$\log p(D\mid \theta)=\sum_{i=1}^m \log \sum_y p(y\mid x_i, \theta) + \sum_{i=1}^m \log p(x_i)=\sum_{i=1}^m \log \sum_y p(y\mid x_i, \theta) + K$$ where $K$ is an unknown constant. The logarithm of the posterior parameter distribution then reads $$\log p(\theta \mid D)=\log p(D\mid \theta) + \log p(\theta)=\log p(\theta) + \sum_{i=1}^m \log \sum_y p(y\mid x_i, \theta) + K$$
The $\log p(\theta\mid D)$ is sufficient to implement the Metropolis Hasting algorithm for sampling the parameters $\theta$ from $p(\theta \mid D)$. 
### Inference

We run the Metropolis Hasting algorithm to generate a sequence $(\theta_1,\ldots,\theta_N)$ sampled from $p(\theta \mid D)$. Given the sequence, we predcit a lable for each test input $x\in D$ by: $$h(x,D)=\arg\min_{\hat{y}}\sum_y \hat p(y\mid x,D)\ell(\hat{y},y)$$ where $\ell(y,y')$ is a selected loss function, e.g. the 0/1-loss, and label posterior will be approximated by $$\hat{p}(y\mid x,D)=\frac{1}{N}\sum_{i=1}^N p(y\mid \theta_i,x)\,.$$ is an estimate of the label posterior.  

# Numerical experiment

The proposed approach will be evaluated on synthetic data where the generating distribution is known and hence the optimal solution can be computed and used as a baseline. 

The 0/1-loss will be used in the experiment.

**Data generating model** There will be classes, $Y=4$. The data in each class will be generated from 2-dimensional normal distribution, i.e. $p_{tr}(x\mid y)=p_{te}(x\mid y)={\cal N}(x; \mu_y,\Sigma_y)$. The means $\mu_y$ and covariance matrices $\Sigma_y$ will be set such that for uniform prior $p_{tr}(y)=\frac{1}{Y}$ the Bayes classification error is around $10\%$. 

**Training the base model** We generate $m$ supervised training examples $T=((x_1,y_1),\ldots,(x_m,y_m))$ from the distribution $p_{tr}(x,y)=p(x\mid y)p_{tr}(y)$. The supervised examples will be used to estimate the training label prior $$\hat{p}_{tr}(y)=\frac{1}{m}\sum_{i=1}^m \delta(y,y_i)$$  and the class posterior $\hat{p}_{tr}(y\mid x)$ by fitting the logistic model (multinomial logistic regression) to the training data. The obtained $\hat p_{tr}(y)$ and $\hat p_{tr}(y\mid x)$ will be used defined the parametric label posterior $p(y\mid x,\theta)\propto \hat{p}_{tr}(y\mid x) \frac{\alpha(y)}{\hat{p}_{tr}(y)}$. 

**Inference on test data** We generate test examples $D=((x_1,y_1),\ldots,(x_n,y_n)$ from the distribution $p_{te}(x,y)=p_{te}(x\mid y)p_{te}(y)$ where the label prior $p_{te}(y)$ will be significantly different from the training prior $p_{tr}(y)$.  We run the Bayesian inference described above which will produce a  label prediction $h(x,D)$ for each test example $x\in D$. 

**Baseline predictors** The following predictors will be evaluated on the test data:
1. Optimal Bayes predictor for the training distribution: $$h^*_{tr}(x)=\arg\min_{\hat y}\sum_{y}p_{tr}(x\mid y)p_{tr}(y)\ell(y,\hat{y})$$
2. Optimal Bayes predictor for the test distribution:  $$h^*_{te}(x)=\arg\min_{\hat y}\sum_{y}p_{te}(x\mid y)p_{te}(y)\ell(y,\hat{y})$$
3. Plugin Bayes predictor using the prior learned from the training data: $$\hat{h}_{tr}(x)=\arg\min_{\hat y}\sum_{y}{\hat p}_{tr}(x\mid y){\hat p}_{tr}(y)\ell(y,\hat{y})$$
4. Plugin Bayes predictor using the true test prior: $$\hat{h}_{true-prior}(x)=\arg\min_{\hat y}\sum_{y}{\hat p}_{tr}(x\mid y){\hat p}_{tr}(y)\ell(y,\hat{y})$$
**Evaluation metrics** Each predictor will be evaluated by the test accuracy $$TestAcc=\frac{1}{n}\sum_{i=1}^n \ell(y_i,\hat{y}_i)$$where $y_i$ and $\hat y_i$ is the true and the predicted label.




