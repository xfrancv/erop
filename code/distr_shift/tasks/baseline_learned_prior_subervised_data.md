# Plan

Add a new baseline predictor to the experiment which implements the plugin Bayes predictor with prior adaption based on supervised examples.

# Method

The baseline is the plugin Bayes-predictor with the test prior estiamted from the supervised test data (they are available due to using the synthetic generator):
$$
\hat{h}_{learned-prior}(x)=\arg\min_{\hat y}\sum_{y}{\hat p}_{tr}(y\mid x)\frac{\hat{p}_{te}(y)}{\hat{p}_{tr}(y)}\ell(y,\hat{y})
$$
where $\hat{p}_{te}(y)$ is the test prior computed from the frequencies of the test labels. 

