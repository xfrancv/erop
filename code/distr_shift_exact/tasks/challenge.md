# Plan 

I want to organize a challange for students with the task to design a reject-option predictor adapting to the label prior shift. The students will get a model trained on one label distribution and test data from another. They will have to build a predictor that answers or rejects each input. It will be scored on regret at 80% coverage. 
The setting is described in detail in README.md .

Ideally, some students will discover the optimal predictor, i.e. the base predictor using Bayesian learning and rejection strategy based on the epistemic uncertainty. 

# Kaggle

The challange will run on Kaggle: https://www.kaggle.com/. 

A private competition on Kaggle has been created for this challenge:
https://www.kaggle.com/competitions/know-what-you-dont-know-under-a-new-prior/

# Data

The competition data will be generated from the `tissuemnist` dataset. The studens will be required to submit their code at the end of the competition. Their code will reveal possible cheating. It will be stated in the competition rules that they cannot use any other data than provided.

The data will be split (class stratified) in the same way as it is done in the recent code base, i.e.
| Split | Fraction | Use |
| --- | --- | --- |
| development |  | ↓ |
| ↳ training | 0.80 of development | fit the network weights |
| ↳ validation | 0.20 of development | model selection, calibration, per-class error rates |
| evaluation |  | generate all test sets |

`tissuemnist` datasets comes with train/val/test splits. **val + test merged** will be used for evaluation only. Training split is used for development, i.e. it is randomly split into training and validation part. 

## Base model and development data provided to students
The students will get the base CNN predictor which will be trained on the training split and its posterior will be calibrated on the validation spplit. The students will not get the training data, however, they will get the validation split for developing their models, e.g. they can use the data for evaluating their developed predictors before submitting. They will get the validation split (student development data) in the form of images and CSV file with labels (file `dev.csv`). 


## Generating the test batches

The evalution data will be used to generate test batches. A procedure derived from README.md paragraph 6.3 and 6.4 will be used as follows. The test batch contains unlabeled images. The batch size $m$ will be varied in $m\in{\mathcal M}$, where
$$\mathcal M=\{1,\,2,\,5,\,10,\,20,\,50,\,100,\,200,\,500\}$$
This means that the addaptation set will be $m-1$ if the use the Bayesian learning (it is difference compared to the setting in the current code base). 

For each $m\in\mathcal M$ and each trial $j=1,\ldots,N(m)$:
1. Sample $\theta_*\sim p(\theta)$, where $p(\theta)$ being uniform distribution.
2. Sample $m$ labeled examples from the evaluation set **with replacement**,
   with class frequencies following $\theta_*$, giving
   $D=\{(x_1,y_1),\ldots,(x_{m},y_{m})\}$.

The metrics are computed **per $m$, pooled across trials** -- not per trial and then
averaged. 

**Constant curve budget.** The pooled count $N(m)\times m$ would otherwise
grow 500× across the x-axis, so confidence intervals would narrow left-to-right
and panels 1–3 would show apparent improvement that is really just shrinking
error bars. Fix the budget instead:

$$B=2000,\qquad N(m)=\mathrm{clip}\!\left(\left\lceil \frac{B}{m}\right\rceil,\;N_{\min},\;N_{\max}\right),\quad N_{\min}=200,\;N_{\max}=2000$$

Pool all $N(m)m$ triplets; if that exceeds $B$, subsample exactly $B$ of
them, **stratified so each trial contributes the same number**. Every curve
then has exactly $B$ points, and every $m$ has at least $N_{\min}$ trials to
bootstrap over.

The test batches will be generated 3 times, resulting in public, private, and ignore test batches. The public and private sets will be generated from disjoint image pools. The public test batches will be used to calculate the metrics shown in the leaderboard during the competition. The private test batches will be used at the end of the competition to avoid overfitting. The ignore test bacthes will help to prevent the students to fine tune on the leaderboard results. The three types of test batches will be differentiated by the column `Usage` in the `solution.csv` which gest values `Public`, `Private`, and `Ignore` accordingly. The `Usage` will have the same value for all images in the same test batch. 

The ground truth needed to evaluate the metrics will be storred in the file `solution.csv`. Specifically, it will contain the ground-truth labels and predictions of the reference predictor. This file will be given to Kaggle but will not be accessible for students. The reference predictor is the plugin Bayes predictor adapted to the true test label prior which known when the test data are generated (formulas for addaptaion are in README.md, Paragraph 2.1). 

For convenience, students also get outputs of the base predictor on all the images which will be provided to them (file `predictions.csv`). They could compute the predictions themself, however, if they do not want to modify the base predictor they do not have to.

## Files to be uploaded to Kaggle

* `images/` -- PNG images 28x28; images from the test batches and development data (which means only the validation split as defined in the table above. The image names will be randomly generate to prevent easy matching to the original tissuemnist database.
* `test_priors.csv` -- List of admissible test label priors. It will be created from the file `priors_tissuemnist_challenge.txt`. 
* `train_prior.csv` -- Training prior estimated as frequencies of the training labels.
* `predictions.csv` -- Predictions of the trained posterior-calibrated NN preditor on images `images/`
* `test.csv` -- List of test inputs. Each input is a batch of images. The number of images in the batch varies from 1 to 500 images.
* `test_batches.csv` -- List of images in each test batch.
* `sample_submission.csv` -- Label predictions and confidence scores for test images.
* `solution.csv` -- Reference predictor output and groundtruth labes required for metric evaluation. The number of images in test baych $m$ can be derived from the file. This file will not be visible to students.
* `dev.csv` -- List of labeled development images.

### Column description

**test_priors.csv**  
|column | description|   
|:-- | :--|  
| id | prior id; 0,...,P-1 |
| p0 | prior of label 0 |
| ... | ... |
| p7  | prior of label 7 |

**train_priors.csv**  
|column | description|   
|:-- | :--|  
| p0 | prior of label 0 |
| ... | ... |
| p7  | prior of label 7 |

**predictions.csv**  
|column | description|   
|:-- | :--|  
| img | image path: `images/<image>.png` | 
| p0 | predicted posterior - label 0 |  
| ... | ... | 
| p7  | predicted posterior - label 7 |  

**test.csv**  
|column | description| 
|:-- | :--| 
| id_test | Identifies the test batch| 
| m | Number of unlabeled images in the batch| 

**test_bacthes.csv**  
|column | description|  
|:-- | :--|  
| row_id  | Identifies the test image |  
| id_test | identifies the test batch
| slot    | 0 to m-1; The order is fixed and carries no meaning. |  
| img     | image path: `images/<image>.png` |  

**solution.csv**  
|column    | description|   
|:--       | :--|  
| row_id   | Identifies the test image |  
| id_test | identifies the test batch
| slot     | 0 to m-1; The order is fixed and carries no meaning. The last highest slot+1 for given id_test equal $m$, and it can be thus used to recover the number of images in the test batch|  
| label    | groundtruth label; 0,1,...,7 | 
| pred_ref | output of the reference predictor; 0,1,...,7 |
| Usage    | Public, Private or Ignore; see description above | 

**sample_submission.csv**  
|column | description|   
|:-- | :--|  
| row_id | Identifies the test image |  
| pred   | predicted label; 0,1,...,7; labels will be output of the base (non-adapted) predictor |  
| confidence | real number representing predictor's confidence; posterior of the class predicted by the base predictor |  

**dev.csv**  
|column | description|   
|:--    | :--|  
| img   | image path: `images/<image>.png` |  
| label | groundtruth label |


# Metric

All predictions at a fixed $m$ are **pooled into a single ranking**. Let
$B$ be the number of pooled predictions, ranked by descending `confidence` under the
permutation $\pi$, $x$ a single input image, and $y$ its true label. Let further $h$ denote the evaluated predictor whose `label`s were submited, $h_*$ the optimal predictor using the true test prior, and let $\ell$ be the 0/1-loss. At rank $k$ the predictor accepts the $k$ least uncertain predictions, and is evaluate by the coverage and the regret:
$$coverage(k)=\frac{k}{B}$$
$$regret(k)=\frac1k\sum_{i=1}^{k}\Big(\ell\big(y_{\pi(i)},h(x_{\pi(i)})\big)-\ell\big(y_{\pi(i)},h(x_{\pi(i)},\theta_*^{(i)})\big)\Big)$$

**Selective regret** measures the loss on accepted inputs against the predictor
*using the true test prior*. 

**Regret at coverage $c$** is the selective regret at a *fixed* coverage
budget:

$$\text{Reg}@c = regret\big(\lceil cB\rceil\big)$$

The **regret at coverage $c=0.8$** will be computed on test batches of sizes:
$$n\in\{1,2,5,10,...,500\}
$$
producing a sequence of $\text{Reg}@c(n)$, $n\in\{1,2,5,10,\ldots,500\}$.

**Average Regret at coverage $c=0.8$**. The final evaluation metric will be the average of  $\text{Reg}@c$ calculated for the varying test batch size, i.e.
$$
  \text{AvgRegAtCoverage} = \frac{1}{|{\mathcal M}|}\sum_{m\in {\mathcal M}}\text{Reg}@c(m)
$$
where $c=0.8$.


# Code

All code for the challenge will be in the subdirectory `challenge/`. The challenge code
base will be self-contained, i.e. if `challenge/` subdirectory is copied elsewhere,
it will run without the need to use code from folders above. 

The challenge code will implement the following functionality:

**Downloading tissuemnist dataset** A script to download and preprocess the tissuemnist dataset.

**Training the base model** 
A script to train the base predictor -- CNN predictor trained on the training split and with the posterior calibrated on the validation split. The base predictor will be given to the studens which will use it to create the reject-option predictor with prior label adaptation. 

**Data preparation** 
A script which will generate all the files to be uploaded to Kaggle (describe in section `Data`).

**Metric evaluation**
A `score` function which accepts a submmision and the solution, calculates the $\text{AvgRegAtCoverage}$. The function will be implemented to jupyter notebook, the template of which is provided by Kaggle: `metric-template.ipynb`. 

In addition, there will be a Python script `evaluate.py` which get a submmision file (e.g. `template_submission.csv`) and the solution file (`solution.csv`) and outputs the $\text{AvgRegAtCoverage} when calling the `score` function. Besides this the script creates a graph evaluating the $\text{AvgReg@c}$ for the different test batch sizes. 

**Optimal solution**
A script which implements the optimal predictor and uses the predictor to create the submittion file. The optimal predictor is uses the Baysian learning predictor with 
the epistemic rejector.

**Predictor evaluator on development data**
A script which will use the development data and the same protocol to generate the test batches and the solution file (without the `Usage` column). These files will be given to students along with the `evaluate.py` script and a scipt which creates the sample submision. Thus, the students will se how to evaluate their predictors before submitting.
