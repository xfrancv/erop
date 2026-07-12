# Plan

Implement a script which learns a neural network predictor on given real-life datasets.
The predictors will be later used by additional script which implements the reject-option
predictors in test label addaptation setup. 

# Context

The directory `data/` contains several datasets of classification problems. The task is always to classify an image into Y classes. The goal is to create script which for given dataset trains and evaluates a neural network predictor.

In case the dataset is split into training and test subset:
- split training subset into two parts: i) training part for training the NN predictor and ii) for model selection, i.e. to select the best epoch based on the validation error; use class stratified split
- all test subset will be used for testing.

In case the dataset is split into training, validation and test subset:
- split training subset into two parts: i) training part for training the NN predictor and ii) for model selection, i.e. to select the best epoch based on the validation error; ; use class stratified split
- the validation and test data will be merged and used for testing only.

To evaluate the predictor the script will compute:
- the classification error
- the confusion matrix

The metrics will be evaluated on both training (the fittinf part only) and test data.

The training script will record for each epoch:
- loss function 
- classification error
on both the training and the validation set.

After training the posterior needs to be calibrated. To this end, fit one temperature $T$ by minimizing NLL reusing the the validation split which was used for model-selection during training.

# Task

Implement script run_base_predictor_exp.py which implements the neural network
predictor training and evaluation described above.

**Inputs** 
- name of the dataset (mandatory argument); the name is compatible with the dataset descriptors used by
`download_datasets.py` ;
- output directory (mandatory argument) ; all output files the script produces go to this directory
- portion of training examples used for model validation (optional); default 20% 
- neural network architecture (optional):
  i) LeNet; default for Fashion-MNIST
  ii) ResNet-18 adapted to 32×32 inputs (3×3 stem, no max-pool)), trained from scratch; default for CIFAR-10
  iii) ResNet-18 adapted to 28x28 images), trained from scratch; default for DermaMNIST / BloodMNIST 
- whether to use CPU or GPU (option) ; by default CPU is used
- random seed argument 

**Output**
- file containing:
 i) the training neural network model (best-epoch weights)
 ii) temperature T 
 iii) estimate of the training priors
- TXT report with the evaluation metrics described above
- a figure showing the evolution of the training and validation error and loss


