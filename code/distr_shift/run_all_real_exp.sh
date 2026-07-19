#!/bin/bash
#SBATCH --job-name=erop   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=xfrancv@fel.cvut.cz   # Where to send mail
#SBATCH --mem=20gb                   # Job Memory
#SBATCH --output=./logs/array_%A-%a.log    # Standard output and error log
#SBATCH --error=./logs/error_%A-%a.log    # Standard output and error log
#SBATCH --partition=cpu


source .venv/bin/activate



## blood
python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 1

# hand made setup: class 0 is a decoy with large aleator uncertainty, 3 and 6 are confusable classes
python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0.5 0 0 0.25 0 0 0.25 0

python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05  --test-prior 0.25 0 0 0.4 0.25 0 0.1 0 


python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0.5 0 0 0.25 0 0 0.25 0 --sampler gibbs

python run_real_reject_option_exp.py runs/bloodmnist/model.pt runs/bloodmnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0.45 0.01 0.01 0.25 0.01 0.01 0.25 0.01 --dirichlet 10



## cifar10

python run_real_reject_option_exp.py runs/cifar10/model.pt runs/cifar10/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/cifar10/model.pt runs/cifar10/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/cifar10/model.pt runs/cifar10/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0.5 0 0 0.25 0 0.25 0 0 0 0

## dermamnist

python run_real_reject_option_exp.py runs/dermamnist/model.pt runs/dermamnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/dermamnist/model.pt runs/dermamnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/dermamnist/model.pt runs/dermamnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0 0 0.5 0 0.25 0.25 0 

## fashion

python run_real_reject_option_exp.py runs/fashion_mnist/model.pt runs/fashion_mnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/fashion_mnist/model.pt runs/fashion_mnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/fashion_mnist/model.pt runs/fashion_mnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --test-prior 0.25 0 0.5 0 0 0 0.25 0 0 0


#python run_real_reject_option_exp.py runs/fashion_mnist/model.pt runs/fashion_mnist/ --sweep --sizes 1 2 5 10 50 100 200 500 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
#  --test-prior 0.4 0 0.25 0 0.25 0 0.1 0 0 0

##

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 1

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 3

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 5 --pair-rest-ratio 1 3


python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 10 1 --pair-rest-ratio 1 3

python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 5 1 --pair-rest-ratio 1 3


python run_real_reject_option_exp.py runs/cifar100/model.pt runs/cifar100/ --sweep --sizes 1 2 5 10 50 100 --regret-target 0.0005 0.001 0.005 0.01 0.05 \
  --pair-ratio 1 10 --pair-rest-ratio 1 1 --sampler gibbs --trials 50

