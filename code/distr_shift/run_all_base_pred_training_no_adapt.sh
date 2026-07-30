#!/bin/bash

source .venv/bib/activate

python run_base_predictor_exp.py bloodmnist runs/bloodmnist_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py cifar10 runs/cifar10_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py dermamnist runs/dermamnist_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py fashion_mnist runs/fashion_mnist_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py cifar100 runs/cifar100_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py organamnist runs/organamnist_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py tissuemnist runs/tissuemnist_nocalib --epochs 30 --device cuda
python run_base_predictor_exp.py organsmnist runs/organsmnist_nocalib --epochs 30 --device cuda







