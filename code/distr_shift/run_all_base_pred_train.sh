#!/bin/bash

python run_base_predictor_exp.py bloodmnist runs/bloodmnist --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py cifar10 runs/cifar10 --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py dermamnist runs/dermamnist --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py fashion_mnist runs/fashion_mnist --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py cifar100 runs/cifar100 --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py organamnist runs/organamnist --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py organsmnist runs/organsmnist --epochs 30 --device cuda --calibration bcts
python run_base_predictor_exp.py tissuemnist runs/tissuemnist --epochs 30 --device cuda --calibration bcts





