#!/bin/bash

# data: epistemic_showcase_near.json
python run_synth_bayesian_learning_exp.py --config configs/epistemic_showcase_near.json --out-dir figures/epistemic_showcase_near/ \
    --n-eval 2000 --m-train 10000 --n-test 2000

python run_synth_bayesian_learning_exp.py --config configs/epistemic_showcase_near.json --out-dir figures/epistemic_showcase_near/ \
    --n-eval 2000 --m-train 10000 --sweep --sizes 1 2 5 10 20 50 100 200 1000 2000 5000

python run_synth_reject_option_exp.py --config configs/epistemic_showcase_near.json --out-dir figures/epistemic_showcase_near/ \
    --n-eval 2000 --m-train 10000 --n-test 100

python run_synth_reject_option_exp.py --config configs/epistemic_showcase_near.json --out-dir figures/epistemic_showcase_near/ \
    --n-eval 2000  --sweep --trials 10 --m-train 10000 --sizes 1 2 5 10 20 50 100 200 500 1000 2000 5000 


# data: epistemic_showcase.json
python run_synth_bayesian_learning_exp.py --config configs/epistemic_showcase.json --out-dir figures/epistemic_showcase/ \
    --m-train 10000 --n-eval 2000 --n-test 2000

python run_synth_bayesian_learning_exp.py --config configs/epistemic_showcase.json --out-dir figures/epistemic_showcase/ \
    --n-eval 2000 --m-train 10000 --sweep --sizes 1 2 5 10 20 50 100 200 1000 2000 5000

python run_synth_reject_option_exp.py --config configs/epistemic_showcase.json --out-dir figures/epistemic_showcase/ \
    --m-train 10000 --n-test 500 --n-eval 2000

python run_synth_reject_option_exp.py --config configs/epistemic_showcase.json --out-dir figures/epistemic_showcase/ \
    --sweep --trials 10 --m-train 10000 --sizes 1 2 5 10 20 50 100 200 500 1000 2000 5000 --n-eval 2000 \
    --regret-target 0.002 --risk-target 0.001


# data: model1.json
python run_synth_bayesian_learning_exp.py --config configs/model1.json --out-dir figures/model1/ --n-eval 2000 --m-train 10000 --n-test 50

python run_synth_bayesian_learning_exp.py --config configs/model1.json --out-dir figures/model1/ \
    --n-eval 2000 --m-train 10000 --sweep --sizes 1 2 5 10 20 50 100 200 1000 2000 5000

python run_synth_reject_option_exp.py --config configs/model1.json --out-dir figures/model1/ --n-test 50 --n-eval 2000

python run_synth_reject_option_exp.py --config configs/model1.json --out-dir figures/model1/ --n-eval 2000 \
    --sweep --sizes 1 2 5 10 20 50 100 200 500 1000 2000 5000 --trials 10 --m-train 2000


# data: model2.json
python run_synth_bayesian_learning_exp.py --config configs/model2.json --out-dir figures/model2/ --n-eval 2000 --m-train 10000 --n-test 50

python run_synth_bayesian_learning_exp.py --config configs/model2.json --out-dir figures/model2/ \
    --n-eval 2000 --m-train 10000 --sweep --sizes 1 2 5 10 20 50 100 200 1000 2000 5000

python run_synth_reject_option_exp.py --config configs/model2.json --out-dir figures/model2/ --n-test 50 --n-eval 2000

python run_synth_reject_option_exp.py --config configs/model2.json --out-dir figures/model2/ \
    --n-eval 2000 --sweep --sizes 1 2 5 10 20 50 100 200 500 1000 2000 5000 --trials 10 --m-train 2000
