# Epistemic Reject Option Prediction

Research code that learns the **test label prior from unlabeled test data** by
Bayesian inference, and uses it to correct a discriminative classifier under
*label shift* (a.k.a. prior / target shift).

The outputs are the figures and tables presented in the paper.

For the method, the metrics, the datasets and the command-line options, see
[DESCRIPTION.md](DESCRIPTION.md).

## Replicating experiments in the paper

Run the steps in order; each one needs the outputs of the previous.

### 1. Set up the environment

Requires `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm` and
`torch`/`torchvision`. The script creates a `.venv/` and installs them; every
other script activates it.

```
./setup_environment.sh
```

### 2. Download the data

```
./download_datasets.sh
```

Downloads all eight datasets into `data/` and writes an HTML report per dataset
to `data/reports/`.

### 3. Run the experiments

```
./run_all_base_pred_training.sh          # every dataset, calibrated + nocalib
./run_all_rejopt_eval.sh                 # then every evaluation, all three modes
```

Training uses CUDA; if no GPU is available, change `DEVICE=cuda` to
`DEVICE=cpu` in the header of `run_base_pred_training.sh`. Both scripts take an
optional `sbatch` argument that submits each run as a separate Slurm job
instead of running them sequentially in the current shell
(`./run_all_base_pred_training.sh sbatch`). Results land in
`runs/<dataset>[_nocalib]/<timestamp>/`.

### 4. Create the tables and figures

```
./make_summary_table.sh      # -> summary_table.txt
./make_ablation_table.sh     # -> summary_table_ablation.txt
./make_figures.sh            # -> figures/rc-curves/<dataset>/*.pdf
```

Each script picks up the **latest** run of every dataset automatically, so
re-running an experiment and then re-running these is enough to refresh the
paper's numbers.
