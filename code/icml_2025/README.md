# Epistemic Reject Option Prediction

This repository contains scripts that reproduce the figures and experiments presented in the accompanying paper.

---
## 🛠️ Installation

To generate high-quality figures, **LaTeX** must be installed on your system.
If LaTeX is not available, comment out the line:

```python
plt.style.use('halfstyle.mplstyle')
```

at the beginning of each notebook to avoid errors.

### Steps:

1. **Create a Python environment:**

   ```bash
   python3.12 -m venv venv
   ```

2. **Activate the environment and install dependencies:**

   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Register the environment as a Jupyter kernel:**

   ```bash
   python -m ipykernel install --user --name=venv --display-name "erop environment"
   ```

4. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```
---

## 📓 Notebooks

The following notebooks visualize and compare three types of reject-option predictors on synthetic datasets, as described in the paper:

1. **`figure_aleatoric.ipynb`** – Visualizes the *aleatoric* reject-option predictor.
2. **`figure_total.ipynb`** – Visualizes the *Bayesian* reject-option predictor.
3. **`figure_epistemic.ipynb`** – Visualizes the *epistemic* reject-option predictor.
4. **`synthetic_experiment.ipynb`** – Compares all three predictors on synthetic data by plotting the Area Under the Regret-Coverage (AuReC) curve with respect to the number of training examples.

## StyleGAN3 Experiments
To reproduce the Stylegan3 experiments, please unzip the ICML_2025_stylegan_experiments.zip archive and follow the instructions within.
