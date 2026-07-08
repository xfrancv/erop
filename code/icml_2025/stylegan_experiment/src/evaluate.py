import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)

def load_oracle_ground_truth(path):
    """
    Loads Oracle Mean and Oracle Aleatoric Variance (True Noise).
    Source of Truth: data/test/predictions.npz
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Oracle file not found at {path}")
    data = np.load(path)
    fnames = data['filenames']
    mu = data['target_mean'] # True Mu
    sigma = data['target_std'] # True Sigma
    
    # Create a dictionary for O(1) access
    # Key: filename, Value: (mu, sigma)
    return {os.path.basename(f): (m, s) for f, m, s in zip(fnames, mu, sigma)}

def load_student_estimates(path):
    """
    Loads student estimates from src/train.py output.
    """
    if not os.path.exists(path):
        return None
        
    try:
        data = np.load(path)
        # Check integrity
        required = ['filenames', 'regression_mean', 'var_epi_student', 'var_ale_student']
        if not all(k in data for k in required):
            return None
            
        return data
    except:
        return None

def get_curve(metric, uncertainty):
    """
    Computes the Risk-Coverage Curve.
    Returns: coverage (0..1), mean_metric_at_coverage
    """
    n = len(metric)
    # Sort by uncertainty (ascending: most confident first)
    idx = np.argsort(uncertainty)
    sorted_metric = metric[idx]
    
    # Coverage: Fraction of data kept (1/N ... N/N)
    cov = np.arange(1, n + 1) / n
    
    # Cumulative Mean of the metric
    # At index i, this is the mean of the top-i most confident samples
    mean_metric = np.cumsum(sorted_metric) / np.arange(1, n + 1)
    
    return cov, mean_metric

def plot_and_save(experiment_name, output_dir, metrics_dict, uncertainties_dict):
    """
    Generates and saves Regret and Risk plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    computed_stats = {} 
    PLOT = False
    
    for m_name, m_vals in metrics_dict.items():
        if PLOT: plt.figure(figsize=(10, 7))
        
        # 1. Optimal Curve (Oracle ordering)
        # If we knew the error perfectly, what would the curve look like?
        # We sort by the metric itself (cheating)
        cov, val = get_curve(m_vals, m_vals)
        if PLOT: plt.plot(cov, val, 'k--', label='Optimal', linewidth=2, alpha=0.6)
        
        # 2. Random Baseline
        if PLOT: plt.hlines(np.mean(m_vals), 0, 1, 'gray', ':', label='Random')
        
        # 3. Strategy Curves
        for u_name, u_vals in uncertainties_dict.items():
            cov, val = get_curve(m_vals, u_vals)
            aurc = np.trapz(val, cov) # Area Under Risk-Coverage
            
            key = f"AURC_{m_name}_{u_name}" # e.g., AURC_Regret_Epistemic
            computed_stats[key] = aurc
            
            # Styling
            if "Total" in u_name:
                ls, lw, c = '-', 3.0, 'firebrick'
            elif "Epistemic" in u_name:
                ls, lw, c = '-', 2.0, 'royalblue'
            elif "Aleatoric" in u_name:
                ls, lw, c = '--', 2.0, 'forestgreen'
            else:
                ls, lw, c = '-.', 1.5, 'gray'
            
            if PLOT: plt.plot(cov, val, label=f'{u_name} (AURC: {aurc:.3f})', linestyle=ls, linewidth=lw, color=c)
            
        if PLOT: plt.title(f'{m_name} vs Coverage\n{experiment_name}')
        if PLOT: plt.xlabel('Coverage (Fraction of Data Retained)')
        if PLOT: plt.ylabel(f'Mean {m_name}')
        if PLOT: plt.legend()
        if PLOT: plt.xlim(0, 1)
        
        # Y-Limit: ignore extreme outliers in the first few percentiles for scaling
        y_max = np.percentile(val, 98) * 1.1
        if PLOT: plt.ylim(0, y_max)
        
        if PLOT: plt.tight_layout()
        if PLOT: plt.savefig(os.path.join(output_dir, f"{m_name}_curve.png"), dpi=150)
        if PLOT: plt.close()
        
    return computed_stats

def evaluate_experiment(exp_path, oracle_map):
    """
    Evaluates a single experiment directory.
    """
    pred_path = os.path.join(exp_path, "predictions.npz")
    student_data = load_student_estimates(pred_path)
    
    if student_data is None:
        return None

    # Unpack Student
    s_fnames = student_data['filenames']
    mu_pred = student_data['regression_mean']
    var_epi = student_data['var_epi_student']
    var_ale = student_data['var_ale_student']
    
    # Metadata
    frac = float(student_data['fraction']) if 'fraction' in student_data else 1.0
    fold = int(student_data['fold']) if 'fold' in student_data else 0

    # Align Data
    valid_mask = []
    mu_true_list = []
    sigma_true_list = []
    
    for f in s_fnames:
        bn = os.path.basename(f)
        if bn in oracle_map:
            valid_mask.append(True)
            m, s = oracle_map[bn]
            mu_true_list.append(m)
            sigma_true_list.append(s)
        else:
            valid_mask.append(False)
            
    if np.sum(valid_mask) == 0:
        return None
        
    mu_pred = mu_pred[valid_mask]
    var_epi = var_epi[valid_mask]
    var_ale = var_ale[valid_mask]
    
    mu_true = np.array(mu_true_list)
    sigma_true = np.array(sigma_true_list)
    
    # --- CALCULATIONS ---
    
    regret = (mu_true - mu_pred) ** 2
    risk = regret + (sigma_true ** 2)
    
    uncertainties = {
        "Total":     var_epi + var_ale,
        "Epistemic": var_epi,
        "Aleatoric": var_ale
    }
    
    # 1. Create the basic results dictionary
    results = {
        "Name": os.path.basename(exp_path),
        "Fold": fold,
        "Frac": frac,
        "MSE_Regret": np.mean(regret),
        "MSE_Risk": np.mean(risk)
    }
    
    # 2. Define metrics for plotting
    metrics = {"Regret": regret, "Risk": risk}
    
    # 3. Generate plots AND get the calculated AURC stats back
    # (Relies on plot_and_save returning a dict like {'AURC_Regret_Epistemic': 12.3, ...})
    aurc_stats = plot_and_save(results["Name"], exp_path, metrics, uncertainties)
    
    # 4. Merge the specific AURC stats into the final results
    results.update(aurc_stats)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    # 1. Load Oracle
    oracle_path = os.path.join(args.data_dir, "test", "predictions.npz")
    print(f"Loading Oracle from {oracle_path}...")
    try:
        oracle_map = load_oracle_ground_truth(oracle_path)
    except Exception as e:
        print(f"Error loading oracle: {e}")
        return

    # 2. Scan Results
    exp_dirs = glob.glob(os.path.join(args.results_dir, "*"))
    print(f"Found {len(exp_dirs)} directories in {args.results_dir}")
    
    summary_list = []
    
    # Note: Import tqdm if you haven't already, or remove tqdm() wrapper if not installed
    from tqdm import tqdm 
    for exp_path in tqdm(sorted(exp_dirs)):
        if not os.path.isdir(exp_path): continue
        
        # print(f"Evaluating {os.path.basename(exp_path)}...") # Optional: commented out to not spam with tqdm
        stats = evaluate_experiment(exp_path, oracle_map)
        if stats:
            summary_list.append(stats)
            
    # 3. Print Summary Table
    if summary_list:
        df = pd.DataFrame(summary_list)
        # Sort by Fraction then Name
        df = df.sort_values(by=["Frac", "Name"])
        
        print("\n" + "="*85)
        # Updated Header to reflect we are showing Total Uncertainty AURC
        print(f"{'Experiment Name':<40} | {'Frac':<5} | {'Regret':<8} | {'Risk':<8} | {'AURC(Tot)':<8}")
        print("-" * 85)
        for _, r in df.iterrows():
            # --- FIX IS HERE: Use 'AURC_Regret_Total' instead of 'AURC_Regret' ---
            print(f"{r['Name']:<40} | {r['Frac']:<5.2f} | {r['MSE_Regret']:.4f}   | {r['MSE_Risk']:.4f}   | {r['AURC_Regret_Total']:.4f}")
        print("="*85 + "\n")
        
        # Save CSV
        out_csv = os.path.join(args.results_dir, "summary_metrics.csv")
        df.to_csv(out_csv, index=False)
        print(f"Summary saved to {out_csv}")
    else:
        print("No valid experiments found.")

if __name__ == "__main__":
    main()