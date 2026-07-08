import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os

# --- Configuration ---

# Total number of images corresponding to Frac = 1.0
TOTAL_IMAGES = 80000

# Cut off the plot after this many samples to focus on the "Crossover/Z" behavior
ZOOM_LIMIT = 10000 

# Line Styles for Configuration in Comparison plots
DASH_STYLES = {
    'Standard': (None, None),  # Solid
    'Oracle': (2.5, 2.5)       # Dashed
}

def setup_plotting_style():
    style_path = os.path.join("src", "halfstyle.mplstyle")
    
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

    try:
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        palette = {
            'Aleatoric': cycle_colors[2], # Greenish
            'Epistemic': cycle_colors[4], # Blueish
            'Total': cycle_colors[5]      # Reddish
        }
    except (IndexError, KeyError, AttributeError):
        palette = {
            'Total': '#C0392B',
            'Epistemic': '#2980B9',
            'Aleatoric': '#27AE60'
        }
    
    return palette

def format_xaxis(x, pos):
    if x >= 1000:
        val = x / 1000
        if val.is_integer():
            return f'{int(val)}k'
        return f'{val:.1f}k'
    return f'{int(x)}'

def parse_metadata(row):
    name = row['Name']
    is_oracle = 'oracle' in name
    
    if 'farl' in name:
        feature = 'FARL'
    elif 'resnet' in name:
        feature = 'ResNet'
    else:
        feature = 'Other'

    if 'gp_linear' in name:
        method = 'GP (Linear)'
    elif 'gp_rbf' in name:
        method = 'GP (RBF)'
    elif 'blr' in name:
        method = 'BLR'
    else:
        method = 'Other'
        
    return pd.Series([feature, method, is_oracle], index=['Feature', 'Method', 'Is_Oracle'])

def prepare_long_format(df, metric_type):
    cols = [f'AURC_{metric_type}_Total', 
            f'AURC_{metric_type}_Epistemic', 
            f'AURC_{metric_type}_Aleatoric']
    
    id_vars = ['Num_Samples', 'Is_Oracle', 'Fold']
    
    melted = df.melt(id_vars=id_vars, value_vars=cols, 
                     var_name='Uncertainty_Raw', value_name='AURC')
    
    melted['Uncertainty'] = melted['Uncertainty_Raw'].apply(lambda x: x.split('_')[-1])
    
    melted['Configuration'] = melted['Is_Oracle'].map({
        True: 'Oracle', 
        False: 'Standard'
    })
    
    return melted

def generate_plot(df, feature, method, output_dir, mode, palette):
    # Filter Data based on Mode
    if mode == 'oracle':
        subset = df[df['Is_Oracle'] == True].copy()
        suffix = "oracle"
        title_extra = "(Oracle Only)"
    elif mode == 'standard':
        subset = df[df['Is_Oracle'] == False].copy()
        suffix = "standard"
        title_extra = "(No Oracle)"
    else: 
        subset = df.copy()
        suffix = "comparison"
        title_extra = "(Oracle vs Standard)"

    # --- APPLY ZOOM (FILTERING) ---
    subset = subset[subset['Num_Samples'] <= ZOOM_LIMIT]

    if subset.empty:
        return

    # --- MODIFIED LAYOUT: 2 Rows, 1 Column, Shared X ---
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    # Optional: General Title
    # fig.suptitle(f"{feature} - {method} {title_extra}", fontsize=16, y=0.98, weight='bold')
    
    metrics = ['Regret', 'Risk']
    hue_order = ['Total', 'Epistemic', 'Aleatoric'] 
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_data = prepare_long_format(subset, metric)
        
        kwargs = {
            'data': plot_data,
            'x': 'Num_Samples',
            'y': 'AURC',
            'hue': 'Uncertainty',
            'hue_order': hue_order,
            'palette': palette,
            'marker': 'o',
            'markers': False,
            'markersize': 5,       
            'markeredgecolor': 'white', 
            'markeredgewidth': 0.0,
            'linewidth': 2.5,       
            'ax': ax,
            'alpha': 1.0,           
            'err_kws': {'alpha': 0.1} 
        }
        
        if mode == 'comparison':
            kwargs['style'] = 'Configuration'
            kwargs['dashes'] = DASH_STYLES
            continue
        
        
        
        sns.lineplot(**kwargs)
        
        # --- Axis Polish (Matching Code 2 Style) ---
        ax.set_yscale('log') 
        ax.set_xscale('log') # Explicitly set for both
        
        # Grid settings similar to Code 2
        ax.grid(True, which="both", ls="--", alpha=0.6)
        
        # Labels
        # Using newlines to stack text like Code 2
        ax.set_ylabel(f'Area Under\n{metric}-Coverage', fontsize=12)
        ax.set_title("") # Clearing individual titles for cleaner look
        
        # Handle X-axis labeling
        if i == 1:
            # Bottom Plot
            ax.set_xlabel('Number of Training Samples', fontsize=12)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_xaxis))
        else:
            # Top Plot
            ax.set_xlabel('')
            # ax.tick_params(labelbottom=False) # Handled by sharex=True
        
        # Handle Legend
        # Place legend on the top plot (Regret), inside upper right
        if i == 0:
            ax.legend(
                loc='lower left', 
                title="",
                frameon=True,
                fontsize=10,
                ncol=1
            )
        else:
            if ax.get_legend():
                ax.get_legend().remove()
        
    # Modifications to suit the plot to a specific visualization for the paper appendix
    #axes[0].set_ylim([4, 50])
    #axes[1].set_ylim([20, 70])
    #axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    #axes[0].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.0f'))
    #axes[0].set_yticks([4, 5, 6, 7, 8, 9, 10, 20, 40])
    #axes[0].set_yticklabels(["", "5", "", "", "", "", "10", "20", "40"])
    #axes[1].set_yticks([20, 30, 40, 50, 60, 70])
    #axes[1].set_yticklabels(["", "30", "40", "50", "60", ""])
    #axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    #axes[1].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.0f'))

    plt.tight_layout()
    # Adjust spacing to bring plots closer together if desired
    plt.subplots_adjust(hspace=0.05)
    
    clean_feat = feature.lower().replace(" ", "")
    clean_meth = method.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    out_path = os.path.join(output_dir, f"{suffix}_{clean_feat}_{clean_meth}.pdf")
    print(f"Saving {mode} plot to {out_path}...")
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate UQ comparison plots")
    parser.add_argument('csv_file', type=str, help="Path to summary_metrics.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: {args.csv_file} does not exist.")
        return

    palette = setup_plotting_style()
    print("Loading data...")
    df = pd.read_csv(args.csv_file)
    
    meta = df.apply(parse_metadata, axis=1)
    df = pd.concat([df, meta], axis=1)
    df['Num_Samples'] = df['Frac'] * TOTAL_IMAGES
    
    output_dir = os.path.dirname(args.csv_file)
    features = df['Feature'].unique()
    methods = df['Method'].unique()
    
    print(f"Found Features: {features}")
    print(f"Found Methods:  {methods}")
    
    for feat in features:        
        for meth in methods:
            subset = df[(df['Feature'] == feat) & (df['Method'] == meth)]
            if subset.empty: continue
                
            generate_plot(subset, feat, meth, output_dir, mode='comparison', palette=palette)
            generate_plot(subset, feat, meth, output_dir, mode='oracle', palette=palette)
            generate_plot(subset, feat, meth, output_dir, mode='standard', palette=palette)
            
    print("Done.")

if __name__ == "__main__":
    main()