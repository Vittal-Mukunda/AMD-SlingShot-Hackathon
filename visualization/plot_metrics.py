"""
Visualization script for Day 5: The "Money Plot".
Grouped bar chart comparing RL vs Baselines with SEM error bars and significance stars.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

def plot_metrics(
    baseline_file='results/baseline_performance.csv',
    rl_file='results/rl_test_performance.csv',
    output_file='results/money_plot.png',
    metric='composite_score',
    title='Project Management Performance: RL vs Baselines'
):
    """
    Generate the main evaluation plot (Money Plot).
    """
    print("Generating Money Plot...")
    
    if not os.path.exists(baseline_file) or not os.path.exists(rl_file):
        print("Error: Results files not found.")
        return

    # Load data
    df_base = pd.read_csv(baseline_file)
    df_rl = pd.read_csv(rl_file)
    
    # Filter RL for Standard condition
    rl_standard = df_rl[df_rl['condition'] == 'Standard'].copy()
    rl_standard['baseline'] = 'RL Agent' # Treat RL as a "baseline" category for plotting
    
    # Combine data
    # Select relevant columns
    cols = ['baseline', metric]
    df_combined = pd.concat([df_base[cols], rl_standard[cols]], ignore_index=True)
    
    # Order: Random, Greedy, STF, Skill, Hybrid, RL Agent
    order = ['Random', 'Greedy', 'STF', 'Skill', 'Hybrid', 'RL Agent']
    
    # Calculate means and SEMs
    summary = df_combined.groupby('baseline')[metric].agg(['mean', 'sem', 'count']).reindex(order)
    
    # Plot setup
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Bar plot
    ax = sns.barplot(
        x='baseline', y=metric, data=df_combined, order=order,
        capsize=0.1, palette='viridis', ci=None # Plot means only, handle errors manually
    )
    
    # Add error bars manually (since seaborn ci usually implies CI not SEM, though ci=68 is approx SEM)
    # But creating manually allows precise control
    x_coords = range(len(order))
    plt.errorbar(
        x=x_coords, 
        y=summary['mean'], 
        yerr=summary['sem'], 
        fmt='none', 
        c='black', 
        capsize=5
    )
    
    # Add significance stars (RL vs others)
    # Get RL data
    rl_data = rl_standard[metric].values
    max_val = df_combined[metric].max()
    y_start = max_val + (max_val * 0.05)
    
    for i, base_name in enumerate(order[:-1]): # Compare all except RL itself
        if base_name == 'RL Agent': continue
        
        base_data = df_base[df_base['baseline'] == base_name][metric].values
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(rl_data, base_data, equal_var=False, alternative='greater')
        
        # Determine stars
        if p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = 'ns'
            
        if marker != 'ns':
            # Draw bracket
            x1, x2 = i, 5 # 5 is RL Agent index
            y = y_start + (i * max_val * 0.05) # Stagger height
            
            # Draw line
            plt.plot([x1, x1, x2, x2], [y, y+max_val*0.02, y+max_val*0.02, y], lw=1.5, c='black')
            
            # Add text
            plt.text((x1+x2)*0.5, y+max_val*0.02, marker, ha='center', va='bottom', color='black')

    # Auto-annotate if RL surpasses Hybrid baseline
    rl_mean = summary.loc['RL Agent', 'mean'] if 'RL Agent' in summary.index else None
    hybrid_mean = summary.loc['Hybrid', 'mean'] if 'Hybrid' in summary.index else None
    if rl_mean is not None and hybrid_mean is not None and rl_mean > hybrid_mean:
        margin_pct = ((rl_mean - hybrid_mean) / max(abs(hybrid_mean), 1e-6)) * 100
        ax.annotate(
            f'✓ RL Agent surpasses Hybrid by {margin_pct:.1f}%',
            xy=(0.02, 0.97), xycoords='axes fraction',
            fontsize=11, color='darkgreen', fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='darkgreen', alpha=0.9)
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Policy', fontsize=14)
    plt.ylabel('Composite Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Money Plot saved to {output_file}")


if __name__ == "__main__":
    plot_metrics()
