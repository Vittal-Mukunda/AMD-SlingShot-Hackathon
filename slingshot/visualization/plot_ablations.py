"""
Visualization script for Day 6 Ablation Studies.
Generates bar chart comparing RL agent performance across 4 conditions:
Standard, No Fatigue, No Shocks, Full Info.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_ablation_results(input_file='results/ablation_results.csv', 
                          output_file='results/ablation_study.png'):
    """
    Generate ablation study plot.
    """
    print("Generating Ablation Study Plot...")
    
    if not os.path.exists(input_file):
        print(f"Error: Results file {input_file} not found. Run evaluation/ablation_studies.py first.")
        return

    # Load data
    df = pd.read_csv(input_file)
    
    # Plot setup
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create colors for conditions
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Bar plot with error bars
    ax = plt.bar(
        df['Condition'], 
        df['Mean_Score'], 
        yerr=df['Std_Dev'], 
        capsize=10, 
        color=colors, 
        alpha=0.8,
        edgecolor='black'
    )
    
    # Add value labels
    for i, rect in enumerate(ax):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width()/2., 
            height + 5,
            f"{height:.1f}",
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )
    
    # Customize plot
    plt.title('Impact of Environmental Dynamics on Agent Performance', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Composite Score', fontsize=14)
    plt.xlabel('Condition', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add interpretation text
    plt.figtext(0.5, -0.05, 
                "Standard: Control | No Fatigue: Testing robustness to tiredness | "
                "No Shocks: Testing robustness to crises | Full Info: Value of perfect information", 
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Ablation Plot saved to {output_file}")

if __name__ == "__main__":
    plot_ablation_results()
