"""
Script to perform statistical tests (Welch's t-test) and effect size analysis (Cohen's d).
Compares RL Agent (Standard Condition) vs Baselines.
Generates statistical summary for Day 5 defense.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

def compute_cohens_d(group1, group2):
    """
    Compute Cohen's d effect size for two independent samples.
    d = (mean1 - mean2) / pooled_std
    averaged std = sqrt((std1^2 + std2^2) / 2)
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation (assuming equal sample sizes ~200)
    # n1, n2 = len(group1), len(group2)
    # var1, var2 = std1**2, std2**2
    # pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Using simple formula for similar sample sizes or just pooled
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    return (mean1 - mean2) / pooled_std

def run_statistical_tests(
    baseline_file='results/baseline_performance.csv',
    rl_file='results/rl_test_performance.csv',
    output_file='results/statistical_summary.csv',
    metric='composite_score'
):
    """
    Run Welch's t-tests comparing RL vs Baselines.
    Applying Bonferroni correction for 5 comparisons (target alpha 0.05 / 5 = 0.01).
    """
    print("Running Statistical Tests...")
    
    if not os.path.exists(baseline_file) or not os.path.exists(rl_file):
        print("Error: Results files not found. Run training/evaluation scripts first.")
        return

    df_base = pd.read_csv(baseline_file)
    df_rl = pd.read_csv(rl_file)
    
    # Filter RL for Standard condition
    rl_data = df_rl[df_rl['condition'] == 'Standard'][metric].values
    
    if len(rl_data) == 0:
        print("Error: No RL data for 'Standard' condition found.")
        return

    baselines = df_base['baseline'].unique()
    results = []
    
    # Bonferroni correction
    num_comparisons = len(baselines)
    alpha_target = 0.05
    alpha_corrected = alpha_target / num_comparisons
    
    print(f"Comparisons: {num_comparisons}")
    print(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")
    
    for baseline in baselines:
        base_data = df_base[df_base['baseline'] == baseline][metric].values
        
        # Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(rl_data, base_data, equal_var=False, alternative='greater') 
        # alternative='greater' tests if RL > Baseline
        
        # Cohen's d
        d = compute_cohens_d(rl_data, base_data)
        
        # Significance
        significant = p_val < alpha_corrected
        
        results.append({
            'Comparison': f'RL vs {baseline}',
            'Metric': metric,
            'T-Statistic': t_stat,
            'P-Value': p_val,
            'Cohen-d': d,
            'Significant': significant,
            'Alpha-Corrected': alpha_corrected
        })
        
        print(f"\nRL vs {baseline}:")
        print(f"  Means: RL={np.mean(rl_data):.2f}, Base={np.mean(base_data):.2f}")
        print(f"  t={t_stat:.4f}, p={p_val:.4e}")
        print(f"  d={d:.4f} ({'Large' if abs(d)>0.8 else 'Medium' if abs(d)>0.5 else 'Small'})")
        print(f"  Significant? {'YES' if significant else 'NO'}")

    # Save summary
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_results.to_csv(output_file, index=False)
    print(f"\nStatistical summary saved to {output_file}")


if __name__ == "__main__":
    run_statistical_tests()
