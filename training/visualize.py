"""
Visualization utilities for training results
Generates learning curves and comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_learning_curve(
    log_path: str = None,
    output_path: str = None,
    figsize: tuple = (12, 6)
):
    """
    Plot DQN learning curve from training log
    
    Args:
        log_path: Path to training_log.csv (default: results/training_log.csv)
        output_path: Path to save plot (default: results/learning_curve.png)
        figsize: Figure size
    """
    # Default paths
    if log_path is None:
        log_path = os.path.join(config.RESULTS_DIR, 'training_log.csv')
    if output_path is None:
        output_path = os.path.join(config.RESULTS_DIR, 'learning_curve.png')
    
    # Load data
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Error: Training log not found at {log_path}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('DQN Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode returns with moving average
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['episode_return'], alpha=0.3, label='Episode Return', color='blue')
    ax1.plot(df['episode'], df['moving_avg_return'], label='Moving Avg (50)', color='red', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (Scaled)')
    ax1.set_title('Learning Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['epsilon'], color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Decay')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Q-values and TD error
    ax3 = axes[1, 0]
    ax3.plot(df['episode'], df['mean_q_value'], label='Mean Q-value', color='purple', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['episode'], df['mean_td_error'], label='TD Error', color='orange', alpha=0.7)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mean Q-value', color='purple')
    ax3_twin.set_ylabel('TD Error', color='orange')
    ax3.set_title('Value Function & TD Error')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    # Plot 4: Task completion metrics
    ax4 = axes[1, 1]
    ax4.plot(df['episode'], df['tasks_completed'], label='Tasks Completed', color='teal', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df['episode'], df['deadline_hit_rate'], label='Deadline Hit Rate', color='crimson', alpha=0.7)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Tasks Completed', color='teal')
    ax4_twin.set_ylabel('Deadline Hit Rate', color='crimson')
    ax4.set_title('Task Metrics')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='teal')
    ax4_twin.tick_params(axis='y', labelcolor='crimson')
    ax4_twin.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning curve saved to {output_path}")


def plot_comparison_with_baselines(
    dqn_log_path: str = None,
    baseline_results: dict = None,
    output_path: str = None,
    figsize: tuple = (10, 6)
):
    """
    Plot DQN learning curve overlaid with baseline performance
    
    Args:
        dqn_log_path: Path to DQN training_log.csv
        baseline_results: Dict of {baseline_name: mean_return}
        output_path: Path to save plot
        figsize: Figure size
    """
    # Default paths
    if dqn_log_path is None:
        dqn_log_path = os.path.join(config.RESULTS_DIR, 'training_log.csv')
    if output_path is None:
        output_path = os.path.join(config.RESULTS_DIR, 'dqn_vs_baselines.png')
    
    # Load DQN data
    try:
        df = pd.read_csv(dqn_log_path)
    except FileNotFoundError:
        print(f"Error: DQN log not found at {dqn_log_path}")
        return
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot DQN
    plt.plot(df['episode'], df['moving_avg_return'], 
             label='DQN (Moving Avg)', color='blue', linewidth=2)
    plt.fill_between(df['episode'], df['episode_return'], df['moving_avg_return'],
                     alpha=0.2, color='blue')
    
    # Plot baselines as horizontal lines
    if baseline_results:
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for (name, value), color in zip(baseline_results.items(), colors):
            plt.axhline(y=value, label=name, linestyle='--', color=color, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Return (Scaled)', fontsize=12)
    plt.title('DQN vs Baselines', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to {output_path}")


if __name__ == "__main__":
    # Test plotting
    plot_learning_curve()
