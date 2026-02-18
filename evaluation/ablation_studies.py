"""
Script to run ablation studies for Day 6 analysis.
Compares RL agent performance across 4 conditions:
1. Standard (Control)
2. No Fatigue (Ablation)
3. No Deadline Shocks (Ablation)
4. Full Observability (Information Gain)
"""

import numpy as np
import pandas as pd
import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent
from utils.metrics import compute_composite_score

def run_ablation_studies(model_path='checkpoints/best_model.pth', 
                         output_file='results/ablation_results.csv',
                         num_episodes=50, seed=42):
    """
    Run ablation studies.
    """
    print("Starting Day 6 Ablation Studies...")
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=config.STATE_DIM, 
        action_dim=config.ACTION_DIM,
        device=None
    )
    
    # Attempt to load model
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        agent.load(model_path)
    else:
        print(f"WARNING: Model {model_path} not found!")
        print("Running with UNTRAINED random agent for demonstration purposes.")
        # Continue anyway to verify script logic
    
    # Set to eval mode
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    # Define conditions
    conditions = {
        'Standard': {},  # Default
        'No Fatigue': {'enable_fatigue': False},
        'No Shocks': {'enable_deadline_shocks': False},
        'Full Info': {'fully_observable': True}
    }
    
    results = []
    
    for condition_name, overrides in conditions.items():
        print(f"\nEvaluating Condition: {condition_name}")
        print(f"  Overrides: {overrides}")
        
        scores = []
        
        for ep in tqdm(range(num_episodes)):
            episode_seed = seed + ep
            
            # Initialize env with specific overrides
            env = ProjectEnv(
                seed=episode_seed, 
                reward_scale=1.0,
                config_overrides=overrides
            )
            
            state = env.reset()
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                # Select greedy action
                action = agent.select_action(state, valid_actions, greedy=True)
                
                state, reward, done, info = env.step(action)
            
            # Calculate composite score
            metrics = env.compute_metrics()
            
            # Reconstruct metric dict for composite score function
            metric_dict = {
                'tasks_completed': metrics['throughput'],
                'avg_delay': metrics['avg_delay'],
                'overload_events': metrics['overload_events'],
                'deadline_misses': len(env.failed_tasks)
            }
            
            score = compute_composite_score(metric_dict)
            scores.append(score)
            
        # Log summary stats
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  Mean Score: {mean_score:.2f} ± {std_score:.2f}")
        
        results.append({
            'Condition': condition_name,
            'Mean_Score': mean_score,
            'Std_Dev': std_score
        })
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nAblation results saved to {output_file}")
    print(df)

if __name__ == "__main__":
    run_ablation_studies()
