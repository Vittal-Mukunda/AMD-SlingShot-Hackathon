"""
Script to evaluate the trained RL agent under various conditions.
Conditions: Standard, High Variance, Frequent Shocks, Fixed Seed.
Generates performance data for Day 5 statistical comparison.
"""

import numpy as np
import pandas as pd
import torch
import os
import sys
from tqdm import tqdm
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent
from utils.metrics import compute_composite_score

def evaluate_agent(model_path='checkpoints/best_model.pth', 
                   output_file='results/rl_test_performance.csv',
                   num_episodes=200, seed=42):
    """
    Evaluate RL agent under different conditions.
    """
    print(f"Evaluating RL Agent from {model_path}...")
    
    # Load agent
    agent = DQNAgent(
        state_dim=config.STATE_DIM, 
        action_dim=config.ACTION_DIM,
        device=None # Auto-detect
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model path {model_path} not found. using random agent for testing? No, aborting.")
        # If best_model doesn't exist, try checkpoint_ep2000.pth or similar?
        # Or just list dir to find one.
        # But for now assuming it exists as per Day 4.
        return

    # Set to eval mode (epsilon = 0 for greedy evaluation)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    conditions = {
        'Standard': {},
        'HighVariance': {'COMPLETION_TIME_NOISE': config.COMPLETION_TIME_NOISE * 1.5},
        'FrequentShocks': {'DEADLINE_SHOCK_PROB': 0.3},
        'FixedSeed': {'FIXED_SEED': True}
    }
    
    results = []
    
    # Store original config values to restore later
    original_config = {
        'COMPLETION_TIME_NOISE': config.COMPLETION_TIME_NOISE,
        'DEADLINE_SHOCK_PROB': config.DEADLINE_SHOCK_PROB
    }
    
    for condition_name, params in conditions.items():
        print(f"\nEvaluating Condition: {condition_name}")
        
        # Apply config modifications
        if 'COMPLETION_TIME_NOISE' in params:
            config.COMPLETION_TIME_NOISE = params['COMPLETION_TIME_NOISE']
        if 'DEADLINE_SHOCK_PROB' in params:
            config.DEADLINE_SHOCK_PROB = params['DEADLINE_SHOCK_PROB']
            
        fixed_seed_mode = params.get('FIXED_SEED', False)
        
        condition_metrics = []
        
        for ep in tqdm(range(num_episodes)):
            # Seed logic
            if fixed_seed_mode:
                episode_seed = seed # Constant seed
            else:
                episode_seed = seed + ep # Varying seed
                
            env = ProjectEnv(seed=episode_seed, reward_scale=1.0) # Unscaled for evaluation metrics
            state = env.reset()
            
            episode_return = 0
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                # Select greedy action
                action = agent.select_action(state, valid_actions, greedy=True)
                
                state, reward, done, info = env.step(action)
                episode_return += reward
            
            # Compute metrics
            metrics = env.compute_metrics()
            
            throughput = metrics['throughput']
            avg_delay = metrics['avg_delay']
            overload_events = metrics['overload_events']
            deadline_misses = len(env.failed_tasks)
            
            # Derived metrics
            # Composite score inputs
            metric_dict = {
                'tasks_completed': throughput,
                'avg_delay': avg_delay,
                'overload_events': overload_events,
                'deadline_misses': deadline_misses
            }
            composite_score = compute_composite_score(metric_dict)
            
            results.append({
                'condition': condition_name,
                'episode': ep,
                'return': episode_return,
                'composite_score': composite_score,
                'throughput': throughput,
                'deadline_hit_rate': metrics['deadline_hit_rate'],
                'avg_delay': avg_delay,
                'overload_events': overload_events,
                'deadline_misses': deadline_misses
            })
            
        # Restore config
        config.COMPLETION_TIME_NOISE = original_config['COMPLETION_TIME_NOISE']
        config.DEADLINE_SHOCK_PROB = original_config['DEADLINE_SHOCK_PROB']
        
    # Save results
    df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nRL evaluation results saved to {output_file}")
    
    # Print summary
    summary = df.groupby('condition')[['composite_score', 'return', 'throughput']].mean()
    print("\nRL Agent Performance Summary:")
    print(summary)


if __name__ == "__main__":
    evaluate_agent()
