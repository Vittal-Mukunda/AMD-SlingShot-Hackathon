"""
Script to run and evaluate all baselines (Random, Greedy, STF, Skill, Hybrid)
Generates baseline performance data for Day 5 statistical comparison.
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from baselines.random_baseline import RandomBaseline
from baselines.greedy_baseline import GreedyBaseline
from baselines.stf_baseline import STFBaseline
from baselines.skill_baseline import SkillBaseline
from baselines.hybrid_baseline import HybridBaseline
from utils.metrics import compute_composite_score

def run_baselines(num_episodes=200, seed=42, output_file='results/baseline_performance.csv'):
    """
    Run all baselines for specified number of episodes and save metrics.
    
    Args:
        num_episodes: Number of evaluation episodes per baseline
        seed: Base seed for reproducibility
        output_file: Path to save results CSV
    """
    print(f"Running 5 baselines for {num_episodes} episodes each...")
    
    # Initialize a dummy env to initialize policies
    # Policies will have their env updated each episode
    dummy_env = ProjectEnv(seed=seed)
    
    baselines = {
        'Random': RandomBaseline(dummy_env),
        'Greedy': GreedyBaseline(dummy_env),
        'STF': STFBaseline(dummy_env),
        'Skill': SkillBaseline(dummy_env),
        'Hybrid': HybridBaseline(dummy_env)
    }
    
    results = []
    
    for name, policy in baselines.items():
        print(f"\nEvaluating {name} Baseline...")
        
        # Observation Phase for Skill/Hybrid
        if hasattr(policy, 'is_observing') and policy.is_observing:
            print(f"  Running observation phase for {name}...")
            obs_episodes = getattr(policy, 'observation_episodes', 10)
            
            for i in range(obs_episodes):
                # Use different seeds for observation to learn general skills
                obs_seed = seed + 10000 + i
                env = ProjectEnv(seed=obs_seed, reward_scale=1.0)
                policy.env = env # Update policy's environment reference
                state = env.reset()
                
                done = False
                while not done:
                    # During observation, policy might use random or greedy actions
                    # select_action usually handles this logic internally based on is_observing
                    valid_actions = env.get_valid_actions()
                    if not valid_actions:
                        break
                    
                    action = policy.select_action(state)
                    state, reward, done, info = env.step(action)
                
                # Update skill estimates
                if hasattr(policy, 'observe_episode'):
                    policy.observe_episode(env)
            
            print(f"  Observation complete. Learned estimates: {len(getattr(policy, 'skill_estimates', []))} workers")

        # Evaluation Phase
        policy_metrics = []
        
        for ep in tqdm(range(num_episodes), desc=f"{name}"):
            # Reproducible seed for evaluation (Same sequence for all baselines)
            episode_seed = seed + ep  # Seeds 42, 43, ..., 241
            
            env = ProjectEnv(seed=episode_seed, reward_scale=1.0)
            policy.env = env # Crucial: Update env reference
            state = env.reset()
            
            # Reset policy ephemeral state if any (but keep learned skills)
            policy.reset() 
            
            episode_reward = 0
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                    
                action = policy.select_action(state)
                state, reward, done, info = env.step(action)
                episode_reward += reward
            
            # Compute metrics
            metrics = env.compute_metrics()
            
            # Add composite score
            # Note: ProjectEnv.metrics might differ slightly from what compute_composite_score expects
            # metrics['deadline_misses'] needs to be derived.
            # env.metrics has 'deadline_hit_rate'. 
            # tasks_completed = throughput
            # total_finished = completed + failed
            # deadline_misses = failed_tasks
            # Let's check env.failed_tasks
            
            deadline_misses = len(env.failed_tasks)
            overload_events = metrics['overload_events']
            avg_delay = metrics['avg_delay']
            throughput = metrics['throughput']
            
            metric_dict = {
                'tasks_completed': throughput,
                'avg_delay': avg_delay,
                'overload_events': overload_events,
                'deadline_misses': deadline_misses
            }
            
            composite_score = compute_composite_score(metric_dict)
            
            results.append({
                'baseline': name,
                'episode': ep,
                'return': episode_reward,
                'composite_score': composite_score,
                'throughput': throughput,
                'deadline_hit_rate': metrics['deadline_hit_rate'],
                'avg_delay': avg_delay,
                'overload_events': overload_events,
                'bad_assignment_rate': 0.0 # Placeholder or calculate if possible
            })
            
    # Save results
    df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"\nBaseline results saved to {output_file}")
    
    # Print summary
    summary = df.groupby('baseline')[['composite_score', 'return', 'deadline_hit_rate']].mean()
    print("\nBaseline Performance Summary:")
    print(summary)

if __name__ == "__main__":
    run_baselines()
