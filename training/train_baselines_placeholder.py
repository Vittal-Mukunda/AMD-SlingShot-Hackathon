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
    """
    print(f"Running 5 baselines for {num_episodes} episodes each...")
    
    # Initialize baselines
    baselines = {
        'Random': RandomBaseline(),
        'Greedy': GreedyBaseline(),
        'STF': STFBaseline(),
        'Skill': SkillBaseline(),
        'Hybrid': HybridBaseline()
    }
    
    results = []
    
    # Run each baseline
    for name, policy in baselines.items():
        print(f"\nEvaluating {name} Baseline...")
        
        # Set specific seed for this baseline's run series to ensure fair comparison across baselines
        # But we want the SAME seeds across baselines for paired testing if possible, 
        # or at least consistent seeds. Let's use the provided seed to initialize envs.
        
        policy_returns = []
        
        for ep in tqdm(range(num_episodes)):
            # Reproducible seed for each episode
            episode_seed = seed + ep
            env = ProjectEnv(seed=episode_seed, reward_scale=1.0) # Baselines use unscaled rewards
            state = env.reset()
            policy.reset() # Reset policy internal state if any
            
            episode_return = 0
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                    
                action = policy.select_action(state, valid_actions, env.task_manager, env.workers) 
                # Note: Baselines might need access to env internals (task_manager, workers) 
                # depending on implementation. BasePolicy usually takes state and valid_actions.
                # Let's check BasePolicy signature.
                # If select_action signature is different, I will adjust. 
                # BasePolicy usually: select_action(state, valid_actions)
                # But some heuristic baselines need full env info which is passed in state or separately.
                # Given strict "do not modify environment", I assume baselines are designed to work with 
                # state or have a way to access what they need. 
                # BUT, heuristic baselines often cheat/need env access. 
                # Let's assume standard signature for now, but Hybrid might need more.
                # Actually, looking at typical heuristic implementations, they often take the env object or components.
                # Let's optimistically pass just state/valid_actions, but check if they accept **kwargs or similar.
                # A safer bet for heuristics is often passing the env itself if allowed, or just the standard arguments.
                # Let's assume `select_action(state, valid_actions)` is the interface based on standard RL.
                # However, for things like "Greedy" (least loaded), it needs to know worker loads.
                # The state vector has this.
                # I will stick to `select_action(state, valid_actions)` and hope baselines parse state correctly.
                # Wait, I should check `base_policy.py` text I just viewed.
                pass 
                # Retrying logic in next block after viewing file.
                
                # ACTUAL CODE GENERATION CONTINUED BELOW AFTER CHECKING FILE
            
    # Placeholder - I need to confirm the select_action signature from the previous tool output 
    # before writing the full file.
    pass

# I will write the actual file content in the next turn after verifying the signature.
# This tool call is just to verify the file content, but I'll combine it with the write operation 
# if I'm confident. 
# actually, I'll view the file first in the tool execution and THEN write.
# But I can't view and write in the same turn if I want to use the view result.
# So I will just write a placeholder comment and fix it? No, that's wasteful.
# I'll view `base_policy.py` in the previous turn (I already did call it). 
# I will wait for the output of `view_file` to be sure.
