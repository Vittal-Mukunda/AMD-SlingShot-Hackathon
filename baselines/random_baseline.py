"""
B1: Random Assignment Baseline
Randomly selects from valid actions (sanity check - RL must beat this)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class RandomBaseline(BasePolicy):
    """
    Random assignment policy - selects uniformly from valid actions
    
    Purpose: Sanity check baseline (RL must beat random)
    Expected performance: ~30% deadline miss rate
    """
    
    def __init__(self, env: ProjectEnv):
        super().__init__(env)
        self.name = "Random"
    
    def select_action(self, state) -> int:
        """
        Select random valid action
        
        Args:
            state: Current state (unused)
        
        Returns:
            Random valid action index
        """
        valid_actions = self.get_valid_actions()
        
        if len(valid_actions) == 0:
            # No valid actions - this shouldn't happen, but return defer for task 0 as fallback
            return self.encode_action(0, action_type='defer')
        
        return np.random.choice(valid_actions)


if __name__ == "__main__":
    # Unit test
    print("Testing RandomBaseline...")
    
    from environment.project_env import ProjectEnv
    
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    policy = RandomBaseline(env)
    
    state = env.reset()
    total_reward = 0
    
    for t in range(100):
        action = policy.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode ended at t={t}, reason={info['termination_reason']}")
            break
    
    metrics = env.compute_metrics()
    print(f"✓ Random policy: reward={total_reward:.1f}, throughput={metrics['throughput']}, "
          f"deadline_hit={metrics['deadline_hit_rate']:.2f}")
    
    print("RandomBaseline test passed!")
