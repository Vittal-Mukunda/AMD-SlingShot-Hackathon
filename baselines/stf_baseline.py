"""
B3: Shortest-Task-First Baseline
Assigns easiest tasks first to maximize throughput
Ignores deadlines and critical path dependencies
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class STFBaseline(BasePolicy):
    """
    Shortest-Task-First (STF) policy
    
    Strategy:
    - Sort tasks by complexity (easiest first)
    - Assign to least loaded available worker
    - Maximize throughput but ignore deadlines
    
    Weakness: Can starve critical path, ignores deadlines
    Expected performance: High task count, but ~25% deadline miss on urgent tasks
    """
    
    def __init__(self, env: ProjectEnv):
        super().__init__(env)
        self.name = "STF"
    
    def select(self, state) -> int:
        """
        STF assignment: easiest task → least loaded worker
        
        Args:
            state: Current state (unused)
        
        Returns:
            Action index
        """
        # Get pending tasks and sort by complexity (ascending)
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks 
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if len(pending_tasks) == 0:
            return self.encode_action(0, action_type='defer')
        
        # Sort by complexity (ascending)
        pending_tasks = sorted(pending_tasks, key=lambda t: t.complexity)
        
        # Get available workers sorted by load
        available_workers = [w for w in self.env.workers if w.availability == 1]
        
        if len(available_workers) == 0:
            return self.encode_action(pending_tasks[0].task_id, action_type='defer')
        
        available_workers = sorted(available_workers, key=lambda w: w.load)
        
        # Assign easiest task to least loaded worker
        selected_task = pending_tasks[0]
        selected_worker = available_workers[0]
        
        return self.encode_action(selected_task.task_id, selected_worker.worker_id, 'assign')
    
    # Note: typo fix - should be select_action, not select
    def select_action(self, state) -> int:
        return self.select(state)


if __name__ == "__main__":
    # Unit test
    print("Testing STFBaseline...")
    
    from environment.project_env import ProjectEnv
    
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    policy = STFBaseline(env)
    
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
    print(f"✓ STF policy: reward={total_reward:.1f}, throughput={metrics['throughput']}, "
          f"deadline_hit={metrics['deadline_hit_rate']:.2f}")
    
    print("STFBaseline test passed!")
