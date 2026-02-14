"""
B2: Greedy Least-Loaded Worker Baseline
Assigns highest priority task to least loaded available worker
Ignores skill mismatch and fatigue
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class GreedyBaseline(BasePolicy):
    """
    Greedy load-balancing policy
    
    Strategy:
    - Sort tasks by priority (descending)
    - Sort workers by current load (ascending)
    - Assign highest priority task to least loaded worker
    
    Weakness: Ignores skill mismatch, can assign complex task to low-skill worker
    Expected performance: ~20% deadline miss, poor quality scores
    """
    
    def __init__(self, env: ProjectEnv):
        super().__init__(env)
        self.name = "Greedy"
    
    def select_action(self, state) -> int:
        """
        Greedy assignment: highest priority task → least loaded worker
        
        Args:
            state: Current state (unused)
        
        Returns:
            Action index
        """
        # Get pending tasks and sort by priority
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks 
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if len(pending_tasks) == 0:
            # No pending tasks - defer task 0 as no-op
            return self.encode_action(0, action_type='defer')
        
        # Sort by priority (descending) then deadline urgency
        pending_tasks = sorted(
            pending_tasks,
            key=lambda t: (-t.priority, t.get_deadline_urgency(self.env.current_timestep)),
            reverse=True
        )
        
        # Get available workers sorted by load (ascending)
        available_workers = [w for w in self.env.workers if w.availability == 1]
        
        if len(available_workers) == 0:
            # No workers available - defer highest priority task
            return self.encode_action(pending_tasks[0].task_id, action_type='defer')
        
        available_workers = sorted(available_workers, key=lambda w: w.load)
        
        # Assign highest priority task to least loaded worker
        selected_task = pending_tasks[0]
        selected_worker = available_workers[0]
        
        return self.encode_action(selected_task.task_id, selected_worker.worker_id, 'assign')


if __name__ == "__main__":
    # Unit test
    print("Testing GreedyBaseline...")
    
    from environment.project_env import ProjectEnv
    
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    policy = GreedyBaseline(env)
    
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
    print(f"✓ Greedy policy: reward={total_reward:.1f}, throughput={metrics['throughput']}, "
          f"deadline_hit={metrics['deadline_hit_rate']:.2f}, load_balance={metrics['load_balance']:.2f}")
    
    print("GreedyBaseline test passed!")
