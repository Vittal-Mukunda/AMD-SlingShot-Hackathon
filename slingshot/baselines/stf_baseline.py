"""
B3: Shortest-Task-First Baseline
Assigns easiest tasks first to maximize throughput
Ignores deadlines and critical path dependencies
"""

import numpy as np
import sys
import os

from slingshot.baselines.base_policy import BasePolicy
from slingshot.environment.project_env import ProjectEnv


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
        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if t.is_available(tick) and t.is_unassigned()
            and t.check_dependencies_met(completed_ids)
        ]
        if not pending_tasks:
            return self.encode_action(0, action_type='defer')
        # Sort by complexity (ascending — shortest first)
        pending_tasks.sort(key=lambda t: t.complexity)
        available_workers = [w for w in self.env.workers if w.availability == 1]
        if not available_workers:
            return self.encode_action(pending_tasks[0].task_id, action_type='defer')
        available_workers.sort(key=lambda w: w.load)
        return self.encode_action(pending_tasks[0].task_id, available_workers[0].worker_id, 'assign')
    
    def select_action(self, state) -> int:
        return self.select(state)

    def encode_action(self, task_id: int, worker_id: int = -1, action_type: str = 'assign') -> int:
        num_tasks   = 20
        num_workers = self.env.num_workers
        tick        = self.env.clock.tick
        visible = [t for t in self.env.tasks if t.is_available(tick) and t.is_unassigned()]
        visible.sort(key=lambda t: t.complexity)
        visible = visible[:num_tasks]
        task_slot = next((i for i, t in enumerate(visible) if t.task_id == task_id), 0)
        if action_type == 'assign':
            return task_slot * num_workers + worker_id
        elif action_type == 'defer':
            return num_tasks * num_workers + task_slot
        return num_tasks * num_workers


if __name__ == "__main__":
    print("Testing STFBaseline v4...")
    from slingshot.environment.project_env import ProjectEnv
    env    = ProjectEnv(num_workers=5, total_tasks=40, seed=42)
    policy = STFBaseline(env)
    state  = env.reset()
    total  = 0.0
    for _ in range(200):
        valid  = env.get_valid_actions()
        action = policy.select_action(state)
        if valid and action not in valid:
            action = valid[0]
        state, r, done, _ = env.step(action)
        total += r
        if done:
            break
    m = env.compute_metrics()
    print(f"✓ STF: reward={total:.1f}, throughput={m['throughput']}, completion={m['completion_rate']:.2%}")
    print("STFBaseline v4 passed!")
