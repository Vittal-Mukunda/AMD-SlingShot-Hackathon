"""
B2: Greedy Least-Loaded Worker Baseline (v4 — updated for dynamic arrivals)

Assigns the highest-priority AVAILABLE task to the least-loaded available worker.
Uses env.clock.tick for zero-lookahead task visibility.
"""

import numpy as np
import sys
import os

from slingshot.baselines.base_policy import BasePolicy
from slingshot.environment.project_env import ProjectEnv


class GreedyBaseline(BasePolicy):
    """
    Greedy load-balancing policy.

    Strategy:
      - Sort arrived & unassigned tasks by priority (DESC), then deadline urgency (DESC)
      - Assign highest-priority task to the least-loaded available worker
    """

    def __init__(self, env: ProjectEnv):
        super().__init__(env)
        self.name = "Greedy"

    def select_action(self, state) -> int:
        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]

        pending = [
            t for t in self.env.tasks
            if t.is_available(tick) and t.is_unassigned()
            and t.check_dependencies_met(completed_ids)
        ]

        if not pending:
            return self.encode_action(0, action_type='defer')

        pending.sort(key=lambda t: (-t.priority, -t.get_deadline_urgency(tick)))

        available = [w for w in self.env.workers if w.availability == 1]
        if not available:
            return self.encode_action(pending[0].task_id, action_type='defer')

        available.sort(key=lambda w: w.load)
        return self.encode_action(pending[0].task_id, available[0].worker_id, 'assign')

    def encode_action(self, task_id: int, worker_id: int = -1, action_type: str = 'assign') -> int:
        """v4-compatible encode using visible-task slot indexing."""
        num_tasks   = 20
        num_workers = self.env.num_workers
        tick        = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]

        visible = [t for t in self.env.tasks if t.is_available(tick) and t.is_unassigned()]
        visible.sort(key=lambda t: -t.get_deadline_urgency(tick))
        visible = visible[:num_tasks]

        task_slot = next((i for i, t in enumerate(visible) if t.task_id == task_id), 0)

        if action_type == 'assign':
            return task_slot * num_workers + worker_id
        elif action_type == 'defer':
            return num_tasks * num_workers + task_slot
        elif action_type == 'escalate':
            return num_tasks * num_workers + num_tasks + task_slot
        return num_tasks * num_workers
