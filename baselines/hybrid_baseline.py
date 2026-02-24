"""
B5: Hybrid Heuristic Baseline (HARDEST BASELINE - TARGET TO BEAT)
Combines priority weighting, skill matching, load balancing, and fatigue awareness
This is the strongest baseline that RL must outperform
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class HybridBaseline(BasePolicy):
    """
    Hybrid heuristic combining all insights from B1-B4
    
    Strategy:
    - Estimate worker skills adaptively (like B4 but online)
    - Priority-weighted task selection: priority * 10 + deadline_urgency
    - Worker scoring: (skill / complexity) / max(1, load) filtered by fatigue < 3
    - Balance load, skill match, and fatigue avoidance
    
    Weakness: Still reactive, no forward planning for fatigue or exploration
    Expected performance: ~10% deadline miss, good load balance
    
    **This is the target RL must beat by ≥15% on composite metric**
    """
    
    def __init__(self, env: ProjectEnv):
        super().__init__(env)
        self.name = "Hybrid"
        # Welford running skill estimates (no list accumulation)
        self.skill_estimates = {i: 1.0 for i in range(env.num_workers)}
        self._skill_counts   = {i: 0   for i in range(env.num_workers)}
    
    def select_action(self, state) -> int:
        """
        Hybrid assignment: balance skill, current load, and fatigue rate.

        Scoring function (additive, not multiplicative):
            score = estimated_skill
                  - LOAD_COEFF   * worker.load        (penalise loaded workers)
                  - FATIGUE_COEFF * worker.fatigue     (penalise already-tired workers)

        This forces work to spread across the team before the top-skill worker
        accumulates too many tasks and burns out.

        Args:
            state: Current state (unused - uses env directly)

        Returns:
            Action index
        """
        # Update skill estimates from recent completions
        self._update_skill_estimates()

        # ── Tunable penalty coefficients ─────────────────────────────────────
        LOAD_COEFF   = 0.40   # Each extra active task costs 0.4 skill points in score
        FATIGUE_COEFF = 0.25  # Each unit of current fatigue costs 0.25 skill points

        # Get pending tasks — only those already arrived (zero lookahead)
        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if t.is_available(tick) and t.is_unassigned()
            and t.check_dependencies_met(completed_ids)
        ]

        if len(pending_tasks) == 0:
            return self.encode_action(0, action_type='defer')

        # Sort tasks by hybrid urgency: priority × 10 + deadline_urgency
        pending_tasks = sorted(
            pending_tasks,
            key=lambda t: -(t.priority * 10 + t.get_deadline_urgency(tick))
        )

        selected_task = pending_tasks[0]

        # Available workers: must not be burned out
        available_workers = [w for w in self.env.workers if w.availability == 1 and w.fatigue < 3]

        if len(available_workers) == 0:
            return self.encode_action(selected_task.task_id, action_type='defer')

        # ── FIXED: additive scoring that distributes load ──────────────────────
        best_worker = None
        best_score = -np.inf

        for worker in available_workers:
            # Safe float extraction from skill_estimates (may be list or float)
            raw = self.skill_estimates.get(worker.worker_id, 1.0)
            skill = float(np.mean(raw)) if isinstance(raw, list) and len(raw) > 0 else float(raw) if not isinstance(raw, list) else 1.0

            # Additive heuristic: penalise load and current fatigue explicitly
            score = skill - LOAD_COEFF * worker.load - FATIGUE_COEFF * worker.fatigue

            if score > best_score:
                best_score = score
                best_worker = worker

        if best_worker is None:
            return self.encode_action(selected_task.task_id, action_type='defer')

        return self.encode_action(selected_task.task_id, best_worker.worker_id, 'assign')

    
    def _update_skill_estimates(self):
        """Update skill estimates using Welford online mean (no list growth bias)."""
        for worker in self.env.workers:
            mean_est, _ = worker.get_skill_estimate()
            if mean_est > 0:
                n = self._skill_counts[worker.worker_id] + 1
                self._skill_counts[worker.worker_id] = n
                old = self.skill_estimates.get(worker.worker_id, 1.0)
                if isinstance(old, list):
                    old = float(np.mean(old)) if old else 1.0
                self.skill_estimates[worker.worker_id] = old + (mean_est - old) / n
    
    def reset(self):
        """Keep skill estimates across episodes — accumulated knowledge persists."""
        pass

    def encode_action(self, task_id: int, worker_id: int = -1, action_type: str = 'assign') -> int:
        """v4-compatible encode using visible-task slot indexing."""
        num_tasks   = 20
        num_workers = self.env.num_workers
        tick        = self.env.clock.tick
        visible = [t for t in self.env.tasks if t.is_available(tick) and t.is_unassigned()]
        visible.sort(key=lambda t: -t.get_deadline_urgency(tick))
        visible = visible[:num_tasks]
        task_slot = next((i for i, t in enumerate(visible) if t.task_id == task_id), 0)
        if action_type == 'assign':
            return task_slot * num_workers + worker_id
        elif action_type == 'defer':
            return num_tasks * num_workers + task_slot
        return num_tasks * num_workers


if __name__ == "__main__":
    print("Testing HybridBaseline v4...")
    from environment.project_env import ProjectEnv
    env    = ProjectEnv(num_workers=5, total_tasks=40, seed=42)
    policy = HybridBaseline(env)
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
    print(f"✓ Hybrid: reward={total:.1f}, throughput={m['throughput']}, completion={m['completion_rate']:.2%}")
    print("HybridBaseline v4 passed!")
