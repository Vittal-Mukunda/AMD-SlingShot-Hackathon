"""
B4: Static Skill-Matching Baseline
Pre-computes skill estimates from initial episodes, then matches tasks to workers by skill
Ignores dynamic fatigue state
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class SkillBaseline(BasePolicy):
    """
    Static skill-matching policy
    
    Strategy:
    - Estimate worker skills from first N episodes (observation phase)
    - Match tasks to workers by skill/complexity ratio
    - Prioritize by task priority
    
    Weakness: Static estimates don't adapt, ignores fatigue
    Expected performance: Good quality, but ~15% deadline miss due to overload
    """
    
    def __init__(self, env: ProjectEnv, observation_episodes: int = None):
        super().__init__(env)
        self.name = "Skill"
        
        self.observation_episodes = observation_episodes or config.BASELINE_SKILL_ESTIMATION_EPISODES
        self.skill_estimates = None  # Will be computed after observation
        self.episodes_observed = 0
        self.is_observing = True
    
    def observe_episode(self, env: ProjectEnv):
        """
        Observe one episode to gather skill data
        
        Args:
            env: Environment to observe
        """
        # After episode, extract skill estimates from workers
        for worker_id, worker in enumerate(env.workers):
            if len(worker.completion_history) > 0:
                # Estimate skill from completion history
                mean_skill, _ = worker.get_skill_estimate()
                
                if self.skill_estimates is None:
                    self.skill_estimates = {i: [] for i in range(env.num_workers)}
                
                self.skill_estimates[worker_id].append(mean_skill)
        
        self.episodes_observed += 1
        
        if self.episodes_observed >= self.observation_episodes:
            # Finish observation phase
            self.is_observing = False
            # Compute final skill estimates (mean across episodes)
            for worker_id in range(env.num_workers):
                if len(self.skill_estimates[worker_id]) > 0:
                    self.skill_estimates[worker_id] = np.mean(self.skill_estimates[worker_id])
                else:
                    self.skill_estimates[worker_id] = 1.0  # Default if no data
    
    def select_action(self, state) -> int:
        """
        Skill-matched assignment: assign to the available worker with the
        highest estimated skill who currently has spare capacity.

        Uses an explicit max() loop — no numpy argsort, no list re-indexing —
        so the winning worker's original W-0…W-4 ID is never lost.

        Args:
            state: Current state (unused)

        Returns:
            Encoded action index
        """
        # Fallback to greedy during observation phase
        if self.is_observing or self.skill_estimates is None:
            return self._greedy_fallback()

        # --- Build unassigned pending task list --------------------------------
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks
            if not t.is_completed
            and not t.is_failed
            and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]

        if not pending_tasks:
            return self.encode_action(0, action_type='defer')

        # Sort: highest priority first, then tightest deadline
        pending_tasks = sorted(
            pending_tasks,
            key=lambda t: (-t.priority, -t.get_deadline_urgency(self.env.current_timestep))
        )
        selected_task = pending_tasks[0]

        # --- Stage 1: capacity-limited available subset -----------------------
        # availability == 1 (not burned out) AND load is below overload threshold
        available_workers = [
            w for w in self.env.workers
            if w.availability == 1 and w.load < config.OVERLOAD_THRESHOLD
        ]

        if not available_workers:
            # All workers burned out or at capacity — defer
            return self.encode_action(selected_task.task_id, action_type='defer')

        # --- Stage 2: explicit max() over the available subset ----------------
        # Helper: safely resolve skill estimate to float for any worker
        def get_skill(worker):
            raw = self.skill_estimates.get(worker.worker_id, 1.0)
            if isinstance(raw, list):
                return float(np.mean(raw)) if raw else 1.0
            return float(raw)

        # Explicit loop — preserves original worker.worker_id at all times
        best_worker = None
        best_skill  = -1.0
        for w in available_workers:
            skill = get_skill(w)
            if skill > best_skill:
                best_skill  = skill
                best_worker = w          # Retain the original Worker object

        # best_worker.worker_id is the original W-0…W-4 identifier
        return self.encode_action(selected_task.task_id, best_worker.worker_id, 'assign')


    
    def _greedy_fallback(self) -> int:
        """
        Fallback to greedy policy during observation phase
        
        Returns:
            Greedy action
        """
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks 
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if len(pending_tasks) == 0:
            return self.encode_action(0, action_type='defer')
        
        pending_tasks = sorted(pending_tasks, key=lambda t: -t.priority)
        available_workers = [w for w in self.env.workers if w.availability == 1]
        
        if len(available_workers) == 0:
            return self.encode_action(pending_tasks[0].task_id, action_type='defer')
        
        available_workers = sorted(available_workers, key=lambda w: w.load)
        
        return self.encode_action(pending_tasks[0].task_id, available_workers[0].worker_id, 'assign')
    
    def reset(self):
        """
        Reset observation data
        """
        # Don't reset skill estimates once learned
        pass


if __name__ == "__main__":
    # Unit test
    print("Testing SkillBaseline...")
    
    from environment.project_env import ProjectEnv
    
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    policy = SkillBaseline(env, observation_episodes=5)
    
    # Observation phase
    print("Observation phase (5 episodes)...")
    for ep in range(5):
        state = env.reset()
        for t in range(100):
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            if done:
                break
        policy.observe_episode(env)
        print(f"  Episode {ep+1}: observed")
    
    print(f"Skill estimates: {policy.skill_estimates}")
    
    # Test with learned skills
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
    print(f"✓ Skill policy: reward={total_reward:.1f}, through put={metrics['throughput']}, "
          f"deadline_hit={metrics['deadline_hit_rate']:.2f}, quality={metrics['quality_score']:.2f}")
    
    print("SkillBaseline test passed!")
