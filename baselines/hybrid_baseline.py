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
        
        # Maintain running skill estimates (updated online)
        self.skill_estimates = {i: 1.0 for i in range(env.num_workers)}
        self.skill_observations = {i: [] for i in range(env.num_workers)}
    
    def select_action(self, state) -> int:
        """
        Hybrid assignment: balance priority, skill, load, and fatigue
        
        Args:
            state: Current state (unused - uses env directly)
        
       Returns:
            Action index
        """
        # Update skill estimates from recent completions
        self._update_skill_estimates()
        
        # Get pending tasks
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        pending_tasks = [
            t for t in self.env.tasks 
            if not t.is_completed and not t.is_failed and t.assigned_worker is None
            and t.check_dependencies_met(completed_ids)
        ]
        
        if len(pending_tasks) == 0:
            return self.encode_action(0, action_type='defer')
        
        # Sort tasks by hybrid score: priority * 10 + deadline_urgency
        pending_tasks = sorted(
            pending_tasks,
            key=lambda t: -(t.priority * 10 + t.get_deadline_urgency(self.env.current_timestep))
        )
        
        # Select highest priority task
        selected_task = pending_tasks[0]
        
        # Score available workers
        available_workers = [w for w in self.env.workers if w.availability == 1 and w.fatigue < 3]
        
        if len(available_workers) == 0:
            # All workers unavailable or burned out - defer
            return self.encode_action(selected_task.task_id, action_type='defer')
        
        # Worker scoring: (skill / complexity) / max(1, load) with fatigue penalty
        best_worker = None
        best_score = -np.inf
        
        for worker in available_workers:
            skill = self.skill_estimates[worker.worker_id]
            
            # Skill match score
            skill_match = skill / selected_task.complexity
            
            # Load penalty: prefer less loaded workers
            load_penalty = 1.0 / max(1, worker.load)
            
            # Fatigue penalty: avoid tired workers for complex tasks
            fatigue_penalty = 1.0 - 0.2 * worker.fatigue
            
            # Combined score
            score = skill_match * load_penalty * fatigue_penalty
            
            if score > best_score:
                best_score = score
                best_worker = worker
        
        if best_worker is None:
            # Shouldn't happen, but fallback to defer
            return self.encode_action(selected_task.task_id, action_type='defer')
        
        return self.encode_action(selected_task.task_id, best_worker.worker_id, 'assign')
    
    def _update_skill_estimates(self):
        """
        Update skill estimates from worker completion history (online learning)
        """
        for worker in self.env.workers:
            if len(worker.completion_history) > len(self.skill_observations[worker.worker_id]):
                # New completions since last update
                new_completions = worker.completion_history[len(self.skill_observations[worker.worker_id]):]
                
                for complexity, time, quality in new_completions:
                    # Estimate skill from completion time
                    # skill ≈ complexity / (time / fatigue_correction)
                    estimated_skill = complexity / time * 1.25  # Rough fatigue correction
                    self.skill_observations[worker.worker_id].append(estimated_skill)
                
                # Update running average
                if len(self.skill_observations[worker.worker_id]) > 0:
                    self.skill_estimates[worker.worker_id] = np.mean(
                        self.skill_observations[worker.worker_id]
                    )
    
    def reset(self):
        """
        Reset skill estimates for new episode
        """
        # Keep learned estimates across episodes (like a real manager would)
        pass


if __name__ == "__main__":
    # Unit test
    print("Testing HybridBaseline (TARGET TO BEAT)...")
    
    from environment.project_env import ProjectEnv
    
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    policy = HybridBaseline(env)
    
    # Run multiple episodes to test adaptation
    total_rewards = []
    deadline_hits = []
    
    for ep in range(5):
        state = env.reset()
        total_reward = 0
        
        for t in range(100):
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        metrics = env.compute_metrics()
        total_rewards.append(total_reward)
        deadline_hits.append(metrics['deadline_hit_rate'])
        
        print(f"Episode {ep+1}: reward={total_reward:.1f}, deadline_hit={metrics['deadline_hit_rate']:.2f}, "
              f"throughput={metrics['throughput']}, quality={metrics['quality_score']:.2f}")
    
    print(f"\n✓ Hybrid policy (target): avg_reward={np.mean(total_rewards):.1f}, "
          f"avg_deadline_hit={np.mean(deadline_hits):.2f}")
    print("HybridBaseline test passed!")
    print("\n** RL MUST BEAT THIS BY ≥15% ON COMPOSITE SCORE **")
