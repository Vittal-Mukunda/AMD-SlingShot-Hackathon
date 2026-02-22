"""
Worker class with dynamic state: load, fatigue, availability, and hidden skills
Implements fatigue accumulation, burnout, and recovery mechanics
"""

import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Worker:
    """
    Represents a worker agent with dynamic state and hidden skill level
    
    Attributes:
        worker_id (int): Unique identifier
        load (int): Number of currently assigned tasks
        fatigue (float): Fatigue level [0, 3]: 0=fresh, 1=tired, 2=exhausted, 3=burnout
        availability (int): Binary 0=unavailable, 1=available
        true_skill (float): Hidden ground truth skill multiplier [0.6, 1.4]
        assigned_tasks (List[int]): List of task IDs currently assigned
        completion_history (List[Tuple]): History of (task_complexity, completion_time, quality)
        burnout_timer (int): Timesteps remaining until recovery from burnout
    """
    
    def __init__(self, worker_id: int, skill: float = None):
        """
        Initialize worker with random skill and differentiated hidden parameters.
        
        Args:
            worker_id: Unique worker identifier
            skill: Optional specific skill level (for testing), otherwise random
        """
        self.worker_id = worker_id
        self.load = 0
        self.fatigue = 0.0
        self.availability = 1
        self.assigned_tasks = []
        self.completion_history = []
        self.burnout_timer = 0
        
        # Hidden skill (only observed indirectly through task completion quality)
        if skill is not None:
            self.true_skill = skill
        else:
            self.true_skill = np.random.uniform(config.SKILL_MIN, config.SKILL_MAX)
        
        # ── Differentiated hidden parameters (unique per worker, hidden from agents) ──
        # How fast this worker accumulates fatigue when overloaded (sampled around global default)
        self.fatigue_rate = np.clip(
            np.random.normal(config.FATIGUE_ACCUMULATION_RATE, 0.05), 0.05, 0.5
        )
        # How fast this worker recovers when idle
        self.recovery_rate = np.clip(
            np.random.normal(config.FATIGUE_RECOVERY_RATE, 0.03), 0.02, 0.25
        )
        # Speed multiplier: affects how quickly they process tasks (around 1.0)
        self.speed_multiplier = np.clip(
            np.random.normal(1.0, 0.15), 0.6, 1.5
        )
        # Personal burnout threshold (how resilient they are to burnout)
        self.burnout_resilience = np.clip(
            np.random.normal(config.FATIGUE_THRESHOLD, 0.2), 1.8, 3.0
        )
    
    def assign_task(self, task_id: int):
        """
        Assign a task to this worker
        
        Args:
            task_id: ID of task being assigned
        """
        if self.availability == 0:
            raise ValueError(f"Worker {self.worker_id} is unavailable")
        
        self.assigned_tasks.append(task_id)
        self.load += 1
    
    def complete_task(self, task_id: int, complexity: int) -> Tuple[float, float]:
        """
        Complete a task and return completion time and quality
        
        Args:
            task_id: ID of completed task
            complexity: Task complexity level
            
        Returns:
            Tuple of (completion_time, quality_score)
        """
        if task_id not in self.assigned_tasks:
            raise ValueError(f"Task {task_id} not assigned to worker {self.worker_id}")
        
        # Remove task from assigned list
        self.assigned_tasks.remove(task_id)
        self.load = max(0, self.load - 1)
        
        # Compute base completion time (affected by skill AND per-worker speed multiplier)
        expected_time = complexity / (self.true_skill * self.speed_multiplier)
        
        # Add fatigue penalty
        fatigue_multiplier = 1.0 + 0.5 * self.fatigue
        expected_time *= fatigue_multiplier
        
        # Add stochastic noise (truncated normal)
        noise_std = config.COMPLETION_TIME_NOISE * expected_time
        completion_time = np.random.normal(expected_time, noise_std)
        completion_time = np.clip(completion_time, 0.5 * expected_time, 2.0 * expected_time)
        
        # Compute quality score [0, 1]
        skill_quality = min(1.0, self.true_skill / complexity)
        fatigue_penalty = 1.0 - config.FATIGUE_QUALITY_PENALTY * self.fatigue
        quality_score = skill_quality * fatigue_penalty
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        # Record in history (observable to agent)
        self.completion_history.append((complexity, completion_time, quality_score))
        
        return completion_time, quality_score
    
    def update_fatigue(self):
        """
        Update fatigue based on current load (called every timestep)
        Non-linear dynamics: overload accelerates fatigue, idle recovers
        """
        if self.burnout_timer > 0:
            # Worker is in burnout recovery
            self.burnout_timer -= 1
            if self.burnout_timer == 0:
                self.availability = 1
                self.fatigue = 1.0  # Return as "tired" not fresh
            return
        
        # Fatigue accumulation when overloaded (uses per-worker fatigue_rate)
        if self.load > config.OVERLOAD_THRESHOLD:
            # Stochastic fatigue increase (higher probability when already tired)
            fatigue_prob = 0.3 + 0.1 * self.fatigue  # 30-60% chance
            if np.random.rand() < fatigue_prob:
                self.fatigue = min(3.0, self.fatigue + self.fatigue_rate)
        
        # Recovery when idle (uses per-worker recovery_rate)
        elif self.load == 0:
            self.fatigue = max(0.0, self.fatigue - self.recovery_rate)
        
        # Check for burnout (uses per-worker burnout_resilience threshold)
        if self.fatigue >= self.burnout_resilience:
            self.trigger_burnout()
    
    def trigger_burnout(self):
        """
        Trigger burnout state: worker becomes unavailable
        """
        self.availability = 0
        self.fatigue = 3.0
        self.burnout_timer = config.BURNOUT_RECOVERY_TIME
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get observable state representation for this worker
        
        Returns:
            3-dim vector: [load_normalized, fatigue_level, availability]
        """
        load_normalized = self.load / config.MAX_WORKER_LOAD
        fatigue_level = self.fatigue / 3.0  # Normalize to [0, 1]
        return np.array([load_normalized, fatigue_level, float(self.availability)])
    
    def get_skill_estimate(self) -> Tuple[float, float]:
        """
        Estimate skill from completion history (used by baselines)
        Returns mean and uncertainty
        
        Returns:
            Tuple of (estimated_skill_mean, skill_uncertainty)
        """
        if len(self.completion_history) == 0:
            # No data: return prior mean with high uncertainty
            return 1.0, 0.4
        
        # Simple estimator: skill ≈ complexity / completion_time
        skill_estimates = []
        for complexity, time, quality in self.completion_history:
            # Adjust for fatigue (approximate)
            estimated_skill = complexity / time * 1.25  # Rough fatigue correction
            skill_estimates.append(estimated_skill)
        
        mean_skill = np.mean(skill_estimates)
        uncertainty = np.std(skill_estimates) / np.sqrt(len(skill_estimates))
        
        return mean_skill, uncertainty
    
    def get_hidden_profile(self) -> dict:
        """
        Return all hidden variables for this worker.
        FOR OBSERVER DISPLAY ONLY — never called by agents or included in state.
        
        Returns:
            Dict of hidden worker parameters
        """
        return {
            'worker_id': self.worker_id,
            'true_skill': round(float(self.true_skill), 3),
            'fatigue_rate': round(float(self.fatigue_rate), 3),
            'recovery_rate': round(float(self.recovery_rate), 3),
            'speed_multiplier': round(float(self.speed_multiplier), 3),
            'burnout_resilience': round(float(self.burnout_resilience), 3),
        }
    
    def reset(self, new_skill: float = None):
        """
        Reset worker state for new episode.
        Hidden parameters (true_skill, fatigue_rate, etc.) are preserved across episodes
        since they represent permanent worker traits.
        
        Args:
            new_skill: Optional new skill level, otherwise keep current
        """
        self.load = 0
        self.fatigue = 0.0
        self.availability = 1
        self.assigned_tasks = []
        self.completion_history = []
        self.burnout_timer = 0
        
        if new_skill is not None:
            self.true_skill = new_skill
        # NOTE: fatigue_rate, recovery_rate, speed_multiplier, burnout_resilience
        # are NOT resampled on reset — they are permanent worker traits.
    
    def __repr__(self):
        return (f"Worker({self.worker_id}, load={self.load}, fatigue={self.fatigue:.2f}, "
                f"avail={self.availability}, skill={self.true_skill:.2f})")


if __name__ == "__main__":
    # Unit test
    print("Testing Worker class...")
    
    # Test 1: Initialize worker
    worker = Worker(worker_id=0, skill=1.0)
    print(f"✓ Initialized: {worker}")
    
    # Test 2: Assign task
    worker.assign_task(task_id=1)
    assert worker.load == 1
    print(f"✓ Task assigned, load={worker.load}")
    
    # Test 3: Complete task
    time, quality = worker.complete_task(task_id=1, complexity=3)
    assert worker.load == 0
    print(f"✓ Task completed: time={time:.2f}, quality={quality:.2f}")
    
    # Test 4: Fatigue accumulation
    for i in range(5):
        worker.assign_task(i)
    worker.update_fatigue()
    print(f"✓ Overload fatigue: {worker.fatigue:.2f}")
    
    # Test 5: Burnout
    worker.fatigue = 2.6
    worker.update_fatigue()
    assert worker.availability == 0
    print(f"✓ Burnout triggered, avail={worker.availability}")
    
    print("\nAll Worker tests passed!")
