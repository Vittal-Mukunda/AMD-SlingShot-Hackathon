"""
Worker class — v4: Heterogeneous workers with realistic workday simulation.

Each worker has unique hidden traits:
  - true_skill          : task quality/expertise multiplier [0.5, 1.5]
  - speed_multiplier    : how quickly they process tasks [0.6, 1.5]
  - fatigue_rate        : how fast fatigue accumulates under overload
  - recovery_rate       : how quickly they recover when idle
  - fatigue_sensitivity : how much fatigue degrades productivity
  - burnout_resilience  : fatigue threshold before burnout

Workday model:
  - hours_worked_today  tracks intra-day effort; resets at day boundary
  - daily_reset()       called at start of each workday (optional carry-over)
  - Productivity decays slightly as hours_worked_today increases (intra-day fatigue)
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Worker:
    """
    Heterogeneous worker with hidden skill, speed, fatigue, and workday state.

    State observable to agents (5-dim):
        [load_norm, fatigue_norm, availability, hours_worked_today_norm, productivity_norm]
    """

    def __init__(self, worker_id: int, skill: Optional[float] = None):
        """
        Initialise worker with randomised hidden heterogeneous parameters.

        Args:
            worker_id : Unique identifier 0…N-1
            skill     : Fixed skill override (for testing); otherwise sampled
        """
        self.worker_id = worker_id

        # ── Observable episode state ──────────────────────────────────────────
        self.load                = 0      # Active tasks currently assigned
        self.fatigue             = 0.0    # [0, 3]
        self.availability        = 1      # 1=available, 0=burned out
        self.hours_worked_today  = 0.0    # Resets each workday
        self.assigned_tasks: List[int] = []
        self.completion_history: List[Tuple] = []   # (complexity, time_h, quality)
        self.burnout_timer       = 0      # Slots remaining in burnout recovery

        # ── Hidden permanent traits (never in state vector) ───────────────────
        self.true_skill = (
            float(skill) if skill is not None
            else float(np.random.uniform(config.SKILL_MIN, config.SKILL_MAX))
        )
        self.speed_multiplier = float(np.clip(
            np.random.normal(config.SPEED_MULT_MEAN, config.SPEED_MULT_STD),
            0.6, 1.5
        ))
        self.fatigue_rate = float(np.clip(
            np.random.normal(config.FATIGUE_RATE_MEAN, config.FATIGUE_RATE_STD),
            0.05, 0.5
        ))
        self.recovery_rate = float(np.clip(
            np.random.normal(config.RECOVERY_RATE_MEAN, config.RECOVERY_RATE_STD),
            0.02, 0.25
        ))
        self.fatigue_sensitivity = float(np.clip(
            np.random.normal(config.FATIGUE_SENS_MEAN, config.FATIGUE_SENS_STD),
            0.05, 0.35
        ))
        self.burnout_resilience = float(np.clip(
            np.random.normal(config.BURNOUT_RESIL_MEAN, config.BURNOUT_RESIL_STD),
            1.8, 3.2
        ))

    # ── Task assignment / completion ──────────────────────────────────────────

    def assign_task(self, task_id: int):
        """Assign a task; raises if worker unavailable."""
        if self.availability == 0:
            raise ValueError(f"Worker {self.worker_id} is unavailable (burnout)")
        self.assigned_tasks.append(task_id)
        self.load += 1

    def complete_task(self, task_id: int, complexity: int) -> Tuple[float, float]:
        """
        Complete a task and return (completion_time_hours, quality_score).

        Completion time depends on:
          - complexity
          - true_skill × speed_multiplier
          - fatigue (slows worker down via fatigue_sensitivity)
          - stochastic noise

        Quality depends on:
          - skill-to-complexity ratio
          - intra-day performance decay (hours_worked_today)
          - fatigue penalty
        """
        if task_id not in self.assigned_tasks:
            raise ValueError(f"Task {task_id} not assigned to worker {self.worker_id}")

        self.assigned_tasks.remove(task_id)
        self.load = max(0, self.load - 1)

        # Effective skill = true_skill × speed_multiplier, degraded by fatigue
        fatigue_slowdown   = 1.0 + self.fatigue_sensitivity * self.fatigue
        effective_speed    = self.true_skill * self.speed_multiplier / fatigue_slowdown
        expected_time_h    = float(complexity) / max(effective_speed, 0.1)

        # Clamp expected time to sensible range
        expected_time_h = np.clip(
            expected_time_h,
            config.MIN_TASK_DURATION_H * 0.5,
            config.MAX_TASK_DURATION_H * 1.5
        )

        # Stochastic completion time
        noise_std      = config.COMPLETION_TIME_NOISE * expected_time_h
        completion_h   = float(np.random.normal(expected_time_h, noise_std))
        completion_h   = float(np.clip(completion_h, 0.5 * expected_time_h, 2.0 * expected_time_h))

        # Accumulate hours worked today
        self.hours_worked_today += completion_h

        # Quality: skill vs complexity, degraded by fatigue + intra-day decay
        base_quality    = min(1.0, self.true_skill / max(complexity, 0.1))
        fatigue_penalty = self.fatigue_sensitivity * self.fatigue
        intraday_penalty = config.INTRADAY_DECAY_RATE * min(self.hours_worked_today, 8.0)
        quality         = base_quality * (1.0 - fatigue_penalty) * (1.0 - intraday_penalty)
        quality         = float(np.clip(quality, 0.0, 1.0))

        self.completion_history.append((complexity, completion_h, quality))
        return completion_h, quality

    # ── Fatigue dynamics ──────────────────────────────────────────────────────

    def update_fatigue(self):
        """
        Update fatigue each time slot.
          - Overloaded → probabilistic fatigue increase (per-worker rate)
          - Idle       → recovery (per-worker rate)
          - At burnout threshold → trigger burnout
        """
        if self.burnout_timer > 0:
            self.burnout_timer -= 1
            if self.burnout_timer == 0:
                self.availability = 1
                self.fatigue      = config.FATIGUE_CARRYOVER * 3.0  # Return partially fatigued
            return

        if self.load > config.MAX_WORKER_LOAD // 2:          # Overloaded threshold
            prob = 0.25 + 0.08 * self.fatigue                # Escalates when already tired
            if np.random.rand() < prob:
                self.fatigue = min(3.0, self.fatigue + self.fatigue_rate)
        elif self.load == 0:                                  # Idle → recover
            self.fatigue = max(0.0, self.fatigue - self.recovery_rate)

        if self.fatigue >= self.burnout_resilience:
            self._trigger_burnout()

    def _trigger_burnout(self):
        """Mark worker as burned out and unavailable for BURNOUT_RECOVERY_TIME slots."""
        self.availability  = 0
        self.fatigue       = 3.0
        self.burnout_timer = config.BURNOUT_RECOVERY_TIME

    def daily_reset(self):
        """
        Reset intra-day state at the start of a new workday.
        Fatigue carries over at FATIGUE_CARRYOVER fraction (default 10%).
        """
        self.hours_worked_today = 0.0
        if self.availability == 1:
            # Partial fatigue carry-over; worker is somewhat refreshed
            self.fatigue = float(np.clip(
                self.fatigue * config.FATIGUE_CARRYOVER,
                0.0, 1.5
            ))

    # ── State observation ─────────────────────────────────────────────────────

    def get_state_vector(self) -> np.ndarray:
        """
        5-dim observable state for this worker:
          [load_norm, fatigue_norm, availability,
           hours_worked_today_norm, productivity_norm]
        """
        load_norm          = float(self.load) / config.MAX_WORKER_LOAD
        fatigue_norm       = self.fatigue / 3.0
        hours_norm         = min(1.0, self.hours_worked_today / 8.0)  # 8h = full day
        # Productivity proxy: decreases with fatigue + hours worked
        intraday_pen       = config.INTRADAY_DECAY_RATE * min(self.hours_worked_today, 8.0)
        fatigue_pen        = self.fatigue_sensitivity * self.fatigue
        productivity       = float(np.clip(1.0 - fatigue_pen - intraday_pen, 0.0, 1.0))
        return np.array([
            load_norm, fatigue_norm, float(self.availability),
            hours_norm, productivity
        ], dtype=np.float32)

    def get_skill_estimate(self) -> Tuple[float, float]:
        """
        Estimate skill from task completion history (observable to baselines).
        Uses Welford's online mean for numeric stability.

        Returns: (mean_skill_estimate, uncertainty_std)
        """
        if len(self.completion_history) == 0:
            return 1.0, 0.4

        estimates = []
        for complexity, time_h, _ in self.completion_history:
            # Approximate: skill ≈ complexity / time × fatigue_correction
            est = (float(complexity) / max(time_h, 0.01)) * 1.2
            estimates.append(est)

        mean_est = float(np.mean(estimates))
        std_est  = float(np.std(estimates) / max(np.sqrt(len(estimates)), 1))
        return mean_est, std_est

    def get_hidden_profile(self) -> dict:
        """
        Return all hidden parameters — FOR OBSERVER DISPLAY ONLY.
        Never called during scheduling by any agent.
        """
        return {
            'worker_id':         self.worker_id,
            'true_skill':        round(self.true_skill,         3),
            'speed_multiplier':  round(self.speed_multiplier,   3),
            'fatigue_rate':      round(self.fatigue_rate,       3),
            'recovery_rate':     round(self.recovery_rate,      3),
            'fatigue_sensitivity': round(self.fatigue_sensitivity, 3),
            'burnout_resilience': round(self.burnout_resilience, 3),
        }

    def reset(self, new_skill: Optional[float] = None):
        """
        Reset worker to episode-start state.
        Hidden traits (skill, speed, fatigue params) are PRESERVED — permanent worker traits.
        """
        self.load               = 0
        self.fatigue            = 0.0
        self.availability       = 1
        self.assigned_tasks     = []
        self.completion_history = []
        self.burnout_timer      = 0
        self.hours_worked_today = 0.0
        if new_skill is not None:
            self.true_skill = float(new_skill)

    def __repr__(self):
        return (
            f"Worker(id={self.worker_id}, load={self.load}, "
            f"fatigue={self.fatigue:.2f}, avail={self.availability}, "
            f"skill={self.true_skill:.2f}, spd={self.speed_multiplier:.2f}, "
            f"hrs_today={self.hours_worked_today:.1f})"
        )


if __name__ == "__main__":
    print("Testing Worker v4...")
    w = Worker(worker_id=0, skill=1.1)
    print(f"✓ Init: {w}")
    print(f"✓ Hidden profile: {w.get_hidden_profile()}")

    w.assign_task(1)
    t, q = w.complete_task(1, complexity=3)
    print(f"✓ Complete task: time={t:.2f}h, quality={q:.3f}")

    sv = w.get_state_vector()
    assert len(sv) == 5, f"Expected 5-dim state, got {len(sv)}"
    print(f"✓ State vector (5-dim): {sv}")

    w.daily_reset()
    print(f"✓ Daily reset: hours_today={w.hours_worked_today}, fatigue={w.fatigue:.2f}")

    mu, sigma = w.get_skill_estimate()
    print(f"✓ Skill estimate: μ={mu:.3f}, σ={sigma:.3f}")
    print("Worker v4 tests passed!")
