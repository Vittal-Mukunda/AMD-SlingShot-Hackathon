"""
Task class — v4: Dynamic arrivals with Poisson process, slot-based deadlines.

Key changes from v3:
  - arrival_tick: Slot at which this task becomes visible to schedulers.
                  NO agent can see or schedule a task before arrival_tick.
  - Duration measured in HOURS; converted to slots by the environment.
  - deadline_slots: Absolute slot deadline (arrival_tick + deadline_h / SLOT_HOURS).
  - is_available(current_tick): False until current_tick >= arrival_tick.
  - All agents operate with zero lookahead — tasks appear dynamically.
"""

import numpy as np
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Task:
    """
    Task with dynamic arrival, slot-based deadlines, and dependency constraints.

    Observable state (5-dim per task):
        [priority_norm, complexity_norm, deadline_urgency, deps_met, arrival_elapsed_norm]
    """

    def __init__(
        self,
        task_id: int,
        priority: int,
        complexity: int,
        deadline_h: float,
        dependencies: Optional[List[int]] = None,
        arrival_tick: int = 0,
    ):
        """
        Initialise task.

        Args:
            task_id      : Unique identifier
            priority     : 0=low … 3=critical
            complexity   : 1=simple … 5=very complex
            deadline_h   : Hours from arrival until deadline expires
            dependencies : Task IDs that must complete before this one
            arrival_tick : Absolute sim-slot when this task becomes available
        """
        self.task_id      = task_id
        self.priority     = priority
        self.complexity   = complexity
        self.deadline_h   = deadline_h
        self.dependencies = dependencies if dependencies is not None else []
        self.arrival_tick = arrival_tick

        # Derived: slot-based absolute deadline
        self.deadline_slot = int(arrival_tick + deadline_h / config.SLOT_HOURS)

        # Assignment state
        self.assigned_worker: Optional[int]  = None
        self.start_tick:      Optional[int]  = None
        self.expected_completion_tick:  Optional[int] = None
        self.actual_completion_tick:    Optional[int] = None

        # Progress
        self.completion_progress = 0.0
        self.is_completed = False
        self.is_failed    = False
        self.quality_score = 0.0

    # ── Availability (dynamic arrivals — zero lookahead) ─────────────────────

    def is_available(self, current_tick: int) -> bool:
        """True only if the task has arrived and is not yet complete/failed."""
        return (
            current_tick >= self.arrival_tick
            and not self.is_completed
            and not self.is_failed
        )

    def is_unassigned(self) -> bool:
        """True if the task arrived but hasn't been assigned yet."""
        return self.assigned_worker is None and not self.is_completed and not self.is_failed

    # ── Assignment ────────────────────────────────────────────────────────────

    def assign_to_worker(self, worker_id: int, current_tick: int, worker_skill: float,
                          worker_speed: float = 1.0):
        """
        Assign task to a worker and sample expected completion tick.

        Args:
            worker_id    : Worker being assigned
            current_tick : Current sim slot
            worker_skill : Worker's true_skill (for task duration sampling)
            worker_speed : Worker's speed_multiplier (default 1.0)
        """
        if self.assigned_worker is not None:
            raise ValueError(f"Task {self.task_id} already assigned to W-{self.assigned_worker}")

        self.assigned_worker = worker_id
        self.start_tick      = current_tick

        # Expected duration in hours, then convert to slots
        effective_speed  = max(worker_skill * worker_speed, 0.1)
        expected_h       = float(self.complexity) / effective_speed
        expected_h       = np.clip(expected_h, config.MIN_TASK_DURATION_H * 0.3,
                                               config.MAX_TASK_DURATION_H * 1.2)

        # Stochastic noise
        noise_h          = config.COMPLETION_TIME_NOISE * expected_h
        sampled_h        = float(np.random.normal(expected_h, noise_h))
        sampled_h        = float(np.clip(sampled_h, 0.5 * expected_h, 2.0 * expected_h))

        # Convert to slots (always at least 1 slot)
        duration_slots   = max(1, int(round(sampled_h / config.SLOT_HOURS)))
        self.expected_completion_tick = current_tick + duration_slots

    # ── Progress tracking ─────────────────────────────────────────────────────

    def update_progress(self, current_tick: int) -> bool:
        """
        Update completion fraction; mark complete if expected_tick reached.

        Returns:
            True if task completed THIS slot.
        """
        if self.assigned_worker is None or self.is_completed:
            return False

        if self.expected_completion_tick is None or self.expected_completion_tick <= 0:
            self.expected_completion_tick = self.start_tick + 1

        elapsed  = current_tick - self.start_tick
        total    = self.expected_completion_tick - self.start_tick
        self.completion_progress = min(1.0, elapsed / max(total, 1))

        if self.completion_progress >= 1.0:
            self.is_completed           = True
            self.actual_completion_tick = current_tick
            return True

        return False

    # ── Deadline management ───────────────────────────────────────────────────

    def check_deadline(self, current_tick: int):
        """Fail task if deadline has passed and it isn't complete."""
        if not self.is_completed and current_tick > self.deadline_slot:
            self.is_failed = True

    def apply_deadline_shock(self, shock_slots: Optional[int] = None):
        """Suddenly reduce deadline by shock_slots (environmental stochasticity)."""
        amt = shock_slots if shock_slots is not None else config.DEADLINE_SHOCK_SLOTS
        self.deadline_slot = max(self.deadline_slot - amt, self.arrival_tick + 2)

    def slots_until_deadline(self, current_tick: int) -> int:
        """Remaining slots before deadline (0 if already passed)."""
        return max(0, self.deadline_slot - current_tick)

    # ── State vector ──────────────────────────────────────────────────────────

    def get_state_vector(self, current_tick: int, completed_ids: List[int]) -> np.ndarray:
        """
        5-dim observable feature vector for this task.

        Dims:
          0 priority_norm             [0, 1]
          1 complexity_norm           [0, 1]
          2 deadline_urgency          [0, 1]  (1=very urgent, 0=lots of time)
          3 deps_met                  {0, 1}
          4 arrival_elapsed_norm      [0, 1]  (time in queue since arrival)
        """
        priority_norm     = self.priority / 3.0
        complexity_norm   = (self.complexity - 1) / 4.0

        total_deadline_slots = max(1, self.deadline_slot - self.arrival_tick)
        remaining            = max(0, self.deadline_slot - current_tick)
        deadline_urgency     = float(np.clip(1.0 - remaining / total_deadline_slots, 0.0, 1.0))

        deps_met = 1.0 if self.check_dependencies_met(completed_ids) else 0.0

        elapsed_since_arrival = max(0, current_tick - self.arrival_tick)
        arrival_elapsed_norm  = float(np.clip(elapsed_since_arrival / max(total_deadline_slots, 1), 0.0, 1.0))

        return np.array([
            priority_norm, complexity_norm, deadline_urgency, deps_met, arrival_elapsed_norm
        ], dtype=np.float32)

    def get_deadline_urgency(self, current_tick: int) -> float:
        """Priority-weighted urgency for sorting (higher = schedule first)."""
        slots_left = max(1, self.slots_until_deadline(current_tick))
        return (self.priority + 1) * 10.0 / slots_left

    # ── Dependencies ─────────────────────────────────────────────────────────

    def check_dependencies_met(self, completed_task_ids: List[int]) -> bool:
        """True if all prerequisite task IDs are in completed_task_ids."""
        return all(dep in completed_task_ids for dep in self.dependencies)

    # ── Utility ──────────────────────────────────────────────────────────────

    def reset(self):
        """Reset assignment state (used for re-running same task set in tests)."""
        self.assigned_worker          = None
        self.start_tick               = None
        self.completion_progress      = 0.0
        self.is_completed             = False
        self.is_failed                = False
        self.quality_score            = 0.0
        self.expected_completion_tick = None
        self.actual_completion_tick   = None

    def __repr__(self):
        status = "done" if self.is_completed else ("failed" if self.is_failed else "pending")
        return (
            f"Task(id={self.task_id}, p={self.priority}, c={self.complexity}, "
            f"arrival={self.arrival_tick}, deadline_slot={self.deadline_slot}, "
            f"deps={self.dependencies}, {status})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Poisson task arrival generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_poisson_arrivals(
    total_tasks: int = None,
    arrival_rate_per_day: float = None,
    total_slots: int = None,
    seed: Optional[int] = None,
) -> List[Task]:
    """
    Generate tasks with Poisson-distributed arrival times across the simulation.

    Tasks arrive according to a Poisson process so NO agent has lookahead:
      - Each task's arrival_tick is drawn from a non-homogeneous Poisson process.
      - Tasks within the same slot are ordered by task_id to break ties.

    Args:
        total_tasks          : Number of tasks (default config.TOTAL_TASKS)
        arrival_rate_per_day : Mean arrivals per day (default config.TASK_ARRIVAL_RATE)
        total_slots          : Total simulation slots (default PHASE1+PHASE2 days × SLOTS_PER_DAY)
        seed                 : Random seed

    Returns:
        List of Task objects sorted by arrival_tick.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    total_tasks          = total_tasks or config.TOTAL_TASKS
    arrival_rate_per_day = arrival_rate_per_day or config.TASK_ARRIVAL_RATE
    total_slots          = total_slots or (config.TOTAL_SIM_DAYS * config.SLOTS_PER_DAY)

    # Rate per slot = tasks_per_day / slots_per_day
    rate_per_slot = arrival_rate_per_day / config.SLOTS_PER_DAY

    # Spread arrivals across working slots only (within each day's working window)
    # Convert slot index to (day, slot_in_day) and filter to working hours
    working_slots = []
    for s in range(total_slots):
        slot_in_day = s % config.SLOTS_PER_DAY
        if config.WORK_START_SLOT <= slot_in_day <= config.WORK_END_SLOT:
            working_slots.append(s)

    # Assign each task an arrival tick via Poisson inter-arrival in working slots
    intervals  = rng.exponential(scale=1.0 / rate_per_slot, size=total_tasks)
    cum_slots  = np.cumsum(intervals)
    # Scale to fit within available working slots (cap at last working slot)
    max_working = working_slots[-1] if working_slots else total_slots - 1
    cum_slots  = np.clip(cum_slots / cum_slots.max() * max_working * 0.9, 0, max_working)

    tasks = []
    for tid in range(total_tasks):
        # Snap to nearest working slot
        ideal_slot   = cum_slots[tid]
        arrival_tick = min(working_slots, key=lambda s: abs(s - ideal_slot))

        priority   = int(rng.choice(config.TASK_PRIORITIES))
        complexity = int(rng.choice(config.TASK_COMPLEXITY_LEVELS))

        # Deadline: scales with complexity and priority (more complex → more time)
        base_deadline_h  = config.DEADLINE_MIN_H + (complexity - 1) * 3.0
        # High-priority tasks get somewhat tighter deadlines
        priority_factor  = 1.0 - 0.05 * priority
        deadline_h       = float(rng.uniform(
            base_deadline_h * priority_factor,
            min(config.DEADLINE_MAX_H, base_deadline_h * 2.5)
        ))

        task = Task(
            task_id      = tid,
            priority     = priority,
            complexity   = complexity,
            deadline_h   = deadline_h,
            dependencies = [],   # Dependencies set after graph generation
            arrival_tick = arrival_tick,
        )
        tasks.append(task)

    # Sort by arrival time so environment can index them predictably
    tasks.sort(key=lambda t: (t.arrival_tick, t.task_id))

    # Assign dependencies: form sparse DAG on sorted task list
    # Earlier-arriving tasks may depend on later-arriving ones only within chains
    _assign_dependencies(tasks, rng)

    return tasks


def _assign_dependencies(tasks: List[Task], rng: np.random.Generator):
    """
    Assign a sparse DAG of dependencies.
    Each chain starts with an earlier-arriving task; only tasks with lower task_id
    can be prerequisites (ensures DAG property for execution order).
    """
    n       = len(tasks)
    n_chains = min(config.DEPENDENCY_GRAPH_COMPLEXITY, n // 3)

    for _ in range(n_chains):
        chain_len  = int(rng.integers(2, min(5, n // n_chains + 2)))
        # Pick chain_len tasks in order of task_id
        indices    = sorted(rng.choice(n, chain_len, replace=False))
        for i in range(1, len(indices)):
            dep_task = tasks[indices[i - 1]]
            cur_task = tasks[indices[i]]
            if dep_task.task_id not in cur_task.dependencies:
                cur_task.dependencies.append(dep_task.task_id)


if __name__ == "__main__":
    print("Testing Task v4 + Poisson arrivals...")

    slots_per_sim = config.TOTAL_SIM_DAYS * config.SLOTS_PER_DAY
    tasks = generate_poisson_arrivals(total_tasks=50, arrival_rate_per_day=3.5,
                                       total_slots=slots_per_sim, seed=42)
    print(f"✓ Generated {len(tasks)} tasks")
    for t in tasks[:5]:
        print(f"  {t}")

    # Test state vector
    sv = tasks[0].get_state_vector(current_tick=0, completed_ids=[])
    assert len(sv) == 5, f"Expected 5-dim, got {len(sv)}"
    print(f"✓ State vector (5-dim): {sv}")

    # Test is_available
    t0 = tasks[0]
    assert not t0.is_available(-1)
    assert t0.is_available(t0.arrival_tick)
    print(f"✓ is_available works correctly (arrival_tick={t0.arrival_tick})")

    print("Task v4 tests passed!")
