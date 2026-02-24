"""
ProjectEnv — v4: Continual Online Learning Scheduling Environment.

Key changes from v3:
  - SimClock tracks (day, slot_in_day) within an 8h/5-day workday model.
  - Tasks arrive dynamically (Poisson); agents have ZERO lookahead.
  - get_available_tasks(tick) returns only tasks that have arrived by 'tick'.
  - Outside work-hours slots are skipped automatically (env advances the clock).
  - State dim: 96 (5w×5 + 10t×5 + 10 belief + 6 global + 15 pad).
  - Reward is makespan-centric: throughput, idle penalty, lateness, completion bonus.
  - Baselines and DQN both use this single environment — no lookahead anywhere.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.worker import Worker
from environment.task   import Task, generate_poisson_arrivals
from environment.belief_state import BeliefState


class SimClock:
    """
    Simulated wall-clock that tracks working days and slots.

    slot 0 = 09:00 on Day 0 (Monday).
    Each day has SLOTS_PER_DAY working slots; non-working hours are skipped.
    """

    def __init__(self):
        self.tick         = 0   # global slot counter (only counts work-hour slots)
        self.day          = 0   # working day counter (0-indexed)
        self.slot_in_day  = 0   # 0 … SLOTS_PER_DAY-1
        self.week         = 0

    def advance(self):
        """Advance one work-hour slot."""
        self.tick        += 1
        self.slot_in_day += 1
        if self.slot_in_day >= config.SLOTS_PER_DAY:
            self.slot_in_day = 0
            self.day        += 1
            if self.day % config.WORK_DAYS_PER_WEEK == 0:
                self.week += 1

    def is_start_of_day(self) -> bool:
        return self.slot_in_day == 0

    def hour_in_day(self) -> float:
        """Current hour within workday (0.0 = 09:00 … 8.0 = 17:00)."""
        return self.slot_in_day * config.SLOT_HOURS

    def __repr__(self):
        return (f"SimClock(tick={self.tick}, week={self.week}, "
                f"day={self.day}, slot_in_day={self.slot_in_day}, "
                f"hour={self.hour_in_day():.1f}h)")


class ProjectEnv:
    """
    Continual online scheduling environment.

    State space : 96-dim vector (5w×5 + 10t×5 + 10 belief + 6 global + 15 pad)
    Action space: 140 discrete actions (assign 20t×5w + 20 defer + 20 escalate)
    Reward      : Makespan-centric — throughput, idle penalty, lateness, terminal bonus

    No agent (DQN or baseline) has access to future task arrivals.
    """

    def __init__(
        self,
        num_workers: int = None,
        total_tasks: int = None,
        seed: int = None,
        enable_diagnostics: bool = False,
        reward_scale: float = 1.0,
        total_sim_slots: int = None,
        config_overrides: Dict = None,
    ):
        """
        Args:
            num_workers      : Number of workers (default config.NUM_WORKERS)
            total_tasks      : Total tasks in simulation (default config.TOTAL_TASKS)
            seed             : Random seed
            enable_diagnostics: Log extra diagnostic info
            reward_scale     : Reward scalar (default 1.0)
            total_sim_slots  : Override total simulation length in slots
            config_overrides : Dict to override default behaviours
        """
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.num_workers  = num_workers or config.NUM_WORKERS
        self.num_tasks    = total_tasks or config.TOTAL_TASKS
        self.reward_scale = reward_scale

        self.config_overrides     = config_overrides or {}
        self.enable_fatigue       = self.config_overrides.get('enable_fatigue',        True)
        self.enable_deadline_shocks = self.config_overrides.get('enable_deadline_shocks', True)
        self.fully_observable     = self.config_overrides.get('fully_observable',      False)
        self.enable_diagnostics   = enable_diagnostics

        # Total simulation length (slots)
        self._total_sim_slots = total_sim_slots or (config.TOTAL_SIM_DAYS * config.SLOTS_PER_DAY)

        # Max tasks visible to state (top-10 most urgent)
        self._max_visible_tasks   = 10

        # Workers (permanent across episodes)
        self.workers      = [Worker(worker_id=i) for i in range(self.num_workers)]

        # Belief state for skill tracking (Bayesian)
        self.belief_state = BeliefState(num_workers=self.num_workers)

        # State that resets each episode/simulation run
        self.tasks:           List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks:    List[Task] = []
        self.clock = SimClock()

        # Metrics
        self.metrics = self._empty_metrics()
        self._last_reward_breakdown   = {}
        self._episode_reward_breakdown = {}
        self._all_completed_makespan  = False

        # Diagnostics
        if self.enable_diagnostics:
            self.diagnostics: Optional[Dict] = {
                'state_ranges': [], 'reward_ranges': [],
                'valid_action_counts': [], 'reward_components': []
            }
        else:
            self.diagnostics = None

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset environment for a new simulation run.
        Generates a fresh Poisson task stream; resets workers and clock.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        # Reset workers (hidden traits preserved)
        for w in self.workers:
            w.reset()

        # Generate fresh Poisson task arrivals
        self.tasks = generate_poisson_arrivals(
            total_tasks           = self.num_tasks,
            arrival_rate_per_day  = config.TASK_ARRIVAL_RATE,
            total_slots           = self._total_sim_slots,
            seed                  = self._seed,
        )

        self.completed_tasks = []
        self.failed_tasks    = []
        self.belief_state.reset()
        self.clock           = SimClock()
        self.metrics         = self._empty_metrics()
        self._last_reward_breakdown    = {}
        self._episode_reward_breakdown = {}
        self._all_completed_makespan   = False

        return self._get_state()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one scheduling action and advance simclock by one slot.

        Args:
            action: Action index [0, 139]
              0–99    : assign task_id × 5 + worker_id  (up to 20 tasks × 5 workers)
              100–119 : defer task_id
              120–139 : escalate task_id

        Returns:
            (next_state, reward, done, info)
        """
        # ── Decode & execute action ───────────────────────────────────────────
        task_id, worker_id, action_type = self._decode_action(action)
        action_reward = self._execute_action(task_id, worker_id, action_type)

        # ── Advance task progress ─────────────────────────────────────────────
        completion_reward = self._update_task_progress()

        # ── Advance clock one slot ────────────────────────────────────────────
        if self.enable_fatigue:
            for w in self.workers:
                w.update_fatigue()

        if self.clock.is_start_of_day():
            for w in self.workers:
                w.daily_reset()

        self.clock.advance()

        # ── Check for new deadline failures ───────────────────────────────────
        for t in self.tasks:
            if t.arrival_tick <= self.clock.tick and not t.is_completed:
                t.check_deadline(self.clock.tick)

        self.failed_tasks = [t for t in self.tasks if t.is_failed
                             and t not in self.failed_tasks]

        # ── Random deadline shock ─────────────────────────────────────────────
        if self.enable_deadline_shocks and np.random.rand() < config.DEADLINE_SHOCK_PROB:
            self._apply_deadline_shock()

        # ── Overload tracking ─────────────────────────────────────────────────
        for w in self.workers:
            if w.load > config.MAX_WORKER_LOAD // 2:
                self.metrics['overload_events'] += 1

        # ── Compute reward ────────────────────────────────────────────────────
        idle_penalty     = self._compute_idle_penalty()
        lateness_penalty = self._compute_lateness_penalty()
        urgency_penalty  = self._compute_urgency_penalty()
        overload_penalty = self._compute_overload_penalty()
        delay_penalty    = config.REWARD_DELAY_WEIGHT   # constant nudge

        # Rebuild failed_tasks list correctly
        new_fails = [t for t in self.tasks if t.is_failed and t not in self.failed_tasks]
        deadline_miss_penalty = len(new_fails) * config.REWARD_DEADLINE_MISS_PENALTY
        for t in new_fails:
            self.failed_tasks.append(t)

        reward_raw = (
            action_reward
            + completion_reward
            + idle_penalty
            + lateness_penalty
            + urgency_penalty
            + overload_penalty
            + delay_penalty
            + deadline_miss_penalty
        )

        # Terminal makespan bonus: all tasks complete → bonus inversely proportional to time used
        if (not self._all_completed_makespan
                and len(self.completed_tasks) == len(self.tasks)):
            makespan_bonus = config.REWARD_MAKESPAN_BONUS * max(
                0.0, 1.0 - self.clock.tick / self._total_sim_slots
            )
            reward_raw += makespan_bonus
            self._all_completed_makespan = True

        reward = reward_raw * self.reward_scale

        # ── Reward diagnostics ────────────────────────────────────────────────
        self._last_reward_breakdown = {
            'tick':                self.clock.tick,
            'action_reward':       action_reward,
            'completion_reward':   completion_reward,
            'idle_penalty':        idle_penalty,
            'lateness_penalty':    lateness_penalty,
            'urgency_penalty':     urgency_penalty,
            'overload_penalty':    overload_penalty,
            'deadline_miss':       deadline_miss_penalty,
            'total_unscaled':      reward_raw,
            'total_scaled':        reward,
        }
        for k in ('completion_reward', 'idle_penalty', 'lateness_penalty'):
            self._episode_reward_breakdown[k] = (
                self._episode_reward_breakdown.get(k, 0.0)
                + self._last_reward_breakdown[k]
            )

        if self.enable_diagnostics:
            self.diagnostics['reward_components'].append(self._last_reward_breakdown)
            valid = self.get_valid_actions()
            s = self._get_state()
            self.diagnostics['state_ranges'].append((float(np.min(s)), float(np.max(s))))
            self.diagnostics['reward_ranges'].append(reward)
            self.diagnostics['valid_action_counts'].append(len(valid))

        # ── Termination ───────────────────────────────────────────────────────
        done, reason = self._check_termination()

        next_state = self._get_state()

        info = {
            'tick':               self.clock.tick,
            'day':                self.clock.day,
            'reward':             reward,
            'completed_tasks':    len(self.completed_tasks),
            'failed_tasks':       len(self.failed_tasks),
            'total_tasks':        len(self.tasks),
            'available_tasks':    len(self._get_available_tasks()),
            'termination_reason': reason if done else None,
        }
        return next_state, reward, done, info

    # ── Action helpers ────────────────────────────────────────────────────────

    def _decode_action(self, action: int) -> Tuple[int, int, str]:
        """Decode action index → (task_idx, worker_id, action_type)."""
        if action < 20 * self.num_workers:        # assign (up to 20 tasks × 5 workers)
            task_idx  = action // self.num_workers
            worker_id = action % self.num_workers
            return task_idx, worker_id, 'assign'
        elif action < 20 * self.num_workers + 20: # defer
            task_idx = action - 20 * self.num_workers
            return task_idx, -1, 'defer'
        else:                                      # escalate
            task_idx = action - 20 * self.num_workers - 20
            return task_idx, -1, 'escalate'

    def _execute_action(self, task_idx: int, worker_id: int, action_type: str) -> float:
        """Execute decoded action; return immediate reward component."""
        available = self._get_available_tasks()

        if action_type == 'assign':
            if task_idx >= len(available) or worker_id >= self.num_workers:
                return -0.5
            task   = available[task_idx]
            worker = self.workers[worker_id]

            if not self._is_valid_assign(task, worker):
                return -0.5

            worker.assign_task(task.task_id)
            task.assign_to_worker(
                worker_id    = worker_id,
                current_tick = self.clock.tick,
                worker_skill = worker.true_skill,
                worker_speed = worker.speed_multiplier,
            )
            # Small skill-match bonus
            match = worker.true_skill / max(task.complexity, 1)
            return 0.1 * min(match, 2.0)

        elif action_type == 'defer':
            if task_idx >= len(available):
                return -0.2
            task = available[task_idx]
            # Strategic defer is acceptable if no good worker is available
            avail_skills = [w.true_skill for w in self.workers if w.availability == 1]
            if avail_skills and max(avail_skills) >= task.complexity * 0.6:
                return -0.1   # Mild penalty for deferring when workers are available
            return config.REWARD_STRATEGIC_DEFER

        elif action_type == 'escalate':
            if task_idx >= len(available):
                return 0.0
            task = available[task_idx]
            if task.assigned_worker is not None and task.priority < 3:
                task.priority              = min(3, task.priority + 1)
                # Speed up expected completion by ~20%
                if task.expected_completion_tick is not None:
                    span = task.expected_completion_tick - self.clock.tick
                    task.expected_completion_tick = self.clock.tick + max(1, int(span * 0.8))
                return -1.5   # Cost of escalation
            return 0.0

        return 0.0

    def _is_valid_assign(self, task: Task, worker: Worker) -> bool:
        """Check that task → worker assignment is legal."""
        if task.assigned_worker is not None or task.is_completed or task.is_failed:
            return False
        if worker.availability == 0:
            return False
        completed_ids = [t.task_id for t in self.completed_tasks]
        if not task.check_dependencies_met(completed_ids):
            return False
        return True

    def _get_available_tasks(self) -> List[Task]:
        """Tasks that have arrived AND are still pending assignment."""
        tick = self.clock.tick
        return [
            t for t in self.tasks
            if t.is_available(tick) and t.is_unassigned()
        ]

    # ── Task progress ─────────────────────────────────────────────────────────

    def _update_task_progress(self) -> float:
        """Advance all in-progress tasks; return completion reward."""
        completion_reward = 0.0
        tick = self.clock.tick

        for task in self.tasks:
            if task.assigned_worker is None or task.is_completed:
                continue

            just_completed = task.update_progress(tick)
            if just_completed:
                worker = self.workers[task.assigned_worker]
                _, quality = worker.complete_task(task.task_id, task.complexity)
                task.quality_score = quality
                self.belief_state.update(task.assigned_worker, quality)
                self.completed_tasks.append(task)
                self.metrics['throughput'] += 1

                priority_weight    = (task.priority + 1) * config.REWARD_COMPLETION_BASE
                completion_reward += priority_weight * quality

        return completion_reward

    # ── Reward components ─────────────────────────────────────────────────────

    def _compute_idle_penalty(self) -> float:
        """Penalise each available worker who has no tasks assigned."""
        idle_count = sum(
            1 for w in self.workers
            if w.availability == 1 and w.load == 0
        )
        return config.REWARD_IDLE_PENALTY * idle_count

    def _compute_lateness_penalty(self) -> float:
        """Penalise tasks completed after their deadline (lateness penalty)."""
        penalty    = 0.0
        tick       = self.clock.tick
        for t in self.completed_tasks:
            if t.actual_completion_tick is not None and t.actual_completion_tick > t.deadline_slot:
                slots_late = t.actual_completion_tick - t.deadline_slot
                penalty   += config.REWARD_LATENESS_PENALTY * slots_late
        return penalty

    def _compute_urgency_penalty(self) -> float:
        """Penalise unstarted tasks close to their deadline."""
        penalty    = 0.0
        tick       = self.clock.tick
        for t in self.tasks:
            if t.is_available(tick) and t.is_unassigned():
                slots_left = t.slots_until_deadline(tick)
                if slots_left <= 4:
                    penalty += config.REWARD_URGENCY_PENALTY
        return penalty

    def _compute_overload_penalty(self) -> float:
        """Mild penalty for load imbalance across workers."""
        loads = np.array([w.load for w in self.workers], dtype=float)
        sigma = float(np.std(loads))
        return config.REWARD_OVERLOAD_WEIGHT * sigma

    # ── Deadline shock ────────────────────────────────────────────────────────

    def _apply_deadline_shock(self):
        if not self.enable_deadline_shocks:
            return
        available = self._get_available_tasks()
        if available:
            shocked = np.random.choice(available)
            shocked.apply_deadline_shock()

    # ── Termination ───────────────────────────────────────────────────────────

    def _check_termination(self) -> Tuple[bool, Optional[str]]:
        # All tasks completed
        if len(self.completed_tasks) == len(self.tasks):
            return True, 'all_completed'
        # Simulation time exhausted
        if self.clock.tick >= self._total_sim_slots:
            return True, 'time_limit'
        # Catastrophic failure rate
        failure_rate = len(self.failed_tasks) / max(len(self.tasks), 1)
        if failure_rate >= 0.6:
            return True, 'catastrophic_failure'
        return False, None

    # ── State observation ─────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        96-dim state vector:
          Worker features  : 5 workers × 5 dims = 25
          Task features    : 10 tasks   × 5 dims = 50  (top-10 by urgency)
          Belief           : 10  (5 skill means + 5 variances)
          Global context   : 6   (progress, completion_rate, failure_rate,
                                   n_idle_norm, n_available_tasks_norm, day_progress)
          Padding          : 5   (zeros to reach 96)
        """
        # ── Worker features (5 × 5 = 25) ────────────────────────────────────
        worker_features = np.concatenate([w.get_state_vector() for w in self.workers])

        # ── Task features (top-10 urgent × 5 = 50) ──────────────────────────
        tick = self.clock.tick
        completed_ids = [t.task_id for t in self.completed_tasks]

        # Include both unassigned & in-progress tasks visible to agents
        visible_tasks = [
            t for t in self.tasks
            if t.is_available(tick) and not t.is_completed and not t.is_failed
        ]
        visible_tasks.sort(key=lambda t: -t.get_deadline_urgency(tick))
        visible_tasks = visible_tasks[:self._max_visible_tasks]

        task_features = []
        for t in visible_tasks:
            task_features.append(t.get_state_vector(tick, completed_ids))
        while len(task_features) < self._max_visible_tasks:
            task_features.append(np.zeros(5, dtype=np.float32))
        task_features = np.concatenate(task_features)

        # ── Belief features (10) ─────────────────────────────────────────────
        if self.fully_observable:
            skill_means = [w.true_skill for w in self.workers]
            skill_vars  = [0.0] * self.num_workers
            belief_features = np.array(skill_means + skill_vars, dtype=np.float32)
        else:
            belief_features = self.belief_state.get_state_vector()

        # ── Global context (6) ───────────────────────────────────────────────
        time_progress     = self.clock.tick / self._total_sim_slots
        completion_rate   = len(self.completed_tasks) / max(len(self.tasks), 1)
        failure_rate      = len(self.failed_tasks)    / max(len(self.tasks), 1)
        n_idle            = sum(1 for w in self.workers if w.availability == 1 and w.load == 0)
        n_available       = len(visible_tasks)
        day_progress      = self.clock.slot_in_day / config.SLOTS_PER_DAY

        global_features = np.array([
            time_progress, completion_rate, failure_rate,
            n_idle / max(self.num_workers, 1),
            n_available / max(self._max_visible_tasks, 1),
            day_progress,
        ], dtype=np.float32)

        # ── Concatenate → 25 + 50 + 10 + 6 = 91; pad to 96 ─────────────────
        state = np.concatenate([worker_features, task_features, belief_features, global_features])
        if len(state) < config.STATE_DIM:
            state = np.pad(state, (0, config.STATE_DIM - len(state)), 'constant')

        return state[:config.STATE_DIM].astype(np.float32)

    # ── Valid actions ─────────────────────────────────────────────────────────

    def get_valid_actions(self) -> List[int]:
        """
        Return list of legal action indices given current state.

        All agents (DQN + baselines) call this — guarantees zero lookahead
        because it only considers tasks visible at clock.tick.
        """
        valid  = []
        tick   = self.clock.tick
        completed_ids = [t.task_id for t in self.completed_tasks]

        # Sort available unassigned tasks by urgency (same order as state vector)
        available = self._get_available_tasks()
        available.sort(key=lambda t: -t.get_deadline_urgency(tick))
        available = available[:self._max_visible_tasks]  # cap at 20 (state window)

        for task_slot, task in enumerate(available):
            if not task.check_dependencies_met(completed_ids):
                continue
            for worker_id, worker in enumerate(self.workers):
                if worker.availability == 1:
                    action_idx = task_slot * self.num_workers + worker_id
                    if action_idx < 100:
                        valid.append(action_idx)

        # Defer actions (always valid for unassigned visible tasks)
        for task_slot, task in enumerate(available):
            if task_slot < 20:
                valid.append(20 * self.num_workers + task_slot)

        return valid

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_metrics(self) -> Dict:
        """Compute comprehensive episode-end metrics."""
        metrics = self._empty_metrics()

        metrics['throughput']      = len(self.completed_tasks)
        metrics['failed_tasks']    = len(self.failed_tasks)
        metrics['total_tasks']     = len(self.tasks)
        metrics['completion_rate'] = len(self.completed_tasks) / max(len(self.tasks), 1)

        # Makespan: slots from tick 0 to last task completion
        if self.completed_tasks:
            last_tick = max(t.actual_completion_tick or 0 for t in self.completed_tasks)
            metrics['makespan_slots'] = last_tick
            metrics['makespan_hours'] = last_tick * config.SLOT_HOURS
        else:
            metrics['makespan_slots'] = self.clock.tick
            metrics['makespan_hours'] = self.clock.tick * config.SLOT_HOURS

        # Throughput per working day
        days_elapsed = max(self.clock.day, 1)
        metrics['throughput_per_day'] = len(self.completed_tasks) / days_elapsed

        # Worker utilisation (fraction of slots with at least one task)
        tasks_per_worker = np.zeros(self.num_workers)
        for t in self.tasks:
            if t.assigned_worker is not None:
                tasks_per_worker[int(t.assigned_worker)] += 1
        metrics['load_balance'] = float(np.std(tasks_per_worker))

        # Lateness
        late_tasks = [t for t in self.completed_tasks
                      if t.actual_completion_tick is not None
                      and t.actual_completion_tick > t.deadline_slot]
        metrics['lateness_rate']  = len(late_tasks) / max(len(self.completed_tasks), 1)
        if late_tasks:
            metrics['avg_lateness_h'] = float(np.mean([
                (t.actual_completion_tick - t.deadline_slot) * config.SLOT_HOURS
                for t in late_tasks
            ]))

        # Quality
        if self.completed_tasks:
            metrics['quality_score'] = float(np.mean([t.quality_score for t in self.completed_tasks]))

        # Idle slots (crude: how many workers had load=0 throughout)
        metrics['overload_events'] = self.metrics['overload_events']

        self.metrics.update(metrics)
        return self.metrics

    def get_reward_breakdown(self) -> Dict:
        return dict(self._last_reward_breakdown)

    def get_episode_reward_breakdown(self) -> Dict:
        return dict(self._episode_reward_breakdown)

    @staticmethod
    def _empty_metrics() -> Dict:
        return {
            'throughput':        0,
            'failed_tasks':      0,
            'total_tasks':       0,
            'completion_rate':   0.0,
            'makespan_slots':    0,
            'makespan_hours':    0.0,
            'throughput_per_day': 0.0,
            'load_balance':      0.0,
            'lateness_rate':     0.0,
            'avg_lateness_h':    0.0,
            'quality_score':     0.0,
            'overload_events':   0,
        }

    def __repr__(self):
        return (
            f"ProjectEnv(tick={self.clock.tick}, day={self.clock.day}, "
            f"completed={len(self.completed_tasks)}/{len(self.tasks)}, "
            f"failed={len(self.failed_tasks)})"
        )


if __name__ == "__main__":
    print("Testing ProjectEnv v4...")
    env = ProjectEnv(num_workers=5, total_tasks=30, seed=42)
    state = env.reset()
    assert len(state) == config.STATE_DIM, f"Expected {config.STATE_DIM}-dim, got {len(state)}"
    print(f"✓ Reset: state shape {state.shape}")

    valid = env.get_valid_actions()
    print(f"✓ Valid actions: {len(valid)}")

    if valid:
        ns, r, done, info = env.step(valid[0])
        print(f"✓ Step: reward={r:.3f}, done={done}, info={info}")
    else:
        print("  (no valid actions at tick 0 — tasks may not have arrived yet)")

    # Run 50 slots
    total_reward = 0.0
    for _ in range(50):
        valid = env.get_valid_actions()
        if not valid:
            action = 100  # defer task 0
        else:
            action = np.random.choice(valid)
        _, r, done, _ = env.step(action)
        total_reward += r
        if done:
            break

    metrics = env.compute_metrics()
    print(f"✓ Metrics after 50 slots: {metrics}")
    print(f"✓ Clock: {env.clock}")
    print("ProjectEnv v4 tests passed!")
