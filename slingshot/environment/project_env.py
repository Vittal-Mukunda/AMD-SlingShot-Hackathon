"""
ProjectEnv — v5: 365-Day Adaptive Scheduling Environment.

Key changes from v4:
  - Backlog penalty: every step incurs cost ∝ number of unassigned available tasks.
  - Terminal penalty: strong one-time penalty at episode end for unfinished/in-progress tasks.
  - Adaptive time stepping: advance_to_next_event() skips idle ticks efficiently
    by jumping to the next task arrival or task completion, not a fixed +1.
  - Idle worker penalty strengthened to discourage underutilization.
  - Deadlines generated within DEADLINE_MIN_DAYS … DEADLINE_MAX_DAYS from arrival
    (set in task.py / config.py), creating genuine scheduling trade-offs.
  - No hardcoded horizon limits — total_sim_slots driven by SIM_DAYS from config.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

from slingshot.core.settings import config
from slingshot.environment.worker import Worker
from slingshot.environment.task   import Task, generate_poisson_arrivals
from slingshot.environment.belief_state import BeliefState


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
        
        # Synergy matrix for collaboration factors
        if self._seed is not None:
            rng = np.random.default_rng(self._seed)
            self.synergy_matrix = rng.uniform(0.95, 1.15, size=(self.num_workers, self.num_workers))
        else:
            self.synergy_matrix = np.random.uniform(0.95, 1.15, size=(self.num_workers, self.num_workers))
            
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
        self._last_completion_milestone = 0.0  # tracks 25/50/75/100% milestones
        # Fix 3: overflow drain window state
        self._overflow_ticks   = 0               # ticks elapsed after _total_sim_slots
        self._overflow_started = False            # True once clock > _total_sim_slots

        # v10 Fix 3: Adaptive quality tracking — rolling 50-decision window
        self._quality_window = []         # last 50 quality scores from DQN assignments
        self._quality_boost_remaining = 0 # decisions left with 1.5× quality reward boost
        self._decision_count = 0          # total DQN scheduling decisions

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
        self._episode_reward_breakdown = {
            'action_reward': 0.0,
            'completion_reward': 0.0,
            'idle_penalty': 0.0,
            'lateness_penalty': 0.0,
            'urgency_penalty': 0.0,
            'overload_penalty': 0.0,
            'deadline_miss': 0.0,
            'delay_penalty': 0.0,
            'makespan_bonus': 0.0,
            'backlog_penalty': 0.0,
            'throughput_bonus': 0.0,
            'terminal_penalty': 0.0,
        }
        self._all_completed_makespan   = False
        self._last_completion_milestone = 0.0
        # Fix 3: reset overflow state
        self._overflow_ticks   = 0
        self._overflow_started = False
        # v10 Fix 3: reset adaptive quality tracking
        self._quality_window = []
        self._quality_boost_remaining = 0
        self._decision_count = 0

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
        task_idx, worker_id, action_type = self._decode_action(action)
        action_type_was_defer_or_noop    = (action_type != 'assign')  # for idle-waiting pen
        action_reward = self._execute_action(task_idx, worker_id, action_type)

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

        # ── Overload tracking (v7: threshold = at capacity, not half) ────────
        for w in self.workers:
            if w.load >= config.MAX_WORKER_LOAD:
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

        makespan_bonus = 0.0
        # Terminal makespan bonus: all tasks complete → bonus inversely proportional to time used
        if (not self._all_completed_makespan
                and len(self.completed_tasks) == len(self.tasks)):
            makespan_bonus = config.REWARD_MAKESPAN_BONUS * max(
                0.0, 1.0 - self.clock.tick / self._total_sim_slots
            )
            self._all_completed_makespan = True

        # ── BACKLOG PENALTY (v7: mild, normalized) ─────────────────────────
        available_now = self._get_available_tasks()
        n_backlog = len(available_now)
        backlog_penalty = config.REWARD_BACKLOG_PENALTY * n_backlog

        # ── Idle-waiting penalty ─────────────────────────────────────────
        # Extra penalty when tasks are waiting but action was NOT an assignment.
        idle_waiting_pen = 0.0
        if action_type_was_defer_or_noop and len(available_now) > 0:
            has_free_worker = any(
                w.availability == 1 and w.load < config.MAX_WORKER_LOAD
                for w in self.workers
            )
            if has_free_worker:
                idle_waiting_pen = getattr(config, 'REWARD_IDLE_WAITING_PENALTY', -0.15)

        # ── THROUGHPUT MILESTONE BONUS (v7: scaled for [-2,+1]) ─────────
        completion_rate_now = len(self.completed_tasks) / max(len(self.tasks), 1)
        throughput_bonus = 0.0
        for milestone in [0.25, 0.50, 0.75, 1.0]:
            if completion_rate_now >= milestone > self._last_completion_milestone:
                throughput_bonus += 0.5  # v7: scaled down from 5.0
                self._last_completion_milestone = milestone

        reward_raw = (
            action_reward
            + completion_reward
            + idle_penalty
            + lateness_penalty
            + urgency_penalty
            + overload_penalty
            + delay_penalty
            + deadline_miss_penalty
            + makespan_bonus
            + backlog_penalty
            + idle_waiting_pen
            + throughput_bonus
        )

        # ── Termination check (before applying terminal penalty) ────────
        done, reason = self._check_termination()

        # ── TERMINAL PENALTY (v6 FIX) ────────────────────────────────
        # Strong one-time penalty for unfinished/in-progress tasks at episode end.
        # Applied AFTER clipping so it always reaches the agent undiminished.
        terminal_penalty = 0.0
        if done and reason != 'all_completed':
            terminal_penalty = self._compute_terminal_penalty()

        # Clip the per-step reward (excludes terminal penalty)
        reward_raw = float(np.clip(reward_raw, config.REWARD_CLIP_MIN, config.REWARD_CLIP_MAX))

        # Apply terminal penalty AFTER clipping — must not be masked
        if terminal_penalty != 0.0:
            reward_raw += terminal_penalty

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
            'delay_penalty':       delay_penalty,
            'makespan_bonus':      makespan_bonus,
            'backlog_penalty':     backlog_penalty,
            'throughput_bonus':    throughput_bonus,
            'terminal_penalty':    terminal_penalty,
            'total_unscaled':      reward_raw,
            'total_scaled':        reward,
        }
        for k in self._episode_reward_breakdown:
            self._episode_reward_breakdown[k] += self._last_reward_breakdown.get(k, 0.0)

        if self.enable_diagnostics:
            self.diagnostics['reward_components'].append(self._last_reward_breakdown)
            valid = self.get_valid_actions()
            s = self._get_state()
            self.diagnostics['state_ranges'].append((float(np.min(s)), float(np.max(s))))
            self.diagnostics['reward_ranges'].append(reward)
            self.diagnostics['valid_action_counts'].append(len(valid))


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
        """Execute decoded action; return immediate shaped reward component."""
        available = self._get_available_tasks()

        if action_type == 'assign':
            if task_idx >= len(available) or worker_id >= self.num_workers:
                return -0.5
            task   = available[task_idx]
            worker = self.workers[worker_id]

            if not self._is_valid_assign(task, worker):
                return -0.5

            # v7 Fix 2: Hard block overloaded workers — BEFORE any positive reward
            if worker.load >= config.MAX_WORKER_LOAD:
                return -5.0  # unrecoverable penalty, cannot be offset

            worker.assign_task(task.task_id, task_type=task.task_type)

            synergy_boost = 1.0
            if getattr(config, 'TEAM_SYNERGY_ENABLED', False) and hasattr(self, 'synergy_matrix'):
                for other_w in self.workers:
                    if other_w.worker_id != worker_id and other_w.load > 0:
                        synergy_boost *= self.synergy_matrix[worker_id, other_w.worker_id]

            task.assign_to_worker(
                worker_id    = worker_id,
                current_tick = self.clock.tick,
                worker_skill = worker.true_skill,
                worker_speed = worker.speed_multiplier * synergy_boost,
            )

            # ── Problem 3: Redesigned shaped assignment reward ─────────────────
            # 1. Skill-match quality (belief-state skill estimate vs task complexity)
            belief_skill = self.belief_state.get_skill_mean(worker_id)
            skill_match  = float(np.clip(belief_skill / max(task.complexity, 1), 0.0, 1.5))
            base_reward  = config.REWARD_SKILL_MATCH_WEIGHT * skill_match   # 0..+0.75

            # 2. Load-balance bonus: reward choosing the worker with lowest current load
            loads = [w.load for w in self.workers if w.availability == 1]
            if loads:
                min_load = min(loads)
                load_bonus = config.REWARD_LOAD_BALANCE_BONUS if worker.load == min_load else 0.0
            else:
                load_bonus = 0.0

            # 3. IMMEDIATE overload penalty: (v7: handled above with hard -5.0 return)

            # 4. Lateness penalty: task already past deadline at assign time
            deadline_pen = 0.0
            slots_left   = task.deadline_slot - self.clock.tick
            if slots_left <= 0:
                deadline_pen = 0.5   # already late
            elif slots_left <= 2:
                deadline_pen = 0.2   # very close to deadline

            # 5. Priority bonus: urgent tasks more valuable to assign quickly
            priority_bonus = 0.05 * (task.priority + 1)

            # v7 Fix 6: Quality bonus for skill match
            quality_bonus = config.REWARD_SKILL_MATCH_WEIGHT * skill_match

            raw = base_reward + load_bonus + quality_bonus - deadline_pen + priority_bonus
            return float(np.clip(raw, -2.0, 1.0))

        elif action_type == 'defer':
            if task_idx >= len(available):
                return -0.2
            task = available[task_idx]
            # Acceptable defer only if ALL workers are either overloaded or unavailable
            has_valid = any(
                w.availability == 1 and w.load < config.MAX_WORKER_LOAD
                for w in self.workers
            )
            if has_valid:
                return -0.15  # Mild penalty: should assign when workers are free
            return 0.0  # Acceptable defer when no workers available

        elif action_type == 'escalate':
            if task_idx >= len(available):
                return 0.0
            task = available[task_idx]
            if task.assigned_worker is not None and task.priority < 3:
                task.priority = min(3, task.priority + 1)
                if task.expected_completion_tick is not None:
                    span = task.expected_completion_tick - self.clock.tick
                    task.expected_completion_tick = self.clock.tick + max(1, int(span * 0.8))
                return -1.0
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

                # v10 Fix 2: quality^2.5 completion reward (sharper gradient for skill-matched)
                priority_weight  = (task.priority + 1) / 4.0 * config.REWARD_COMPLETION_BASE
                quality_reward = priority_weight * (quality ** 2.5)

                # v10 Fix 3: Adaptive quality boost — if skill_util rate < 60%,
                # boost quality reward by 1.5× for next 20 decisions
                self._quality_window.append(1.0 if quality > 0.35 else 0.0)
                if len(self._quality_window) > 50:
                    self._quality_window = self._quality_window[-50:]
                self._decision_count += 1

                if self._quality_boost_remaining > 0:
                    quality_reward *= 1.5
                    self._quality_boost_remaining -= 1

                # Check every 10 decisions if we need to activate the boost
                if (self._decision_count % 10 == 0 and
                    len(self._quality_window) >= 20):
                    skill_util_rate = sum(self._quality_window) / len(self._quality_window)
                    if skill_util_rate < 0.60 and self._quality_boost_remaining <= 0:
                        self._quality_boost_remaining = 20
                        print(f"[QualityBoost] skill_util={skill_util_rate:.2f} < 0.60, "
                              f"activating 1.5x reward for next 20 decisions")

                completion_reward += quality_reward

                # v9 Fix 4: penalty for severely mismatched assignment (quality < 0.15)
                if quality < 0.15:
                    completion_reward -= 0.1   # REWARD_MIN_QUALITY_PENALTY

                # v8 Fix 4: Early/on-time completion bonuses
                slots_ahead = task.deadline_slot - tick
                total_window = max(task.deadline_slot - task.arrival_tick, 1)
                pct_ahead = slots_ahead / total_window
                early_bonus = getattr(config, 'REWARD_EARLY_COMPLETION_BONUS', 0.2)
                ontime_bonus = getattr(config, 'REWARD_ONTIME_COMPLETION_BONUS', 0.1)
                if pct_ahead >= 0.20:
                    completion_reward += early_bonus   # +0.2 for 20%+ ahead
                elif slots_ahead >= 0:
                    completion_reward += ontime_bonus  # +0.1 for any on-time

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
        """Penalise tasks completed THIS tick that finished after deadline (one-shot)."""
        penalty = 0.0
        tick = self.clock.tick
        for t in self.completed_tasks:
            if (t.actual_completion_tick is not None
                    and t.actual_completion_tick == tick
                    and t.actual_completion_tick > t.deadline_slot):
                slots_late = t.actual_completion_tick - t.deadline_slot
                penalty += config.REWARD_LATENESS_PENALTY * slots_late
        return penalty

    def _compute_urgency_penalty(self) -> float:
        """Penalise unstarted tasks close to their deadline (8-slot / 4h window)."""
        penalty = 0.0
        tick = self.clock.tick
        for t in self.tasks:
            if t.is_available(tick) and t.is_unassigned():
                slots_left = t.slots_until_deadline(tick)
                if slots_left <= 8:
                    penalty += config.REWARD_URGENCY_PENALTY
        return penalty

    def _compute_overload_penalty(self) -> float:
        """v7: Fatal -5.0 per worker at/above capacity + -0.5 cumulative surcharge."""
        penalty = 0.0
        for w in self.workers:
            if w.load >= config.MAX_WORKER_LOAD:
                penalty += config.REWARD_OVERLOAD_WEIGHT  # -5.0 per worker
                excess = w.load - config.MAX_WORKER_LOAD + 1
                penalty += -0.5 * excess  # cumulative surcharge
        return penalty

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
        """v9 Fix 3: Overflow drain window — don't hard-stop at time limit.

        When configured day count is reached:
          1. Immediately completes any active in-progress tasks (they finish at clock.tick).
          2. Allows up to 2 extra working days (2 × SLOTS_PER_DAY overflow ticks) to drain
             tasks that have already arrived but are unassigned.
          3. After overflow exhausted, marks all remaining non-failed tasks as failed
             with reason='simulation_boundary' and terminates.
        """
        MAX_OVERFLOW_SLOTS = 2 * config.SLOTS_PER_DAY  # 2 extra working days

        # All tasks completed — best case
        if len(self.completed_tasks) == len(self.tasks):
            return True, 'all_completed'

        # Catastrophic failure rate — terminate early regardless of overflow window
        failure_rate = len(self.failed_tasks) / max(len(self.tasks), 1)
        if failure_rate >= 0.6:
            return True, 'catastrophic_failure'

        # Primary time limit reached — enter or continue overflow window
        if self.clock.tick >= self._total_sim_slots:
            self._overflow_started = True
            self._overflow_ticks = self.clock.tick - self._total_sim_slots

            # Check if any task is still in-progress or arrived-unassigned
            in_progress = [
                t for t in self.tasks
                if t.assigned_worker is not None and not t.is_completed and not t.is_failed
            ]
            arrived_unassigned = [
                t for t in self.tasks
                if t.arrival_tick <= self.clock.tick
                and t.assigned_worker is None
                and not t.is_completed
                and not t.is_failed
            ]

            still_workable = in_progress + arrived_unassigned

            if not still_workable or self._overflow_ticks >= MAX_OVERFLOW_SLOTS:
                # Mark all remaining non-failed, non-completed tasks as failed
                boundary_count = 0
                for t in self.tasks:
                    if not t.is_completed and not t.is_failed:
                        t.is_failed = True
                        t.failure_reason = 'simulation_boundary'
                        self.failed_tasks.append(t)
                        boundary_count += 1
                if boundary_count:
                    print(f"  [Termination] Marked {boundary_count} tasks as 'simulation_boundary' "
                          f"after {self._overflow_ticks}-tick overflow window.")
                return True, 'time_limit'

            # Still within overflow — continue running
            return False, None

        return False, None

    def _compute_terminal_penalty(self) -> float:
        """
        One-time terminal penalty applied at episode end.

        Penalises tasks that are:
          - Still in the queue (unstarted, unassigned)
          - In-progress but not yet completed

        This prevents the agent from gaming the episode by ending with backlog.
        """
        unfinished_count = sum(
            1 for t in self.tasks
            if not t.is_completed and not t.is_failed
        )
        penalty = config.REWARD_TERMINAL_UNFINISHED_PENALTY * unfinished_count
        return float(penalty)

    # ── Adaptive Event-Driven Time Stepping ──────────────────────────────────

    def advance_to_next_event(self) -> int:
        """
        Advance the simulation clock to the tick of the next meaningful event,
        rather than advancing one tick at a time.

        A 'meaningful event' is defined as:
          1. A task arrival (a task becomes available that wasn't before)
          2. A task completion (an in-progress task is expected to finish)
          3. A deadline expiration (a task's deadline slot)

        If no future events exist within the simulation horizon, advance to
        the end.

        Returns:
            Number of ticks advanced.

        Design:
          Applied consistently in both DQN and baseline runners when there are
          no valid assign actions — skips idle ticks efficiently without losing
          scheduling accuracy.
        """
        current_tick = self.clock.tick
        horizon      = self._total_sim_slots
        # Cap per advance to avoid skipping past multiple events at once
        MAX_JUMP     = max(1, config.SLOTS_PER_DAY)  # at most 1 day forward

        candidate_ticks = []

        # Next task arrival (task not yet visible)
        for t in self.tasks:
            if t.arrival_tick > current_tick and not t.is_completed and not t.is_failed:
                candidate_ticks.append(t.arrival_tick)

        # Next expected task completion (in-progress tasks)
        for t in self.tasks:
            if (t.assigned_worker is not None
                    and not t.is_completed
                    and t.expected_completion_tick is not None
                    and t.expected_completion_tick > current_tick):
                candidate_ticks.append(t.expected_completion_tick)

        # Next deadline (to avoid missing failures)
        for t in self.tasks:
            if (not t.is_completed and not t.is_failed
                    and t.deadline_slot > current_tick):
                candidate_ticks.append(t.deadline_slot)

        if candidate_ticks:
            next_event = min(candidate_ticks)
            # Clamp: don't jump past horizon, and don't jump too far in one go
            jump_to = min(next_event, current_tick + MAX_JUMP, horizon)
        else:
            jump_to = min(current_tick + MAX_JUMP, horizon)

        # Advance clock tick-by-tick through the gap, applying fatigue/deadline checks
        ticks_advanced = 0
        while self.clock.tick < jump_to:
            # Fatigue update
            if self.enable_fatigue:
                for w in self.workers:
                    w.update_fatigue()

            if self.clock.is_start_of_day():
                for w in self.workers:
                    w.daily_reset()

            self.clock.advance()
            ticks_advanced += 1

            # Check and update task progress for in-progress tasks
            for task in self.tasks:
                if task.assigned_worker is not None and not task.is_completed:
                    completed = task.update_progress(self.clock.tick)
                    if completed:
                        worker = self.workers[task.assigned_worker]
                        _, quality = worker.complete_task(task.task_id, task.complexity)
                        task.quality_score = quality
                        self.belief_state.update(task.assigned_worker, quality)
                        self.completed_tasks.append(task)
                        self.metrics['throughput'] += 1

            # Check deadline failures
            for t in self.tasks:
                if t.arrival_tick <= self.clock.tick and not t.is_completed:
                    t.check_deadline(self.clock.tick)
            self.failed_tasks = [
                t for t in self.tasks
                if t.is_failed and t not in self.failed_tasks
            ] + self.failed_tasks
            # Deduplicate failed_tasks
            seen = set()
            deduped = []
            for t in self.failed_tasks:
                if t.task_id not in seen:
                    seen.add(t.task_id)
                    deduped.append(t)
            self.failed_tasks = deduped

            # If a new assign action becomes possible, stop advancing early
            if self._get_available_tasks():
                break

        return ticks_advanced

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

        # ── Global context (6 → 7) ───────────────────────────────────────────
        time_progress     = self.clock.tick / self._total_sim_slots
        completion_rate   = len(self.completed_tasks) / max(len(self.tasks), 1)
        failure_rate      = len(self.failed_tasks)    / max(len(self.tasks), 1)
        n_idle            = sum(1 for w in self.workers if w.availability == 1 and w.load == 0)
        n_available       = len(visible_tasks)
        day_progress      = self.clock.slot_in_day / config.SLOTS_PER_DAY
        # Problem 4 fix: add overloaded-workers count so agent can see imbalance
        n_overloaded      = sum(1 for w in self.workers if w.load >= config.MAX_WORKER_LOAD)

        global_features = np.array([
            time_progress, completion_rate, failure_rate,
            n_idle / max(self.num_workers, 1),
            n_available / max(self._max_visible_tasks, 1),
            day_progress,
            n_overloaded / max(self.num_workers, 1),  # fraction of workers overloaded
        ], dtype=np.float32)

        # ── Concatenate → 25 + 50 + 10 + 7 = 92
        # v10 Fix 2: skill-match features for top task vs ALL workers = 5 dims
        # Each value = clip(worker_skill_mean / match_target, 0, 1)
        # This gives the DQN direct observability of which worker best matches a task.
        top_task = visible_tasks[0] if visible_tasks else None
        if top_task:
            req_skill = max(getattr(top_task, 'required_skill', 0.5), 0.1)
            complexity_proxy = max(top_task.complexity, 1)
            match_target = max(req_skill, complexity_proxy * 0.3)  # blend both signals
            skill_matches = np.array([
                float(np.clip(
                    self.belief_state.get_skill_mean(i) / match_target, 0.0, 1.0
                ))
                for i in range(min(self.num_workers, 5))
            ] + [0.0] * max(0, 5 - self.num_workers), dtype=np.float32)
        else:
            skill_matches = np.zeros(5, dtype=np.float32)

        # Total: 92 + 5 = 97 → truncated to STATE_DIM (96), or padded if <96
        state = np.concatenate([worker_features, task_features, belief_features, global_features, skill_matches])
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
                # Problem 5 fix: NEVER allow assigning to an overloaded worker
                if worker.availability == 1 and worker.load < config.MAX_WORKER_LOAD:
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
        metrics['deadline_hit_rate'] = 1.0 - metrics['lateness_rate']
        
        avg_lateness = 0.0
        if late_tasks:
            avg_lateness = float(np.mean([
                (t.actual_completion_tick - t.deadline_slot) * config.SLOT_HOURS
                for t in late_tasks
            ]))
        metrics['avg_lateness_h'] = avg_lateness
        metrics['avg_delay'] = avg_lateness  # Alias for evaluation scripts

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
            'deadline_hit_rate': 0.0,
            'avg_lateness_h':    0.0,
            'avg_delay':         0.0,
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
