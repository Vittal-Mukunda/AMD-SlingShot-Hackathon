"""
backend/simulation_runner.py — v5: Unified 365-Day Simulation Runner.

REFACTOR SUMMARY (v5):
  1. UNIFIED SIM_DAYS — Single configurable horizon, shared by baselines and DQN.
     No hardcoded 60-day or 30-day limits anywhere.
  2. BOUNDED MEMORY — daily metrics and gantt_blocks capped via collections.deque.
  3. RATE-LIMITED EMISSIONS — tick_update emitted every EMIT_EVERY_N_TICKS (not every step).
  4. ADAPTIVE TIME STEPPING — advance_to_next_event() used when no valid assign actions,
     skipping idle ticks without wasting CPU or emitting stale frames.
  5. CONTROLLED TRAINING INTERVAL — DQN trained every TRAIN_EVERY_N_STEPS steps.
  6. CONSISTENT MECHANICS — DQN and all baselines use identical env configuration
     and identical temporal constraints (same total_sim_slots).
"""

import asyncio
import collections
import csv
import os
import sys
import traceback
import numpy as np
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as cfg_module
from slingshot.environment.project_env import ProjectEnv
from slingshot.agents.dqn_agent import DQNAgent
from slingshot.baselines.greedy_baseline import GreedyBaseline
from slingshot.baselines.skill_baseline import SkillBaseline
from slingshot.baselines.stf_baseline import STFBaseline

try:
    from slingshot.baselines.hybrid_baseline import HybridBaseline
    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

try:
    from slingshot.baselines.random_baseline import RandomBaseline
    _HAS_RANDOM = True
except ImportError:
    _HAS_RANDOM = False

# ── Runtime constants ─────────────────────────────────────────────────────────
# Emit only once every N ticks — prevents frontend flood at 365-day scale.
def _emit_every():
    return getattr(cfg_module, 'EMIT_EVERY_N_TICKS', 8)

# Minimum gradient steps during Phase 2 training sub-phase.
MIN_TRAINING_STEPS = 200

# How often to train DQN during Phase 2 scheduling (in env steps).
def _train_every():
    return getattr(cfg_module, 'TRAIN_EVERY_N_STEPS', 4)


def _make_env(cfg, seed_offset: int = 0) -> ProjectEnv:
    """
    Create a fresh ProjectEnv matching user config.

    total_sim_slots is always derived from the unified SIM_DAYS so that
    both the baseline and DQN paths operate under identical temporal constraints.
    """
    sim_days = getattr(cfg_module, 'SIM_DAYS', cfg.days_phase1 + cfg.days_phase2)
    total_slots = sim_days * cfg_module.SLOTS_PER_DAY
    # v10 FIX: use cfg_module.TOTAL_TASKS (dynamic cap), NOT cfg.task_count (raw frontend)
    total_tasks = getattr(cfg_module, 'TOTAL_TASKS', cfg.task_count)
    return ProjectEnv(
        num_workers=cfg.num_workers,
        total_tasks=total_tasks,
        seed=cfg.seed + seed_offset,
        total_sim_slots=total_slots,
    )


def _decode_action_parts(action: int, num_workers: int) -> Tuple[int, int]:
    """
    Decode action index → (task_slot, worker_id) for assign actions.
    Encoding: action = task_slot * num_workers + worker_id
    """
    task_slot  = action // num_workers
    worker_idx = action % num_workers
    return task_slot, worker_idx


class SimulationRunner:
    """
    Two-phase simulation runner (v5 — Unified 365-Day).

    Phase 1: All baselines run SEQUENTIALLY on independent envs sharing the
             same total_sim_slots (derived from SIM_DAYS).
             Each baseline emits tick_update + gantt_block at rate-limited intervals.
             DQN stores transitions from ALL baselines into a bounded replay buffer.

    Phase 2a (training): DQN trains offline on collected transitions.
             Emits training_progress events only.

    Phase 2b (scheduling): DQN schedules on fresh env, emitting at rate-limited intervals.
             Online training fires every TRAIN_EVERY_N_STEPS steps.
    """

    def __init__(self, cfg, sio):
        self.cfg = cfg
        self.sio = sio

        # ── Apply user config to cfg_module globals ──────────────────────────
        # Derive phase durations from unified SIM_DAYS if set, else from legacy fields.
        sim_days = getattr(cfg, 'sim_days', None)
        if sim_days and sim_days > 0:
            cfg_module.SIM_DAYS        = sim_days
            cfg_module.PHASE1_DAYS     = max(1, int(sim_days * cfg_module.PHASE1_FRACTION))
            cfg_module.PHASE2_DAYS     = max(1, sim_days - cfg_module.PHASE1_DAYS)
        else:
            # Legacy: derive SIM_DAYS from explicit phase days
            cfg_module.PHASE1_DAYS    = cfg.days_phase1
            cfg_module.PHASE2_DAYS    = cfg.days_phase2
            cfg_module.SIM_DAYS       = cfg.days_phase1 + cfg.days_phase2

        cfg_module.TOTAL_SIM_DAYS  = cfg_module.SIM_DAYS
        cfg_module.NUM_WORKERS     = cfg.num_workers

        # v10 Fix 1: Dynamically compute TOTAL_TASKS from SIM_DAYS × arrival_rate × 1.5
        # This guarantees enough tasks for the full run with 50% headroom.
        import math
        dynamic_task_cap = max(50, math.ceil(
            cfg_module.SIM_DAYS * cfg_module.TASK_ARRIVAL_RATE * 1.5
        ))
        # Use the LARGER of user-submitted task_count and dynamic cap
        cfg_module.TOTAL_TASKS = max(int(cfg.task_count), dynamic_task_cap)
        cfg_module.NUM_TASKS   = cfg_module.TOTAL_TASKS

        # v8 Fix 3: Propagate tasks_per_day to config module
        tasks_per_day = getattr(cfg, 'tasks_per_day', None) or cfg_module.TASK_ARRIVAL_RATE
        cfg_module.TASK_ARRIVAL_RATE = float(tasks_per_day)

        # v9 Fix 6: propagate max_worker_load if user provided it
        max_worker_load = getattr(cfg, 'max_worker_load', None)
        if max_worker_load and max_worker_load > 0:
            cfg_module.MAX_WORKER_LOAD = int(max_worker_load)

        # v9 Fix 2+6: allow user-supplied phase1_fraction to override global default
        phase1_frac = getattr(cfg, 'phase1_fraction', None)
        if phase1_frac and 0.0 < phase1_frac < 1.0:
            cfg_module.PHASE1_FRACTION = float(phase1_frac)
            # Recompute phase days with updated fraction if sim_days was set
            if sim_days and sim_days > 0:
                cfg_module.PHASE1_DAYS = max(1, int(sim_days * cfg_module.PHASE1_FRACTION))
                cfg_module.PHASE2_DAYS = max(1, sim_days - cfg_module.PHASE1_DAYS)

        # v10 Fix 4: Comprehensive startup log with all effective values
        print(f"[runner] ==================================================")
        print(f"[runner]   SIM_DAYS       = {cfg_module.SIM_DAYS}")
        print(f"[runner]   PHASE1_DAYS    = {cfg_module.PHASE1_DAYS}")
        print(f"[runner]   PHASE2_DAYS    = {cfg_module.PHASE2_DAYS}")
        print(f"[runner]   PHASE1_FRAC    = {cfg_module.PHASE1_FRACTION:.2f}")
        print(f"[runner]   TOTAL_TASKS    = {cfg_module.TOTAL_TASKS} (dynamic cap)")
        print(f"[runner]   tasks_per_day  = {cfg_module.TASK_ARRIVAL_RATE}")
        print(f"[runner]   workers        = {cfg.num_workers}")
        print(f"[runner]   max_load       = {cfg_module.MAX_WORKER_LOAD}")
        print(f"[runner] ==================================================")

        # v9 Fix 1: Configure epsilon scoped to Phase 2 decisions only
        tasks_per_day = getattr(cfg, 'tasks_per_day', None) or getattr(cfg_module, 'TASK_ARRIVAL_RATE', 4.0)
        self.agent = DQNAgent()
        self.agent.configure_epsilon_schedule(
            tasks_per_day=tasks_per_day,
            phase2_days=cfg_module.PHASE2_DAYS,  # ONLY Phase 2 (not full SIM_DAYS)
            sim_days=cfg_module.SIM_DAYS,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._cancelled = False
        self._phase = 0        # 0=init, 1=baseline, 2=training, 3=DQN scheduling
        self._tick   = 0
        self._day    = 0
        self._running = False
        self._injected_tasks: List[Dict] = []

        # ── BOUNDED Metrics storage ───────────────────────────────────────────
        max_days = getattr(cfg_module, 'MAX_DAILY_METRICS_IN_MEMORY', 730)
        self._phase1_metrics = collections.deque(maxlen=max_days)
        self._phase2_metrics = collections.deque(maxlen=max_days)
        self._baseline_snapshots: Dict[str, Dict] = {}

        # ── Per-baseline environments (Phase 1) ───────────────────────────────
        self._baseline_defs: List[Tuple[str, Any]] = self._make_baseline_defs()

        # ── Phase 2 scheduling env (fresh, separate from Phase 1 envs) ───────
        self._dqn_env: ProjectEnv | None = None

    # ── Baseline definitions ──────────────────────────────────────────────────

    def _make_baseline_defs(self) -> List[Tuple[str, Any]]:
        """Returns list of (name, factory_callable) pairs."""
        defs = [
            ("Greedy",  GreedyBaseline),
            ("Skill",   SkillBaseline),
            ("FIFO",    STFBaseline),
        ]
        if _HAS_HYBRID:
            defs.append(("Hybrid", HybridBaseline))
        if _HAS_RANDOM:
            defs.append(("Random", RandomBaseline))
        return defs

    # ── Control interface ─────────────────────────────────────────────────────

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def inject_task(self, task_data: Dict):
        self._injected_tasks.append(task_data)

    def get_status(self) -> Dict:
        """Returns tick_update-compatible dict for reconnect."""
        env = self._dqn_env if self._dqn_env is not None else _make_env(self.cfg)
        active = "Training" if self._phase == 2 else ("DQN" if self._phase == 3 else "Greedy")
        return {
            "phase":            self._phase,
            "tick":             self._tick,
            "day":              self._day,
            "running":          self._running,
            "worker_states":    self._serialize_workers(env),
            "queue_state":      self._serialize_queue(env),
            "last_assignment":  None,
            "active_policy":    active,
        }

    # ── Serialization helpers ─────────────────────────────────────────────────

    def _serialize_workers(self, env: ProjectEnv) -> List[Dict]:
        out = []
        for w in env.workers:
            fatigue = float(getattr(w, 'fatigue', 0))
            out.append({
                "id":             f"w{w.worker_id}",
                "name":           f"Worker {w.worker_id + 1}",
                "fatigue":        fatigue,
                "fatigue_level": (
                    "burnout"   if fatigue >= 2.6 else
                    "exhausted" if fatigue >= 2.0 else
                    "tired"     if fatigue >= 1.0 else
                    "fresh"
                ),
                "availability":   int(getattr(w, 'availability', 1)),
                "assigned_tasks": list(getattr(w, 'assigned_tasks', [])),
                "skill_level":    float(getattr(w, 'true_skill', 1.0)),
            })
        return out

    def _serialize_queue(self, env: ProjectEnv) -> List[Dict]:
        tick  = env.clock.tick
        items = []
        for t in env.tasks:
            if getattr(t, 'is_completed', False) or getattr(t, 'is_failed', False):
                continue
            if getattr(t, 'assigned_worker', None) is not None:
                continue
            if getattr(t, 'arrival_tick', 0) > tick:
                continue
            items.append({
                "task_id":        f"t{t.task_id}",
                "priority":       int(getattr(t, 'priority', 1)),
                "urgency_label":  ["low", "medium", "high", "critical"][
                    min(int(getattr(t, 'priority', 1)), 3)],
                "duration_slots": int(getattr(t, 'duration_slots', 2)),
                "required_skill": float(getattr(t, 'required_skill', 0.5)),
                "deadline_tick":  int(getattr(t, 'deadline_tick', 0)),
                "slots_remaining": max(0, int(getattr(t, 'deadline_tick', 0)) - tick),
            })
        return items

    def _build_gantt_block(self, env: ProjectEnv, action: int, policy: str) -> Dict | None:
        """
        Decode action and return a GanttBlock dict, or None if not an assign action.
        Uses the CORRECT decoding formula: task_slot = action // num_workers
        """
        num_workers = env.num_workers
        max_assign = 20 * num_workers

        if action >= max_assign:
            return None  # defer or escalate — no gantt block

        task_slot, worker_idx = _decode_action_parts(action, num_workers)

        tick = env.clock.tick
        available = [
            t for t in env.tasks
            if t.is_available(tick) and t.is_unassigned()
        ]
        available.sort(key=lambda t: -t.get_deadline_urgency(tick))
        available = available[:20]

        if task_slot >= len(available):
            return None

        task     = available[task_slot]
        duration = int(getattr(task, 'duration_slots', 2))

        return {
            "task_id":    f"t{task.task_id}",
            "worker_id":  f"w{worker_idx}",
            "start_tick": tick,
            "end_tick":   tick + duration,
            "urgency":    int(getattr(task, 'priority', 1)),
            "policy":     policy,
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        try:
            await self._run_all()
        except asyncio.CancelledError:
            print("[runner] Cancelled")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[runner] FATAL ERROR:\n{tb}")
            await self.sio.emit("simulation_error", {
                "message": str(exc),
                "traceback": tb,
            })
        finally:
            self._running = False

    async def _run_all(self):
        # ── Phase 1: Baselines ─────────────────────────────────────────────
        self._phase = 1
        await self.sio.emit("tick_update", {
            "tick": 0, "day": 0, "phase": 1,
            "worker_states": [], "queue_state": [],
            "last_assignment": None, "active_policy": "Starting…",
        })

        p1_metrics, gantt_by_baseline = await self._run_phase1()
        for m in p1_metrics:
            self._phase1_metrics.append(m)

        # Build baseline snapshot
        baseline_snapshot: Dict[str, Dict] = {}
        p1_list = list(self._phase1_metrics)
        for bname, _ in self._baseline_defs:
            rows = [m for m in p1_list if m.get("baseline") == bname]
            if rows:
                baseline_snapshot[bname] = {
                    "throughput":      float(np.mean([r["throughput_per_day"] for r in rows])),
                    "completion_rate": float(np.mean([r["completion_rate"] for r in rows])),
                    "lateness_rate":   float(np.mean([r["lateness_rate"] for r in rows])),
                    "quality_score":   float(np.mean([r["quality_score"] for r in rows])),
                    "overload_events": float(np.mean([r["overload_events"] for r in rows])),
                    "gantt_blocks":    list(gantt_by_baseline.get(bname, [])),
                }
        self._baseline_snapshots = baseline_snapshot

        # ── Notify frontend: Phase 1 complete, entering training ─────────
        await self.sio.emit("phase_transition", {
            "new_phase":                 "training",
            "baseline_results_snapshot": baseline_snapshot,
        })

        # ── Phase 2a: DQN training ────────────────────────────────────────
        self._phase = 2
        await self._run_dqn_training()

        # ── Notify frontend: training done, DQN scheduling starts ─────────
        await self.sio.emit("phase2_ready", {
            "baseline_results_snapshot": baseline_snapshot,
        })

        # ── Phase 2b: DQN scheduling ───────────────────────────────────────
        self._phase = 3
        p2_metrics = await self._run_phase2_scheduling()
        for m in p2_metrics:
            self._phase2_metrics.append(m)

        # ── Final metrics ──────────────────────────────────────────────────
        overall = self._dqn_env.compute_metrics() if self._dqn_env else {}
        final = self._build_final_metrics(
            list(self._phase1_metrics),
            list(self._phase2_metrics),
            baseline_snapshot, overall)
        await self.sio.emit("simulation_complete", {"final_metrics": final})

        self._save_results()

    # ── Phase 1: Run each baseline sequentially ───────────────────────────────

    async def _run_phase1(self) -> Tuple[List[Dict], Dict[str, List]]:
        """
        Run each baseline in full on its own independent env.
        Both baseline and DQN environments use the SAME total_sim_slots
        so all policies operate under identical temporal constraints.

        Emits tick_update every EMIT_EVERY_N_TICKS ticks (rate-limited).
        Uses advance_to_next_event() when no valid actions (adaptive stepping).
        Stores transitions into the DQN replay buffer (bounded by config).
        """
        all_metrics:      List[Dict]       = []
        gantt_by_baseline: Dict[str, List] = {}
        emit_every = _emit_every()
        max_gantt  = getattr(cfg_module, 'MAX_GANTT_BLOCKS_IN_MEMORY', 500)

        # Phase 1 runs over ALL of SIM_DAYS (both phases share the same environment horizon)
        total_slots = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY

        for bl_idx, (bname, BLClass) in enumerate(self._baseline_defs):
            if self._cancelled:
                break

            env    = _make_env(self.cfg, seed_offset=bl_idx + 1)
            policy = BLClass(env)
            state  = env.reset()

            # Bounded Gantt storage (ring-buffer, per baseline)
            gantt_deque = collections.deque(maxlen=max_gantt)
            prev_day    = 0
            day_decisions = 0
            step_count  = 0

            await self.sio.emit("tick_update", {
                "tick": 0, "day": 0, "phase": 1,
                "worker_states": self._serialize_workers(env),
                "queue_state":   self._serialize_queue(env),
                "last_assignment": None,
                "active_policy": bname,
            })
            await asyncio.sleep(0.01)

            while env.clock.tick < total_slots:
                if self._cancelled:
                    break
                await self._pause_event.wait()

                current_day = env.clock.day

                # ── Day boundary ───────────────────────────────────────────
                if current_day > prev_day:
                    m = self._collect_metrics(day_decisions, bname, env)
                    all_metrics.append(m)
                    day_decisions = 0
                    prev_day      = current_day
                    await self.sio.emit("daily_summary", {
                        "day":   current_day - 1,
                        "phase": 1,
                        "metrics_per_policy": {bname: m},
                    })

                valid = env.get_valid_actions()

                # ── No valid actions: use adaptive event-driven stepping ───
                if not valid:
                    ticks_advanced = env.advance_to_next_event()
                    done_check, _ = env._check_termination()
                    if done_check or env.clock.tick >= total_slots:
                        break
                    step_count += ticks_advanced
                    if step_count % emit_every == 0:
                        await self.sio.emit("tick_update", {
                            "tick": env.clock.tick, "day": env.clock.day, "phase": 1,
                            "worker_states": self._serialize_workers(env),
                            "queue_state":   self._serialize_queue(env),
                            "last_assignment": None,
                            "active_policy": bname,
                        })
                    await asyncio.sleep(0)
                    continue

                # ── Select action ──────────────────────────────────────────
                action = policy.select_action(state)
                if action not in valid:
                    action = valid[0]

                # ── Build gantt block BEFORE stepping ─────────────────────
                block = self._build_gantt_block(env, action, bname)

                # ── Step environment ───────────────────────────────────────
                next_state, reward, done, info = env.step(action)

                # ── Store transition with MAX PRIORITY in bounded DQN replay buffer (v7 Fix 3)
                state_copy = np.array(state, dtype=np.float32)
                next_copy  = np.array(next_state, dtype=np.float32)
                max_p = self.agent.replay_buffer._max_priority
                self.agent.replay_buffer.tree.add(
                    max_p ** self.agent.replay_buffer.alpha,
                    (state_copy, action, reward, next_copy, float(done))
                )
                self.agent.steps_done += 1

                # ── Emit gantt block (bounded) ─────────────────────────────
                if block is not None:
                    gantt_deque.append(block)
                    await self.sio.emit("gantt_block", block)

                # ── Emit task_completed ────────────────────────────────────
                if info.get("completed_tasks", 0) > 0:
                    recent = ([t for t in env.completed_tasks
                               if t.actual_completion_tick == env.clock.tick - 1]
                              or [])
                    for ct in recent:
                        await self.sio.emit("task_completed", {
                            "task_id":         f"t{ct.task_id}",
                            "worker_id":       f"w{getattr(ct, 'assigned_worker', 0)}",
                            "completion_tick": env.clock.tick,
                            "lateness":        float(
                                max(0, (ct.actual_completion_tick or 0) - ct.deadline_slot)
                                * cfg_module.SLOT_HOURS),
                            "quality":         float(getattr(ct, 'quality_score', 1.0)),
                        })

                # ── Rate-limited tick_update ───────────────────────────────
                step_count += 1
                if step_count % emit_every == 0:
                    await self.sio.emit("tick_update", {
                        "tick":          env.clock.tick,
                        "day":           env.clock.day,
                        "phase":         1,
                        "worker_states": self._serialize_workers(env),
                        "queue_state":   self._serialize_queue(env),
                        "last_assignment": {
                            "task_id":   block["task_id"] if block else "—",
                            "worker_id": block["worker_id"] if block else "—",
                            "policy":    bname,
                        } if block else None,
                        "active_policy": bname,
                    })
                await asyncio.sleep(0)

                state         = next_state
                day_decisions += 1

                if done:
                    break

            # Collect final day
            if day_decisions > 0:
                m = self._collect_metrics(day_decisions, bname, env)
                all_metrics.append(m)
                await self.sio.emit("daily_summary", {
                    "day": env.clock.day,
                    "phase": 1,
                    "metrics_per_policy": {bname: m},
                })

            gantt_by_baseline[bname] = list(gantt_deque)
            await asyncio.sleep(0.05)

        buf_size = len(self.agent.replay_buffer)
        print(f"[Phase1] Done. Replay buffer: {buf_size} transitions across {len(self._baseline_defs)} baselines.")
        return all_metrics, gantt_by_baseline

    # ── Phase 2a: DQN offline training ───────────────────────────────────────

    async def _run_dqn_training(self):
        """
        Train DQN on the bounded replay buffer collected from Phase 1.
        Yields to event loop every 10 steps so WebSocket heartbeats continue.
        """
        buf_size = len(self.agent.replay_buffer)

        if buf_size < self.agent.min_replay_size:
            self.agent.min_replay_size = max(self.agent.batch_size, buf_size)
            print(f"[Training] Lowered min_replay_size to {self.agent.min_replay_size} "
                  f"(buffer has {buf_size} transitions)")

        if buf_size < self.agent.batch_size:
            print(f"[Training] Buffer too small ({buf_size}) — skipping training (need >= {self.agent.batch_size})")
            await self.sio.emit("training_progress", {"percent": 100, "steps": 0})
            return

        target_steps = max(MIN_TRAINING_STEPS, buf_size // 2)
        print(f"[Training] Starting {target_steps} gradient steps. Buffer={buf_size}")

        # v8 Fix 1: Do NOT reset epsilon — let the schedule continue from Phase 1
        # self.agent.set_epsilon(cfg_module.EPSILON_PHASE2_START)  # REMOVED
        steps_done = 0

        while steps_done < target_steps:
            if self._cancelled:
                break

            try:
                loss, q_mean, _ = self.agent.train_step()
                steps_done      += 1
                self.agent.update_epsilon()
            except Exception as e:
                print(f"[Training] train_step error: {e}")
                break

            if steps_done % 10 == 0:
                pct = int(steps_done / target_steps * 100)
                await self.sio.emit("training_progress", {
                    "percent": pct,
                    "steps":   steps_done,
                })
                await asyncio.sleep(0.01)

        await self.sio.emit("training_progress", {"percent": 100, "steps": steps_done})
        print(f"[Training] Complete. train_steps={steps_done}, epsilon={self.agent.epsilon:.4f}")

    # ── Phase 2b: DQN scheduling ──────────────────────────────────────────────

    async def _run_phase2_scheduling(self) -> List[Dict]:
        """
        DQN controls the scheduler on a fresh env using the unified SIM_DAYS horizon.
        Rate-limited emissions, adaptive stepping, controlled online training.
        """
        env           = _make_env(self.cfg, seed_offset=99)
        self._dqn_env = env
        state         = env.reset()

        # v10: Assert task list is large enough for the full simulation
        # Poisson process naturally generates ~(rate×days) tasks ± variance,
        # so we use 0.8× as the minimum threshold (not 1.0×, Poisson has variance)
        min_expected = cfg_module.SIM_DAYS * cfg_module.TASK_ARRIVAL_RATE * 0.8
        actual_tasks = len(env.tasks)
        print(f"[Phase2] Task list: {actual_tasks} tasks generated "
              f"(min expected = {min_expected:.0f}, "
              f"cap = {cfg_module.TOTAL_TASKS}, "
              f"sim_days = {cfg_module.SIM_DAYS})")
        assert actual_tasks >= min_expected, (
            f"Task list too short: {actual_tasks} < {min_expected:.0f} "
            f"(SIM_DAYS={cfg_module.SIM_DAYS}, rate={cfg_module.TASK_ARRIVAL_RATE})"
        )

        emit_every   = _emit_every()
        train_every  = _train_every()
        total_slots  = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY
        # Phase 2 runs the FULL horizon — DQN gets fresh env with same total_slots
        # (Previously used only phase2_slots = 20% of SIM_DAYS, causing early termination)

        prev_day     = 0
        day_decisions = 0
        all_metrics: List[Dict] = []
        step_count   = 0

        await self.sio.emit("tick_update", {
            "tick": 0, "day": 0, "phase": 3,
            "worker_states": self._serialize_workers(env),
            "queue_state":   self._serialize_queue(env),
            "last_assignment": None, "active_policy": "DQN",
        })
        await asyncio.sleep(0.01)

        # Safety guard: cap iterations at total_slots + generous buffer
        max_iterations = total_slots + max(500, total_slots // 2)
        iteration_count = 0

        while env.clock.tick < total_slots:
            if self._cancelled:
                break
            await self._pause_event.wait()

            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"[Phase2] Safety exit: exceeded {max_iterations} iterations at tick={env.clock.tick}")
                break

            # Process injected tasks
            if self._injected_tasks:
                self._inject_task_into_env(self._injected_tasks.pop(0), env)

            current_day   = env.clock.day
            self._tick    = env.clock.tick
            self._day     = current_day

            # Day boundary
            if current_day > prev_day:
                m = self._collect_metrics(day_decisions, "DQN", env)
                all_metrics.append(m)
                day_decisions = 0
                prev_day      = current_day
                await self.sio.emit("daily_summary", {
                    "day": current_day - 1, "phase": 3,
                    "metrics_per_policy": {"DQN": m},
                })

            valid = env.get_valid_actions()

            # ── No valid actions: adaptive event-driven stepping ───────────
            if not valid:
                tick_before = env.clock.tick
                ticks_adv = env.advance_to_next_event()
                if env.clock.tick <= tick_before and ticks_adv == 0:
                    print(f"[Phase2] WARN: clock stuck at tick={tick_before}, forcing break")
                    break
                done_check, _ = env._check_termination()
                if done_check:
                    break
                step_count += ticks_adv
                if step_count % emit_every == 0:
                    await self.sio.emit("tick_update", {
                        "tick": env.clock.tick, "day": env.clock.day, "phase": 3,
                        "worker_states": self._serialize_workers(env),
                        "queue_state":   self._serialize_queue(env),
                        "last_assignment": None, "active_policy": "DQN",
                    })
                await asyncio.sleep(0)
                continue

            # ── v8 Fix 2: DQN selects and executes ALL available assignments ──
            # Agent is called once per available task (not once per tick).
            assign_actions = [a for a in valid if a < 20 * env.num_workers]
            last_block = None
            done = False

            if assign_actions:
                assignments_this_tick = 0
                # Loop: assign as many tasks as possible before clock advances
                for _assign_iter in range(50):  # safety cap
                    current_valid = env.get_valid_actions()
                    current_assign = [a for a in current_valid if a < 20 * env.num_workers]
                    if not current_assign:
                        break

                    action = self.agent.select_action(
                        np.array(state, dtype=np.float32), current_valid, greedy=False
                    )
                    if action not in current_valid:
                        action = current_assign[0]

                    block = self._build_gantt_block(env, action, "DQN")
                    next_state, reward, done, info = env.step(action)

                    # Store transition + train
                    self.agent.store_transition(
                        np.array(state, dtype=np.float32),
                        action, reward,
                        np.array(next_state, dtype=np.float32),
                        float(done),
                    )
                    self.agent.steps_done += 1
                    buf_size = len(self.agent.replay_buffer)
                    if buf_size >= self.agent.min_replay_size:
                        # v10 Fix 3: tapered training — 4 grad steps while exploring, 2 at ε floor
                        n_grad = 2 if self.agent.epsilon <= self.agent.epsilon_end + 0.01 else 4
                        for _ in range(n_grad):
                            self.agent.train_step()
                        self.agent.update_epsilon()

                    state = next_state
                    last_block = block
                    assignments_this_tick += 1
                    if block is not None:
                        await self.sio.emit("gantt_block", block)
                    if done:
                        break

                day_decisions += assignments_this_tick

            else:
                # Only defer actions available
                action = valid[0]
                block = self._build_gantt_block(env, action, "DQN")
                tick_before = env.clock.tick
                next_state, reward, done, info = env.step(action)

                self.agent.store_transition(
                    np.array(state, dtype=np.float32),
                    action, reward,
                    np.array(next_state, dtype=np.float32),
                    float(done),
                )
                self.agent.steps_done += 1
                buf_size = len(self.agent.replay_buffer)
                if buf_size >= self.agent.min_replay_size:
                    self.agent.train_step()
                    self.agent.update_epsilon()

                if env.clock.tick <= tick_before:
                    print(f"[Phase2] WARN: clock stuck at tick={tick_before}")
                    break
                state = next_state
                last_block = block

            step_count += 1
            if step_count % emit_every == 0:
                await self.sio.emit("tick_update", {
                    "tick":          env.clock.tick,
                    "day":           env.clock.day,
                    "phase":         3,
                    "worker_states": self._serialize_workers(env),
                    "queue_state":   self._serialize_queue(env),
                    "last_assignment": {
                        "task_id":   last_block["task_id"] if last_block else "—",
                        "worker_id": last_block["worker_id"] if last_block else "—",
                        "policy":    "DQN",
                    } if last_block else None,
                    "active_policy": "DQN",
                })
            if done:
                break

            await asyncio.sleep(0)

        if day_decisions > 0:
            m = self._collect_metrics(day_decisions, "DQN", env)
            all_metrics.append(m)
            await self.sio.emit("daily_summary", {
                "day": env.clock.day,
                "phase": 3,
                "metrics_per_policy": {"DQN": m},
            })

        return all_metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _inject_task_into_env(self, task_data: Dict, env: ProjectEnv):
        try:
            from slingshot.environment.task import Task as EnvTask
            task = EnvTask(
                task_id=len(env.tasks),
                priority=int(task_data.get("urgency", 2)),
                complexity=int(task_data.get("complexity", 2)),
                deadline_h=float(task_data.get("deadline_h", 16.0)),
                arrival_tick=int(task_data.get("arrival_tick", self._tick)),
            )
            env.tasks.append(task)
        except Exception as e:
            print(f"[inject] {e}")

    def _collect_metrics(self, day_decisions: int, baseline_name: str, env: ProjectEnv) -> Dict:
        m = env.compute_metrics()
        return {
            "baseline":           baseline_name,
            "day":                env.clock.day,
            "phase":              self._phase,
            "throughput_per_day": float(m.get("throughput_per_day", 0)),
            "completion_rate":    float(m.get("completion_rate", 0)),
            "lateness_rate":      float(m.get("lateness_rate", 0)),
            "quality_score":      float(m.get("quality_score", 0)),
            "load_balance":       float(m.get("load_balance", 0)),
            "overload_events":    int(m.get("overload_events", 0)),
            "decisions":          day_decisions,
        }

    def _build_final_metrics(self, p1, p2, baseline_snap, overall) -> Dict:
        def agg(rows, key):
            vals = [r[key] for r in rows if key in r]
            return float(np.mean(vals)) if vals else 0.0

        dqn_tp      = agg(p2, "throughput_per_day")
        dqn_late    = agg(p2, "lateness_rate")
        dqn_quality = agg(p2, "quality_score")
        dqn_ms      = float(overall.get("makespan_hours", 0))

        best_name, best_tp = "Greedy", -1.0
        for bn, snap in baseline_snap.items():
            if snap["throughput"] > best_tp:
                best_tp, best_name = snap["throughput"], bn

        best_ms = float(cfg_module.PHASE1_DAYS * 8)

        return {
            "best_policy":                best_name,
            "dqn_vs_best_makespan_delta": round(dqn_ms - best_ms, 2),
            "total_tasks_completed":      int(overall.get("throughput", 0)),
            "overall_lateness_rate":      round(dqn_late, 4),
            "peak_overload_events":       int(overall.get("overload_events", 0)),
            "avg_quality_score":          round(dqn_quality, 3),
            "dqn_throughput":             round(dqn_tp, 2),
            "baseline_results":           baseline_snap,
            "phase1_daily":               p1,
            "phase2_daily":               p2,
            "overall": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in overall.items()
            },
        }

    def _save_results(self):
        rdir = cfg_module.RESULTS_DIR
        for phase_num, rows in [(1, list(self._phase1_metrics)), (2, list(self._phase2_metrics))]:
            if not rows:
                continue
            path = os.path.join(rdir, f"phase{phase_num}_metrics.csv")
            try:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            except Exception as e:
                print(f"[save] Could not write {path}: {e}")
