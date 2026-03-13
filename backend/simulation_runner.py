"""
backend/simulation_runner.py -- v7: Silent-hang fixes.

FIX SUMMARY (v7 over v6):

  FIX 6 -- _pause_event.wait() gets a timeout so a missed resume() can't hang forever
    OLD: await self._pause_event.wait()   (blocks indefinitely if resume() never fires)
    NEW: await asyncio.wait_for(self._pause_event.wait(), timeout=30.0)
         On timeout: log a warning and check self._cancelled; if not cancelled, treat
         as a spurious pause and continue so the simulation always makes forward progress.
    WHY: A client disconnect mid-run calls pause() but never calls resume(). The coroutine
         then blocks at this line forever, producing a silent hang with no
         simulation_complete event.

  FIX 7 -- advance_to_next_event() zero-advance guard strengthened
    OLD: if env.clock.tick <= tick_before AND ticks_adv == 0: break
         (misses the case where advance returns 0 but clock did tick -- loop re-enters
          and calls advance again with no valid actions, cycling silently)
    NEW: if ticks_adv == 0: break
         Any zero-advance is a stuck clock regardless of tick comparison, because
         advance_to_next_event() is only called when valid actions are empty --
         if it can't move forward at all the env has stalled.

  FIX 8 -- Inner assign loop tracks clock tick and breaks on no forward progress
    OLD: for _assign_iter in range(50): ... (no clock-progress check)
         If env.step() on an assign action doesn't advance the clock AND doesn't
         set done=True, the 50-iteration cap is exhausted, outer iteration_count
         increments by 1, and the outer while immediately re-enters, repeating
         indefinitely -- max_iterations counts outer iterations, not inner ones,
         so it offers no protection here.
    NEW: record tick_at_inner_start before the inner loop; after each step check
         if the clock has not advanced AND assignments_this_tick > 0; if so, break
         the inner loop immediately and log a warning.

  FIX 9 -- simulation_complete payload stripped of per-day arrays
    OLD: emit({"final_metrics": final}) where final includes phase1_daily
         and phase2_daily -- for a 365-day run these lists total 1000+ rows
         and easily exceed the socket.io default 1 MB frame limit, causing
         the client to silently drop the event and never navigate to analytics.
    NEW: strip phase1_daily and phase2_daily from the socket payload;
         replace with compact phase1_days_count / phase2_days_count ints.
         Daily data is already written to CSV by _save_results() beforehand.
         _save_results() is also moved BEFORE the emit and wrapped in its own
         try/except so a disk error can never suppress simulation_complete.

  RETAINED from v6:
  - FIX 1: _phase1_metrics deque maxlen scaled by actual baseline count
  - FIX 2: Phase 2 task-count assertion replaced with soft warning + clamp
  - FIX 3: Phase 1 final-day metrics flush uses a snapshot of day_decisions
  - FIX 4: _build_final_metrics handles empty baseline_snap gracefully
  - FIX 5: _save_results ensures output directory exists before open()
  - max_iterations = total_slots * 50 + 10000  (DQN outer loop safety cap)
  - BOUNDED Gantt deques per baseline
  - Adaptive advance_to_next_event() stepping
  - Rate-limited tick_update emissions
  - Dynamic replay buffer sizing
"""

import asyncio
import collections
import csv
import os
import sys
import traceback
import numpy as np

# Force UTF-8 on stdout/stderr so unicode in print() doesn't crash on
# Windows (cp1252) or any other narrow-charset terminal.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as cfg_module
from slingshot.environment.project_env import ProjectEnv
from slingshot.agents.dqn_agent import DQNAgent
try:
    from slingshot.baselines.greedy_baseline import GreedyBaseline
    _HAS_GREEDY = True
except Exception as _greedy_err:
    import traceback as _tb
    print(f"[FATAL] GreedyBaseline import failed: {_greedy_err}")
    _tb.print_exc()
    _HAS_GREEDY = False

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

# -- Runtime constants ---------------------------------------------------------
def _emit_every():
    return getattr(cfg_module, 'EMIT_EVERY_N_TICKS', 8)

MIN_TRAINING_STEPS = 200

def _train_every():
    return getattr(cfg_module, 'TRAIN_EVERY_N_STEPS', 4)


async def _wait_for_resume(pause_event: asyncio.Event, cancelled_flag_getter, location: str = ""):
    """
    FIX 6: Await a pause event with a 30-second timeout so that a missed
    resume() call (e.g. from client disconnect) can never hang the simulation
    forever.  On timeout we log a warning, auto-set the event, and return so
    the caller can check _cancelled and continue or abort.
    Returns True if the loop should continue, False if cancelled.
    """
    try:
        await asyncio.wait_for(pause_event.wait(), timeout=30.0)
        return not cancelled_flag_getter()
    except asyncio.TimeoutError:
        loc = f" [{location}]" if location else ""
        print(f"[runner]{loc} WARN: _pause_event timed out after 30 s -- "
              f"resume() was never called. Auto-resuming to prevent hang.")
        pause_event.set()   # prevent re-hang on next iteration
        return not cancelled_flag_getter()


def _make_env(cfg, seed_offset: int = 0) -> ProjectEnv:
    sim_days   = getattr(cfg_module, 'SIM_DAYS', cfg.days_phase1 + cfg.days_phase2)
    total_slots = sim_days * cfg_module.SLOTS_PER_DAY
    total_tasks = getattr(cfg_module, 'TOTAL_TASKS', cfg.task_count)
    return ProjectEnv(
        num_workers=cfg.num_workers,
        total_tasks=total_tasks,
        seed=cfg.seed + seed_offset,
        total_sim_slots=total_slots,
    )


def _decode_action_parts(action: int, num_workers: int) -> Tuple[int, int]:
    task_slot  = action // num_workers
    worker_idx = action % num_workers
    return task_slot, worker_idx


class SimulationRunner:
    """
    Two-phase simulation runner (v6 -- end-of-simulation crash fixes).
    See module docstring for full fix summary.
    """

    def __init__(self, cfg, sio):
        self.cfg = cfg
        self.sio = sio

        # -- Apply user config to cfg_module globals --------------------------
        sim_days = getattr(cfg, 'sim_days', None)
        if sim_days and sim_days > 0:
            cfg_module.SIM_DAYS        = sim_days
            cfg_module.PHASE1_DAYS     = max(1, int(sim_days * cfg_module.PHASE1_FRACTION))
            cfg_module.PHASE2_DAYS     = max(1, sim_days - cfg_module.PHASE1_DAYS)
        else:
            cfg_module.PHASE1_DAYS    = cfg.days_phase1
            cfg_module.PHASE2_DAYS    = cfg.days_phase2
            cfg_module.SIM_DAYS       = cfg.days_phase1 + cfg.days_phase2

        cfg_module.TOTAL_SIM_DAYS  = cfg_module.SIM_DAYS
        cfg_module.NUM_WORKERS     = cfg.num_workers

        import math
        dynamic_task_cap = max(50, math.ceil(
            cfg_module.SIM_DAYS * cfg_module.TASK_ARRIVAL_RATE * 1.5
        ))
        cfg_module.TOTAL_TASKS = max(int(cfg.task_count), dynamic_task_cap)
        cfg_module.NUM_TASKS   = cfg_module.TOTAL_TASKS

        tasks_per_day = getattr(cfg, 'tasks_per_day', None) or cfg_module.TASK_ARRIVAL_RATE
        cfg_module.TASK_ARRIVAL_RATE = float(tasks_per_day)

        max_worker_load = getattr(cfg, 'max_worker_load', None)
        if max_worker_load and max_worker_load > 0:
            cfg_module.MAX_WORKER_LOAD = int(max_worker_load)

        phase1_frac = getattr(cfg, 'phase1_fraction', None)
        if phase1_frac and 0.0 < phase1_frac < 1.0:
            cfg_module.PHASE1_FRACTION = float(phase1_frac)
            if sim_days and sim_days > 0:
                cfg_module.PHASE1_DAYS = max(1, int(sim_days * cfg_module.PHASE1_FRACTION))
                cfg_module.PHASE2_DAYS = max(1, sim_days - cfg_module.PHASE1_DAYS)

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

        import math as _math
        _n_baselines = len(self._make_baseline_defs())
        estimated_p1_decisions = int(
            cfg_module.PHASE1_DAYS
            * cfg_module.TASK_ARRIVAL_RATE
            * _n_baselines
            * 1.3
        )
        dynamic_cap = max(2000, min(estimated_p1_decisions, 12000))
        cfg_module.REPLAY_BUFFER_MAX_CAPACITY = dynamic_cap
        cfg_module.REPLAY_BUFFER_SIZE         = dynamic_cap
        cfg_module.MIN_REPLAY_SIZE = max(cfg_module.BATCH_SIZE, dynamic_cap // 20)
        print(f"[runner]   BUFFER_CAP     = {dynamic_cap} (dynamic, targeting 75%+ fill)")
        print(f"[runner]   MIN_REPLAY_SIZE= {cfg_module.MIN_REPLAY_SIZE}")
        print(f"[runner] ==========================================================")

        tasks_per_day = getattr(cfg, 'tasks_per_day', None) or getattr(cfg_module, 'TASK_ARRIVAL_RATE', 4.0)

        cfg_module.ACTION_DIM = (20 * cfg_module.NUM_WORKERS) + 40
        print(f"[runner]   ACTION_DIM     = {cfg_module.ACTION_DIM} (dynamic for {cfg_module.NUM_WORKERS} workers)")

        self.agent = DQNAgent(action_dim=cfg_module.ACTION_DIM)
        self.agent.configure_epsilon_schedule(
            tasks_per_day=tasks_per_day,
            phase2_days=cfg_module.PHASE2_DAYS,
            sim_days=cfg_module.SIM_DAYS,
        )

        # -- State -------------------------------------------------------------
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._cancelled = False
        self._phase = 0
        self._tick   = 0
        self._day    = 0
        self._running = False
        self._injected_tasks: List[Dict] = []

        # -- BOUNDED Metrics storage -------------------------------------------
        # FIX 1: Scale deque maxlen by actual number of baselines so that early
        # baselines are never evicted before baseline_snapshot is assembled.
        # Old code used max_days * 5 which silently dropped metrics for runs
        # with >5 baselines or >730-day equivalent throughput.
        max_days      = getattr(cfg_module, 'MAX_DAILY_METRICS_IN_MEMORY', 20000)
        n_baselines   = len(self._make_baseline_defs())
        p1_deque_size = max(max_days, cfg_module.SIM_DAYS * (n_baselines + 2))
        self._phase1_metrics = collections.deque(maxlen=p1_deque_size)
        self._phase2_metrics = collections.deque(maxlen=max_days)
        self._baseline_snapshots: Dict[str, Dict] = {}
        print(f"[runner]   p1_metrics deque maxlen = {p1_deque_size} "
              f"({cfg_module.SIM_DAYS} days x {n_baselines + 2} baselines+margin)")

        # -- Per-baseline environments (Phase 1) -------------------------------
        self._baseline_defs: List[Tuple[str, Any]] = self._make_baseline_defs()

        # -- Phase 2 scheduling env --------------------------------------------
        self._dqn_env: ProjectEnv | None = None

    # -- Baseline definitions --------------------------------------------------

    def _make_baseline_defs(self) -> List[Tuple[str, Any]]:
        defs = [
            ("Skill",   SkillBaseline),
            ("FIFO",    STFBaseline),
        ]
        if _HAS_GREEDY:
            defs.insert(0, ("Greedy", GreedyBaseline))
        else:
            print("[WARNING] Greedy baseline not loaded -- check import errors above")
        if _HAS_HYBRID:
            defs.append(("Hybrid", HybridBaseline))
        if _HAS_RANDOM:
            defs.append(("Random", RandomBaseline))
        assert any(name == "Greedy" for name, _ in defs), \
            "FATAL: Greedy baseline failed to load -- check greedy_baseline.py and imports above"
        return defs

    # -- Control interface -----------------------------------------------------

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def inject_task(self, task_data: Dict):
        self._injected_tasks.append(task_data)

    def get_status(self) -> Dict:
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

    # -- Serialization helpers -------------------------------------------------

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
        num_workers = env.num_workers
        max_assign  = 20 * num_workers

        if action >= max_assign:
            return None

        task_slot, worker_idx = _decode_action_parts(action, num_workers)

        tick      = env.clock.tick
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

    # -- Main entry point ------------------------------------------------------

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
        self._phase = 1
        await self.sio.emit("tick_update", {
            "tick": 0, "day": 0, "phase": 1,
            "worker_states": [], "queue_state": [],
            "last_assignment": None, "active_policy": "Starting...",
        })

        p1_metrics, gantt_by_baseline = await self._run_phase1()
        for m in p1_metrics:
            self._phase1_metrics.append(m)

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
                    "gantt_blocks":    list(gantt_by_baseline.get(bname, []))[-400:],
                }
        self._baseline_snapshots = baseline_snapshot

        await self.sio.emit("phase_transition", {
            "new_phase":                 "training",
            "baseline_results_snapshot": baseline_snapshot,
        })

        self._phase = 2
        await self._run_dqn_training()

        await self.sio.emit("phase2_ready", {
            "baseline_results_snapshot": baseline_snapshot,
        })

        self._phase = 3
        p2_metrics = await self._run_phase2_scheduling()
        for m in p2_metrics:
            self._phase2_metrics.append(m)

        overall = self._dqn_env.compute_metrics() if self._dqn_env else {}
        final = self._build_final_metrics(
            list(self._phase1_metrics),
            list(self._phase2_metrics),
            baseline_snapshot, overall)

        # Save to disk BEFORE emitting so a save error never blocks the
        # simulation_complete event. Wrapped so disk failure is logged but
        # does not prevent navigation to the analytics page.
        try:
            self._save_results()
        except Exception as save_exc:
            print(f"[runner] WARNING: _save_results failed (non-fatal): {save_exc}")

        # FIX 9: Strip the large per-day arrays from the socket payload.
        # For a 365-day run, phase1_daily + phase2_daily can exceed 1 MB,
        # causing socket.io to silently drop the frame -- so simulation_complete
        # never arrives and the analytics page never loads.
        # The daily data is already persisted to CSV via _save_results().
        # The frontend only needs the compact summary fields to navigate.
        payload = {k: v for k, v in final.items()
                   if k not in ("phase1_daily", "phase2_daily")}
        payload["phase1_days_count"] = len(final.get("phase1_daily", []))
        payload["phase2_days_count"] = len(final.get("phase2_daily", []))

        print(f"[runner] Emitting simulation_complete "
              f"(payload keys: {list(payload.keys())})")
        await self.sio.emit("simulation_complete", {"final_metrics": payload})

    # -- Phase 1: Run each baseline sequentially -------------------------------

    async def _run_phase1(self) -> Tuple[List[Dict], Dict[str, List]]:
        """
        Run each baseline in full on its own independent env.

        FIX 3 is applied here: last_day_decisions is tracked separately from
        day_decisions so that the final partial day is always flushed after the
        loop, even when the loop exits via `done` (which resets day_decisions
        to 0 on the same iteration as the loop break).
        """
        all_metrics:       List[Dict]       = []
        gantt_by_baseline: Dict[str, List]  = {}
        emit_every = _emit_every()
        max_gantt  = getattr(cfg_module, 'MAX_GANTT_BLOCKS_IN_MEMORY', 50000)

        total_slots = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY

        for bl_idx, (bname, BLClass) in enumerate(self._baseline_defs):
            if self._cancelled:
                break

            env    = _make_env(self.cfg, seed_offset=bl_idx + 1)
            policy = BLClass(env)
            state  = env.reset()

            gantt_deque       = collections.deque(maxlen=max_gantt)
            prev_day          = 0
            day_decisions     = 0
            last_day_decisions = 0   # FIX 3: snapshot before reset
            step_count        = 0

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
                # FIX 6: timeout prevents hang if resume() is never called
                if not await _wait_for_resume(self._pause_event, lambda: self._cancelled, f"Phase1/{bname}"):
                    break

                current_day = env.clock.day

                if current_day > prev_day:
                    m = self._collect_metrics(day_decisions, bname, env)
                    all_metrics.append(m)
                    last_day_decisions = day_decisions  # FIX 3
                    day_decisions      = 0
                    prev_day           = current_day
                    await self.sio.emit("daily_summary", {
                        "day":   current_day - 1,
                        "phase": 1,
                        "metrics_per_policy": {bname: m},
                    })

                valid = env.get_valid_actions()

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
                        await asyncio.sleep(0.005)
                    else:
                        await asyncio.sleep(0)
                    continue

                action = policy.select_action(state)
                if action not in valid:
                    action = valid[0]

                block = self._build_gantt_block(env, action, bname)
                next_state, reward, done, info = env.step(action)

                state_copy = np.array(state, dtype=np.float32)
                next_copy  = np.array(next_state, dtype=np.float32)
                max_p = self.agent.replay_buffer._max_priority
                self.agent.replay_buffer.tree.add(
                    max_p ** self.agent.replay_buffer.alpha,
                    (state_copy, action, reward, next_copy, float(done))
                )
                self.agent.steps_done += 1

                if block is not None:
                    gantt_deque.append(block)
                    await self.sio.emit("gantt_block", block)

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

                step_count += 1
                if step_count % emit_every == 0:
                    await self.sio.emit("tick_update", {
                        "tick":          env.clock.tick,
                        "day":           env.clock.day,
                        "phase":         1,
                        "worker_states": self._serialize_workers(env),
                        "queue_state":   self._serialize_queue(env),
                        "last_assignment": {
                            "task_id":   block["task_id"] if block else "--",
                            "worker_id": block["worker_id"] if block else "--",
                            "policy":    bname,
                        } if block else None,
                        "active_policy": bname,
                    })
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0)

                state         = next_state
                day_decisions += 1
                last_day_decisions = day_decisions  # FIX 3: keep in sync

                if done:
                    break

            # FIX 3: Always flush the last partial day, regardless of how the
            # loop exited. Use last_day_decisions (never 0 after first step).
            flush_decisions = day_decisions if day_decisions > 0 else last_day_decisions
            if flush_decisions > 0 or env.clock.day > prev_day:
                m = self._collect_metrics(flush_decisions, bname, env)
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

    # -- Phase 2a: DQN offline training ---------------------------------------

    async def _run_dqn_training(self):
        buf_size = len(self.agent.replay_buffer)

        if buf_size < self.agent.min_replay_size:
            self.agent.min_replay_size = max(self.agent.batch_size, buf_size)
            print(f"[Training] Lowered min_replay_size to {self.agent.min_replay_size} "
                  f"(buffer has {buf_size} transitions)")

        if buf_size < self.agent.batch_size:
            print(f"[Training] Buffer too small ({buf_size}) -- skipping training (need >= {self.agent.batch_size})")
            await self.sio.emit("training_progress", {"percent": 100, "steps": 0})
            return

        target_steps = max(500, min(buf_size * 3, 8000))
        print(f"[Training] Starting {target_steps} gradient steps. Buffer={buf_size}")

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

    # -- Phase 2b: DQN scheduling ----------------------------------------------

    async def _run_phase2_scheduling(self) -> List[Dict]:
        """
        FIX 2 is applied here: the hard assert on task count is replaced with a
        soft warning that clamps total_slots so the simulation still completes
        rather than raising an AssertionError at the very end of setup.
        """
        env           = _make_env(self.cfg, seed_offset=99)
        self._dqn_env = env
        state         = env.reset()

        # FIX 2: Soft warning instead of hard assert.
        # Poisson arrivals have variance ~= mean, so actual_tasks is typically
        # +/-sqrt(mean) of expected. The old assert fired on any run where variance
        # went slightly negative (common for 365-day simulations).
        min_expected = cfg_module.SIM_DAYS * cfg_module.TASK_ARRIVAL_RATE * 0.8
        actual_tasks = len(env.tasks)
        print(f"[Phase2] Task list: {actual_tasks} tasks generated "
              f"(min expected ~= {min_expected:.0f}, "
              f"cap = {cfg_module.TOTAL_TASKS}, "
              f"sim_days = {cfg_module.SIM_DAYS})")
        if actual_tasks < min_expected:
            # Clamp the horizon to what we actually have rather than crashing.
            safe_days   = max(1, int(actual_tasks / max(cfg_module.TASK_ARRIVAL_RATE, 1e-6)))
            safe_slots  = safe_days * cfg_module.SLOTS_PER_DAY
            print(f"[Phase2] WARNING: task list shorter than expected "
                  f"({actual_tasks} < {min_expected:.0f}). "
                  f"Clamping total_slots from "
                  f"{cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY} -> {safe_slots} "
                  f"({safe_days} days) to avoid over-running the task list.")
            total_slots = safe_slots
        else:
            total_slots = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY

        emit_every   = _emit_every()
        train_every  = _train_every()

        prev_day      = 0
        day_decisions = 0
        all_metrics: List[Dict] = []
        step_count    = 0

        await self.sio.emit("tick_update", {
            "tick": 0, "day": 0, "phase": 3,
            "worker_states": self._serialize_workers(env),
            "queue_state":   self._serialize_queue(env),
            "last_assignment": None, "active_policy": "DQN",
        })
        await asyncio.sleep(0.01)

        # Raised from total_slots * 1.5 -> total_slots * 50 + 10000 in v5 to
        # accommodate multiple defer actions per tick without false-positive exits.
        max_iterations  = total_slots * 50 + 10000
        iteration_count = 0

        while env.clock.tick < total_slots:
            if self._cancelled:
                break
            # FIX 6: timeout prevents hang if resume() is never called
            if not await _wait_for_resume(self._pause_event, lambda: self._cancelled, "Phase2/DQN"):
                break

            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"[Phase2] Safety exit: exceeded {max_iterations} iterations at tick={env.clock.tick}")
                break

            if self._injected_tasks:
                self._inject_task_into_env(self._injected_tasks.pop(0), env)

            current_day   = env.clock.day
            self._tick    = env.clock.tick
            self._day     = current_day

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

            if not valid:
                tick_before = env.clock.tick
                ticks_adv = env.advance_to_next_event()
                # FIX 7: any zero-advance means the env has stalled -- break
                # unconditionally rather than only when both conditions hold.
                # The old guard (tick <= tick_before AND ticks_adv == 0) missed
                # the case where advance_to_next_event returns 0 but the clock
                # moved by 1 due to a side effect, causing an infinite re-entry.
                if ticks_adv == 0:
                    print(f"[Phase2] WARN: advance_to_next_event() returned 0 at "
                          f"tick={env.clock.tick} -- env stalled, exiting loop.")
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
                    await asyncio.sleep(0.005)
                else:
                    await asyncio.sleep(0)
                continue

            assign_actions = [a for a in valid if a < 20 * env.num_workers]
            last_block = None
            done = False

            if assign_actions:
                assignments_this_tick = 0
                tick_at_inner_start   = env.clock.tick  # FIX 8: baseline for progress check
                for _assign_iter in range(50):
                    current_valid  = env.get_valid_actions()
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

                    self.agent.store_transition(
                        np.array(state, dtype=np.float32),
                        action, reward,
                        np.array(next_state, dtype=np.float32),
                        float(done),
                    )
                    self.agent.steps_done += 1
                    buf_size = len(self.agent.replay_buffer)
                    if buf_size >= self.agent.min_replay_size:
                        if self.agent.epsilon > 0.5:
                            n_grad = 8
                        elif self.agent.epsilon > 0.3:
                            n_grad = 6
                        elif self.agent.epsilon > 0.1:
                            n_grad = 4
                        else:
                            n_grad = 2
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

                    # FIX 8: If the clock hasn't moved after at least one assign,
                    # env.step() is not advancing time -- break to prevent the outer
                    # loop from re-entering this branch endlessly. max_iterations
                    # counts outer iterations only and won't catch this inner cycle.
                    if assignments_this_tick > 0 and env.clock.tick == tick_at_inner_start:
                        print(f"[Phase2] WARN: inner assign loop made {assignments_this_tick} "
                              f"step(s) but clock did not advance from tick={tick_at_inner_start}. "
                              f"Breaking inner loop to prevent silent hang.")
                        break
                    
                    await asyncio.sleep(0)

                day_decisions += assignments_this_tick

            else:
                action = valid[0]
                block  = self._build_gantt_block(env, action, "DQN")
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
                state      = next_state
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
                        "task_id":   last_block["task_id"] if last_block else "--",
                        "worker_id": last_block["worker_id"] if last_block else "--",
                        "policy":    "DQN",
                    } if last_block else None,
                    "active_policy": "DQN",
                })
                await asyncio.sleep(0.005)
            else:
                await asyncio.sleep(0)
            
            if done:
                break

        if day_decisions > 0:
            m = self._collect_metrics(day_decisions, "DQN", env)
            all_metrics.append(m)
            await self.sio.emit("daily_summary", {
                "day": env.clock.day,
                "phase": 3,
                "metrics_per_policy": {"DQN": m},
            })

        return all_metrics

    # -- Helpers ---------------------------------------------------------------

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

        # FIX 4: Handle empty baseline_snap gracefully.
        # Previously best_name defaulted to "Greedy" even when no baselines ran,
        # producing a misleading result payload with best_tp stuck at -1.0.
        if baseline_snap:
            best_name, best_tp = "Greedy", -1.0
            for bn, snap in baseline_snap.items():
                if snap["throughput"] > best_tp:
                    best_tp, best_name = snap["throughput"], bn
        else:
            best_name = "no_baselines"
            best_tp   = 0.0
            print("[WARNING] _build_final_metrics: baseline_snap is empty -- "
                  "no baseline data collected during Phase 1.")

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
        # FIX 5: Ensure output directory exists before attempting to write.
        # A bare open() would crash at teardown if RESULTS_DIR was deleted mid-run.
        os.makedirs(rdir, exist_ok=True)
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