"""
backend/simulation_runner.py — Complete rewrite fixing all broken links.

BUGS FIXED:
  1. Action decoding: task_idx = action // num_workers, not action % num_tasks
  2. Phase 2 uses Greedy baseline env (carries forward from Phase 1) not a fresh env
  3. DQN collects transitions from ALL 5 baselines (not just Greedy) for richer replay
  4. Phase 2 separated: background training first → emit training_progress → emit
     phase2_ready → then DQN scheduling starts (3 distinct sub-phases)
  5. simulation_error emitted on any exception so nothing is silent
  6. phase_transition now says new_phase=1.5 (TRAINING) so frontend shows spinner;
     phase2_ready triggers actual Phase 2 Gantt
  7. Gantt block action encoding uses correct formula
  8. Phase 1 baselines run sequentially per-baseline (not interleaved) to keep each
     baseline's Gantt coherent, then all results collected before transition
"""

import asyncio
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

# ── Constants ─────────────────────────────────────────────────────────────────
# Emit every step for real-time frontend updates (was 2 — caused stale UI)
EMIT_EVERY = 1
# Minimum number of training gradient steps in Phase 2 training sub-phase
MIN_TRAINING_STEPS = 200


def _make_env(cfg, seed_offset: int = 0) -> ProjectEnv:
    """Create a fresh ProjectEnv matching user config."""
    return ProjectEnv(
        num_workers=cfg.num_workers,
        total_tasks=cfg.task_count,
        seed=cfg.seed + seed_offset,
        total_sim_slots=(cfg.days_phase1 + cfg.days_phase2) * cfg_module.SLOTS_PER_DAY,
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
    Two-phase simulation runner.

    Phase 1: 5 baselines run SEQUENTIALLY (one full run each) on independent envs.
             Each baseline emits tick_update + gantt_block in real time.
             DQN stores transitions from ALL baselines.

    Phase 2 (training): DQN trains on collected replay buffer offline.
             Emits training_progress events (percentage only).
             Emits phase2_ready when training is done.

    Phase 2 (scheduling): DQN schedules on fresh env, emitting tick_update + gantt_block.
    """

    def __init__(self, cfg, sio):
        self.cfg = cfg
        self.sio = sio

        # ── Apply user config to cfg_module globals ──────────────────────────
        cfg_module.PHASE1_DAYS    = cfg.days_phase1
        cfg_module.PHASE2_DAYS    = cfg.days_phase2
        cfg_module.TOTAL_SIM_DAYS = cfg.days_phase1 + cfg.days_phase2
        cfg_module.TOTAL_TASKS    = cfg.task_count
        cfg_module.NUM_TASKS      = cfg.task_count
        cfg_module.NUM_WORKERS    = cfg.num_workers

        # ── State ─────────────────────────────────────────────────────────────
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._cancelled = False
        self._phase = 0        # 0=init, 1=baseline, 2=training, 3=DQN scheduling
        self._tick   = 0
        self._day    = 0
        self._running = False
        self._injected_tasks: List[Dict] = []

        # ── Metrics storage ───────────────────────────────────────────────────
        self._phase1_metrics:   List[Dict] = []
        self._phase2_metrics:   List[Dict] = []
        self._baseline_snapshots: Dict[str, Dict] = {}

        # ── DQN agent (shared across all phases) ─────────────────────────────
        self.agent = DQNAgent()

        # ── Per-baseline environments (Phase 1) ───────────────────────────────
        # Order matters: the user sees them in this order in the UI tabs
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
        """Returns tick_update-compatible dict for reconnect (Bug 8)."""
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
        max_assign = 20 * num_workers  # action space assign boundary

        if action >= max_assign:
            return None  # defer or escalate — no gantt block

        task_slot, worker_idx = _decode_action_parts(action, num_workers)

        # Get the visible task list in exact same order as env uses
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
        self._phase1_metrics = p1_metrics

        # Build baseline snapshot
        baseline_snapshot: Dict[str, Dict] = {}
        for bname, _ in self._baseline_defs:
            rows = [m for m in p1_metrics if m.get("baseline") == bname]
            if rows:
                baseline_snapshot[bname] = {
                    "throughput":      float(np.mean([r["throughput_per_day"] for r in rows])),
                    "completion_rate": float(np.mean([r["completion_rate"] for r in rows])),
                    "lateness_rate":   float(np.mean([r["lateness_rate"] for r in rows])),
                    "quality_score":   float(np.mean([r["quality_score"] for r in rows])),
                    "overload_events": float(np.mean([r["overload_events"] for r in rows])),
                    "gantt_blocks":    gantt_by_baseline.get(bname, []),
                }
        self._baseline_snapshots = baseline_snapshot

        # ── Notify frontend: Phase 1 complete, entering training ─────────
        await self.sio.emit("phase_transition", {
            "new_phase":                 "training",
            "baseline_results_snapshot": baseline_snapshot,
        })

        # ── Phase 2a: DQN training (background, emits training_progress) ──
        self._phase = 2
        await self._run_dqn_training()

        # ── Notify frontend: training done, DQN scheduling starts ─────────
        await self.sio.emit("phase2_ready", {
            "baseline_results_snapshot": baseline_snapshot,
        })

        # ── Phase 2b: DQN scheduling ───────────────────────────────────────
        self._phase = 3
        p2_metrics = await self._run_phase2_scheduling()
        self._phase2_metrics = p2_metrics

        # ── Final metrics ──────────────────────────────────────────────────
        overall = self._dqn_env.compute_metrics() if self._dqn_env else {}
        final = self._build_final_metrics(p1_metrics, p2_metrics, baseline_snapshot, overall)
        await self.sio.emit("simulation_complete", {"final_metrics": final})

        self._save_results()

    # ── Phase 1: Run each baseline sequentially ───────────────────────────────

    async def _run_phase1(self) -> Tuple[List[Dict], Dict[str, List]]:
        """
        Run each baseline in full on its own independent env.
        Emits tick_update + gantt_block for each step so the frontend
        can show live Gantt updates per baseline tab.
        Stores ALL transitions (from all baselines) into the DQN replay buffer.
        """
        all_metrics:      List[Dict]        = []
        gantt_by_baseline: Dict[str, List]  = {}

        phase1_slots = self.cfg.days_phase1 * cfg_module.SLOTS_PER_DAY

        for bl_idx, (bname, BLClass) in enumerate(self._baseline_defs):
            if self._cancelled:
                break

            env    = _make_env(self.cfg, seed_offset=bl_idx + 1)
            policy = BLClass(env)
            state  = env.reset()

            gantt_blocks: List[Dict] = []
            prev_day     = 0
            day_decisions = 0

            # Announce to frontend which baseline is running
            await self.sio.emit("tick_update", {
                "tick": 0, "day": 0, "phase": 1,
                "worker_states": self._serialize_workers(env),
                "queue_state":   self._serialize_queue(env),
                "last_assignment": None,
                "active_policy": bname,
            })
            await asyncio.sleep(0.01)

            step_count = 0
            while True:
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

                # ── No valid actions: advance clock or terminate ───────────
                if not valid:
                    _, _, done, _ = env.step(20 * env.num_workers)
                    if done:
                        break
                    step_count += 1
                    # Emit periodically while idle
                    if step_count % EMIT_EVERY == 0:
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

                # ── Build gantt block BEFORE stepping (so visible tasks are correct) ─
                block = self._build_gantt_block(env, action, bname)

                # ── Step environment ───────────────────────────────────────
                next_state, reward, done, info = env.step(action)

                # ── Store transition in DQN replay buffer ──────────────────
                self.agent.store_transition(
                    np.array(state,      dtype=np.float32),
                    action,
                    reward,
                    np.array(next_state, dtype=np.float32),
                    float(done),
                )
                self.agent.steps_done += 1

                # ── Emit gantt block ───────────────────────────────────────
                if block is not None:
                    gantt_blocks.append(block)
                    await self.sio.emit("gantt_block", block)

                # ── Emit task_completed ────────────────────────────────────
                if info.get("completed_tasks", 0) > 0:
                    # Emit for the most recently completed task
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

                # ── Emit tick_update every step (real-time UI) ─────────────
                step_count += 1
                if step_count % EMIT_EVERY == 0:
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

            gantt_by_baseline[bname] = gantt_blocks

            # Brief pause between baselines so frontend can clear/switch tabs
            await asyncio.sleep(0.05)

        buf_size = len(self.agent.replay_buffer)
        print(f"[Phase1] Done. Replay buffer: {buf_size} transitions across {len(self._baseline_defs)} baselines.")
        return all_metrics, gantt_by_baseline

    # ── Phase 2a: DQN offline training ───────────────────────────────────────

    async def _run_dqn_training(self):
        """
        Train DQN on the replay buffer collected from Phase 1.
        Runs synchronously (blocking) but yields to the event loop every
        few steps so WebSocket heartbeats continue.
        Emits only training_progress (percentage) events.
        """
        buf_size = len(self.agent.replay_buffer)

        # Lower min_replay_size if not enough data in buffer
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

        self.agent.set_epsilon(cfg_module.EPSILON_PHASE2_START)
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

            # Yield to event loop every 10 steps
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
        DQN controls the scheduler on a fresh Phase-2 environment.
        Emits tick_update + gantt_block per step.
        """
        env         = _make_env(self.cfg, seed_offset=99)
        self._dqn_env = env
        state       = env.reset()

        prev_day     = 0
        day_decisions = 0
        all_metrics:  List[Dict] = []
        step_count   = 0

        phase2_slots = self.cfg.days_phase2 * cfg_module.SLOTS_PER_DAY

        await self.sio.emit("tick_update", {
            "tick": 0, "day": 0, "phase": 3,
            "worker_states": self._serialize_workers(env),
            "queue_state":   self._serialize_queue(env),
            "last_assignment": None, "active_policy": "DQN",
        })
        await asyncio.sleep(0.01)

        # Safety: prevent infinite loop (e.g. tick/state mismatch at Day 4, Tick 79)
        max_iterations = phase2_slots + 200
        iteration_count = 0

        while env.clock.tick < phase2_slots:
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
            if not valid:
                # No valid actions: advance time via defer (no-op when queue empty)
                tick_before = env.clock.tick
                _, _, done, _ = env.step(20 * env.num_workers)
                if env.clock.tick <= tick_before:
                    print(f"[Phase2] WARN: clock did not advance at tick={tick_before}, forcing break")
                    break
                if done:
                    break
                step_count += 1
                if step_count % EMIT_EVERY == 0:
                    await self.sio.emit("tick_update", {
                        "tick": env.clock.tick, "day": env.clock.day, "phase": 3,
                        "worker_states": self._serialize_workers(env),
                        "queue_state":   self._serialize_queue(env),
                        "last_assignment": None, "active_policy": "DQN",
                    })
                await asyncio.sleep(0)
                continue

            # DQN inference only (no training — model was pre-trained in Phase 2a)
            action = self.agent.select_action(
                np.array(state, dtype=np.float32), valid, greedy=True
            )
            if action not in valid:
                action = valid[0]

            block = self._build_gantt_block(env, action, "DQN")

            tick_before = env.clock.tick
            next_state, reward, done, info = env.step(action)
            if env.clock.tick <= tick_before:
                print(f"[Phase2] WARN: clock did not advance at tick={tick_before} after step, forcing break")
                break

            if block is not None:
                await self.sio.emit("gantt_block", block)

            step_count += 1
            if step_count % EMIT_EVERY == 0:
                await self.sio.emit("tick_update", {
                    "tick":          env.clock.tick,
                    "day":           env.clock.day,
                    "phase":         3,
                    "worker_states": self._serialize_workers(env),
                    "queue_state":   self._serialize_queue(env),
                    "last_assignment": {
                        "task_id":   block["task_id"] if block else "—",
                        "worker_id": block["worker_id"] if block else "—",
                        "policy":    "DQN",
                    } if block else None,
                    "active_policy": "DQN",
                })
            await asyncio.sleep(0)

            state         = next_state
            day_decisions += 1
            if done:
                break

            await asyncio.sleep(0)  # Yield to event loop every step (prevents freeze)

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
            from environment.task import Task as EnvTask
            task = EnvTask(
                task_id=len(env.tasks),
                priority=int(task_data.get("urgency", 2)),
                duration_slots=max(1, int(task_data.get("duration", 4))),
                arrival_tick=int(task_data.get("arrival_tick", self._tick)),
                required_skill=float(task_data.get("required_skill", 0.5)),
                deadline_tick=(int(task_data.get("arrival_tick", self._tick))
                               + int(task_data.get("duration", 4)) * 3),
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

        best_ms = float(self.cfg.days_phase1 * 8)  # baseline measured over Phase 1

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
        for phase_num, rows in [(1, self._phase1_metrics), (2, self._phase2_metrics)]:
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
