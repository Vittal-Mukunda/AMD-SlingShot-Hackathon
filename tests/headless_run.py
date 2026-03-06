"""
tests/headless_run.py — Headless simulation runner for hyperparameter sweeps.

Bypasses FastAPI and Socket.IO entirely. Runs Phase 1 baselines + Phase 2 DQN
synchronously and prints results in a format the sweep script can parse.

Usage:
    python tests/headless_run.py --sim_days 30 --phase1_fraction 0.6 --tasks_per_day 4.0
"""

import argparse
import asyncio
import collections
import math
import os
import sys
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
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


# ── Null Socket.IO (no-op, swallows all emits) ────────────────────────────────
class NullSio:
    async def emit(self, *args, **kwargs):
        pass


# ── Minimal SimConfig ─────────────────────────────────────────────────────────
class HeadlessConfig:
    def __init__(self, sim_days, phase1_fraction, tasks_per_day,
                 num_workers, max_worker_load, seed):
        self.sim_days         = sim_days
        self.phase1_fraction  = phase1_fraction
        self.tasks_per_day    = tasks_per_day
        self.num_workers      = num_workers
        self.max_worker_load  = max_worker_load
        self.seed             = seed
        self.task_count       = max(50, math.ceil(sim_days * tasks_per_day * 1.5))
        # Legacy fields expected by _make_env fallback
        self.days_phase1      = max(1, int(sim_days * phase1_fraction))
        self.days_phase2      = max(1, sim_days - self.days_phase1)


# ── Environment factory (mirrors simulation_runner._make_env) ─────────────────
def make_env(cfg, seed_offset=0):
    sim_days    = cfg_module.SIM_DAYS
    total_slots = sim_days * cfg_module.SLOTS_PER_DAY
    total_tasks = cfg_module.TOTAL_TASKS
    return ProjectEnv(
        num_workers=cfg.num_workers,
        total_tasks=total_tasks,
        seed=cfg.seed + seed_offset,
        total_sim_slots=total_slots,
    )


def apply_config(cfg):
    """Mirror SimulationRunner.__init__ config application logic."""
    cfg_module.SIM_DAYS           = cfg.sim_days
    cfg_module.PHASE1_FRACTION    = cfg.phase1_fraction
    cfg_module.PHASE1_DAYS        = max(1, int(cfg.sim_days * cfg.phase1_fraction))
    cfg_module.PHASE2_DAYS        = max(1, cfg.sim_days - cfg_module.PHASE1_DAYS)
    cfg_module.TOTAL_SIM_DAYS     = cfg.sim_days
    cfg_module.NUM_WORKERS        = cfg.num_workers
    cfg_module.MAX_WORKER_LOAD    = cfg.max_worker_load
    cfg_module.TASK_ARRIVAL_RATE  = cfg.tasks_per_day

    dynamic_cap = max(50, math.ceil(cfg.sim_days * cfg.tasks_per_day * 1.5))
    cfg_module.TOTAL_TASKS        = max(int(cfg.task_count), dynamic_cap)
    cfg_module.NUM_TASKS          = cfg_module.TOTAL_TASKS


# ── Synchronous baseline runner ───────────────────────────────────────────────
def run_baseline_sync(bname, BLClass, cfg, agent, seed_offset):
    env    = make_env(cfg, seed_offset=seed_offset)
    policy = BLClass(env)
    state  = env.reset()

    total_slots = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY

    while env.clock.tick < total_slots:
        valid = env.get_valid_actions()

        if not valid:
            ticks = env.advance_to_next_event()
            done_check, _ = env._check_termination()
            if done_check or env.clock.tick >= total_slots:
                break
            continue

        action = policy.select_action(state)
        if action not in valid:
            action = valid[0]

        next_state, reward, done, info = env.step(action)

        # Store into DQN replay buffer
        state_np      = np.array(state,      dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        max_p = agent.replay_buffer._max_priority
        agent.replay_buffer.tree.add(
            max_p ** agent.replay_buffer.alpha,
            (state_np, action, reward, next_state_np, float(done))
        )
        agent.steps_done += 1

        state = next_state
        if done:
            break

    return env.compute_metrics()


# ── Synchronous DQN training ──────────────────────────────────────────────────
def run_dqn_training_sync(agent, min_steps=200):
    buf_size = len(agent.replay_buffer)
    if buf_size < agent.batch_size:
        return 0

    target_steps = max(min_steps, buf_size // 2)
    steps = 0
    while steps < target_steps:
        try:
            agent.train_step()
            agent.update_epsilon()
            steps += 1
        except Exception:
            break
    return steps


# ── Synchronous DQN scheduling ────────────────────────────────────────────────
def run_dqn_scheduling_sync(cfg, agent):
    env   = make_env(cfg, seed_offset=99)
    state = env.reset()

    total_slots = cfg_module.SIM_DAYS * cfg_module.SLOTS_PER_DAY
    max_iter    = total_slots + max(500, total_slots // 2)
    iteration   = 0

    while env.clock.tick < total_slots:
        iteration += 1
        if iteration > max_iter:
            break

        valid = env.get_valid_actions()

        if not valid:
            tick_before = env.clock.tick
            env.advance_to_next_event()
            if env.clock.tick <= tick_before:
                break
            done_check, _ = env._check_termination()
            if done_check:
                break
            continue

        assign_actions = [a for a in valid if a < 20 * env.num_workers]

        if assign_actions:
            done = False
            for _ in range(50):  # safety cap per tick
                current_valid  = env.get_valid_actions()
                current_assign = [a for a in current_valid if a < 20 * env.num_workers]
                if not current_assign:
                    break

                action = agent.select_action(
                    np.array(state, dtype=np.float32), current_valid, greedy=False
                )
                if action not in current_valid:
                    action = current_assign[0]

                next_state, reward, done, info = env.step(action)

                agent.store_transition(
                    np.array(state,      dtype=np.float32),
                    action, reward,
                    np.array(next_state, dtype=np.float32),
                    float(done),
                )
                agent.steps_done += 1

                if len(agent.replay_buffer) >= agent.min_replay_size:
                    n_grad = 2 if agent.epsilon <= agent.epsilon_end + 0.01 else 4
                    for _ in range(n_grad):
                        agent.train_step()
                    agent.update_epsilon()

                state = next_state
                if done:
                    break
            if done:
                break
        else:
            # Defer only
            tick_before = env.clock.tick
            next_state, reward, done, info = env.step(valid[0])

            agent.store_transition(
                np.array(state,      dtype=np.float32),
                valid[0], reward,
                np.array(next_state, dtype=np.float32),
                float(done),
            )
            agent.steps_done += 1
            if len(agent.replay_buffer) >= agent.min_replay_size:
                agent.train_step()
                agent.update_epsilon()

            if env.clock.tick <= tick_before:
                break
            state = next_state
            if done:
                break

    return env.compute_metrics()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_days",        type=int,   default=30)
    parser.add_argument("--phase1_fraction", type=float, default=0.6)
    parser.add_argument("--tasks_per_day",   type=float, default=4.0)
    parser.add_argument("--num_workers",     type=int,   default=5)
    parser.add_argument("--max_worker_load", type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    cfg = HeadlessConfig(
        sim_days        = args.sim_days,
        phase1_fraction = args.phase1_fraction,
        tasks_per_day   = args.tasks_per_day,
        num_workers     = args.num_workers,
        max_worker_load = args.max_worker_load,
        seed            = args.seed,
    )

    apply_config(cfg)

    # Build baseline list
    baseline_defs = [
        ("Greedy", GreedyBaseline),
        ("Skill",  SkillBaseline),
        ("FIFO",   STFBaseline),
    ]
    if _HAS_HYBRID:
        baseline_defs.append(("Hybrid", HybridBaseline))
    if _HAS_RANDOM:
        baseline_defs.append(("Random", RandomBaseline))

    # Initialise agent
    agent = DQNAgent()
    agent.configure_epsilon_schedule(
        tasks_per_day = cfg.tasks_per_day,
        phase2_days   = cfg_module.PHASE2_DAYS,
        sim_days      = cfg_module.SIM_DAYS,
    )

    # ── Phase 1: baselines ────────────────────────────────────────────────────
    baseline_results = {}
    for i, (bname, BLClass) in enumerate(baseline_defs):
        m = run_baseline_sync(bname, BLClass, cfg, agent, seed_offset=i+1)
        baseline_results[bname] = m

    buf_size = len(agent.replay_buffer)

    # ── Phase 2a: offline training ────────────────────────────────────────────
    train_steps = run_dqn_training_sync(agent)

    # ── Phase 2b: DQN scheduling ──────────────────────────────────────────────
    dqn_metrics = run_dqn_scheduling_sync(cfg, agent)

    # ── Print results in parseable format ─────────────────────────────────────
    dqn_quality    = float(dqn_metrics.get("quality_score",      0.0))
    dqn_throughput = float(dqn_metrics.get("throughput_per_day", 0.0))
    dqn_lateness   = float(dqn_metrics.get("lateness_rate",      0.0))
    dqn_overload   = int(dqn_metrics.get("overload_events",      0))
    epsilon_final  = float(agent.epsilon)
    buffer_fill    = buf_size / cfg_module.REPLAY_BUFFER_MAX_CAPACITY

    # These exact lines are parsed by the sweep script
    print(f"dqn_quality:{dqn_quality:.6f}")
    print(f"dqn_throughput:{dqn_throughput:.6f}")
    print(f"dqn_lateness:{dqn_lateness:.6f}")
    print(f"dqn_overload:{dqn_overload}")
    print(f"epsilon_final:{epsilon_final:.6f}")
    print(f"buffer_fill:{buffer_fill:.4f}")
    print(f"train_steps:{train_steps}")

    # Baseline comparison lines (parsed optionally)
    for bname, m in baseline_results.items():
        bq = float(m.get("quality_score",      0.0))
        bt = float(m.get("throughput_per_day", 0.0))
        print(f"baseline_{bname}_quality:{bq:.6f}")
        print(f"baseline_{bname}_throughput:{bt:.6f}")

    # Human-readable summary (ignored by parser)
    print("---")
    print(f"[summary] DQN quality={dqn_quality:.4f} throughput={dqn_throughput:.4f} "
          f"lateness={dqn_lateness:.4f} overload={dqn_overload} "
          f"epsilon={epsilon_final:.4f} buffer={buffer_fill:.2%} "
          f"train_steps={train_steps}")


if __name__ == "__main__":
    main()
