"""
sweep.py — Standalone Hyperparameter Sweep for AMD SlingShot DQN Agent
=======================================================================

Runs entirely standalone from the command line WITHOUT FastAPI, Socket.IO,
or the frontend. Uses the actual project code (ProjectEnv, DQNAgent, all
baselines) synchronously.

Usage:
    python sweep.py                           # 200 random combos, 30 days each
    python sweep.py --sim_days 50             # 200 combos, 50 days each
    python sweep.py --full                    # all 5120 combos
    python sweep.py --n 100 --sim_days 40    # 100 random combos, 40 days
    python sweep.py --output my_results.json

Search Space: 5 × 4 × 4 × 4 × 4 × 4 = 5120 combinations
Default fast mode: random sample of 200 combos.

Scoring: 50% quality + 30% throughput - 15% lateness - 5% overload
"""

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List

# ── Make sure project root is on PATH ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
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


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "LEARNING_RATE":          [5e-5, 1e-4, 3e-4, 5e-4, 8e-4],
    "GAMMA":                  [0.85, 0.90, 0.95, 0.99],
    "EPSILON_DECAY":          [0.9990, 0.9995, 0.9998, 0.9999],
    "REWARD_COMPLETION_BASE": [0.8, 1.0, 1.2, 1.5],
    "PER_ALPHA":              [0.3, 0.4, 0.5, 0.6],
    "TARGET_UPDATE_FREQ":     [100, 200, 400, 600],
}

# Total combos: 5 × 4 × 4 × 4 × 4 × 4 = 5120
ALL_COMBOS = [
    dict(zip(SEARCH_SPACE.keys(), vals))
    for vals in itertools.product(*SEARCH_SPACE.values())
]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score(metrics: dict) -> float:
    """
    Composite score:
      50% quality + 30% throughput (normalised to 8/day) – 15% lateness – 5% overload_flag
    """
    q  = metrics.get('quality_score',      0.0)
    tp = metrics.get('throughput_per_day', 0.0) / 8.0   # normalize
    la = metrics.get('lateness_rate',      0.0)
    ov = 1.0 if metrics.get('overload_events', 0) > 0 else 0.0
    return round(0.50 * q + 0.30 * tp - 0.15 * la - 0.05 * ov, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Apply hyperparams
# ─────────────────────────────────────────────────────────────────────────────

def apply_hyperparams(params: dict):
    """Patch cfg_module globals directly. Called before each run."""
    for k, v in params.items():
        setattr(cfg_module, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(sim_days: int, tasks_per_day: float, seed: int) -> ProjectEnv:
    total_slots = sim_days * cfg_module.SLOTS_PER_DAY
    total_tasks = max(50, math.ceil(sim_days * tasks_per_day * 1.5))
    return ProjectEnv(
        num_workers=cfg_module.NUM_WORKERS,
        total_tasks=total_tasks,
        seed=seed,
        total_sim_slots=total_slots,
    )


def _make_agent(sim_days: int) -> DQNAgent:
    return DQNAgent(
        learning_rate=cfg_module.LEARNING_RATE,
        gamma=cfg_module.GAMMA,
        epsilon_start=cfg_module.EPSILON_START,
        epsilon_end=cfg_module.EPSILON_END,
        epsilon_decay=cfg_module.EPSILON_DECAY,
        replay_capacity=cfg_module.REPLAY_BUFFER_SIZE,
        batch_size=cfg_module.BATCH_SIZE,
        target_update_freq=cfg_module.TARGET_UPDATE_FREQ,
        per_alpha=cfg_module.PER_ALPHA,
        per_beta_start=cfg_module.PER_BETA_START,
        per_beta_frames=cfg_module.PER_BETA_FRAMES,
        lr_scheduler_t0=cfg_module.LR_SCHEDULER_T0,
        min_replay_size=cfg_module.MIN_REPLAY_SIZE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_baseline_sync
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_sync(name: str, BLClass, sim_days: int, tasks_per_day: float,
                      agent: DQNAgent, seed: int) -> dict:
    """Run one baseline synchronously. Stores transitions into agent's replay buffer."""
    env = _make_env(sim_days, tasks_per_day, seed)
    state = env.reset()
    policy = BLClass(env)

    total_reward = 0.0
    steps = 0
    max_steps = sim_days * cfg_module.SLOTS_PER_DAY * 4

    while steps < max_steps:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            done, _ = env.advance_to_next_event()
            if done:
                break
            continue

        action = policy.select_action(state)
        if action not in valid_actions:
            action = valid_actions[0]

        next_state, reward, done, info = env.step(action)

        # Store transition into agent replay buffer (passive observation)
        agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    metrics = env.compute_metrics()
    metrics['reward'] = total_reward
    metrics['steps'] = steps
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# run_training_sync
# ─────────────────────────────────────────────────────────────────────────────

def run_training_sync(agent: DQNAgent, min_steps: int = 200) -> int:
    """Run offline gradient steps. Returns number of steps completed."""
    if len(agent.replay_buffer) < agent.min_replay_size:
        return 0

    steps = 0
    target_steps = max(min_steps, len(agent.replay_buffer) // 4)
    while steps < target_steps:
        agent.train_step()
        steps += 1
    return steps


# ─────────────────────────────────────────────────────────────────────────────
# run_dqn_sync
# ─────────────────────────────────────────────────────────────────────────────

def run_dqn_sync(sim_days: int, tasks_per_day: float, agent: DQNAgent,
                 seed: int) -> dict:
    """Run Phase 2 DQN scheduling synchronously. Returns compute_metrics()."""
    # Reset epsilon to Phase 2 start value
    agent.epsilon = cfg_module.EPSILON_PHASE2_START

    env = _make_env(sim_days, tasks_per_day, seed + 10000)
    state = env.reset()

    total_reward = 0.0
    steps = 0
    max_steps = sim_days * cfg_module.SLOTS_PER_DAY * 4

    while steps < max_steps:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            done, _ = env.advance_to_next_event()
            if done:
                break
            continue

        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)

        # Online learning during Phase 2
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        agent.update_epsilon()

        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    return env.compute_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# run_single_combo
# ─────────────────────────────────────────────────────────────────────────────

def run_single_combo(params: dict, sim_days: int, phase1_fraction: float,
                     tasks_per_day: float, seed: int) -> dict:
    """
    Full Phase1 + offline training + Phase2 run.
    Returns dict with all metrics + params + score.
    """
    t0 = time.time()
    try:
        apply_hyperparams(params)

        phase1_days = max(5, int(sim_days * phase1_fraction))
        phase2_days = max(5, sim_days - phase1_days)

        agent = _make_agent(sim_days)

        # ── Phase 1: run all baselines, fill replay buffer ────────────────────
        baselines = [("Greedy", GreedyBaseline), ("Skill", SkillBaseline),
                     ("FIFO", STFBaseline)]
        if _HAS_HYBRID:
            baselines.append(("Hybrid", HybridBaseline))
        if _HAS_RANDOM:
            baselines.append(("Random", RandomBaseline))

        for i, (bl_name, BLClass) in enumerate(baselines):
            run_baseline_sync(bl_name, BLClass, phase1_days, tasks_per_day,
                              agent, seed + i)

        # ── Offline training ──────────────────────────────────────────────────
        train_steps = run_training_sync(agent, min_steps=200)

        # ── Phase 2: DQN scheduling ───────────────────────────────────────────
        dqn_metrics = run_dqn_sync(phase2_days, tasks_per_day, agent, seed)

        buf_fill = len(agent.replay_buffer) / max(agent.replay_buffer.capacity, 1)

        result = {
            # Hyperparams
            **params,
            # DQN metrics
            'dqn_quality':    dqn_metrics.get('quality_score',      0.0),
            'dqn_throughput': dqn_metrics.get('throughput_per_day', 0.0),
            'dqn_lateness':   dqn_metrics.get('lateness_rate',      0.0),
            'dqn_overload':   dqn_metrics.get('overload_events',    0),
            'epsilon_final':  round(agent.epsilon, 5),
            'buffer_fill':    round(buf_fill, 3),
            'train_steps':    train_steps,
            'score':          score({
                'quality_score':      dqn_metrics.get('quality_score',      0.0),
                'throughput_per_day': dqn_metrics.get('throughput_per_day', 0.0),
                'lateness_rate':      dqn_metrics.get('lateness_rate',      0.0),
                'overload_events':    dqn_metrics.get('overload_events',    0),
            }),
            # Run metadata
            'sim_days':  sim_days,
            'seed':      seed,
            'elapsed_s': round(time.time() - t0, 2),
            'failed':    False,
        }
        return result

    except Exception as e:
        import traceback
        return {
            **params,
            'failed':    True,
            'error':     str(e),
            'traceback': traceback.format_exc(),
            'sim_days':  sim_days,
            'seed':      seed,
            'elapsed_s': round(time.time() - t0, 2),
            'score':     -999.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: List[dict], top_n: int = 20):
    """Print top-N results as a formatted table."""
    valid = [r for r in results if not r.get('failed', False)]
    valid.sort(key=lambda r: r.get('score', -999), reverse=True)
    failed = len(results) - len(valid)

    print("\n" + "=" * 72)
    print(f" SWEEP COMPLETE — {len(results)} runs, {failed} failed")
    print("=" * 72)
    header = (f" {'Rank':>4}  {'Score':>7}  {'Quality':>7}  {'Thpt':>6}"
              f"  {'LR':>8}  {'GAMMA':>5}  {'EPS':>6}  {'BASE':>5}"
              f"  {'ALPHA':>5}  {'TGT':>5}")
    print(header)
    print("-" * 72)
    for i, r in enumerate(valid[:top_n], 1):
        print(
            f" {i:>4}  {r['score']:>7.4f}  {r['dqn_quality']:>7.4f}"
            f"  {r['dqn_throughput']:>6.2f}"
            f"  {r['LEARNING_RATE']:>8.1e}  {r['GAMMA']:>5.2f}"
            f"  {r['EPSILON_DECAY']:>6.4f}  {r['REWARD_COMPLETION_BASE']:>5.1f}"
            f"  {r['PER_ALPHA']:>5.2f}  {r['TARGET_UPDATE_FREQ']:>5}"
        )

    if valid:
        best = valid[0]
        print("\n" + "-" * 72)
        print(" BEST CONFIG — paste into config.py:")
        print(f"   LEARNING_RATE          = {best['LEARNING_RATE']}")
        print(f"   GAMMA                  = {best['GAMMA']}")
        print(f"   EPSILON_DECAY          = {best['EPSILON_DECAY']}")
        print(f"   REWARD_COMPLETION_BASE = {best['REWARD_COMPLETION_BASE']}")
        print(f"   PER_ALPHA              = {best['PER_ALPHA']}")
        print(f"   TARGET_UPDATE_FREQ     = {best['TARGET_UPDATE_FREQ']}")
    print("=" * 72)


def save_results(results: List[dict], path: str = 'sweep_results.json'):
    """Save full results list as JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[sweep] Results saved to {path} ({len(results)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AMD SlingShot Hyperparameter Sweep — standalone, no server required"
    )
    parser.add_argument('--sim_days',        type=int,   default=30,
                        help='Total simulation days per combo (default: 30)')
    parser.add_argument('--phase1_fraction', type=float, default=0.55,
                        help='Fraction of sim_days for Phase 1 baselines (default: 0.55)')
    parser.add_argument('--tasks_per_day',   type=float, default=4.0,
                        help='Mean task arrival rate per day (default: 4.0)')
    parser.add_argument('--full',            action='store_true',
                        help='Run all 5120 combos instead of random sample')
    parser.add_argument('--n',               type=int,   default=200,
                        help='Number of random combos to sample (default: 200, ignored if --full)')
    parser.add_argument('--seed',            type=int,   default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output',          type=str,   default='sweep_results.json',
                        help='Output JSON path (default: sweep_results.json)')
    args = parser.parse_args()

    # ── Select combos ─────────────────────────────────────────────────────────
    if args.full:
        combos = ALL_COMBOS
        print(f"[sweep] Full sweep: {len(combos)} combos × {args.sim_days} days each")
    else:
        rng = random.Random(args.seed)
        combos = rng.sample(ALL_COMBOS, min(args.n, len(ALL_COMBOS)))
        print(f"[sweep] Random sample: {len(combos)} combos × {args.sim_days} days each")

    total = len(combos)
    results: List[dict] = []
    best_score = -999.0
    start_time = time.time()

    # ── Run sweep ─────────────────────────────────────────────────────────────
    for idx, params in enumerate(combos, 1):
        seed = args.seed + idx * 37   # deterministic per-run seed

        result = run_single_combo(
            params,
            sim_days=args.sim_days,
            phase1_fraction=args.phase1_fraction,
            tasks_per_day=args.tasks_per_day,
            seed=seed,
        )
        results.append(result)

        if not result.get('failed', False):
            best_score = max(best_score, result.get('score', -999))

        # ── Progress print every 10 or at end ─────────────────────────────────
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            eta_s = (elapsed / idx) * (total - idx)
            eta_min = eta_s / 60
            sc   = result.get('score', 0.0)
            q    = result.get('dqn_quality', 0.0)
            tp   = result.get('dqn_throughput', 0.0)
            eps  = result.get('epsilon_final', 0.0)
            print(
                f"[{idx:>4}/{total}]  score={sc:.4f}  best={best_score:.4f}"
                f"  q={q:.4f}  tp={tp:.2f}  eps={eps:.3f}  ETA={eta_min:.0f}min"
            )

    # ── Save + print table ────────────────────────────────────────────────────
    save_results(results, args.output)
    print_results_table(results)


if __name__ == '__main__':
    main()
