"""
continual_scheduler.py — Two-Phase Adaptive Scheduling Framework

Phase 1 (PHASE1_DAYS working days):
  - Baselines (Round-robin: Greedy → Skill → Hybrid) drive all scheduling
  - DQN PASSIVELY observes: collects (s, a, r, s', done) into replay buffer
  - No gradient updates during Phase 1 (buffer warm-up only)
  - Metrics logged per day: throughput, idle%, lateness, completion rate

Phase 2 (PHASE2_DAYS working days):
  - DQN takes full control with epsilon starting at EPSILON_PHASE2_START
  - Online learning: train_step() called every decision slot
  - Target network synced every TARGET_UPDATE_FREQ decisions
  - Epsilon decays progressively per decision throughout Phase 2
  - Metrics logged per day: same as Phase 1 + loss, Q-mean, TD-error

Comparison:
  - Phase 1 vs Phase 2 metrics saved to results/phase{1,2}_metrics.csv
  - Console summary printed at end
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

from slingshot.core.settings import config
from slingshot.environment.project_env import ProjectEnv
from slingshot.agents.dqn_agent import DQNAgent

# ── Baseline imports ──────────────────────────────────────────────────────────
from slingshot.baselines.greedy_baseline  import GreedyBaseline
from slingshot.baselines.skill_baseline   import SkillBaseline
try:
    from slingshot.baselines.hybrid_baseline  import HybridBaseline
    _HAS_HYBRID = True
except ImportError:
    _HAS_HYBRID = False

try:
    from slingshot.baselines.random_baseline  import RandomBaseline
    _HAS_RANDOM = True
except ImportError:
    _HAS_RANDOM = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_action(action: int, valid: List[int], fallback: int = None) -> int:
    """Return action if valid; else fallback or first valid."""
    if action in valid:
        return action
    if fallback is not None and fallback in valid:
        return fallback
    return valid[0] if valid else 100   # defer as last resort


def _build_baselines(env: ProjectEnv, skill_baseline: SkillBaseline) -> List:
    """Build round-robin baseline pool."""
    pool = [GreedyBaseline(env)]
    if _HAS_HYBRID:
        pool.append(HybridBaseline(env))
    pool.append(skill_baseline)
    if _HAS_RANDOM:
        pool.append(RandomBaseline(env))
    return pool


def _collect_day_metrics(env: ProjectEnv, day_decisions: int,
                          losses: List[float], q_means: List[float]) -> Dict:
    """Aggregate per-day metrics."""
    m = env.compute_metrics()
    return {
        'throughput_per_day':  m['throughput_per_day'],
        'completion_rate':    m['completion_rate'],
        'lateness_rate':      m['lateness_rate'],
        'quality_score':      m['quality_score'],
        'load_balance':       m['load_balance'],
        'overload_events':    m['overload_events'],
        'decisions':          day_decisions,
        'mean_loss':          float(np.nanmean(losses))  if losses  else 0.0,
        'mean_q':             float(np.nanmean(q_means)) if q_means else 0.0,
    }


def _write_csv(path: str, rows: List[Dict], fieldnames: Optional[List[str]] = None):
    """Append rows to CSV; create with header if new file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    fnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _print_day_header(phase: int, day: int, tick: int):
    print(f"\n{'─'*60}")
    print(f"  Phase {phase} | Working Day {day:>3} | Tick {tick:>5}")
    print(f"{'─'*60}")


def _print_decision_log(tick: int, phase: int, decision_n: int,
                        action: int, reward: float, epsilon: float,
                        loss: float = 0.0, q_mean: float = 0.0,
                        policy_name: str = "?", train_steps: int = 0):
    # Log every 20 decisions in Phase 2 (for visible training progress)
    interval = 20 if phase == 2 else 50
    if decision_n % interval == 0 or decision_n == 1:
        train_tag = f" | trained={train_steps}" if phase == 2 else ""
        print(f"  [{phase}] tick={tick:>4}, dec={decision_n:>5}, "
              f"a={action:>3}, r={reward:>+6.2f}, "
              f"eps={epsilon:.3f}, loss={loss:.4f}, Q={q_mean:.3f}"
              f"{train_tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Baseline-driven + Passive DQN Observation
# ─────────────────────────────────────────────────────────────────────────────

def run_phase1(env: ProjectEnv, agent: DQNAgent, args) -> List[Dict]:
    """
    Run Phase 1 (PHASE1_DAYS working days).

    Baselines drive scheduling; DQN stores transitions but does NOT learn.
    Returns list of per-day metric dicts.
    """
    print("\n" + "═"*60)
    print("  PHASE 1 — Baseline Observation  (DQN passive)")
    print(f"  {config.PHASE1_DAYS} working days | {config.PHASE1_DAYS * config.SLOTS_PER_DAY} slots")
    print("═"*60)

    state        = env.reset()
    all_metrics  = []
    day_losses, day_qmeans = [], []
    day_decisions = 0
    total_decisions = 0
    prev_day = 0

    # Build baseline pool
    skill_bl  = SkillBaseline(env)
    baselines = _build_baselines(env, skill_bl)
    baseline_idx = 0

    phase1_slots = config.PHASE1_DAYS * config.SLOTS_PER_DAY

    for slot in range(phase1_slots):
        current_day = env.clock.day

        # Day boundary: log metrics and reset daily accumulators
        if current_day > prev_day:
            m = _collect_day_metrics(env, day_decisions, day_losses, day_qmeans)
            m['phase'] = 1; m['day'] = prev_day
            all_metrics.append(m)
            _print_day_header(1, prev_day, env.clock.tick)
            print(f"    throughput/day={m['throughput_per_day']:.2f}, "
                  f"completion={m['completion_rate']:.2%}, "
                  f"lateness={m['lateness_rate']:.2%}, "
                  f"overload={m['overload_events']}")
            # End of baseline observation day: update skill estimates
            skill_bl.observe_episode(env)

            day_losses, day_qmeans = [], []
            day_decisions = 0
            prev_day      = current_day
            baseline_idx  = (baseline_idx + 1) % len(baselines)  # rotate baseline

        valid = env.get_valid_actions()
        if not valid:
            # Nothing to do this slot — advance with a no-op defer
            _, _, done, _ = env.step(20 * env.num_workers)
            if done:
                break
            continue

        # Baseline selects action
        bl      = baselines[baseline_idx % len(baselines)]
        action  = _safe_action(bl.select_action(state), valid)

        # DQN passively stores this transition
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, float(done))

        _print_decision_log(env.clock.tick, 1, total_decisions + 1,
                            action, reward, agent.epsilon,
                            policy_name=bl.name)

        state           = next_state
        total_decisions += 1
        day_decisions   += 1
        agent.steps_done += 1

        if done:
            break

    # Final day metrics
    if day_decisions > 0:
        m = _collect_day_metrics(env, day_decisions, day_losses, day_qmeans)
        m['phase'] = 1; m['day'] = env.clock.day
        all_metrics.append(m)

    print(f"\n[Phase 1 complete] replay_buffer={len(agent.replay_buffer)} transitions, "
          f"total_decisions={total_decisions}")
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — DQN Online Learning
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(env: ProjectEnv, agent: DQNAgent, args) -> List[Dict]:
    """
    Run Phase 2 (PHASE2_DAYS working days).

    DQN controls all scheduling; learns online with per-decision gradient updates.
    Epsilon starts at EPSILON_PHASE2_START (lower than 1.0 due to Phase 1 warm-up).
    Returns list of per-day metric dicts.
    """
    print("\n" + "═"*60)
    print("  PHASE 2 — DQN Online Scheduling (continuous learning)")
    print(f"  {config.PHASE2_DAYS} working days | {config.PHASE2_DAYS * config.SLOTS_PER_DAY} slots")
    print("═"*60)

    # Set Phase 2 starting epsilon
    agent.set_epsilon(config.EPSILON_PHASE2_START)
    print(f"  Starting ε = {agent.epsilon:.3f}")

    # Continue from Phase 1 clock — do NOT reset env (seamless continuation)
    # The env's task list continues; new tasks still arriving per Poisson schedule
    state       = env._get_state()
    all_metrics = []
    day_losses, day_qmeans = [], []
    day_decisions  = 0
    total_decisions = 0
    prev_day = env.clock.day

    ckpt_dir  = config.CHECKPOINT_DIR
    ckpt_path = os.path.join(ckpt_dir, 'dqn_phase2_latest.pth')
    best_throughput = -np.inf

    phase2_slots = config.PHASE2_DAYS * config.SLOTS_PER_DAY
    phase2_start_tick = env.clock.tick

    while (env.clock.tick - phase2_start_tick) < phase2_slots:
        current_day = env.clock.day

        # Day boundary
        if current_day > prev_day:
            m = _collect_day_metrics(env, day_decisions, day_losses, day_qmeans)
            m['phase'] = 2; m['day'] = current_day - 1
            all_metrics.append(m)
            _print_day_header(2, current_day - 1, env.clock.tick)
            print(f"    throughput/day={m['throughput_per_day']:.2f}, "
                  f"completion={m['completion_rate']:.2%}, "
                  f"eps={agent.epsilon:.3f}, "
                  f"loss={m['mean_loss']:.4f}, Q={m['mean_q']:.3f}, "
                  f"train_steps={agent.train_steps}, skipped={agent.train_skipped}")

            # Checkpoint if best
            if m['throughput_per_day'] > best_throughput:
                best_throughput = m['throughput_per_day']
                agent.save(ckpt_path)

            day_losses, day_qmeans = [], []
            day_decisions  = 0
            prev_day       = current_day

        valid = env.get_valid_actions()
        if not valid:
            _, _, done, _ = env.step(20 * env.num_workers)
            if done:
                break
            continue

        # DQN online_step: select → execute → store → learn → decay ε
        action, reward, next_state, done, loss, q_mean = agent.online_step(
            state, valid, env, train_every=1
        )

        _print_decision_log(env.clock.tick, 2, total_decisions + 1,
                            action, reward, agent.epsilon, loss, q_mean,
                            policy_name="DQN", train_steps=agent.train_steps)

        if not np.isnan(loss):
            day_losses.append(loss)
        if not np.isnan(q_mean):
            day_qmeans.append(q_mean)

        state            = next_state
        total_decisions += 1
        day_decisions   += 1

        if done:
            break

    # Final day metrics
    if day_decisions > 0:
        m = _collect_day_metrics(env, day_decisions, day_losses, day_qmeans)
        m['phase'] = 2; m['day'] = env.clock.day
        all_metrics.append(m)

    print(f"\n[Phase 2 complete] train_steps={agent.train_steps}, "
          f"final_ε={agent.epsilon:.4f}, "
          f"best_throughput/day={best_throughput:.2f}")

    # Save final model
    final_path = os.path.join(ckpt_dir, 'dqn_online_final.pth')
    agent.save(final_path)
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Comparison & Results
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(p1_metrics: List[Dict], p2_metrics: List[Dict]):
    """Print a summary comparison table of Phase 1 vs Phase 2."""
    def agg(rows, key):
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else 0.0

    keys = ['throughput_per_day', 'completion_rate', 'lateness_rate',
            'quality_score', 'overload_events']
    labels = ['Throughput/day', 'Completion rate', 'Lateness rate',
              'Avg quality', 'Overload events']

    print("\n" + "═"*60)
    print("  PERFORMANCE COMPARISON: Phase 1 (Baseline) vs Phase 2 (DQN)")
    print("═"*60)
    print(f"  {'Metric':<22} {'Phase 1':>10} {'Phase 2':>10} {'Δ':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*8}")
    for key, label in zip(keys, labels):
        v1 = agg(p1_metrics, key)
        v2 = agg(p2_metrics, key)
        delta = v2 - v1
        sign  = '+' if delta >= 0 else ''
        print(f"  {label:<22} {v1:>10.3f} {v2:>10.3f} {sign}{delta:>7.3f}")
    print("═"*60)


def save_results(p1_metrics: List[Dict], p2_metrics: List[Dict]):
    """Save phase metrics to CSV files."""
    results_dir = config.RESULTS_DIR
    for phase, rows in [(1, p1_metrics), (2, p2_metrics)]:
        if not rows:
            continue
        path = os.path.join(results_dir, f'phase{phase}_metrics.csv')
        # Overwrite with fresh results
        if os.path.exists(path):
            os.remove(path)
        _write_csv(path, rows)
        print(f"  Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Continual Online Learning — Two-Phase DQN Scheduler'
    )
    parser.add_argument('--seed',       type=int, default=42,
                        help='Random seed')
    parser.add_argument('--days-p1',    type=int, default=config.PHASE1_DAYS,
                        help='Phase 1 working days')
    parser.add_argument('--days-p2',    type=int, default=config.PHASE2_DAYS,
                        help='Phase 2 working days')
    parser.add_argument('--tasks',      type=int, default=config.TOTAL_TASKS,
                        help='Total tasks')
    parser.add_argument('--workers',    type=int, default=config.NUM_WORKERS,
                        help='Number of workers')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Fast smoke test (2+3 days, no sleep)')
    parser.add_argument('--debug-skill', action='store_true',
                        help='Enable verbose skill baseline logging')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to DQN checkpoint to resume from')
    parser.add_argument('--phase1-only', action='store_true',
                        help='Run Phase 1 only')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Run Phase 2 only (requires checkpoint)')
    parser.add_argument('--debug-training', action='store_true',
                        help='Enable verbose per-step DQN training logs')
    args = parser.parse_args()

    if args.debug_skill:
        config.BASELINE_DEBUG_SKILL = True

    if args.smoke_test:
        args.days_p1 = 2
        args.days_p2 = 3
        args.tasks   = 60   # increased for fuller buffer warm-up
        print(f"🔥 Smoke-test mode: {args.days_p1}+{args.days_p2} days, {args.tasks} tasks")

    # Override phase days in config for env slot calculation
    config.PHASE1_DAYS   = args.days_p1
    config.PHASE2_DAYS   = args.days_p2
    config.TOTAL_SIM_DAYS = args.days_p1 + args.days_p2
    total_slots = config.TOTAL_SIM_DAYS * config.SLOTS_PER_DAY

    np.random.seed(args.seed)

    print(f"\n{'═'*60}")
    print(f"  Continual DQN Scheduler  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Seed={args.seed} | Workers={args.workers} | Tasks={args.tasks}")
    print(f"  Phase1={args.days_p1}d | Phase2={args.days_p2}d | Slots={total_slots}")
    print(f"{'═'*60}\n")

    # ── Build environment ────────────────────────────────────────────────────
    env = ProjectEnv(
        num_workers     = args.workers,
        total_tasks     = args.tasks,
        seed            = args.seed,
        total_sim_slots = total_slots,
    )

    # ── Build DQN agent ──────────────────────────────────────────────────────
    agent = DQNAgent()
    if hasattr(args, 'debug_training') and args.debug_training:
        agent.debug_training = True
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        agent.load(args.load_checkpoint)

    print(f"  [Agent] device={agent.device}, batch={agent.batch_size}, "
          f"min_replay={agent.min_replay_size}, "
          f"buf_capacity={agent.replay_buffer.tree.capacity}")


    # ── Run phases ───────────────────────────────────────────────────────────
    p1_metrics, p2_metrics = [], []

    if not args.phase2_only:
        p1_metrics = run_phase1(env, agent, args)

    if not args.phase1_only:
        p2_metrics = run_phase2(env, agent, args)

    # ── Final metrics & comparison ───────────────────────────────────────────
    print_comparison(p1_metrics, p2_metrics)
    save_results(p1_metrics, p2_metrics)

    overall = env.compute_metrics()
    print(f"\n  Overall simulation metrics:")
    print(f"    Makespan:        {overall['makespan_hours']:.1f}h")
    print(f"    Throughput/day:  {overall['throughput_per_day']:.2f}")
    print(f"    Completion rate: {overall['completion_rate']:.2%}")
    print(f"    Lateness rate:   {overall['lateness_rate']:.2%}")
    print(f"    Quality score:   {overall['quality_score']:.3f}")
    print(f"    DQN train steps: {agent.train_steps}")
    print(f"    Final ε:         {agent.epsilon:.4f}")
    print(f"\n  🟢 Done — results in: {config.RESULTS_DIR}")


if __name__ == '__main__':
    main()
