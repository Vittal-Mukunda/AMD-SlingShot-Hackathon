"""
run_pipeline.py — End-to-end pipeline orchestrator for RL-Driven Agentic Project Manager.

Usage:
    python run_pipeline.py --full                  # Run entire pipeline
    python run_pipeline.py --train                 # Train DQN only
    python run_pipeline.py --baselines             # Run baselines only
    python run_pipeline.py --evaluate              # Evaluate RL agent only
    python run_pipeline.py --stats                 # Statistical tests only
    python run_pipeline.py --plots                 # Generate all plots
    python run_pipeline.py --train --episodes 500  # Train with 500 episodes
    python run_pipeline.py --full --seed 99        # Full pipeline with seed 99
"""

import argparse
import os
import sys
import csv
import random
import time
import numpy as np
import torch
from datetime import datetime

# ── Ensure project root is on path ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config


# ── Utilities ────────────────────────────────────────────────────────────────

def set_global_seeds(seed: int):
    """Set deterministic seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"  [seed] random={seed}, numpy={seed}, torch={seed}")


def ensure_dirs():
    """Create required output directories (idempotent)."""
    for d in [config.RESULTS_DIR, config.CHECKPOINT_DIR, config.LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"  [dirs] results/ checkpoints/ logs/ ready")


def print_banner(title: str):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 72)


def save_reward_breakdown(breakdown_rows: list, output_file: str):
    """Write per-episode reward diagnostics to CSV."""
    if not breakdown_rows:
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fieldnames = list(breakdown_rows[0].keys())
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(breakdown_rows)
    print(f"  [diag] Reward breakdown saved → {output_file}")


def validate_stability(stats: dict):
    """Print stability warnings from training summary."""
    if not stats:
        return
    q_max = abs(stats.get('q_value_max', 0))
    if q_max > config.CONVERGENCE_THRESHOLD:
        print(f"  ⚠️  WARNING: Q-value magnitude {q_max:.1f} exceeds threshold {config.CONVERGENCE_THRESHOLD}")
    if not stats.get('training_stable', True):
        print("  ⚠️  WARNING: Training reported instability (NaN or explosion detected)")
    if stats.get('early_stopping_triggered', False):
        ep = stats.get('total_episodes', '?')
        print(f"  ℹ️  Early stopping triggered after {ep} episodes")


# ── Phase 1: Train DQN ───────────────────────────────────────────────────────

def phase_train(args):
    print_banner("PHASE 1 — DQN Training")
    from training.train_dqn import train_dqn
    from training.visualize import plot_learning_curve
    from environment.project_env import ProjectEnv
    from agents.dqn_agent import DQNAgent
    from utils.metrics import compute_composite_score

    set_global_seeds(args.seed)
    ensure_dirs()

    t0 = time.time()

    # Run training — reuse existing function (zero logic duplication)
    summary = train_dqn(
        max_episodes=args.episodes,
        seed=args.seed,
        results_dir=config.RESULTS_DIR,
        checkpoints_dir=config.CHECKPOINT_DIR,
    )

    validate_stability(summary)

    # ── Reward Breakdown: replay training_log episodes through env ──────────
    # We do a lightweight post-training pass to collect per-episode breakdowns.
    # This is a separate, greedy-policy replay on TRAIN seeds; it does not re-train.
    print("\n  Collecting reward breakdown diagnostics (greedy replay, 50 episodes)...")
    breakdown_rows = []

    model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(model_path):
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        agent.load(model_path)
        agent.epsilon = 0.0
        agent.policy_net.eval()

        replay_episodes = min(50, args.episodes)
        for ep in range(replay_episodes):
            env = ProjectEnv(seed=args.seed + ep, reward_scale=0.1)
            state = env.reset()
            done = False
            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                action = agent.select_action(state, valid, greedy=True)
                state, _, done, _ = env.step(action)

            bd = env.get_episode_reward_breakdown()
            bd['episode'] = ep
            bd['seed'] = args.seed + ep
            metrics = env.compute_metrics()
            bd['tasks_completed'] = metrics['throughput']
            breakdown_rows.append(bd)
    else:
        print("  [skip] best_model.pth not found — skipping reward breakdown collection")

    save_reward_breakdown(
        breakdown_rows,
        os.path.join(config.RESULTS_DIR, 'reward_breakdown.csv')
    )

    # Learning curve
    print("\n  Generating learning curve...")
    plot_learning_curve()

    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 1 complete in {elapsed/60:.1f} min")
    return summary


# ── Phase 2: Baseline Evaluation ─────────────────────────────────────────────

def phase_baselines(args):
    print_banner("PHASE 2 — Baseline Evaluation")
    from training.train_baselines import run_baselines

    set_global_seeds(args.seed)
    ensure_dirs()

    t0 = time.time()
    run_baselines(
        num_episodes=args.baseline_episodes,
        seed=args.seed,
        output_file=os.path.join(config.RESULTS_DIR, 'baseline_performance.csv')
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 2 complete in {elapsed/60:.1f} min")


# ── Phase 3: RL Agent Evaluation ─────────────────────────────────────────────

def phase_evaluate(args):
    print_banner("PHASE 3 — RL Agent Evaluation (4 conditions)")
    from evaluation.evaluate_agent import evaluate_agent

    set_global_seeds(args.seed)
    ensure_dirs()

    t0 = time.time()
    evaluate_agent(
        model_path=os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
        output_file=os.path.join(config.RESULTS_DIR, 'rl_test_performance.csv'),
        num_episodes=args.eval_episodes,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Phase 3 complete in {elapsed/60:.1f} min")


# ── Phase 4: Statistical Tests ───────────────────────────────────────────────

def phase_stats(args):
    print_banner("PHASE 4 — Statistical Testing (Welch's t-test + Bonferroni + Cohen's d)")
    from evaluation.statistical_tests import run_statistical_tests

    ensure_dirs()

    run_statistical_tests(
        baseline_file=os.path.join(config.RESULTS_DIR, 'baseline_performance.csv'),
        rl_file=os.path.join(config.RESULTS_DIR, 'rl_test_performance.csv'),
        output_file=os.path.join(config.RESULTS_DIR, 'statistical_summary.csv'),
    )
    print(f"\n  ✓ Phase 4 complete")


# ── Phase 5: Visualizations ──────────────────────────────────────────────────

def phase_plots(args):
    print_banner("PHASE 5 — Visualization (learning curve + money plot)")
    from training.visualize import plot_learning_curve
    from visualization.plot_metrics import plot_metrics

    ensure_dirs()

    print("  Generating learning curve...")
    plot_learning_curve(
        log_path=os.path.join(config.RESULTS_DIR, 'training_log.csv'),
        output_path=os.path.join(config.RESULTS_DIR, 'learning_curve.png'),
    )

    print("  Generating money plot...")
    plot_metrics(
        baseline_file=os.path.join(config.RESULTS_DIR, 'baseline_performance.csv'),
        rl_file=os.path.join(config.RESULTS_DIR, 'rl_test_performance.csv'),
        output_file=os.path.join(config.RESULTS_DIR, 'money_plot.png'),
    )

    print(f"\n  ✓ Phase 5 complete")
    print(f"  Outputs:")
    for f in ['learning_curve.png', 'money_plot.png']:
        path = os.path.join(config.RESULTS_DIR, f)
        status = '✓' if os.path.exists(path) else '✗ MISSING'
        print(f"    {status}  {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RL-Driven Agentic Project Manager — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --full
  python run_pipeline.py --train --episodes 500
  python run_pipeline.py --train --baselines --evaluate
  python run_pipeline.py --stats --plots
        """
    )

    # Pipeline stage flags
    parser.add_argument('--full',      action='store_true', help='Run entire pipeline end-to-end')
    parser.add_argument('--train',     action='store_true', help='Run DQN training')
    parser.add_argument('--baselines', action='store_true', help='Run baseline evaluations')
    parser.add_argument('--evaluate',  action='store_true', help='Evaluate trained RL agent')
    parser.add_argument('--stats',     action='store_true', help='Run statistical tests')
    parser.add_argument('--plots',     action='store_true', help='Generate plots')

    # Runtime overrides
    parser.add_argument('--episodes',          type=int,   default=config.TRAIN_EPISODES,
                        help=f'Training episodes (default: {config.TRAIN_EPISODES})')
    parser.add_argument('--baseline-episodes', type=int,   default=1000,
                        help='Episodes per baseline (default: 1000)')
    parser.add_argument('--eval-episodes',     type=int,   default=1000,
                        help='Evaluation episodes per condition (default: 1000)')
    parser.add_argument('--seed',              type=int,   default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--variance-mult',     type=float, default=config.TEST_VARIANCE_MULTIPLIER,
                        help=f'High-variance multiplier (default: {config.TEST_VARIANCE_MULTIPLIER})')
    parser.add_argument('--shock-prob',        type=float, default=config.TEST_SHOCK_PROB_HIGH,
                        help=f'Frequent-shock probability (default: {config.TEST_SHOCK_PROB_HIGH})')

    args = parser.parse_args()

    # Apply CLI overrides to config so downstream modules pick them up
    if args.variance_mult != config.TEST_VARIANCE_MULTIPLIER:
        config.TEST_VARIANCE_MULTIPLIER = args.variance_mult
        print(f"  [override] TEST_VARIANCE_MULTIPLIER → {args.variance_mult}")
    if args.shock_prob != config.TEST_SHOCK_PROB_HIGH:
        config.TEST_SHOCK_PROB_HIGH = args.shock_prob
        print(f"  [override] TEST_SHOCK_PROB_HIGH → {args.shock_prob}")

    # Resolve --full as all stages
    if args.full:
        args.train = args.baselines = args.evaluate = args.stats = args.plots = True

    # Require at least one stage
    if not any([args.train, args.baselines, args.evaluate, args.stats, args.plots]):
        parser.print_help()
        sys.exit(0)

    overall_start = time.time()

    print_banner("RL-DRIVEN AGENTIC PROJECT MANAGER — PIPELINE START")
    print(f"  Episodes (train): {args.episodes}")
    print(f"  Episodes (baselines): {args.baseline_episodes}")
    print(f"  Episodes (eval): {args.eval_episodes}")
    print(f"  Base seed: {args.seed}")
    print(f"  Stages: "
          f"{'TRAIN ' if args.train else ''}"
          f"{'BASELINES ' if args.baselines else ''}"
          f"{'EVALUATE ' if args.evaluate else ''}"
          f"{'STATS ' if args.stats else ''}"
          f"{'PLOTS' if args.plots else ''}")

    # ── Execute stages in order ──────────────────────────────────────────────
    train_summary = None

    if args.train:
        train_summary = phase_train(args)

    if args.baselines:
        phase_baselines(args)

    if args.evaluate:
        phase_evaluate(args)

    if args.stats:
        phase_stats(args)

    if args.plots:
        phase_plots(args)

    # ── Final summary ────────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print_banner("PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} min")
    print()
    print("  Expected output files:")
    output_files = {
        'Training log':         'results/training_log.csv',
        'Reward breakdown':     'results/reward_breakdown.csv',
        'Best model':           'checkpoints/best_model.pth',
        'Baseline results':     'results/baseline_performance.csv',
        'RL eval results':      'results/rl_test_performance.csv',
        'Statistical summary':  'results/statistical_summary.csv',
        'Learning curve':       'results/learning_curve.png',
        'Money plot':           'results/money_plot.png',
    }
    for label, rel_path in output_files.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        status = '✓' if os.path.exists(full_path) else '·'
        print(f"    {status}  {rel_path:<35} — {label}")

    if train_summary:
        print()
        print(f"  Training Summary:")
        print(f"    Best moving avg return: {train_summary.get('best_moving_avg_return', 0):.2f}")
        print(f"    Training stable:        {'YES' if train_summary.get('training_stable') else 'NO'}")
        print(f"    Early stopped:          {'YES' if train_summary.get('early_stopping_triggered') else 'NO'}")


if __name__ == '__main__':
    main()
