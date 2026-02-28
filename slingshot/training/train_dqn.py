"""
Complete DQN Training Loop for Project Task Allocation
Implements: Training with logging, checkpointing, early stopping, and stability monitoring

Changes (v3 — Dueling DQN + PER):
  - Agent now uses DuelingQNetwork + PrioritizedReplayBuffer + Double DQN
  - Passes episode index to update_epsilon() for CosineAnnealingWarmRestarts
  - Learning rate and batch size driven from config.py (v3 tuned values)
  - PER beta logged in CSV for monitoring IS-correction progress
  - Backward-incompatible checkpoint format: delete old best_model.pth before re-training
"""

import numpy as np
import torch
import random
import sys
import os
import csv
import time
from pathlib import Path


from slingshot.core.settings import config
from slingshot.environment.project_env import ProjectEnv
from slingshot.agents.dqn_agent import DQNAgent


class TrainingLogger:
    """Efficient CSV logger for training metrics including reward breakdowns."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.fieldnames = [
            'episode', 'epsilon', 'episode_return', 'moving_avg_return',
            'mean_step_reward', 'mean_q_value', 'mean_td_error',
            'tasks_completed', 'deadline_hit_rate', 'overload_events',
            # Per-component reward diagnostics (raw, unscaled)
            'comp_reward', 'delay_penalty', 'deadline_penalty',
            'overload_penalty', 'throughput_bonus',
            'early_stopping_triggered'
        ]

        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_episode(self, metrics: dict):
        row = {k: metrics.get(k, 0.0) for k in self.fieldnames}
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def train_dqn(
    max_episodes: int = 5000,
    min_replay_size: int = 1000,
    checkpoint_freq: int = 100,
    early_stopping_patience: int = 1000,
    moving_avg_window: int = 50,
    reward_scale: float = 0.1,
    learning_rate: float = None,      # None → use config.LEARNING_RATE
    batch_size: int = None,            # None → use config.BATCH_SIZE
    seed: int = 42,
    results_dir: str = None,
    checkpoints_dir: str = None,
    enable_diagnostics: bool = False,
):
    """
    Main DQN training loop (v3 — Dueling DQN + PER).

    Args:
        max_episodes:           Maximum training episodes (5000)
        min_replay_size:        Replay warmup size (1000)
        checkpoint_freq:        Episodes between periodic checkpoints (100)
        early_stopping_patience:Episodes without improvement before stopping (1000)
        moving_avg_window:      Window for moving average return (50)
        reward_scale:           Environment reward scaling (0.1)
        learning_rate:          DQN learning rate (None → config.LEARNING_RATE=0.001)
        batch_size:             Mini-batch size (None → config.BATCH_SIZE=128)
        seed:                   Random seed for reproducibility
        results_dir:            Directory for results
        checkpoints_dir:        Directory for checkpoints
        enable_diagnostics:     Enable environment diagnostics

    Returns:
        Dictionary with training summary
    """
    # Setup directories
    results_dir = results_dir or config.RESULTS_DIR
    checkpoints_dir = checkpoints_dir or config.CHECKPOINT_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Set seeds for reproducibility (all 3 libraries)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize environment
    env = ProjectEnv(
        seed=seed,
        reward_scale=reward_scale,
        enable_diagnostics=enable_diagnostics
    )

    # Resolve defaults from config
    lr         = learning_rate if learning_rate is not None else config.LEARNING_RATE
    batch      = batch_size    if batch_size    is not None else config.BATCH_SIZE

    # Initialize DQN agent (v3)
    agent = DQNAgent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        learning_rate=lr,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        replay_capacity=config.REPLAY_BUFFER_SIZE,
        batch_size=batch,
        target_update_freq=config.TARGET_UPDATE_FREQ
    )

    # Initialize logger
    log_path = os.path.join(results_dir, 'training_log.csv')
    logger = TrainingLogger(log_path)

    # Training state
    episode_returns = []
    moving_avg_returns = []
    best_moving_avg = -np.inf
    episodes_since_improvement = 0
    early_stopped = False
    retried_with_reduced_lr = False
    training_stable = True

    # Q-value tracking for stability
    all_q_values = []

    print("=" * 80)
    print("DQN TRAINING START (v3 — Dueling DQN + PER + Double DQN + Cosine LR)")
    print("=" * 80)
    print(f"Device:                  {agent.device}")
    print(f"Architecture:            DuelingQNetwork  {config.HIDDEN_LAYERS} shared")
    print(f"Max episodes:            {max_episodes}")
    print(f"Reward scale:            {reward_scale}")
    print(f"Learning rate:           {lr}  (CosineAnnealingWarmRestarts T0={config.LR_SCHEDULER_T0})")
    print(f"Batch size:              {batch}")
    print(f"Epsilon decay:           {config.EPSILON_DECAY}  (→ 0.05 ~ep 5000)")
    print(f"Replay buffer:           {config.REPLAY_BUFFER_SIZE:,} transitions  [PER α={config.PER_ALPHA}]")
    print(f"Min replay size:         {min_replay_size}")
    print(f"Target update freq:      {config.TARGET_UPDATE_FREQ} steps")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Best-model guard:        episodes ≥ {moving_avg_window}")
    print("=" * 80)

    start_time = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_q_values = []
        episode_td_errors = []
        done = False
        timestep = 0

        while not done and timestep < config.EPISODE_HORIZON:
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                break

            action = agent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            # Train only once replay buffer is warm
            if len(agent.replay_buffer) >= min_replay_size:
                loss, q_mean, td_error = agent.train_step()

                if np.isnan(loss) or np.isnan(q_mean):
                    print(f"\n  WARNING: NaN detected at episode {episode}, step {timestep}")
                    training_stable = False
                    if not retried_with_reduced_lr:
                        retried_with_reduced_lr = True

                episode_q_values.append(q_mean)
                episode_td_errors.append(td_error)

                if abs(q_mean) > 1000:
                    print(f"\n  WARNING: Exploding Q-values (|Q|={abs(q_mean):.1f}) at episode {episode}")
                    training_stable = False

            episode_reward += reward
            state = next_state
            timestep += 1

        # Update epsilon + LR scheduler after each episode
        agent.update_epsilon(episode=episode)

        # Compute task metrics
        metrics = env.compute_metrics()
        episode_returns.append(episode_reward)

        # Moving average
        if len(episode_returns) >= moving_avg_window:
            moving_avg = np.mean(episode_returns[-moving_avg_window:])
        else:
            moving_avg = np.mean(episode_returns)
        moving_avg_returns.append(moving_avg)

        if episode_q_values:
            all_q_values.extend(episode_q_values)

        # ── Best model guard ─────────────────────────────────────────────────
        # Only consider saving best model AFTER the moving average window is full.
        # This prevents a noisy early episode from locking in a "best" checkpoint
        # when the agent is still mostly random (Root Cause 4).
        if episode >= moving_avg_window and moving_avg > best_moving_avg:
            best_moving_avg = moving_avg
            episodes_since_improvement = 0
            best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
            agent.save(best_model_path)
        elif episode >= moving_avg_window:
            episodes_since_improvement += 1

        # Early stopping
        if episodes_since_improvement >= early_stopping_patience:
            early_stopped = True
            print(f"\n  Early stopping triggered at episode {episode}")
            print(f"  No improvement for {early_stopping_patience} episodes")
            print(f"  Best moving avg: {best_moving_avg:.2f}")

        # ── Reward breakdown (unscaled) ──────────────────────────────────────
        breakdown = env.get_episode_reward_breakdown()
        # Unscale for human-readable logging (breakdown values are already raw)

        # ── CSV logging ──────────────────────────────────────────────────────
        log_metrics = {
            'episode':                   episode,
            'epsilon':                   agent.epsilon,
            'episode_return':            episode_reward,
            'moving_avg_return':         moving_avg,
            'mean_step_reward':          episode_reward / max(1, timestep),
            'mean_q_value':              np.mean(episode_q_values) if episode_q_values else 0.0,
            'mean_td_error':             np.mean(episode_td_errors) if episode_td_errors else 0.0,
            'tasks_completed':           metrics['throughput'],
            'deadline_hit_rate':         metrics['deadline_hit_rate'],
            'overload_events':           metrics['overload_events'],
            'comp_reward':               breakdown.get('completion_reward', 0.0),
            'delay_penalty':             breakdown.get('delay_penalty', 0.0),
            'deadline_penalty':          breakdown.get('deadline_miss', 0.0) + breakdown.get('lateness_penalty', 0.0),
            'overload_penalty':          breakdown.get('overload_penalty', 0.0),
            'throughput_bonus':          breakdown.get('makespan_bonus', 0.0),
            'early_stopping_triggered':  early_stopped,
        }
        logger.log_episode(log_metrics)


        # Periodic checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_ep{episode+1}.pth')
            agent.save(checkpoint_path)

        # Progress logging (every 100 episodes)
        if (episode + 1) % 100 == 0:
            elapsed = time.time() - start_time
            q_display = np.mean(episode_q_values) if episode_q_values else 0.0
            print(f"Episode {episode+1:>5}/{max_episodes} | "
                  f"Return: {episode_reward:7.2f} | "
                  f"MA-50: {moving_avg:7.2f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Q: {q_display:6.2f} | "
                  f"Tasks: {metrics['throughput']:2}/{config.NUM_TASKS} | "
                  f"Time: {elapsed:.0f}s")

        if early_stopped:
            break

    # ── Training complete ─────────────────────────────────────────────────────
    total_time = time.time() - start_time

    # Save final best_model if not yet saved (< moving_avg_window episodes ran)
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        agent.save(best_model_path)
        print("  [warn] best_model.pth saved at training end (guard window not reached)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    final_epsilon = agent.epsilon
    final_moving_avg = moving_avg_returns[-1] if moving_avg_returns else 0.0
    q_min = np.min(all_q_values) if all_q_values else 0.0
    q_max = np.max(all_q_values) if all_q_values else 0.0

    summary = {
        'total_episodes':            len(episode_returns),
        'final_epsilon':             final_epsilon,
        'best_moving_avg_return':    best_moving_avg,
        'final_moving_avg_return':   final_moving_avg,
        'q_value_min':               q_min,
        'q_value_max':               q_max,
        'training_stable':           training_stable,
        'retried_with_reduced_lr':   retried_with_reduced_lr,
        'early_stopping_triggered':  early_stopped,
        'total_training_time':       total_time,
    }

    print(f"Total Episodes:          {summary['total_episodes']}")
    print(f"Final Epsilon:           {summary['final_epsilon']:.4f}")
    print(f"Best Moving Avg Return:  {summary['best_moving_avg_return']:.2f}")
    print(f"Final Moving Avg Return: {summary['final_moving_avg_return']:.2f}")
    print(f"Q-value Range:           [{summary['q_value_min']:.2f}, {summary['q_value_max']:.2f}]")
    print(f"Training Stable:         {'YES' if summary['training_stable'] else 'NO'}")
    print(f"Early Stopping:          {'YES' if summary['early_stopping_triggered'] else 'NO'}")
    print(f"Training Time:           {summary['total_training_time']:.1f}s "
          f"({summary['total_training_time']/60:.1f}min)")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    summary = train_dqn(
        max_episodes=5000,
        reward_scale=0.1,
        # learning_rate and batch_size default to config values
        seed=42,
    )
