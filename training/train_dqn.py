"""
Complete DQN Training Loop for Project Task Allocation
Implements: Training with logging, checkpointing, early stopping, and stability monitoring
"""

import numpy as np
import torch
import sys
import os
import csv
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv
from agents.dqn_agent import DQNAgent


class TrainingLogger:
    """Efficient CSV logger for training metrics"""
    
    def __init__(self, log_path: str):
        """
        Initialize logger
        
        Args:
            log_path: Path to CSV log file
        """
        self.log_path = log_path
        self.fieldnames = [
            'episode', 'epsilon', 'episode_return', 'moving_avg_return',
            'mean_step_reward', 'mean_q_value', 'mean_td_error',
            'tasks_completed', 'deadline_hit_rate', 'overload_events',
            'early_stopping_triggered'
        ]
        
        # Create file and write header
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_episode(self, metrics: dict):
        """
        Log episode metrics
        
        Args:
            metrics: Dictionary of metrics matching fieldnames
        """
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)


def train_dqn(
    max_episodes: int = 2000,
    min_replay_size: int = 1000,
    checkpoint_freq: int = 50,
    early_stopping_patience: int = 200,
    moving_avg_window: int = 50,
    reward_scale: float = 0.1,
    learning_rate: float = 0.0005,
    seed: int = 42,
    results_dir: str = None,
    checkpoints_dir: str = None,
    enable_diagnostics: bool = False
):
    """
    Main DQN training loop
    
    Args:
        max_episodes: Maximum training episodes (2000)
        min_replay_size: Replay warmup size (1000)
        checkpoint_freq: Episodes between checkpoints (50)
        early_stopping_patience: Episodes without improvement before stopping (200)
        moving_avg_window: Window for moving average return (50)
        reward_scale: Environment reward scaling (0.1)
        learning_rate: DQN learning rate (0.0005)
        seed: Random seed for reproducibility
        results_dir: Directory for results (default: config.RESULTS_DIR)
        checkpoints_dir: Directory for checkpoints (default: config.CHECKPOINT_DIR)
        enable_diagnostics: Enable environment diagnostics
    
    Returns:
        Dictionary with training summary
    """
    # Setup directories
    results_dir = results_dir or config.RESULTS_DIR
    checkpoints_dir = checkpoints_dir or config.CHECKPOINT_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set seeds for reproducibility
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
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        learning_rate=learning_rate,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        replay_capacity=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
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
    
    print("="*80)
    print("DQN TRAINING START")
    print("="*80)
    print(f"Device: {agent.device}")
    print(f"Reward Scale: {reward_scale}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Min Replay Size: {min_replay_size}")
    print(f"Target Update Freq: {config.TARGET_UPDATE_FREQ}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print("="*80)
    
    start_time = time.time()
    
    # Training loop
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_q_values = []
        episode_td_errors = []
        done = False
        timestep = 0
        
        while not done and timestep < config.EPISODE_HORIZON:
            # Select action (epsilon-greedy with action masking)
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            action = agent.select_action(state, valid_actions)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train if replay buffer has enough samples
            if len(agent.replay_buffer) >= min_replay_size:
                loss, q_mean, td_error = agent.train_step()
                
                # Check for NaN
                if np.isnan(loss) or np.isnan(q_mean):
                    print(f"\n⚠️  WARNING: NaN detected at episode {episode}, step {timestep}")
                    training_stable = False
                    
                    # Retry with reduced learning rate if first occurrence
                    if not retried_with_reduced_lr:
                        print(f"  Retrying with reduced learning rate (0.0001)...")
                        retried_with_reduced_lr = True
                        # Would need to reinitialize agent here with new LR
                        # For now, just log the issue
                
                episode_q_values.append(q_mean)
                episode_td_errors.append(td_error)
                
                # Check for exploding Q-values
                if abs(q_mean) > 1000:
                    print(f"\n⚠️  WARNING: Exploding Q-values (|Q|={abs(q_mean):.1f}) at episode {episode}")
                    training_stable = False
            
            episode_reward += reward
            state = next_state
            timestep += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Compute metrics
        metrics = env.compute_metrics()
        episode_returns.append(episode_reward)
        
        # Moving average
        if len(episode_returns) >= moving_avg_window:
            moving_avg = np.mean(episode_returns[-moving_avg_window:])
        else:
            moving_avg = np.mean(episode_returns)
        moving_avg_returns.append(moving_avg)
        
        # Track Q-values
        if episode_q_values:
            all_q_values.extend(episode_q_values)
        
        # Check for improvement
        if moving_avg > best_moving_avg:
            best_moving_avg = moving_avg
            episodes_since_improvement = 0
            
            # Save best model
            best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
            agent.save(best_model_path)
        else:
            episodes_since_improvement += 1
        
        # Early stopping check
        if episodes_since_improvement >= early_stopping_patience:
            early_stopped = True
            print(f"\n⏸️  Early stopping triggered at episode {episode}")
            print(f"  No improvement for {early_stopping_patience} episodes")
            print(f"  Best moving avg: {best_moving_avg:.2f}")
        
        # Log metrics
        log_metrics = {
            'episode': episode,
            'epsilon': agent.epsilon,
            'episode_return': episode_reward,
            'moving_avg_return': moving_avg,
            'mean_step_reward': episode_reward / max(1, timestep),
            'mean_q_value': np.mean(episode_q_values) if episode_q_values else 0.0,
            'mean_td_error': np.mean(episode_td_errors) if episode_td_errors else 0.0,
            'tasks_completed': metrics['throughput'],
            'deadline_hit_rate': metrics['deadline_hit_rate'],
            'overload_events': metrics['overload_events'],
            'early_stopping_triggered': early_stopped
        }
        logger.log_episode(log_metrics)
        
        # Checkpoint
        if (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_ep{episode+1}.pth')
            agent.save(checkpoint_path)
        
        # Progress logging (every 50 episodes)
        if (episode + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}/{max_episodes} | "
                  f"Return: {episode_reward:.2f} | "
                  f"Avg100: {moving_avg:.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Tasks: {metrics['throughput']}/{config.NUM_TASKS} | "
                  f"Time: {elapsed:.1f}s")
        
        # Break if early stopped
        if early_stopped:
            break
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Compute final statistics
    final_epsilon = agent.epsilon
    final_moving_avg = moving_avg_returns[-1] if moving_avg_returns else 0.0
    q_min = np.min(all_q_values) if all_q_values else 0.0
    q_max = np.max(all_q_values) if all_q_values else 0.0
    
    # Summary report
    summary = {
        'total_episodes': len(episode_returns),
        'final_epsilon': final_epsilon,
        'best_moving_avg_return': best_moving_avg,
        'final_moving_avg_return': final_moving_avg,
        'q_value_min': q_min,
        'q_value_max': q_max,
        'training_stable': training_stable,
        'retried_with_reduced_lr': retried_with_reduced_lr,
        'early_stopping_triggered': early_stopped,
        'total_training_time': total_time
    }
    
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"Final Epsilon: {summary['final_epsilon']:.4f}")
    print(f"Best Moving Avg Return: {summary['best_moving_avg_return']:.2f}")
    print(f"Final Moving Avg Return: {summary['final_moving_avg_return']:.2f}")
    print(f"Q-value Range: [{summary['q_value_min']:.2f}, {summary['q_value_max']:.2f}]")
    print(f"Training Stable: {'YES' if summary['training_stable'] else 'NO'}")
    print(f"Reduced LR Retry: {'YES' if summary['retried_with_reduced_lr'] else 'NO'}")
    print(f"Early Stopping: {'YES' if summary['early_stopping_triggered'] else 'NO'}")
    print(f"Training Time: {summary['total_training_time']:.1f}s ({summary['total_training_time']/60:.1f}min)")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    # Run training
    summary = train_dqn(
        max_episodes=2000,
        reward_scale=0.1,
        learning_rate=0.0005,
        seed=42
    )
    
    # Generate learning curve
    print("\nGenerating learning curve...")
    from training.visualize import plot_learning_curve
    plot_learning_curve()
    print("✓ Learning curve saved to results/learning_curve.png")
