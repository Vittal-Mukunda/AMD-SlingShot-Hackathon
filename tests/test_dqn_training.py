"""
Quick test script to verify DQN training setup before full run
Tests 10 episodes to ensure everything works
"""

import sys
sys.path.append('.')

from training.train_dqn import train_dqn
import config

print("Running quick DQN training test (10 episodes)...")
print("="*80)

summary = train_dqn(
    max_episodes=10,
    min_replay_size=100,  # Reduced for quick test
    checkpoint_freq=5,
    early_stopping_patience=1000,  # Disabled for test
    reward_scale=0.1,
    learning_rate=0.0005,
    seed=42
)

print("\n" + "="*80)
print("QUICK TEST SUMMARY")
print("="*80)
for key, value in summary.items():
    print(f"{key}: {value}")
print("="*80)

print("\n✓ DQN training infrastructure verified!")
print("Ready for full 2000-episode training run")
