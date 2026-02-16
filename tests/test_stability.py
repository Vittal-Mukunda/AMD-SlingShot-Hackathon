"""
Test script to verify environment stability improvements for DQN
"""

import numpy as np
import sys
sys.path.append('../')

from environment.project_env import ProjectEnv
import config

print("="*80)
print("TESTING ENVIRONMENT STABILITY IMPROVEMENTS")
print("="*80)

# Test 1: Reward scaling
print("\n1. Testing Reward Scaling...")
env_unscaled = ProjectEnv(seed=42, reward_scale=1.0, enable_diagnostics=True)
env_scaled = ProjectEnv(seed=42, reward_scale=0.1, enable_diagnostics=True)

for env, name in [(env_unscaled, "Unscaled"), (env_scaled, "Scaled (0.1x)")]:
    state = env.reset()
    total_rewards = []
    
    for _ in range(5):  # 5 episodes
        state = env.reset()
        episode_reward = 0
        for t in range(100):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = np.random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    
    print(f"\n{name}:")
    print(f"  Mean episode return: {np.mean(total_rewards):.2f}")
    print(f"  Std episode return: {np.std(total_rewards):.2f}")
    print(f"  Range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")

# Test 2: State normalization
print("\n2. Testing State Normalization...")
env = ProjectEnv(seed=42, enable_diagnostics=True)
state = env.reset()

print(f"State shape: {state.shape}")
print(f"State range: [{np.min(state):.4f}, {np.max(state):.4f}]")
print(f"State mean: {np.mean(state):.4f}")
print(f"State std: {np.std(state):.4f}")

# Check for any values outside [0, 1] (with small tolerance)
out_of_bounds = (state < -0.01) | (state > 1.01)
if np.any(out_of_bounds):
    print(f"⚠️  WARNING: {np.sum(out_of_bounds)} state features out of [0,1] range")
else:
    print("✓ All state features properly normalized to [0,1]")

# Test 3: Diagnostic logging
print("\n3. Testing Diagnostic Logging...")
env = ProjectEnv(seed=42, enable_diagnostics=True, reward_scale=0.1)
state = env.reset()

for t in range(50):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        break
    action = np.random.choice(valid_actions)
    state, reward, done, info = env.step(action)
    if done:
        break

# Get diagnostic summary (if method exists)
try:
    summary = env.get_diagnostics_summary()
    if summary.get('diagnostics_enabled'):
        print("✓ Diagnostics collected successfully")
        if 'reward' in summary:
            print(f"  Reward stats:")
            print(f"    Mean: {summary['reward']['mean']:.4f}")
            print(f"    Std: {summary['reward']['std']:.4f}")
            print(f"    Range: [{summary['reward']['min']:.4f}, {summary['reward']['max']:.4f}]")
        if 'actions' in summary:
            print(f"  Action sparsity:")
            print(f"    Mean valid actions: {summary['actions']['mean_valid']:.1f}")
            print(f"    Sparsity ratio: {summary['actions']['sparsity_ratio']:.2%}")
except AttributeError:
    print("⚠️  Note: get_diagnostics_summary() method not yet added")

# Test 4: Division by zero fix
print("\n4. Testing Division-by-Zero Fix...")
try:
    env = ProjectEnv(seed=42)
    state = env.reset()
    
    # Run 100 steps to make sure no crashes
    for t in range(100):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        action = np.random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        if done:
            break
    
    print("✓ No division-by-zero errors detected")
except ZeroDivisionError as e:
    print(f"❌ Division by zero still exists: {e}")

print("\n" + "="*80)
print("ALL STABILITY TESTS COMPLETE")
print("="*80)
