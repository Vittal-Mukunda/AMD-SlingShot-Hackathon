# Environment Parameter Guide

This README comprehensively documents every configurable parameter in the environment, their effects, recommended ranges, and examples for experimentation.

---

## Table of Contents
1. [Seed Behaviour & Reproducibility](#1-seed-behaviour--reproducibility)
2. [Worker Hidden Parameters](#2-worker-hidden-parameters)
3. [Task Generation Parameters](#3-task-generation-parameters)
4. [Episode & Termination Settings](#4-episode--termination-settings)
5. [Fatigue & Burnout Dynamics](#5-fatigue--burnout-dynamics)
6. [Reward Function Weights](#6-reward-function-weights)
7. [DQN Hyperparameters](#7-dqn-hyperparameters)
8. [Configuration Examples](#8-configuration-examples)

---

## 1. Seed Behaviour & Reproducibility

### How Seeding Works
The environment accepts an optional `seed` parameter that controls **all randomness**:
- Worker hidden skill/fatigue/speed parameters
- Task complexity, priority, and deadline generation
- Dependency graph structure
- Stochastic completion time noise
- Deadline shock events

```python
# In demo_run.py / ProjectEnv usage:
env = ProjectEnv(seed=42)    # Everything is deterministic
env.reset()                  # Re-applies seed → same tasks, same RNG state
```

> **Important:** The seed is stored as `env._env_seed` (private). It is **never** included in the 88-dim state vector or accessible through `get_valid_actions()`. Agents observe only load, fatigue, and availability.

### Custom vs Auto Seed
When you run `python demo_run.py`:
- **Press Enter** → auto-generated seed between 1000–99999 (different every run)
- **Type a number** (e.g. `42`) → fully deterministic, identical runs every time

### seed in config.py
```python
# config.py has no fixed DEMO_SEED — seed is chosen at runtime via demo_run.py
TRAIN_RANDOM_SEEDS = [42, 123, 456, 789, 1011]   # Seeds used during DQN training
```

---

## 2. Worker Hidden Parameters

Each of the 5 workers has **unique hidden parameters** sampled at init from a seeded RNG. The agents can never observe these directly — they only observe `[load_normalized, fatigue_level, availability]`.

### Parameters (per worker)

| Parameter | Where Set | Range | Effect |
|---|---|---|---|
| `true_skill` | `worker.py` | `[SKILL_MIN=0.6, SKILL_MAX=1.4]` | Quality of completed work; higher = better output |
| `fatigue_rate` | `worker.py` | `~N(0.20, 0.05)` clipped `[0.05, 0.5]` | How quickly this worker tires when overloaded |
| `recovery_rate` | `worker.py` | `~N(0.10, 0.03)` clipped `[0.02, 0.25]` | How fast they recover when idle |
| `speed_multiplier` | `worker.py` | `~N(1.0, 0.15)` clipped `[0.6, 1.5]` | Scales task completion time (1.0=avg, 1.5=fast) |
| `burnout_resilience` | `worker.py` | `~N(2.5, 0.2)` clipped `[1.8, 3.0]` | Personal burnout threshold; lower = more fragile |

### Modifying Skill Range
```python
# config.py
SKILL_MIN = 0.6   # Weakest possible worker skill (0.4 for more spread)
SKILL_MAX = 1.4   # Strongest possible worker skill (1.8 for experts)
```
**Effect:** Wider range → more differentiation between workers → more reward for skill matching.

### Changing Fatigue Noise
In `worker.py`, the distribution parameters are hardcoded around config values:
```python
self.fatigue_rate = np.clip(np.random.normal(config.FATIGUE_ACCUMULATION_RATE, 0.05), 0.05, 0.5)
```
- Increase `0.05` std to `0.12` → workers become much more varied in stamina
- Increase clip upper bound to `0.8` → some workers tire extremely fast

### Burnout Behavior
```python
# config.py
FATIGUE_THRESHOLD = 2.5        # Global fallback (individual workers use burnout_resilience)
BURNOUT_RECOVERY_TIME = 5      # Timesteps unavailable after burnout (increase for harsher env)
```

---

## 3. Task Generation Parameters

Tasks are generated fresh on every `env.reset()` using the environment seed.

### Core Task Settings
```python
# config.py
NUM_TASKS = 20                              # Total tasks per episode
TASK_COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]  # Sampled uniformly per task
TASK_PRIORITIES = [0, 1, 2, 3]            # 0=low, 1=medium, 2=high, 3=critical
DEADLINE_MIN = 20                           # Minimum deadline (timesteps)
DEADLINE_MAX = 60                           # Maximum deadline
```

**Effect of increasing `NUM_TASKS`:**
- More tasks → more complex allocation problem → agent needs more training
- Increases state complexity but action space stays at 140 (capped at 20 tasks × 5 workers)

**Effect of tightening deadlines:**
```python
DEADLINE_MIN = 10   # Very tight → increases deadline miss penalties
DEADLINE_MAX = 30   # Reduces slack → RL must be more strategic
```

### Task Dependencies
```python
DEPENDENCY_GRAPH_COMPLEXITY = 3   # Number of dependency chains created
MAX_DEPENDENCY_DEPTH = 3          # Max depth of dependency tree
```
Dependencies prevent tasks from being assigned until prerequisites complete. Increasing depth makes the problem harder (agent must plan ahead).

### Deadline Shocks
```python
DEADLINE_SHOCK_PROB = 0.15       # 15% chance per step of a deadline shock
DEADLINE_SHOCK_AMOUNT = 10       # How many timesteps are removed from a task's deadline
```
Set `DEADLINE_SHOCK_PROB = 0.0` to disable shocks (use `config_overrides={'enable_deadline_shocks': False}`).

---

## 4. Episode & Termination Settings

```python
EPISODE_HORIZON = 100    # Max timesteps before timeout termination
FAILURE_THRESHOLD = 0.5  # 50% of tasks failing ends the episode early
```

**Termination conditions:**
1. **Success:** All `NUM_TASKS` tasks completed
2. **Timeout:** `current_timestep >= EPISODE_HORIZON`
3. **Failure:** `len(failed_tasks) / NUM_TASKS >= FAILURE_THRESHOLD`

**Recommended experimentation:**
- Increase `EPISODE_HORIZON` to 150 for longer episodes (more time to learn deadline juggling)
- Decrease `FAILURE_THRESHOLD` to 0.3 → harder; agent fails out earlier when behind

---

## 5. Fatigue & Burnout Dynamics

```python
OVERLOAD_THRESHOLD = 3          # worker.load > this triggers accelerated fatigue
FATIGUE_ACCUMULATION_RATE = 0.2 # Default rate (workers have individual rates around this)
FATIGUE_RECOVERY_RATE = 0.1     # Default recovery rate
FATIGUE_THRESHOLD = 2.5         # Default burnout threshold (overridden per worker)
BURNOUT_RECOVERY_TIME = 5       # Timesteps worker is unavailable after burnout
```

### Fatigue Formula (per worker, per step)
```
if load > OVERLOAD_THRESHOLD and random() < (0.3 + 0.1 * fatigue):
    fatigue += worker.fatigue_rate   ← per-worker, not global!
elif load == 0:
    fatigue -= worker.recovery_rate  ← per-worker!
if fatigue >= worker.burnout_resilience:
    trigger_burnout()                ← unavailable for BURNOUT_RECOVERY_TIME steps
```

### Effect on Completion Time
```
completion_time = complexity / (true_skill × speed_multiplier) × (1 + 0.5 × fatigue)
```
A fatigued expert worker (fatigue=2.0) is 2× slower than the same worker when fresh.

### Disabling Fatigue (Ablation)
```python
env = ProjectEnv(config_overrides={'enable_fatigue': False})
```

---

## 6. Reward Function Weights

All reward components are defined in `config.py`. Final reward is scaled by `reward_scale` (default 0.1).

| Component | Config Key | Default | Effect |
|---|---|---|---|
| Task completion | `REWARD_COMPLETION_BASE` | 15.0 | Multiplied by `(priority+1) × quality` |
| Queue delay penalty | `REWARD_DELAY_WEIGHT` | -0.3 | Per-step penalty for tasks waiting |
| Worker overload penalty | `REWARD_OVERLOAD_WEIGHT` | -2.0 | Quadratic: `(load - threshold)²` |
| Deadline miss | `REWARD_DEADLINE_MISS_PENALTY` | -15.0 | One-time penalty per failed task |
| Strategic defer bonus | `REWARD_STRATEGIC_DEFER` | 1.0 | Bonus for deferring when no worker is skilled enough |
| Skill exploration bonus | `REWARD_EXPLORATION_BONUS` | 0.5 | Bonus for assigning above-skill tasks |

```python
# Reward scaling (applied to all components)
reward = reward_unscaled * self.reward_scale   # reward_scale default = 0.1
```

**Making completion more important:**
```python
REWARD_COMPLETION_BASE = 25.0
REWARD_DEADLINE_MISS_PENALTY = -10.0   # Reduce catastrophic penalty
```

**Harsher load balancing:**
```python
REWARD_OVERLOAD_WEIGHT = -5.0   # Was -2.0; severely penalizes hoarding tasks
```

---

## 7. DQN Hyperparameters

```python
STATE_DIM = 88        # State vector dimension (fixed — do not change)
ACTION_DIM = 140      # Action space: 20×5 assign + 20 defer + 20 escalate
HIDDEN_LAYERS = [128, 128]   # Q-network architecture
LEARNING_RATE = 0.0005
GAMMA = 0.95          # Discount factor (0.99 for longer-horizon reasoning)
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9994      # Per-episode; reaches 0.05 around episode 3500
MAX_EPISODES = 5000
EARLY_STOPPING_PATIENCE = 1000
TARGET_UPDATE_FREQ = 100
```

### Training the Agent
```bash
python run_pipeline.py --train --episodes 5000 --seed 42
```

---

## 8. Configuration Examples

### Easy Environment (for quick testing)
```python
NUM_TASKS = 10
EPISODE_HORIZON = 50
DEADLINE_MIN = 30
DEADLINE_MAX = 80
DEADLINE_SHOCK_PROB = 0.0
FAILURE_THRESHOLD = 0.7
```

### Hard Environment (stress test)
```python
NUM_TASKS = 20
EPISODE_HORIZON = 80
DEADLINE_MIN = 15
DEADLINE_MAX = 40
DEADLINE_SHOCK_PROB = 0.30
OVERLOAD_THRESHOLD = 2
FAILURE_THRESHOLD = 0.3
```

### Research Ablation: Fully Observable Skills
```python
env = ProjectEnv(config_overrides={'fully_observable': True})
# Workers' true_skill is included directly in state (belief uncertainty = 0)
# Use to measure how much hidden info hurts performance
```

### Reproduce a Specific Run
```python
env = ProjectEnv(seed=42, reward_scale=0.1)
# or via demo_run.py: type '42' at the seed prompt
```
