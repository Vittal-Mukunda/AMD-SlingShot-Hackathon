# `agents/` — DQN Agent

Contains the core reinforcement learning agent used in Phase 2 scheduling.

## Files

| File | Description |
|------|------------|
| `dqn_agent.py` | Full DQN agent: Dueling DQN + PER + Double DQN + Cosine LR |
| `__init__.py` | Package init |

## Architecture

```
Input (96-dim state)
    ↓
Shared Backbone: Linear(256) → LayerNorm → ReLU → Linear(256) → LayerNorm → ReLU
    ├── Value Stream:      Linear(128) → ReLU → Linear(1)
    └── Advantage Stream:  Linear(128) → ReLU → Linear(140)
    ↓
Q(s,a) = V(s) + A(s,a) − mean(A)   [Dueling combination]
```

## Key Methods

| Method | Description |
|--------|-------------|
| `select_action(state, valid_actions)` | ε-greedy with action masking |
| `store_transition(s, a, r, s', done)` | Push to PER buffer |
| `online_step(state, valid, env)` | Full one-step: select → execute → store → train → decay ε |
| `train_step()` | One mini-batch gradient update (Double DQN + PER IS-weights + Huber loss) |
| `update_epsilon()` | Per-decision exponential decay |
| `set_epsilon(eps)` | Manually set epsilon (called at Phase 2 start) |
| `save(path)` / `load(path)` | Checkpoint management |

## Training Parameters

```python
BATCH_SIZE         = 32       # Small batch for fast online updates
MIN_REPLAY_SIZE    = 64       # Must equal BATCH_SIZE to start training immediately
LEARNING_RATE      = 0.0003
GAMMA              = 0.97
TARGET_UPDATE_FREQ = 100      # Sync target net every 100 gradient steps
EPSILON_DECAY      = 0.999    # Per-decision; reaches 0.05 in ~3000 decisions
```

## Debug

```python
agent.debug_training = True   # Enable verbose per-step training logs
# Shows: [DQN-TRAIN] step=N, buf=N, loss=X, Q=Y, ε=Z, train_steps=N
# Shows: [DQN-SKIP] buf=N < min=64 (need N more transitions)
# Shows: [DQN-TARGET] Synced target net at train_step=N
```

## Quick Start

```bash
python agents/dqn_agent.py   # Built-in smoke test
```
