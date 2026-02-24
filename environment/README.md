# `environment/` — Scheduling Environment

The core OpenAI Gym-style environment for the v4 continual online scheduling framework.

## Files

| File | Description |
|------|------------|
| `project_env.py` | Main `ProjectEnv` class implementing the scheduling MDP + `SimClock` |
| `worker.py` | `Worker` class with heterogeneous hidden traits and intra-day fatigue |
| `task.py` | `Task` class + `generate_poisson_arrivals()` for dynamic task creation |
| `belief_state.py` | Bayesian Beta-distribution belief tracker over worker skills |
| `diagnostics.py` | Diagnostic logging utilities |
| `__init__.py` | Package init |

## Key Classes

### `ProjectEnv` (project_env.py)
Main environment. Implements `reset()`, `step(action)`, `get_valid_actions()`, `compute_metrics()`.

**State space (96-dim):**
```
[worker_features (5×5=25)] + [task_features (10×5=50)] + [belief (10)] + [global (6)] + [pad (5)]
```

**Action space (140 actions):**
- 0–99: assign task_slot × worker_id (20 tasks × 5 workers)
- 100–119: defer task at slot 0–19
- 120–139: escalate task at slot 0–19

**Zero-lookahead guarantee:** Only tasks with `arrival_tick ≤ clock.tick` appear in `get_valid_actions()`.

### `SimClock` (project_env.py)
Tracks simulation time in 30-minute slots (SLOTS_PER_DAY=16 → 8h workday).
- `advance()` — move forward one slot
- `is_start_of_day()` — triggers `worker.daily_reset()`

### `Worker` (worker.py)
5-dim observable state: `[load_norm, fatigue_norm, availability, hours_norm, productivity]`

Hidden traits (never in state vector, sampled once at init):
- `true_skill`, `speed_multiplier`, `fatigue_rate`, `recovery_rate`, `fatigue_sensitivity`, `burnout_resilience`

### `Task` (task.py)
5-dim state: `[priority_norm, complexity_norm, deadline_urgency, deps_met, arrival_elapsed]`
- `arrival_tick` — task not visible before this tick
- `is_available(tick)` — returns `arrival_tick <= tick and is_unassigned()`

## Reward Function (Makespan-centric)

```
R = +COMPLETION_BASE × (priority+1) × quality   (at task completion)
  − IDLE_PENALTY × n_idle_workers               (per slot)
  − LATENESS_PENALTY × slots_late               (at completion, if late)
  − OVERLOAD_WEIGHT × std(loads)               (per slot)
  − URGENCY_PENALTY × n_urgent_unstarted        (per slot)
  − DEADLINE_MISS_PENALTY                       (one-shot per failure)
  − DELAY_WEIGHT                                (per step)
  + MAKESPAN_BONUS                              (terminal: all tasks done)
```

## Quick Start

```python
from environment.project_env import ProjectEnv
env = ProjectEnv(num_workers=5, total_tasks=200, seed=42)
state = env.reset()
valid = env.get_valid_actions()
next_state, reward, done, info = env.step(valid[0])
metrics = env.compute_metrics()
```
