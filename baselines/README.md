# `baselines/` — Scheduling Baseline Policies

All baselines implement the same `BasePolicy` interface and use **zero-lookahead** (only see tasks that have already arrived).

## Files

| File | Baseline | Strategy |
|------|----------|----------|
| `greedy_baseline.py` | **Greedy** | Highest-priority arrived task → least-loaded worker |
| `hybrid_baseline.py` | **Hybrid** | Skill + load + fatigue additive score (hardest baseline) |
| `skill_baseline.py` | **Skill** | Bayesian skill estimation (Welford online mean) → best-skill worker |
| `stf_baseline.py` | **STF** | Shortest-Task-First → least-loaded worker |
| `random_baseline.py` | **Random** | Uniform random from valid actions (sanity check) |
| `base_policy.py` | `BasePolicy` | Abstract base class |

## Interface

All baselines implement:
```python
def select_action(self, state) -> int:
    ...  # returns action index in [0, 139]

def encode_action(self, task_id, worker_id, action_type='assign') -> int:
    ...  # v4 slot-based encoding (zero-lookahead aware)
```

## Phase 1 Rotation

In `continual_scheduler.py` Phase 1, baselines rotate daily:
```
Day 0: Greedy
Day 1: Hybrid
Day 2: Skill
Day 3: (repeat...)
```

## Skill Baseline Details

The `SkillBaseline` maintains Bayesian Beta posteriors per worker:
- **Prior**: Beta(2, 2) → mean = 0.5
- **Update**: `observe_episode(env)` called at each day boundary
- **Welford mean**: prevents list-accumulation bias in skill estimates
- **Selection**: deterministic max-skill worker with assertion guard

Enable debug logs:
```python
config.BASELINE_DEBUG_SKILL = True
# or via CLI: python continual_scheduler.py --debug-skill
```

## Adding a New Baseline

1. Create `baselines/my_baseline.py` extending `BasePolicy`:
```python
from baselines.base_policy import BasePolicy
class MyBaseline(BasePolicy):
    def __init__(self, env): super().__init__(env); self.name = "MyBaseline"
    def select_action(self, state): ...
```
2. Add to `_build_baselines()` pool in `continual_scheduler.py`.
