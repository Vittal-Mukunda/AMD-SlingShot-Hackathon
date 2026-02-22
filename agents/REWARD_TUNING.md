# DQN Reward Function Tuning Guide

This guide explains every configurable parameter in the reward function, its real-world effect on agent behaviour, and how to adjust it when fine-tuning.

---

## Reward Architecture Overview

The total reward at each step is:

```
reward = reward_scale × (
    action_reward         ← immediate action validity signal
  + completion_reward     ← PRIMARY: task finished successfully
  + deadline_penalty      ← GATING:  catastrophic hit per dropped task
  + overload_penalty      ← SECONDARY: mild quadratic cost for overloaded workers
  + delay_penalty         ← TERTIARY: tiny per-step queue tick (almost noise)
)
+ terminal_success_bonus  ← flat +5 (unscaled) only when ALL tasks complete
```

The hierarchy is intentional. If the agent can earn more by letting tasks expire than by completing them, it will. The weights enforce that **completing tasks always dominates**.

---

## Parameters in `config.py`

### `REWARD_COMPLETION_BASE = 18.0`

The base multiplier for task completion.

**Formula:**
```
completion_reward += REWARD_COMPLETION_BASE × (task.priority + 1) × quality_score
```

| Priority | Max quality (1.0) | Min quality (0.3) |
|---|---|---|
| 0 (Low) | +18 | +5.4 |
| 1 (Medium) | +36 | +10.8 |
| 2 (High) | +54 | +16.2 |
| 3 (Critical) | +72 | +21.6 |

**Tuning:**
- Increase (e.g. `25.0`) → agent becomes more aggressive about completing tasks, accepts higher fatigue/overload risk
- Decrease (e.g. `10.0`) → agent becomes more conservative; risk of reward hacking returns
- **Do not go below `12.0`** or deadline penalties start dominating again

---

### `REWARD_DEADLINE_MISS_PENALTY = -20.0`

One-shot penalty applied exactly once when a task's deadline expires without completion.

**Anti-hacking math:** The agent might consider intentionally ignoring a task (letting it expire) to save the ~6-10 delay ticks it would cost in `delay_penalty`. With `delay_penalty = -0.005` per task per step and `deadline_miss = -20.0`:
```
Cost of dropping one task: -20.0 × 0.1 × reward_scale = -2.0 (scaled)
Benefit saved (10 steps of delay): 10 × 0.005 × 0.1 = 0.005 (scaled)
```
Dropping a task is **400× more costly** than the delay saved. Reward hacking is not profitable.

**Tuning:**
- Increase (e.g. `-25.0`) → even harsher punishment for drops; agent may become overly conservative and defer risky tasks too much
- Decrease (e.g. `-10.0`) → risk of reward hacking resumes; not recommended below `-15.0`
- **Keep the ratio `|deadline_miss / (delay_weight × horizon)|` above 10×**

---

### `REWARD_DELAY_WEIGHT = -0.005`

Per-step, per-pending-task penalty. Applied to all tasks not yet completed or failed.

**Formula:**
```
delay_penalty = REWARD_DELAY_WEIGHT × Σ (time_in_queue / task.deadline)
```

With 20 tasks and horizon=100: total delay burden ≈ `-0.005 × 20 × 100 × 0.5 ≈ -5.0` unscaled.
One high-priority task completion = `+72`. The ratio is 14:1 in favour of completion.

**Tuning:**
- Increase magnitude (e.g. `-0.05`) → agent becomes more urgent about clearing queue; can lead to quality degradation (assigning skilled work to wrong workers to save ticks)
- Decrease magnitude (e.g. `-0.001`) → queue management pressure disappears; agent ignores urgency
- **Recommended range: `-0.002` to `-0.02`**

---

### `REWARD_OVERLOAD_WEIGHT = -0.5`

Quadratic penalty for workers loaded above `OVERLOAD_THRESHOLD` (default: 3 tasks).

**Formula:**
```
overload_penalty = REWARD_OVERLOAD_WEIGHT × Σ max(0, worker.load - OVERLOAD_THRESHOLD)²
```

With one worker at load=5 (2 above threshold): `-0.5 × 4 = -2.0` unscaled per step.

**Tuning:**
- Increase magnitude (e.g. `-2.0`) → agent learns to distribute load earlier; may refuse to overload even when one expert worker would be ideal
- Decrease magnitude (e.g. `-0.1`) → agent ignores load balance; burnout events increase
- **Recommended range: `-0.3` to `-1.5`**

---

### `reward_scale = 0.1`  *(passed as constructor arg)*

Global multiplier applied to the full unscaled reward before returning to the DQN.

**Purpose:** Keeps Q-values in a numerically stable range for the neural network.
- Value of `0.1` means Q-values converge around `[-5, +10]` — well within float32 precision.

**Tuning:**
- Increase (e.g. `0.5`) → larger gradient signals; can cause Q-value divergence if learning rate is not reduced proportionally
- Decrease (e.g. `0.01`) → very small gradients; learning slows down significantly
- **If you change reward weights significantly, adjust `reward_scale` so unscaled reward stays in `[-100, +400]` range**

---

### `REWARD_STRATEGIC_DEFER = 1.0`

Small bonus when the agent defers a task because no skilled worker is available.

**Condition:** `max(available_skills) < task.complexity × 0.5`

**Tuning:**
- Keep this small (≤ 2.0) — higher values teach the agent to strategically defer valid tasks to collect bonuses
- Set to `0.0` to disable entirely if the agent over-defers

---

## Terminal Success Bonus *(hardcoded in `project_env.py` step())*

```python
if len(self.completed_tasks) == self.num_tasks:
    reward_unscaled += 5.0   # Flat bonus for 100% completion
```

This is applied **before** `reward_scale`, so it contributes `+0.5` scaled. It fires exactly once and cannot be exploited by dropping tasks (you need all 20 complete to trigger it).

**Tuning:** Increase to `20.0` if you want to strongly incentivise 100% completion in shorter episodes. Keep scaled contribution below the completion reward for a single critical task.

---

## Quick Reference: Recommended Profiles

### Profile 1 — Default (anti-reward-hacking, balanced)
```python
REWARD_COMPLETION_BASE   = 18.0
REWARD_DEADLINE_MISS_PENALTY = -20.0
REWARD_DELAY_WEIGHT      = -0.005
REWARD_OVERLOAD_WEIGHT   = -0.5
# reward_scale = 0.1
```

### Profile 2 — Deadline-Critical (tight deadlines, harsh shocks)
```python
REWARD_COMPLETION_BASE   = 20.0
REWARD_DEADLINE_MISS_PENALTY = -25.0
REWARD_DELAY_WEIGHT      = -0.01
REWARD_OVERLOAD_WEIGHT   = -0.5
```
Use when `DEADLINE_SHOCK_PROB > 0.20` or `DEADLINE_MAX < 35`.

### Profile 3 — Load-Balanced Research (study skill allocation)
```python
REWARD_COMPLETION_BASE   = 15.0
REWARD_DEADLINE_MISS_PENALTY = -15.0
REWARD_DELAY_WEIGHT      = -0.005
REWARD_OVERLOAD_WEIGHT   = -2.0   # Stronger load balance pressure
```

### Profile 4 — Quality-Focused
```python
REWARD_COMPLETION_BASE   = 25.0   # Higher: quality×skill matters more
REWARD_DEADLINE_MISS_PENALTY = -20.0
REWARD_DELAY_WEIGHT      = -0.002
REWARD_OVERLOAD_WEIGHT   = -0.3
```
Favours matching high-skill workers to complex tasks.

---

## Diagnosing Reward Hacking

If the DQN achieves very low `avg_delay` and high `load_balance` scores but low `throughput` and `deadline_hit_rate`, it is reward hacking. Check the episode reward breakdown:

```python
# In demo_run.py or training loop:
breakdown = env.get_episode_reward_breakdown()
print(breakdown)
# Expected healthy episode:
#   completion_reward  : +800 to +1200   (dominant)
#   delay_penalty      : -5 to -15       (tiny)
#   overload_penalty   : -2 to -8        (secondary)
#   deadline_penalty   : 0 to -40        (should be small)
```

If `deadline_penalty > completion_reward` in absolute magnitude, the agent has dropped tasks intentionally — increase `REWARD_COMPLETION_BASE` or decrease `|REWARD_DEADLINE_MISS_PENALTY|`.

---

## Files to Edit

| What to change | File |
|---|---|
| Reward weights (completion, delay, overload, deadline) | [`config.py`](file:///c:/Users/vitta/OneDrive/Desktop/Python%203.10/AMD_Repo/AMD-SlingShot-Hackathon/config.py) lines 84–95 |
| Reward computation logic / terminal bonus | [`environment/project_env.py`](file:///c:/Users/vitta/OneDrive/Desktop/Python%203.10/AMD_Repo/AMD-SlingShot-Hackathon/environment/project_env.py) lines 213–234 |
| Skill-match action bonus (`_execute_action`) | [`environment/project_env.py`](file:///c:/Users/vitta/OneDrive/Desktop/Python%203.10/AMD_Repo/AMD-SlingShot-Hackathon/environment/project_env.py) lines 315–358 |
| `reward_scale` constructor argument | Passed when calling `ProjectEnv(reward_scale=0.1)` |
