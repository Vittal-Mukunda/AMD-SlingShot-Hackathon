# Mathematical Modeling

The simulation incorporates several mathematical models to realistically dictate a worker's performance degradation and reward balancing over an ongoing shift.

## Overload Degradation Metric
By default, the worker's efficiency acts linearly using their `true_skill` vs the `complexity` of a task. However, when overloaded (concurrent tasks cross `MAX_WORKER_LOAD // 2`), a dynamic efficiency penalty triggers. 

**Core Penalty Function**
```math
base_prob = 0.25 + 0.08 * current_fatigue
fragility_factor = fatigue_rate / max(true_skill, 0.1)
probability = base_prob * (1.0 + fragility_factor)
```
Under this model, lower-skill workers or those with a naturally high fatigue rate will see an exponential drop-off relative to highly skilled workers during periods of sustained overload.

## Quality of Work
The quality is determined by combining the true skill against task complexity, while degrading over time:

```math
base_quality = min(1.0, true_skill / max(complexity, 0.1))
overload_penalty = (current_load / MAX_WORKER_LOAD) * 0.2
fatigue_penalty = (fatigue_sensitivity * current_fatigue) + overload_penalty
intraday_penalty = INTRADAY_DECAY_RATE * min(hours_worked_today, 8.0)

quality = base_quality * (1.0 - fatigue_penalty) * (1.0 - intraday_penalty)
```
This forces the DQN agent to proactively monitor worker load. If a worker is overloaded, their resulting task completion generates a substantially worse positive completion reward.
