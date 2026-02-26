# Tweaking Parameters

The core settings for the simulation are unified within `config.py`. Modifying this file directly adjusts the hyperparameter behavior for both the environment and the DQN agent.

## Simulation Framework
- `PHASE1_DAYS`: The initial observation period (currently set to 20 days / 1 month). The DQN learns passively here alongside baseline models.
- `PHASE2_DAYS`: Evaluation operational phase (currently set to 5 days / 1 week). This is the only phase where visualization plots are rendered to evaluate the active DQN control and directly compare against baselines.

## Reward Tuning
The reward function is configured to heavily penalize inefficient makespan operations. Crucial weights:
- `REWARD_OVERLOAD_WEIGHT`: An explicit penalty against keeping workers overloaded compared to their peers.
- `REWARD_IDLE_PENALTY`: A negative push to keep available workers busy.
- `REWARD_LATENESS_PENALTY`: Severe drop-offs for failing to meet generated project deadlines.

## DQN Settings
The reinforcement learning agent parameters such as `LEARNING_RATE` (0.0003), `GAMMA` (0.97), and `BATCH_SIZE` (32) are optimized for an continuous online setting, allowing rapid adjustments without needing massive replay buffers initially.

## Running the Simulation
When utilizing `demo_run.py`, you will be met with an interactive configuration prompt before the simulation actively begins. This prompt will permit you to override `num_tasks`, `random_seed`, and worker profiles dynamically without modifying the actual python constants in `config.py`.
