# Worker Profiles

Each `Worker` object initialized in the simulation has unique traits defining their behavior within the `project_env`. You cannot observe these traits during active schedule allocation—they are hidden factors driving their efficiency dynamically.

## Predefined Traits
- `true_skill`: Represents the raw expertise of the worker (0.5 to 1.5). Directly influences generated task quality.
- `speed_multiplier`: Alters task completion speed (0.6 to 1.5).
- `fatigue_rate`: The speed at which fatigue increases (0.05 to 0.5) when assigned new jobs.
- `burnout_resilience`: Threshold indicating how much fatigue limits productivity.

## Integrating Parameters
To integrate new bounds on how these initial traits fall, see the `Worker` class initializations inside `environment/worker.py`. 
You can provide manual parameter overrides dynamically via `interactive_config.py` upon simulation boot. If `manual_workers` is engaged, it loops through each worker and constructs their traits.
