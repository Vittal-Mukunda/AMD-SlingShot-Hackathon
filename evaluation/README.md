# `evaluation/` — Results & Comparison

Scripts for evaluating and comparing Phase 1 (baselines) vs Phase 2 (DQN) performance.

## Files

| File | Description |
|------|------------|
| `compare_phases.py` | Load CSV results and print a comparison table |

## Output Files (auto-generated into `results/`)

| File | Description |
|------|------------|
| `results/phase1_metrics.csv` | Per-day metrics from Phase 1 (baselines) |
| `results/phase2_metrics.csv` | Per-day metrics from Phase 2 (DQN online) |
| `results/phase_comparison.csv` | Aggregated Phase 1 vs Phase 2 delta table |

## Usage

```bash
# After running continual_scheduler.py, compare results:
python evaluation/compare_phases.py

# Output example:
# Metric             Phase1    Phase2         Δ
# Throughput/day      2.25      8.34    + 6.09
# Completion rate     0.10      0.80    + 0.70
# Lateness rate       0.00      0.02    + 0.02
# Avg quality         0.30      0.42    + 0.12
# Overload events     0.00      1.50    + 1.50
```

## Metrics Glossary

| Metric | Description |
|--------|------------|
| `throughput_per_day` | Completed tasks / working days elapsed |
| `completion_rate` | Fraction of total tasks completed |
| `lateness_rate` | Fraction of completed tasks that missed deadline |
| `quality_score` | Mean quality across completed tasks (0–1 scale) |
| `load_balance` | Std-dev of worker loads (lower = better) |
| `overload_events` | Times any worker reached burnout-level fatigue |
| `makespan_hours` | Total wall-clock hours to complete all tasks |

## Interpreting Results

- **Throughput/day ↑**: DQN learning to assign tasks more efficiently
- **Lateness rate ↑**: Expected early on as DQN explores; should decrease as ε decays
- **Loss decreasing**: Confirms gradient updates are converging
- **Q-value increasing**: Policy estimating higher expected returns (good sign)

> [!TIP]
> Monitor `train_steps` in Phase 2 logs. A value of `0` means training never started — check `MIN_REPLAY_SIZE` and buffer fill rate.
