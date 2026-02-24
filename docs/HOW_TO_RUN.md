# How to Run — Continual Online Learning Scheduler

## Prerequisites

```bash
pip install torch numpy
```

---

## Quick Start (Recommended)

```bash
# Run the full two-phase online scheduler (10 + 15 working days)
python run_pipeline.py --online

# Or run the scheduler directly with custom settings
python continual_scheduler.py --days-p1 10 --days-p2 15 --seed 42
```

---

## All Run Modes

### Online Learning Mode (v4 — Recommended)

```bash
# Standard run (2 weeks baseline obs + 3 weeks DQN control)
python continual_scheduler.py

# Quick smoke test (2+3 days, 40 tasks — finishes in <1 minute)
python continual_scheduler.py --smoke-test

# With debug output for the skill baseline
python continual_scheduler.py --debug-skill

# Resume from a checkpoint
python continual_scheduler.py --load-checkpoint checkpoints/dqn_phase2_latest.pth

# Phase 1 only (build replay buffer)
python continual_scheduler.py --phase1-only

# Phase 2 only (requires existing checkpoint with warm buffer)
python continual_scheduler.py --phase2-only

# Via run_pipeline.py dispatch
python run_pipeline.py --online --smoke-test
python run_pipeline.py --online --days-p1 5 --days-p2 10 --debug-skill
```

### Legacy Pipeline (train-then-run)

```bash
python run_pipeline.py --full           # Full train + evaluate
python run_pipeline.py --train          # Train only
python run_pipeline.py --baselines      # Baselines only
python run_pipeline.py --evaluate       # Evaluate only
python run_pipeline.py --plots          # Generate plots
```

---

## View Results After Running

```bash
# Print comparison table of Phase 1 vs Phase 2
python evaluation/compare_phases.py
```

Results are saved to:
- `results/phase1_metrics.csv`
- `results/phase2_metrics.csv`
- `results/phase_comparison.csv`
- `checkpoints/dqn_online_final.pth`
- `checkpoints/dqn_phase2_latest.pth` (best checkpoint by throughput/day)

---

## Key Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHASE1_DAYS` | 10 | Working days for baseline observation phase |
| `PHASE2_DAYS` | 15 | Working days for DQN online learning phase |
| `TOTAL_TASKS` | 200 | Total tasks to schedule across simulation |
| `TASK_ARRIVAL_RATE` | 3.5 | Mean tasks arriving per working day (Poisson) |
| `SLOT_HOURS` | 0.5 | Each time slot = 30 minutes |
| `SLOTS_PER_DAY` | 16 | 8-hour workday |
| `EPSILON_PHASE2_START` | 0.40 | DQN exploration rate at start of Phase 2 |
| `BASELINE_DEBUG_SKILL` | False | Enable skill baseline per-assignment logging |

---

## Output Format

During Phase 1, you'll see:
```
  Phase 1 | Working Day   0 | Tick    16
    throughput/day=3.50, completion=17.50%, lateness=5.00%, overload=2
```

During Phase 2, you'll see every 100 decisions:
```
  ⟳ [P2] tick=  432, dec=  200, a= 12, r=+12.34, ε=0.385, loss=0.0043, Q=8.241, policy=DQN
```

Final comparison table:
```
═══════════════════════════════════════════════════════════════════════
  PHASE 1 vs PHASE 2 — Performance Comparison
  Metric                    Phase 1 (Baseline)  Phase 2 (DQN)        Δ
  Throughput / day                       3.20           4.10     +0.90
  Completion rate                      64.00%         82.00%   +18.00%
```

---

## Architecture Diagram

```
Tasks (Poisson arrivals)
      ↓
SimClock (slot=30min, 8h/day, 5d/week)
      ↓
ProjectEnv (96-dim state, 140 actions)
      ↓
  ┌────────────────────────────────────┐
  │  Phase 1: Baselines rotate         │
  │  Greedy → Hybrid → Skill           │
  │  DQN passively stores transitions  │
  └────────────────────────────────────┘
      ↓  (replay buffer warmed up)
  ┌────────────────────────────────────┐
  │  Phase 2: DQN online control       │
  │  online_step() every decision:     │
  │    select → execute → store        │
  │    → train_step() → decay ε        │
  └────────────────────────────────────┘
      ↓
  results/phase{1,2}_metrics.csv  →  compare_phases.py
```
