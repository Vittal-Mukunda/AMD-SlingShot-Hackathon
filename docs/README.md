# `docs/` — Architecture and Usage Documentation

## Files

| File | Description |
|------|------------|
| `ONLINE_LEARNING.md` | Full architecture reference: two-phase framework, state/action/reward spaces |
| `HOW_TO_RUN.md` | Step-by-step run instructions, CLI arguments, output interpretation |

## Quick Reference

### Run the System

```bash
# Standard full run (recommended)
python continual_scheduler.py

# Smoke test (~60s)
python continual_scheduler.py --smoke-test

# With training debug logs
python continual_scheduler.py --smoke-test --debug-training

# Via run_pipeline.py
python run_pipeline.py --online
```

### Key Configuration Parameters (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATE_DIM` | 96 | State vector dimensionality |
| `ACTION_DIM` | 140 | Action space size |
| `BATCH_SIZE` | 32 | Mini-batch size for gradient updates |
| `MIN_REPLAY_SIZE` | 64 | Buffer size before training starts |
| `EPSILON_DECAY` | 0.999 | Per-decision ε decay (reaches 0.05 in ~3000 steps) |
| `PHASE1_DAYS` | 10 | Baseline observation phase duration |
| `PHASE2_DAYS` | 15 | DQN online learning phase duration |

## Architecture Overview

```
Phase 1 (baselines observe):          Phase 2 (DQN controls):
┌─────────────────────────┐          ┌──────────────────────────────────────┐
│  Greedy / Hybrid / Skill│  →warm→  │  DQN online_step (every decision)    │
│  drive scheduling        │  buffer  │   select → execute → store → train   │
│  DQN passively stores   │          │  ε decays: 0.35 → 0.05 over Phase 2  │
│  transitions             │          │  Target net synced every 100 steps   │
└─────────────────────────┘          └──────────────────────────────────────┘
```
