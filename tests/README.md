# `tests/` — Test Suite

Unit and integration tests for the v4 continual online learning scheduler.

## Files

| File | Description |
|------|------------|
| `test_stability.py` | DQN training stability tests (convergence, loss, reward consistency) |

## Quick Smoke Test

The fastest way to verify all v4 components work:

```bash
python smoke_test.py
```

Expected output:
```
[1] Worker v4:       ✓ state_dim=5
[2] Task v4:         ✓ 30 tasks, zero-lookahead works
[3] ProjectEnv v4:   ✓ state_dim=96
[4] DQN Agent v4:    ✓ train_step, epsilon decay
[5] SkillBaseline:   ✓ Welford estimates
[6] GreedyBaseline:  ✓ v4 API
ALL SMOKE TESTS PASSED
```

## Full Training Verification

```bash
# Quick 2+3 day run to confirm training triggers
python continual_scheduler.py --smoke-test --debug-training

# Confirm in output:
# [DQN-TRAIN] step=65, buf=65, loss=0.3412, Q=-0.821, ε=0.3490, train_steps=1
# train_steps=XX (should be > 0 in Phase 2)
```

## Running Full Tests

```bash
python -m pytest tests/ -v
```
