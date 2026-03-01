# Repository Cleanup Log

**Date:** 2026-02-28  
**Performed by:** Antigravity (automated audit)

## Deleted Files & Rationale

| File / Directory | Type | Reason |
|---|---|---|
| `app/mcp/` (entire dir) | Directory | MCP server scaffold — never integrated with RL engine, all stubs |
| `app/tests/` (entire dir) | Directory | Test stubs importing deleted modules — would fail on import |
| `app/rl/policy.py` | Empty stub | No implementation, not imported anywhere |
| `app/rl/trainer.py` | Empty stub | No implementation, not imported anywhere |
| `app/db/session.py` | Empty stub | SQLAlchemy session never used; simulation uses in-memory state |
| `app/db/crud.py` | Empty stub | CRUD layer never implemented or imported |
| `app/utils/helpers.py` | Empty stub | No implementation, not imported anywhere |
| `app/utils/validators.py` | Empty stub | No implementation, not imported anywhere |
| `app/utils/metrics.py` | Empty stub | Duplicates `utils/metrics.py` at root; never imported from here |
| `app/api/routes/agents.py` | Empty stub | Agent routes never implemented |
| `app/core/constants.py` | Empty stub | No constants defined, not imported anywhere |
| `scripts/seed_data.py` | Empty stub | Database seeding for removed DB layer |
| `scripts/simulate.py` | Empty stub | Superseded by `continual_scheduler.py` |
| `smoke_output.txt` | Stale artifact | Output from a previous manual test run, not needed in repo |

## Retained Files (Core Engine)

The following are **kept** as they form the functional RL engine:

- `continual_scheduler.py` — Two-phase DQN scheduler orchestrator
- `config.py` — All hyperparameters and simulation config
- `run_pipeline.py` — CLI entry point
- `environment/project_env.py` — RL environment (ProjectEnv)
- `agents/dqn_agent.py` — DQN agent with Dueling + PER + Double DQN
- `baselines/greedy_baseline.py`, `skill_baseline.py`, `hybrid_baseline.py`, `random_baseline.py`
- `visualization/plot_metrics.py`, `task_grid_viz.py`
- `results/`, `checkpoints/`, `logs/` — Output directories
- `app/` (remaining) — Partial FastAPI skeleton, extended by new `backend/`
