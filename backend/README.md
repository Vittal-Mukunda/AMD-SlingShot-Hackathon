# Backend — DQN Workforce Scheduler API

FastAPI + python-socketio ASGI server that orchestrates the two-phase DQN simulation and streams real-time data to the frontend.

## Stack

- **FastAPI** — REST API
- **python-socketio** (AsyncServer, ASGI mode) — WebSocket streaming
- **uvicorn** — ASGI server
- **asyncio** — non-blocking simulation loop

## Starting

```bash
# From project root (with .venv active):
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the unified command from the project root:

```bash
npm run dev
```

## REST API

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/api/initialize` | `SimConfig` JSON | `{"status":"started","days_phase1":N,"days_phase2":M}` |
| `POST` | `/api/pause` | — | `{"status":"paused"}` |
| `POST` | `/api/resume` | — | `{"status":"running"}` |
| `GET` | `/api/status` | — | Current `{phase, tick, day, running, worker_states, queue_state}` |
| `GET` | `/api/export` | — | CSV file download |
| `POST` | `/api/generate-readme` | — | Triggers README generation; progress via WebSocket |

### SimConfig (POST /api/initialize)

```json
{
  "days_phase1": 20,
  "days_phase2": 5,
  "worker_mode": "auto",
  "worker_seed": 42,
  "num_workers": 5,
  "manual_workers": [],
  "arrival_distribution": "poisson",
  "arrival_params": { "rate": 3.5 },
  "task_count": 200,
  "seed": 42
}
```

## WebSocket Events (Server → Client)

All events go to every connected client (broadcast):

| Event | Description |
|-------|-------------|
| `connected` | Emitted to new client on connect — also replays current state |
| `tick_update` | Every N decisions: `{tick, day, phase, worker_states, queue_state, last_assignment, active_policy}` |
| `gantt_block` | Per assignment: `{task_id, worker_id, start_tick, end_tick, urgency, policy}` |
| `daily_summary` | End of each simulated day: `{day, phase, metrics_per_policy}` |
| `phase_transition` | Phase 1 → 2: `{new_phase, baseline_results_snapshot}` |
| `simulation_complete` | All done: `{final_metrics}` |
| `readme_progress` | README generation log: `{file_path, status, preview_snippet}` |

## WebSocket Events (Client → Server)

| Event | Description |
|-------|-------------|
| `pause_simulation` | Pause the simulation loop |
| `resume_simulation` | Resume the simulation loop |
| `inject_task` | Add a priority task: `{task_id, duration, urgency, required_skill, arrival_tick}` |
| `generate_readme` | Trigger README regeneration |

## Source Files

| File | Purpose |
|------|---------|
| `main.py` | ASGI app = `socketio.ASGIApp(sio, fastapi_app)`. Registers REST routes and Socket.IO events. |
| `simulation_runner.py` | `SimulationRunner` class — creates 5 independent baseline envs, runs Phase 1 concurrently, then Phase 2 DQN. All emits happen here. |
| `readme_generator.py` | Async README generation with per-file WebSocket progress. |

## ASGI Architecture

```
Request (HTTP or WS)
    ↓
socketio.ASGIApp
    ├── /socket.io/* → python-socketio WebSocket handling
    └── /* → FastAPI (REST routes)
```

CORS is configured in FastAPI middleware allowing `http://localhost:5173` and `http://localhost:3000`.
