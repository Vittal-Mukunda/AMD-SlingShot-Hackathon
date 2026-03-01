"""
backend/readme_generator.py — Idempotent async README generator.

Traverses the project repo and regenerates all README.md files.
Emits readme_progress Socket.IO events per file.
Idempotency: SHA256 hash comparison skips files that haven't changed.
"""

import asyncio
import hashlib
import os
import sys
from datetime import datetime
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as cfg_module

# ── README templates ──────────────────────────────────────────────────────────

ROOT_README_TEMPLATE = """# AMD SlingShot — Continual Online DQN Workforce Scheduler

> A two-phase adaptive scheduling system using Deep Q-Networks for real-time workforce optimization.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React + Vite)                        │
│  ConfigPage → SimulationPage (Gantt + LiveMetrics) → AnalyticsPage  │
│  Socket.io-client ←──────── WebSocket ──────────────────────────────┤
├─────────────────────────────────────────────────────────────────────┤
│                     BACKEND (FastAPI + Socket.IO)                    │
│   /api/initialize → SimulationRunner → asyncio background task      │
├─────────────────────────────────────────────────────────────────────┤
│                       RL ENGINE (Python)                             │
│  ProjectEnv ←→ Phase1(Baselines) ←→ Phase2(DQNAgent online_step())  │
│  ReplayBuffer (PER) │ DuelingDQN │ DoubleQ │ CosineScheduler         │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Tree

```
AMD-SlingShot-Hackathon/
├── backend/                  # FastAPI + Socket.IO server
│   ├── main.py               # ASGI app (FastAPI + socketio combined)
│   ├── simulation_runner.py  # Two-phase RL orchestrator
│   └── readme_generator.py   # Idempotent README generator
├── frontend/                 # Vite + React + TypeScript dashboard
│   ├── src/
│   │   ├── pages/            # ConfigPage, SimulationPage, AnalyticsPage
│   │   ├── components/       # config/, simulation/, analytics/, readme/, shared/
│   │   ├── store/            # Zustand simulationStore.ts
│   │   ├── hooks/            # useSocket.ts, useSimulation.ts
│   │   └── types/            # simulation.ts, config.ts, metrics.ts
│   └── tailwind.config.ts
├── environment/
│   └── project_env.py        # Stochastic RL environment
├── agents/
│   └── dqn_agent.py          # Dueling DQN + PER + Double Q + Cosine LR
├── baselines/                # 4 baseline scheduling policies
├── continual_scheduler.py    # Two-phase CLI runner (Phase1→Phase2)
├── config.py                 # All hyperparameters
├── run_pipeline.py           # End-to-end CLI orchestrator
└── results/                  # CSV outputs, plots
```

## Setup & Running

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
# Create virtual environment
python -m venv .venv
.venv\\Scripts\\activate          # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start backend server (port 8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Configuration

Use the dashboard Config Page to set:
- Simulation horizon (Phase 1 + Phase 2 working days)
- Worker configuration (auto-seed or manual)
- Task arrival distribution (Poisson / Uniform / Burst / Custom)
- Priority task injections

### CLI (headless mode)

```bash
# Recommended: online continual learning
python run_pipeline.py --online --days-p1 20 --days-p2 5

# Smoke test (2+3 days)
python run_pipeline.py --online --smoke-test
```

## Modules

### Environment (`environment/project_env.py`)
Stochastic scheduling environment with heterogeneous workers, Poisson task arrivals,
per-worker fatigue dynamics, deadline shocks, and slot-based 8h/day workday clock.

### Baselines (`baselines/`)
Four baseline policies: **Greedy** (highest-priority first), **Skill** (best-fit skill matching),
**Hybrid** (composite), **Random** (uniform sampling).

### DQN Agent (`agents/dqn_agent.py`)
Dueling DQN with Prioritized Experience Replay (PER), Double Q-learning,
and a Cosine Annealing learning rate scheduler. State dim: 96. Action dim: 140.

### Replay Buffer
PrioritizedReplayBuffer using a SumTree for O(log n) priority sampling.
Alpha=0.6, Beta anneals from 0.4 to 1.0 over training.

### Scheduler Loop (`continual_scheduler.py`)
- **Phase 1** ({phase1_days} working days): Baselines drive all assignments;
  DQN passively stores transitions (no gradient updates — buffer warm-up).
- **Phase 2** ({phase2_days} working days): DQN takes full control with online
  per-decision gradient updates starting at ε={eps_p2_start}.

### Frontend
4-module React dashboard: Config wizard → real-time Gantt + metrics → post-simulation analytics → README generator.

## Performance Benchmarks

{results_table}

---
*Generated: {timestamp}*
"""

FRONTEND_README = """# Frontend — DQN Workforce Dashboard

React 18 + Vite + TypeScript dashboard for the DQN Workforce Scheduling System.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 18 + Vite |
| Language | TypeScript (strict) |
| Styling | Tailwind CSS (custom palette) |
| Charts | Recharts + custom SVG |
| State | Zustand |
| Routing | React Router v6 |
| WebSocket | socket.io-client |

## Structure

```
src/
  pages/          # ConfigPage, SimulationPage, AnalyticsPage
  components/     # Organized by module (config, simulation, analytics, readme, shared)
  store/          # simulationStore.ts (Zustand)
  hooks/          # useSocket.ts, useSimulation.ts
  types/          # simulation.ts, config.ts, metrics.ts
```

## Running

```bash
npm install
npm run dev       # Dev server at http://localhost:5173
npm run build     # Production build
npm run lint      # ESLint check
```

## Design

- Background: `#0D0F12` (near-black)
- Accent: `#F59E0B` (electric amber)
- Secondary text: `#94A3B8` (cool slate)
- Alerts: `#EF4444` (red)
- Success: `#22C55E` (green)
- Fonts: JetBrains Mono (data), DM Sans (headers)
"""

BACKEND_README = """# Backend — FastAPI + Socket.IO Server

Python FastAPI server with Socket.IO for real-time simulation streaming.

## Endpoints

### REST

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/initialize` | Start a new simulation with config |
| POST | `/api/pause` | Pause running simulation |
| POST | `/api/resume` | Resume paused simulation |
| POST | `/api/inject-task` | Inject a priority task |
| GET | `/api/status` | Current simulation status |
| GET | `/api/export` | Download results CSV |
| POST | `/api/generate-readme` | Trigger README generation |

### WebSocket Events (Socket.IO)

**Emitted by server:**
- `tick_update` — every N decisions
- `task_completed` — on task completion
- `phase_transition` — Phase 1→2 boundary
- `daily_summary` — end of each working day
- `simulation_complete` — end of Phase 2
- `readme_progress` — during README generation

**Received from client:**
- `inject_task`, `pause_simulation`, `resume_simulation`, `generate_readme`

## Running

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
"""


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _load_results_table() -> str:
    """Load last simulation results into a markdown table."""
    p2_path = os.path.join(cfg_module.RESULTS_DIR, "phase2_metrics.csv")
    p1_path = os.path.join(cfg_module.RESULTS_DIR, "phase1_metrics.csv")
    target = p2_path if os.path.exists(p2_path) else p1_path

    if not os.path.exists(target):
        return "_No simulation results available yet. Run a simulation first._"

    try:
        import csv
        with open(target, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:10]  # Limit to 10 rows
        if not rows:
            return "_Empty results file._"
        headers = list(rows[0].keys())[:6]  # First 6 columns
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in rows:
            table += "| " + " | ".join(str(r.get(h, ""))[:12] for h in headers) + " |\n"
        return table
    except Exception:
        return "_Could not parse results CSV._"


class ReadmeGenerator:

    README_FILES = [
        ("", "README.md", "root"),
        ("frontend", "README.md", "frontend"),
        ("backend", "README.md", "backend"),
    ]

    def __init__(self, sio):
        self.sio = sio

    async def run(self):
        for subdir, filename, kind in self.README_FILES:
            await asyncio.sleep(0.1)  # slight delay between files

            rel_path = os.path.join(subdir, filename) if subdir else filename
            abs_path = os.path.join(PROJECT_ROOT, rel_path)

            # Generate content
            if kind == "root":
                content = ROOT_README_TEMPLATE.format(
                    phase1_days=cfg_module.PHASE1_DAYS,
                    phase2_days=cfg_module.PHASE2_DAYS,
                    eps_p2_start=cfg_module.EPSILON_PHASE2_START,
                    results_table=_load_results_table(),
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                )
            elif kind == "frontend":
                content = FRONTEND_README
            elif kind == "backend":
                content = BACKEND_README
            else:
                continue

            # Idempotency check
            new_hash = _sha256(content)
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    existing = f.read()
                if _sha256(existing) == new_hash:
                    await self.sio.emit("readme_progress", {
                        "file_path": rel_path,
                        "status": "unchanged",
                        "preview_snippet": existing[:200],
                    })
                    continue

            # Ensure directory exists
            os.makedirs(os.path.dirname(abs_path) if subdir else PROJECT_ROOT, exist_ok=True)

            await self.sio.emit("readme_progress", {
                "file_path": rel_path,
                "status": "generating",
                "preview_snippet": content[:200],
            })
            await asyncio.sleep(0.3)  # simulate progress

            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            await self.sio.emit("readme_progress", {
                "file_path": rel_path,
                "status": "done",
                "preview_snippet": content[:400],
            })

        await self.sio.emit("readme_progress", {
            "file_path": "__complete__",
            "status": "complete",
            "preview_snippet": "",
        })
