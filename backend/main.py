"""
backend/main.py — FastAPI + Socket.IO ASGI server for DQN Workforce Scheduler Dashboard

Serves:
  REST:      POST /api/initialize, /api/pause, /api/resume, /api/inject-task
             GET  /api/export, /api/status
             POST /api/generate-readme
  WebSocket: Socket.IO namespace / (uses sio)

Run:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import os
import sys

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Ensure project root on path ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Socket.IO (async) ─────────────────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="DQN Workforce Scheduler API", version="2.0.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simulation state ──────────────────────────────────────────────────────────
from backend.simulation_runner import SimulationRunner

_runner: SimulationRunner = None
_sim_task: asyncio.Task = None


# ── Pydantic request models ───────────────────────────────────────────────────

class SimConfig(BaseModel):
    # Simulation horizon
    days_phase1: int = 20
    days_phase2: int = 5
    # Worker config
    worker_mode: str = "auto"        # "auto" | "manual"
    worker_seed: int = 42
    num_workers: int = 5
    manual_workers: list = []        # list of WorkerConfig dicts when mode=manual
    # Task arrival
    arrival_distribution: str = "poisson"  # uniform | poisson | burst | custom
    arrival_params: dict = {}
    task_count: int = 200
    # Meta
    seed: int = 42


class InjectTaskPayload(BaseModel):
    task_id: str
    duration: float
    urgency: int          # 0-3
    required_skill: float
    arrival_tick: int


# ── REST Endpoints ────────────────────────────────────────────────────────────

@fastapi_app.post("/api/initialize")
async def initialize_simulation(cfg: SimConfig):
    """Create and start a new simulation run."""
    global _runner, _sim_task

    # Cancel any existing run
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass

    _runner = SimulationRunner(cfg=cfg, sio=sio)
    _sim_task = asyncio.create_task(_runner.run())
    return {"status": "started", "days_phase1": cfg.days_phase1, "days_phase2": cfg.days_phase2}


@fastapi_app.post("/api/pause")
async def pause_simulation():
    if _runner:
        _runner.pause()
        return {"status": "paused"}
    return {"status": "no_runner"}


@fastapi_app.post("/api/resume")
async def resume_simulation():
    if _runner:
        _runner.resume()
        return {"status": "running"}
    return {"status": "no_runner"}


@fastapi_app.post("/api/inject-task")
async def inject_task(task: InjectTaskPayload):
    if _runner:
        _runner.inject_task(task.dict())
        return {"status": "injected", "task_id": task.task_id}
    return {"status": "no_runner"}


@fastapi_app.get("/api/status")
async def get_status():
    if _runner:
        return _runner.get_status()
    return {"phase": 0, "tick": 0, "day": 0, "running": False}


@fastapi_app.get("/api/export")
async def export_csv():
    """Download the latest phase metrics as CSV."""
    import config as cfg_module
    p2_path = os.path.join(cfg_module.RESULTS_DIR, "phase2_metrics.csv")
    p1_path = os.path.join(cfg_module.RESULTS_DIR, "phase1_metrics.csv")
    target = p2_path if os.path.exists(p2_path) else p1_path
    if os.path.exists(target):
        return FileResponse(target, media_type="text/csv", filename="simulation_results.csv")
    return {"error": "No results CSV found yet"}


@fastapi_app.post("/api/generate-readme")
async def generate_readme():
    """Start async README generation; progress streamed via WebSocket."""
    from backend.readme_generator import ReadmeGenerator
    gen = ReadmeGenerator(sio=sio)
    asyncio.create_task(gen.run())
    return {"status": "generating"}


# ── Socket.IO events (frontend → backend) ────────────────────────────────────

@sio.event
async def connect(sid, environ, auth=None):
    print(f"[WS] Client connected: {sid}")
    await sio.emit("connected", {"message": "Backend ready"}, to=sid)
    if _runner:
        # Bug 8: Send full state to reconnecting client (phase_transition/phase2_ready + tick_update)
        snapshot = _runner.get_status()
        phase = snapshot.get("phase", 0)
        bl_snap = getattr(_runner, "_baseline_snapshots", {})
        if phase == 2 and bl_snap:
            await sio.emit("phase_transition", {
                "new_phase": "training",
                "baseline_results_snapshot": bl_snap,
            }, to=sid)
        elif phase == 3 and bl_snap:
            await sio.emit("phase2_ready", {
                "baseline_results_snapshot": bl_snap,
            }, to=sid)
        await sio.emit("tick_update", snapshot, to=sid)


@sio.event
async def disconnect(sid):
    print(f"[WS] Client disconnected: {sid}")


@sio.event
async def inject_task(sid, data):
    if _runner:
        _runner.inject_task(data)
        await sio.emit("task_injected", {"task_id": data.get("task_id")}, to=sid)


@sio.event
async def pause_simulation(sid, data=None):
    if _runner:
        _runner.pause()


@sio.event
async def resume_simulation(sid, data=None):
    if _runner:
        _runner.resume()


@sio.event
async def generate_readme(sid, data=None):
    from backend.readme_generator import ReadmeGenerator
    gen = ReadmeGenerator(sio=sio)
    asyncio.create_task(gen.run())


# ── ASGI app: combine FastAPI + Socket.IO ────────────────────────────────────
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app, socketio_path="/socket.io")
