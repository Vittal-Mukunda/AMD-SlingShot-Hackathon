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
from fastapi.responses import FileResponse, JSONResponse, Response
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
    # Unified simulation horizon (overrides days_phase1 + days_phase2 when set)
    sim_days: int = 0             # 0 means use days_phase1 + days_phase2 (legacy)
    # Legacy phase durations (kept for backward-compat)
    days_phase1: int = 60
    days_phase2: int = 40
    # v9: Phase 1/2 split fraction (0.40–0.80; default 0.60 = 60% baseline observation)
    phase1_fraction: float = 0.60
    # Worker config
    worker_mode: str = "auto"        # "auto" | "manual"
    worker_seed: int = 42
    num_workers: int = 5
    max_worker_load: int = 5         # v9: max concurrent tasks per worker (3–15)
    manual_workers: list = []        # list of WorkerConfig dicts when mode=manual
    # Task arrival
    arrival_distribution: str = "poisson"  # uniform | poisson | burst | custom
    arrival_params: dict = {}
    task_count: int = 600            # v10: raised to match dynamic cap for 100-day sims
    tasks_per_day: float = 4.0       # v8: fixed daily arrival rate (1-20)
    # Meta
    seed: int = 42
    # Frontend may send pre-queued injected tasks — accepted and ignored at init time
    # (tasks are injected live via WebSocket once simulation is running)
    injected_tasks: list = []


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
    import traceback as _tb
    global _runner, _sim_task

    try:
        # ── Cancel any existing run ──────────────────────────────────────────
        print(f"[init] Received config: days_phase1={cfg.days_phase1}, "
              f"days_phase2={cfg.days_phase2}, workers={cfg.num_workers}, "
              f"tasks_per_day={cfg.tasks_per_day}, seed={cfg.seed}")

        if _sim_task and not _sim_task.done():
            _sim_task.cancel()
            try:
                await _sim_task
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0.1)

        # ── Notify frontend to fully reset before new simulation ─────────────
        print("[init] Emitting simulation_reset")
        await sio.emit("simulation_reset", {"reason": "new_config"})
        await asyncio.sleep(0.05)

        # ── Create simulation runner ─────────────────────────────────────────
        print("[init] Creating SimulationRunner…")
        _runner = SimulationRunner(cfg=cfg, sio=sio)
        print("[init] SimulationRunner created successfully")

        # ── Launch async simulation task ─────────────────────────────────────
        print("[init] Launching simulation task…")
        _sim_task = asyncio.create_task(_runner.run())
        print("[init] Simulation task launched")

        return {"status": "started", "days_phase1": cfg.days_phase1, "days_phase2": cfg.days_phase2}

    except Exception as exc:
        tb_str = _tb.format_exc()
        print(f"[init] ERROR during initialization:\n{tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "detail": tb_str,
                "hint": (
                    "Check that all Python dependencies are installed and the backend "
                    "slingshot package is importable. Run: pip install -r requirements.txt"
                ),
            },
        )


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
    """Download detailed simulation results as CSV.

    Sections:
      1. Per-task DQN records (identity, timing, deadline, quality)
      2. === SUMMARY === (aggregate counts)
      3. === DQN AGENT DIAGNOSTICS === (epsilon, training stats, buffer utilization)
      4. === POLICY COMPARISON === (all baselines + DQN side-by-side)
    """
    import io
    import csv
    import config as cfg_module

    # Try to generate rich per-task CSV from live runner data
    if _runner and hasattr(_runner, '_dqn_env') and _runner._dqn_env is not None:
        env   = _runner._dqn_env
        agent = _runner.agent  # DQNAgent instance
        output = io.StringIO()
        writer = csv.writer(output)

        slots_per_day = cfg_module.SLOTS_PER_DAY
        slot_hours    = cfg_module.SLOT_HOURS

        # ── Section 1: Per-task records ───────────────────────────────────
        writer.writerow([
            # Task identity
            'task_id', 'priority', 'complexity', 'task_type',
            'arrival_tick', 'arrival_day',
            # Assignment
            'assigned_worker', 'start_tick', 'start_day',
            # Completion
            'completion_tick', 'completion_day',
            'duration_slots', 'duration_hours',
            # Deadline
            'deadline_slot', 'deadline_hours_from_arrival',
            'slots_late', 'hours_late', 'deadline_met',
            # Quality
            'quality_score',
            # Status
            'is_completed', 'is_failed',
        ])

        for t in env.tasks:
            arrival_day = t.arrival_tick // slots_per_day if slots_per_day else 0

            if t.is_completed and t.actual_completion_tick is not None:
                comp_tick    = t.actual_completion_tick
                comp_day     = comp_tick // slots_per_day
                start_tick   = t.start_tick or t.arrival_tick
                start_day    = start_tick // slots_per_day
                dur_slots    = comp_tick - start_tick
                dur_hours    = round(dur_slots * slot_hours, 2)
                slots_late   = max(0, comp_tick - t.deadline_slot)
                hours_late   = round(slots_late * slot_hours, 2)
                deadline_met = 1 if slots_late == 0 else 0
                quality      = round(getattr(t, 'quality_score', 0.0), 4)
            else:
                comp_tick = comp_day = start_tick = start_day = ''
                dur_slots = dur_hours = slots_late = hours_late = ''
                deadline_met = 0
                quality = ''

            deadline_h = round(t.deadline_h, 2) if hasattr(t, 'deadline_h') else ''

            writer.writerow([
                t.task_id, t.priority, t.complexity,
                getattr(t, 'task_type', 0),
                t.arrival_tick, arrival_day,
                t.assigned_worker if t.assigned_worker is not None else '',
                start_tick, start_day,
                comp_tick, comp_day,
                dur_slots, dur_hours,
                t.deadline_slot, deadline_h,
                slots_late, hours_late, deadline_met,
                quality,
                int(t.is_completed), int(t.is_failed),
            ])

        # ── Section 2: Summary ────────────────────────────────────────────
        completed = [t for t in env.tasks if t.is_completed]
        failed    = [t for t in env.tasks if t.is_failed]
        late      = [t for t in completed if (t.actual_completion_tick or 0) > t.deadline_slot]
        on_time   = len(completed) - len(late)
        avg_q     = (sum(getattr(t, 'quality_score', 0) for t in completed) / len(completed)) if completed else 0

        writer.writerow([])
        writer.writerow(['=== SUMMARY ==='])
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_tasks',       len(env.tasks)])
        writer.writerow(['completed',         len(completed)])
        writer.writerow(['failed',            len(failed)])
        writer.writerow(['on_time',           on_time])
        writer.writerow(['late',              len(late)])
        writer.writerow(['completion_rate',   f"{len(completed)/max(len(env.tasks),1)*100:.1f}%"])
        writer.writerow(['on_time_rate',      f"{on_time/max(len(completed),1)*100:.1f}%"])
        writer.writerow(['lateness_rate',     f"{len(late)/max(len(completed),1)*100:.1f}%"])
        writer.writerow(['avg_quality_score', f"{avg_q:.4f}"])
        writer.writerow(['simulation_days',   env.clock.day])
        writer.writerow(['total_ticks',       env.clock.tick])

        # ── Section 3: DQN Agent Diagnostics ─────────────────────────────
        writer.writerow([])
        writer.writerow(['=== DQN AGENT DIAGNOSTICS ==='])
        writer.writerow(['metric', 'value', 'description'])

        buf_size     = len(agent.replay_buffer)
        buf_capacity = agent.replay_buffer.tree.capacity
        buf_pct      = f"{buf_size / max(buf_capacity, 1) * 100:.1f}%"

        writer.writerow(['architecture',       'Dueling DQN + Double DQN + PER + Cosine LR',
                         'Network architecture used'])
        writer.writerow(['device',             str(agent.device),
                         'Hardware accelerator used for training'])
        writer.writerow(['state_dim',          agent.state_dim,
                         'Input state vector dimensionality'])
        writer.writerow(['action_dim',         agent.action_dim,
                         'Number of possible actions (task×worker combos)'])
        writer.writerow(['epsilon_final',      f"{agent.epsilon:.6f}",
                         'Final exploration rate (0=fully greedy, 1=fully random)'])
        writer.writerow(['epsilon_start',      f"{agent.epsilon_start:.6f}",
                         'Starting exploration rate at beginning of Phase 2'])
        writer.writerow(['epsilon_end',        f"{agent.epsilon_end:.6f}",
                         'Minimum epsilon floor'])
        writer.writerow(['gamma',              f"{agent.gamma:.4f}",
                         'Discount factor for future rewards'])
        writer.writerow(['learning_rate',      f"{agent.optimizer.param_groups[0]['lr']:.2e}",
                         'Current learning rate (cosine-annealed)'])
        writer.writerow(['batch_size',         agent.batch_size,
                         'Mini-batch size for each gradient update'])
        writer.writerow(['steps_done',         agent.steps_done,
                         'Total decisions taken by agent (Phase 1 + Phase 2)'])
        writer.writerow(['train_steps',        agent.train_steps,
                         'Total gradient updates (Bellman backprop steps)'])
        writer.writerow(['train_skipped',      agent.train_skipped,
                         'Steps skipped due to insufficient replay buffer'])
        writer.writerow(['last_loss',          f"{agent.last_loss:.6f}" if agent.train_steps > 0 else 'n/a',
                         'Final Huber loss value (IS-weighted)'])
        writer.writerow(['last_q_mean',        f"{agent.last_q_mean:.4f}" if agent.train_steps > 0 else 'n/a',
                         'Mean Q-value from last training batch'])
        writer.writerow(['last_q_target',      f"{getattr(agent, 'last_q_target', 0):.4f}" if agent.train_steps > 0 else 'n/a',
                         'Mean target Q-value from last training batch'])
        writer.writerow(['last_td_error',      f"{agent.last_td_error:.6f}" if agent.train_steps > 0 else 'n/a',
                         'Mean |TD error| from last training batch'])
        writer.writerow(['replay_buffer_size', buf_size,
                         'Final number of transitions in replay buffer'])
        writer.writerow(['replay_buffer_capacity', buf_capacity,
                         'Maximum replay buffer capacity'])
        writer.writerow(['replay_buffer_fill', buf_pct,
                         'Percentage of replay buffer filled'])
        writer.writerow(['per_alpha',          f"{agent.replay_buffer.alpha:.3f}",
                         'PER priority exponent (0=uniform, 1=full priority)'])
        writer.writerow(['target_update_freq', agent.target_update_freq,
                         'Target network sync frequency (every N train steps)'])
        writer.writerow(['min_replay_size',    agent.min_replay_size,
                         'Minimum buffer fill before training begins'])

        # ── Section 4: Policy Comparison ─────────────────────────────────
        writer.writerow([])
        writer.writerow(['=== POLICY COMPARISON ==='])
        writer.writerow(['policy', 'throughput_per_day', 'completion_rate_%',
                         'lateness_rate_%', 'quality_score', 'overload_events',
                         'phase'])

        # Baseline results (Phase 1)
        baseline_snapshots = getattr(_runner, '_baseline_snapshots', {})
        for pol_name, snap in baseline_snapshots.items():
            writer.writerow([
                pol_name,
                f"{snap.get('throughput', 0):.3f}",
                f"{snap.get('completion_rate', 0)*100:.1f}",
                f"{snap.get('lateness_rate', 0)*100:.1f}",
                f"{snap.get('quality_score', 0):.4f}",
                int(snap.get('overload_events', 0)),
                'Phase 1 (Baseline)',
            ])

        # DQN results (Phase 2)
        p2_metrics = getattr(_runner, '_phase2_metrics', [])
        if p2_metrics:
            import numpy as np
            dqn_tp   = float(np.mean([r['throughput_per_day'] for r in p2_metrics]))
            dqn_cr   = float(np.mean([r['completion_rate']    for r in p2_metrics]))
            dqn_late = float(np.mean([r['lateness_rate']      for r in p2_metrics]))
            dqn_qual = float(np.mean([r['quality_score']      for r in p2_metrics]))
            dqn_ovld = int(sum(r['overload_events']           for r in p2_metrics))
            writer.writerow([
                'DQN',
                f"{dqn_tp:.3f}",
                f"{dqn_cr*100:.1f}",
                f"{dqn_late*100:.1f}",
                f"{dqn_qual:.4f}",
                dqn_ovld,
                'Phase 2 (DQN Agent)',
            ])

        # ── Section 5: Baseline Daily Metrics ─────────────────────────────
        writer.writerow([])
        writer.writerow(['=== BASELINE DAILY METRICS ==='])
        writer.writerow(['baseline', 'day', 'throughput_per_day', 'completion_rate',
                         'lateness_rate', 'quality_score', 'load_balance',
                         'overload_events', 'decisions'])
        p1_metrics = list(getattr(_runner, '_phase1_metrics', []))
        for row in p1_metrics:
            writer.writerow([
                row.get('baseline', ''),
                row.get('day', ''),
                f"{row.get('throughput_per_day', 0):.3f}",
                f"{row.get('completion_rate', 0):.4f}",
                f"{row.get('lateness_rate', 0):.4f}",
                f"{row.get('quality_score', 0):.4f}",
                f"{row.get('load_balance', 0):.4f}",
                row.get('overload_events', 0),
                row.get('decisions', 0),
            ])

        # ── Section 6: Parameter Documentation ────────────────────────────
        writer.writerow([])
        writer.writerow(['=== PARAMETER DOCUMENTATION ==='])
        writer.writerow(['parameter', 'current_value', 'file', 'description', 'how_to_tweak'])
        param_docs = [
            ['TASK_ARRIVAL_RATE', f"{cfg_module.TASK_ARRIVAL_RATE}", 'config.py',
             'Mean tasks arriving per working day (Poisson rate)',
             'Edit config.py line ~72 or use the Tasks per Day slider in the UI'],
            ['TOTAL_TASKS', f"{cfg_module.TOTAL_TASKS}", 'config.py',
             'Upper cap on total tasks generated',
             'Edit config.py line ~73 or use the Task Cap field in the UI'],
            ['SIM_DAYS', f"{cfg_module.SIM_DAYS}", 'config.py',
             'Total simulation working days (Phase 1 + Phase 2)',
             'Set Phase 1 + Phase 2 days in the UI, or edit config.py line ~26'],
            ['NUM_WORKERS', f"{cfg_module.NUM_WORKERS}", 'config.py',
             'Number of concurrent workers',
             'Use Num Workers field in UI or edit config.py line ~41'],
            ['EPSILON_START', f"{cfg_module.EPSILON_START}", 'config.py',
             'Initial exploration rate (1.0 = fully random)',
             'Edit config.py line ~138. Higher = more exploration early'],
            ['EPSILON_END', f"{cfg_module.EPSILON_END}", 'config.py',
             'Minimum exploration floor (0.05 = 5% random at convergence)',
             'Edit config.py line ~139. Lower = more greedy at end'],
            ['LEARNING_RATE', f"{cfg_module.LEARNING_RATE}", 'config.py',
             'Adam optimizer learning rate for DQN training',
             'Edit config.py line ~131. Reduce to 0.0001 for more stable but slower learning'],
            ['GAMMA', f"{cfg_module.GAMMA}", 'config.py',
             'Discount factor for future rewards (0-1)',
             'Edit config.py line ~132. Higher = agent values future rewards more'],
            ['BATCH_SIZE', f"{cfg_module.BATCH_SIZE}", 'config.py',
             'Mini-batch size for gradient updates',
             'Edit config.py line ~133. Larger = more stable gradients, slower updates'],
            ['REPLAY_BUFFER_SIZE', f"{cfg_module.REPLAY_BUFFER_SIZE}", 'config.py',
             'Priority replay buffer capacity',
             'Edit config.py line ~114. Larger = more diverse samples, more memory'],
            ['REWARD_COMPLETION_BASE', f"{cfg_module.REWARD_COMPLETION_BASE}", 'config.py',
             'Base reward for completing a task (scaled by priority and quality)',
             'Edit config.py line ~163. Higher = stronger completion incentive'],
            ['REWARD_EARLY_COMPLETION_BONUS', f"{getattr(cfg_module, 'REWARD_EARLY_COMPLETION_BONUS', 0.2)}", 'config.py',
             'Bonus for completing a task 20%+ ahead of deadline',
             'Edit config.py line ~166. Higher = stronger speed incentive'],
            ['REWARD_OVERLOAD_WEIGHT', f"{cfg_module.REWARD_OVERLOAD_WEIGHT}", 'config.py',
             'Penalty per overloaded worker (fatal, -5.0 default)',
             'Edit config.py line ~183. More negative = harder overload avoidance'],
            ['DEADLINE_MIN_DAYS', f"{cfg_module.DEADLINE_MIN_DAYS}", 'config.py',
             'Minimum deadline window from task arrival (working days)',
             'Edit config.py line ~83. Shorter = more scheduling pressure'],
            ['DEADLINE_MAX_DAYS', f"{cfg_module.DEADLINE_MAX_DAYS}", 'config.py',
             'Maximum deadline window from task arrival (working days)',
             'Edit config.py line ~84. Shorter = tighter deadlines for all tasks'],
            ['PHASE1_FRACTION', f"{cfg_module.PHASE1_FRACTION}", 'config.py',
             'Fraction of SIM_DAYS used for Phase 1 (baseline observation)',
             'Edit config.py line ~27. Default 0.80 = 80% observation, 20% DQN'],
            ['TARGET_UPDATE_FREQ', f"{cfg_module.TARGET_UPDATE_FREQ}", 'config.py',
             'How often the target Q-network syncs with the policy network',
             'Edit config.py line ~135. Lower = faster but noisier target updates'],
            ['MAX_WORKER_LOAD', f"{cfg_module.MAX_WORKER_LOAD}", 'config.py',
             'Maximum concurrent tasks per worker before overload',
             'Edit config.py line ~42. Higher = more capacity per worker'],
        ]
        for p in param_docs:
            writer.writerow(p)

        csv_bytes = output.getvalue().encode('utf-8')
        return Response(
            content=csv_bytes,
            media_type='text/csv',
            headers={'Content-Disposition': 'attachment; filename="dqn_simulation_analytics.csv"'},
        )

    # Fallback: serve pre-saved CSV file
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
