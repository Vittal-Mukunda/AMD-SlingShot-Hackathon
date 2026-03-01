# Frontend — DQN Workforce Scheduler Dashboard

React + Vite + TypeScript frontend that streams and visualizes real-time simulation data from the backend.

## Stack

- **React 19** with TypeScript
- **Vite 7** dev server with API proxy to `:8000`
- **Tailwind CSS v4** + custom design tokens
- **Recharts** — bar, radar, line charts
- **Custom SVG Gantt** — scrollable worker × tick timeline
- **socket.io-client v4** — direct WebSocket to `http://localhost:8000`
- **Zustand** — global simulation state
- **React Router v6** — three-page SPA

## Running

```bash
npm install
npm run dev
```

App is available at **http://localhost:5173**

## Pages

| Page | Route | Purpose |
|------|-------|---------|
| `ConfigPage` | `/` | Configuration wizard — sets up workers, tasks, schedules, then POSTs to `/api/initialize` |
| `SimulationPage` | `/simulation` | Live view — 5 Gantt tabs (Phase 1), Phase 2 transition screen, DQN live view |
| `AnalyticsPage` | `/analytics` | Post-simulation — 6 summary cards, makespan bar, radar, daily throughput, lateness, heatmap |

## Key Source Files

```
src/
├── App.tsx                      # BrowserRouter + AppShell (socket mounted here)
├── main.tsx                     # React root mount
├── index.css                    # Design system tokens + global styles
│
├── pages/
│   ├── ConfigPage.tsx           # Configuration wizard
│   ├── SimulationPage.tsx       # Real-time simulation (Gantt + workers + task queue)
│   └── AnalyticsPage.tsx        # Post-simulation analytics
│
├── components/
│   ├── simulation/
│   │   ├── GanttChart.tsx       # SVG Gantt: X=ticks, Y=workers, color=urgency
│   │   ├── WorkerSidebar.tsx    # Circular fatigue gauges + task list
│   │   ├── TaskQueue.tsx        # Scrollable horizontal task queue panel
│   │   └── ComparisonStrip.tsx  # Right panel: Phase 1 metrics / Phase 2 head-to-head
│   ├── config/
│   │   └── PriorityInjectionPanel.tsx  # Inject tasks pre/during sim
│   └── readme/
│       └── ReadmeGeneratorPanel.tsx    # Live README generation log
│
├── hooks/
│   ├── useSocket.ts             # Socket.IO client — global singleton, event dispatch
│   └── useSimulation.ts        # Derived selectors (formatTick, useHeadToHead, etc.)
│
├── store/
│   └── simulationStore.ts       # Zustand store — all simulation state
│
└── types/
    ├── simulation.ts            # WebSocket payload types
    ├── config.ts                # SimConfig, validation, defaults
    └── metrics.ts               # PolicyMetrics, RadarDataPoint, POLICY_COLORS
```

## WebSocket Connection

The frontend connects directly to `http://localhost:8000` (not through the Vite proxy) using socket.io-client. This is because `python-socketio` handles WebSocket upgrade as an ASGI app and CORS is configured server-side.

The `/api/*` routes ARE proxied through Vite to avoid CORS on REST calls.

## Design System

Colors (from `index.css`):

| Token | Value | Usage |
|-------|-------|-------|
| `--color-bg` | `#0D0F12` | App background |
| `--color-panel` | `#141721` | Card/panel background |
| `--color-amber` | `#F59E0B` | Primary accent, DQN highlights |
| `--color-success` | `#22C55E` | Throughput, fresh workers |
| `--color-danger` | `#EF4444` | Alerts, burnout, missed deadlines |
| `--color-slate-text` | `#94A3B8` | Secondary text |
