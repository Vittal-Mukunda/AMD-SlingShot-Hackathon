# RL-Driven Agentic Project Manager (AMD SlingShot Hackathon)

A simulation-first backend system designed to train reinforcement learning (RL) agents for optimal project management. The system simulates a dynamic environment with tasks, workers, deadlines, and fatigue, exposing this state via Model Context Protocol (MCP) tools for agent interaction.

## 🚀 Key Features

*   **Simulation Engine**: Realistic modeling of task progress, worker fatigue, deadlines, and skill matching.
*   **RL-Ready Environment**: dedicated `RLEnvironment` class providing observation vectors and rewards for training agents.
*   **MCP Integration**: Exposes simulation control and state inspection as MCP tools (`simulate_step`, `get_rl_observation`).
*   **FastAPI Backend**: Robust REST API for external interaction and monitoring.
*   **Modular Architecture**: Clean separation of concerns (Simulation, MCP, RL, DB).

## 📂 Project Structure

```
app/
├── core/           # Configuration & constants
├── db/             # Pydantic models & in-memory mock database
├── mcp/            # MCP server & tool definitions
├── rl/             # Reinforcement Learning environment & wrappers
├── services/       # Business logic (Simulation, Task assignment)
├── api/            # FastAPI routes
└── main.py         # Application entrypoint
scripts/
├── run_server.py   # Script to start the FastAPI/MCP server
└── verify_backend.py # specific script to verify simulation logic
```

## 🛠️ Getting Started

### Prerequisites

*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### Installation

1.  Clone the repository.
2.  Install dependencies:

```bash
uv sync 
# or 
uv pip install fastapi uvicorn fastmcp numpy
```

### Running the Server

Start the FastAPI server with MCP mounted:

```bash
uv run scripts/run_server.py
```

The API will be available at `http://localhost:8000`.
MCP endpoint (SSE) is available at `http://localhost:8000/mcp/sse` (or similar depending on FastMCP config).

## 🧠 Simulation & RL

### Simulation Logic
The system runs a discrete-time simulation where:
*   **Tasks** have difficulty, deadlines, and required skills.
*   **Workers** have skill levels, efficiency, and fatigue properties.
*   **Time** advances in steps (e.g., 1 hour), updating progress and fatigue.

### Interacting with the Simulation

You can control the simulation using MCP tools or directly via Python scripts.

**Example MCP Tool Usage:**

*   `get_state()`: View all tasks and workers.
*   `assign_task(worker_id, task_id)`: Assign work.
*   `simulate_step(hours=1.0)`: Advance time by 1 hour.
*   `get_rl_observation()`: Get the vectorized state for an RL agent.

### Running the MCP Inspector

To test the MCP integration using the official inspector (note the `-q` flag to silence uv output):

```bash
npx @modelcontextprotocol/inspector uv run -q scripts/run_mcp.py
```

### Verification

Run the included verification script to test the simulation dynamics:

```bash
uv run python -m scripts.verify_backend
```

## 📚 API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the interactive Swagger UI.