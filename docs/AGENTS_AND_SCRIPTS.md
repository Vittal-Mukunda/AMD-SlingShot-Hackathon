# Agents and Scripts — Details, Behavior, and Improvements

This document explains the agent classes and utility scripts in this repository: what each agent is intended to do, what it currently does, how it does it, and suggested improvements. It also documents the main test/run scripts and how to exercise the codebase.

**Contents**
- **Agents**: overview of each agent, responsibilities, current behavior, and improvements
- **Scripts**: test and utility scripts and how to use them
- **Testing**: suggested tests and how to run them
- **Future work**: improvements, integration notes, and design suggestions

**Agents**

- **ContextAgent**: Provides read-only access and lightweight event recording.
  - **File**: [app/agents/context.py](app/agents/context.py#L1)
  - **Intended role**: Serve as a simple façade for planners and workers to query simulation state, inspect backlog (TODO tasks), and record lightweight events/history without mutating simulation state.
  - **What it actually does**: Implements `get_state()`, `get_backlog()`, `record_event(event)`, `get_history()`, and `summary()`:
    - `get_state()` forwards to the `SimulationService.get_state()` and returns a dict (guarding against None/non-dict results).
    - `get_backlog()` filters tasks with `status == "todo"`.
    - `record_event()` appends dict events to an internal `history` list.
    - `get_history()` returns a shallow copy of history.
    - `summary()` returns a small convenience report (backlog length, active task count, recent events, current_time).
  - **How it does it**: Uses typed method signatures and defensive checks; keeps no complex logic or side-effects; relies on the `SimulationService` and `TaskService` for authoritative state and mutation.
  - **Limitations / Improvements**:
    - Events are stored only in-memory; consider integrating a persistent event log or publishing events to an event bus for observability and replay.
    - Backlog filtering assumes tasks are dicts; if services return Pydantic objects, normalize them centrally.
    - Add timestamping/structured schema for events and size caps / rotation for history.

- **PlannerAgent**: Observes state, asks a policy for an action, and executes it.
  - **File**: [app/agents/planner.py](app/agents/planner.py#L1)
  - **Intended role**: Be the decision-making loop that glues a policy to the services — observe, decide, act.
  - **What it actually does**: Implements `observe()`, `decide()` and `act(action)` with the following semantics:
    - `observe()` returns `context.get_state()`.
    - `decide()` calls `policy.select_action(observation)` and returns the action. It wraps policy errors and returns `{ "error": str(e) }` if the policy raises.
    - `act()` accepts an action dict and supports three shapes:
      - Assign: `{ "type": "assign", "worker_id": str, "task_id": str }` → calls `TaskService.assign_task(..., automated=True)`
      - Wait: `{ "type": "wait", "hours": float }` → calls `SimulationService.step(hours)`
      - Unknown: returns `{ "error": "unknown_action" }`
    - `act()` is exception-safe and returns `{ "error": str(e) }` for unexpected errors.
  - **How it does it**: Lightweight glue — it doesn't implement policies itself; the repo includes tests with fake policies to validate integration. It relies on `TaskService` and `SimulationService` for side-effects.
  - **Limitations / Improvements**:
    - Validation for action shapes is minimal; consider using a small schema/TypedDict or pydantic model for actions.
    - Provide richer action set (defer/escalate/prioritize) and standardized result objects.
    - Add logging/tracing around `decide()` and `act()` for debugging and reproducibility.

- **WorkerAgent**: Convenience wrapper around a single worker in `global_state`.
  - **File**: [app/agents/worker.py](app/agents/worker.py#L1)
  - **Intended role**: Provide read-only convenience accessors for worker profiles and simple helpers like `is_busy()` without reimplementing state mutation logic (delegated to `TaskService`).
  - **What it actually does**: Implements `_worker()` to find a worker by id in `global_state.workers`, `profile()` to return key worker fields, and `is_busy()` to check for `current_task_id`.
  - **How it does it**: Looks up the worker object in `app.db.mock_db.global_state` and returns a dict (using `.dict()` if available). It intentionally avoids calling `TaskService` or mutating `global_state`.
  - **Limitations / Improvements**:
    - The wrapper assumes `global_state` is available and mutable; for real deployments, inject a read-only store or accessor interface for testability.
    - Consider optional `TaskService` injection for convenience methods like `assign()` or `release()` with clear separation of responsibility.

- **DQNAgent**: Deep RL agent used as a policy / baseline.
  - **File**: [agents/dqn_agent.py](agents/dqn_agent.py#L1)
  - **Intended role**: Learn a Q-function to map simulation states to actions (task assignment, wait, etc.) using DQN with replay buffer and target network.
  - **What it actually does**: Implements a full DQN stack:
    - `QNetwork` nn model, `ReplayBuffer`, `DQNAgent` with `select_action()`, `store_transition()`, `train_step()`, `save()` and `load()`.
    - Supports action masking by passing a `valid_actions` list to `select_action()`.
    - Includes a small unit-test block under `if __name__ == "__main__"` for quick sanity checks (initialization, forward pass, action selection, store/train cycle, checkpoint save/load).
  - **How it does it**: PyTorch-based network, Smooth L1 loss, Adam optimizer, gradient clipping, target network updates, and basic NaN checks.
  - **Limitations / Improvements**:
    - Hard-coded state/action dimensions come from `config.py`; consider deriving these dynamically or validating the mapping from simulation to vectorized state.
    - The test harness in the file is useful, but larger integration tests should use the `SimulationService` and `TaskService` to ensure end-to-end behavior.
    - Add distributed/accelerated training options, logging (TensorBoard/Weights & Biases), and configurable hyperparameters via `pyproject.toml` or CLI flags.


**Scripts and Utilities**

- `scripts/run_tests.py` and `scripts/run_tests.bat`
  - **Purpose**: Convenience wrappers to run the test suite via `pytest`.
  - **Usage**:
    ```bash
    # Python runner
    python scripts/run_tests.py

    # Windows batch
    scripts\run_tests.bat
    ```
  - **Note**: `pytest` is added to `requirements.txt`; install dependencies with `pip install -r requirements.txt` before running.

- `scripts/run_server.py`, `scripts/run_mcp.py`, `scripts/simulate.py` (if present)
  - **Purpose**: Helper scripts to launch the API server, the MCP server, and simulation driver respectively. Check each script for specific flags or environment variables they expect.
  - **Usage**: Typical usage is `python scripts/run_server.py` or via `uvicorn`/`fastapi` entrypoints if configured.

- Test files under `tests/` (examples added):
  - `tests/test_agents.py` — basic tests for `ContextAgent` (record_event, get_backlog)
  - `tests/test_planner.py` — mock-policy tests for `PlannerAgent`'s `decide()` and `act()` behavior
  - `tests/test_worker_agent.py` — integration tests that add a temporary worker to `global_state` and assert `profile()` / `is_busy()` behavior


**Testing and Running Locally**

- Install dependencies (recommended inside a virtualenv):
  ```bash
  python -m venv .venv
  .venv\Scripts\activate    # Windows
  pip install -r requirements.txt
  ```

- Run tests:
  ```bash
  python -m pytest -q
  # or
  scripts\run_tests.bat
  ```

- If `pytest` is not installed in your environment, install it explicitly:
  ```bash
  pip install pytest
  ```


**Design Notes & Integration Points**

- Services are the source of truth: `SimulationService` and `TaskService` hold and mutate the `global_state` (Pydantic `SimulationState`). Agents should avoid reimplementing mutation logic and instead call service methods.
- `app/db/mock_db.py` exposes `global_state` which the services and agents currently read. For production-like tests, swap in a fixture or adapter that provides a clean state per test.
- Action representation: Agents and policies currently use plain dicts to represent actions. Consider a small action schema (`TypedDict` or `pydantic` model) to validate and document allowed action fields.


**Future Improvements and Roadmap**

- Observability & Telemetry
  - Add structured event logging for `ContextAgent.record_event()` and service operations. Persist to file or external store for replay and debugging.

- Stronger Action Types
  - Replace free-form dict actions with TypedDict or pydantic models and centralize validation in `PlannerAgent.act()`.

- Policy Interface
  - Standardize the policy interface: `select_action(observation, valid_actions=None) -> Action`. Add a base `Policy` abstract class and register simple baselines in `baselines/`.

- Testing & CI
  - Add a GitHub Actions job to install `requirements.txt` and run `pytest`. Enforce code-style and static analysis (flake8/ruff/mypy).

- Separation of Concerns
  - Reduce direct `global_state` usage by agents; inject `StateStore` or read-only adapters for cleaner testing and separation.

- DQN & RL
  - Add training harness, experiment tracking, and hyperparameter sweep support. Decouple vectorization from `config.py` so the DQN network can adapt to changing numbers of workers/tasks.


**Appendix: Quick file reference**
- Agents:
  - [app/agents/context.py](app/agents/context.py#L1)
  - [app/agents/planner.py](app/agents/planner.py#L1)
  - [app/agents/worker.py](app/agents/worker.py#L1)
  - [agents/dqn_agent.py](agents/dqn_agent.py#L1)

- Services:
  - [app/services/simulation_service.py](app/services/simulation_service.py#L1)
  - [app/services/task_service.py](app/services/task_service.py#L1)

- Key scripts:
  - [scripts/run_tests.py](scripts/run_tests.py#L1)
  - [scripts/run_tests.bat](scripts/run_tests.bat#L1)


If you'd like, I can also:
- Add stricter action type definitions (TypedDict/pydantic) and migrate `PlannerAgent` to use them.
- Wire `ContextAgent` events to a simple persistent JSON log file and add rotation.
- Create a CI workflow that installs `requirements.txt` and runs `pytest` on PRs.

---
Generated on project snapshot: see files above for implementation details.






### FUTURE WORK

Integration status (summary)

Agents are present and functional as thin wrappers/glue:
ContextAgent provides read-only queries and an in-memory event history.
PlannerAgent observes via a context, calls a policy.select_action(obs), and executes 3 action types (assign, wait, noop) using TaskService and SimulationService.
WorkerAgent is a small convenience reader over global_state.
RL code (dqn_agent.py and baselines/) implements policies but there is no automatic adapter connecting:
Simulation state → DQN input vector (vectorizer) is missing.
DQN action indices → actionable dicts (assign/wait) (action mapper) is missing.
The MCP layer and services exist and are independent:
app/mcp/tools.py and app/mcp/server.py instantiate TaskService and SimulationService, but they do not use PlannerAgent or ContextAgent currently.
Tests exercise agents in isolation (mock policies); no end-to-end integration test is present that runs a planner + policy against the real SimulationService/TaskService loop.
Conclusion: agents are "simulating" behavior well in isolation, but not yet wired into the runtime/ML stack or MCP endpoints for actual end-to-end operation.

Actionable steps to make agents ready for production use
(ordered, concrete, and minimal changes you can implement incrementally)

Standardize policy & action types

Create a minimal Policy interface and an Action schema:
class Policy: def select_action(self, observation: dict) -> dict
Use a TypedDict or pydantic model for actions: Assign/Wait/Noop. This makes validation easy in PlannerAgent.act().
Location: add app/agents/types.py or app/agents/schemas.py.
Implement a State Vectorizer (simulation → policy input)

Add a small adapter that converts SimulationService.get_state() into the vector expected by DQNAgent (or other policies).
Encapsulate in app/rl/vec.py with functions:
state_to_vector(state: dict) -> np.ndarray
valid_actions(state: dict) -> List[int]
Reason: decouples network input logic from agents and makes policies interchangeable.
Implement an Action Mapper (policy output → actionable dict)

Map DQN's integer action to { "type": "assign", "worker_id": ..., "task_id": ... } or {"type":"wait","hours":...}.
Keep mapping deterministic and documented in config.
Wire a Policy Adapter for the DQN

Implement a thin adapter class that wraps DQNAgent and implements select_action(observation: dict) by:
calling the vectorizer to get state vector and valid actions,
calling DQNAgent.select_action(state_vector, valid_actions),
mapping the returned index into an action dict using the Action Mapper.
Create an orchestration loop / runner

Add a scripts/runner.py or extend scripts/simulate.py to:
instantiate SimulationService, TaskService, ContextAgent, PlannerAgent(policy, ...),
run a loop: observe → decide → act → record events, call simulation.step() for time progression.
Optionally expose this runner as an MCP tool or REST endpoint.
Integrate with MCP / API

Register planner control tools in app/mcp/tools.py (currently registers simulation/task tools). Add functions to:
start/stop a planner loop,
step the planner once,
query planner state/history.
This enables remote orchestration and manual overrides.
Robustness and observability

Add structured logs for actions, policy inputs/outputs, and service responses. Persist events from ContextAgent.record_event() to a rotating JSON/ndjson file or push to a logging backend.
Add metrics: tasks assigned, completion rate, mean time to completion.
Concurrency / Safety

If using an async server or multiple threads, protect global_state mutations (services) with a lock or serialize planner actions via a single-threaded agent loop.
Consider storing state in an in-memory DB or simple sqlite for multi-process experiments.
Tests and CI

Add integration tests that instantiate real SimulationService and TaskService, wire a basic policy (e.g., greedy_baseline), run N steps, and assert system-level invariants (no exceptions, tasks progress and complete/fail correctly).
Add a GitHub Actions workflow to run pip install -r requirements.txt and pytest.
Harden DQN integration

Ensure config.py hyperparameters match the vectorizer/action mapping.
Add training harness invoking the environment in app/rl/trainer.py that uses the PlannerAgent and services for environment stepping. Add checkpoints and observability.
Minimal quick changes I can implement for you right now (pick one):

Add the Policy interface and Action TypedDicts and update PlannerAgent to validate actions.
Add a small example DQNPolicyAdapter that maps DQN outputs to actions and a simple state_to_vector stub.
Create a scripts/runner.py that wires services, a context, planner, and a simple baseline, and runs a short loop to demonstrate end-to-end integration.
Commands to run tests and runner once wired:

pip install -r requirements.txt
# run tests
python -m pytest -q
# run a runner script (if added)
python scripts/runner.py