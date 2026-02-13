from app.services.task_service import TaskService
from app.services.simulation_service import SimulationService

task_service = TaskService()
simulation_service = SimulationService()

def register_tools(mcp):
    """Refactored tool registration to match official SDK patterns."""

    @mcp.tool()
    def get_tasks() -> list:
        """Return all tasks with simulation details.

        Returns:
            list: List of task dictionaries.
        """
        return task_service.get_all_tasks()

    @mcp.tool()
    def assign_task(worker_id: str, task_id: str) -> dict:
        """Assign a task to a worker.

        Args:
            worker_id (str): The ID of the worker.
            task_id (str): The ID of the task.

        Returns:
            dict: Assignment result.
        """
        return task_service.assign_task(worker_id, task_id, automated=False)

    @mcp.tool()
    def update_task_status(task_id: str, status: str) -> dict:
        """Update task status manually.

        Args:
            task_id (str): The ID of the task.
            status (str): New status (todo, in_progress, completed, failed).

        Returns:
            dict: Update result.
        """
        from app.db.models import TaskStatus
        return task_service.update_status(task_id, TaskStatus(status))

    @mcp.tool()
    def get_state() -> dict:
        """Return global project state.

        Returns:
            dict: Global state containing tasks, workers, and time.
        """
        return task_service.get_state()

    # --- Simulation Tools ---

    @mcp.tool()
    def simulate_step(hours: float = 1.0) -> dict:
        """Advance the simulation by N hours.

        Args:
            hours (float): Number of hours to advance.

        Returns:
            dict: Events and new time.
        """
        return simulation_service.step(hours)

    @mcp.tool()
    def get_rl_observation() -> dict:
        """Get the vector-ready observation for RL agents.

        Returns:
            dict: Simplified numeric state vector.
        """
        from app.rl.environment import RLEnvironment
        env = RLEnvironment()
        return env.get_observation()

    @mcp.tool()
    def reset_simulation() -> dict:
        """Resets the simulation to initial seed state.

        Returns:
            dict: New initial state.
        """
        from app.db.mock_db import create_initial_state, global_state
        
        # Reset global state in-place
        new_state = create_initial_state()
        global_state.tasks.clear()
        global_state.tasks.extend(new_state.tasks)
        global_state.workers.clear()
        global_state.workers.extend(new_state.workers)
        global_state.current_time = 0.0
        
        return {"message": "Simulation reset", "state": global_state.dict()}
