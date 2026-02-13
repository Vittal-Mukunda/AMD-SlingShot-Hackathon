from typing import Dict, List, Any
from app.db.models import Task, Worker, SimulationState, TaskStatus, TaskPriority

# Initial seed data
def create_initial_state() -> SimulationState:
    
    initial_tasks = [
        Task(
            id="t1", 
            title="Implement Authentication", 
            description="Add JWT auth",
            required_skill_level=3.0,
            estimated_duration=4.0,
            remaining_work=4.0,
            deadline=24.0,
            priority=TaskPriority.HIGH
        ),
        Task(
            id="t2", 
            title="Design Database Schema", 
            description="Create tables",
            required_skill_level=5.0,
            estimated_duration=2.0,
            remaining_work=2.0,
            deadline=12.0,
            priority=TaskPriority.CRITICAL
        ),
        Task(
            id="t3", 
            title="Write Documentation", 
            description="API docs",
            required_skill_level=1.0,
            estimated_duration=1.0,
            remaining_work=1.0,
            deadline=48.0,
            priority=TaskPriority.LOW
        ),
    ]

    initial_workers = [
        Worker(id="w1", name="Alice", skill_level=8.0, efficiency=1.2),
        Worker(id="w2", name="Bob", skill_level=4.0, efficiency=0.9),
        Worker(id="w3", name="Charlie", skill_level=2.0, efficiency=1.0),
    ]

    return SimulationState(
        tasks=initial_tasks,
        workers=initial_workers,
        current_time=0.0
    )

# Mutable in-memory database
# We expose a 'DB' dictionary to mimic the existing usage patterns,
# but we'll also keep a structured SimulationState for internal logic.

global_state = create_initial_state()

# Helper accessors to maintain compatibility with existing 'DB["tasks"]' pattern
# while backing it with Pydantic models
class MockDBLocal:
    @property
    def tasks(self):
        # Return dict representations for compatibility, or the objects themselves
        return [t.dict() for t in global_state.tasks]

    @property
    def workers(self):
        return [w.dict() for w in global_state.workers]

    def get_task_obj(self, task_id: str):
        return next((t for t in global_state.tasks if t.id == task_id), None)
    
    def get_worker_obj(self, worker_id: str):
        return next((w for w in global_state.workers if w.id == worker_id), None)

mock_db_instance = MockDBLocal()

# Expose as a dict-like object for legacy code compatibility
# CAUTION: This means direct writes to DB['tasks'] won't update global_state unless we handle it.
# Ideally, we should refactor services to use global_state directly. 
# For now, we will expose the global_state directly to the service layer.

DB = {
    "tasks": mock_db_instance.tasks,
    "workers": mock_db_instance.workers,
    "state": global_state # Direct access for simulation service
}
