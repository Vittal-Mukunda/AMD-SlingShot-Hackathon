from typing import List, Dict, Any, Optional
from app.db.mock_db import global_state
from app.db.models import TaskStatus, Task, Worker
from app.core.logging import setup_logger, log_agent_action

logger = setup_logger("TaskService")

class TaskService:
    def __init__(self):
        self.state = global_state # Direct reference for now

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return [t.model_dump() for t in self.state.tasks]

    def assign_task(self, worker_id: str, task_id: str, automated: bool = False) -> Dict[str, Any]:
        """
        Assigns a task to a worker.
        Supports manual assignment (API/MCP) or automated (RL/Agent).
        """
        task = self._get_task(task_id)
        worker = self._get_worker(worker_id)
        
        if not task:
            return {"error": "Task not found"}
        if not worker:
            return {"error": "Worker not found"}
            
        if task.status != TaskStatus.TODO:
            return {"error": f"Task is {task.status}, cannot assign"}
            
        if worker.current_task_id:
             return {"error": f"Worker is busy with {worker.current_task_id}"}
        
        # Assignment logic
        task.assigned_to = worker_id
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = self.state.current_time
        
        worker.current_task_id = task_id
        
        log_agent_action(logger, "ASSIGN", {"task_id": task_id, "worker_id": worker_id})
        
        return {
            "message": "assigned",
            "task": task.model_dump(),
            "worker": worker.model_dump(),
            "automated": automated
        }

    def update_status(self, task_id: str, status: TaskStatus) -> Dict[str, Any]:
        task = self._get_task(task_id)
        if not task:
             return {"error": "Task not found"}
        
        old_status = task.status
        task.status = status
        
        logger.info("Task Status Update", extra={"extra_data": {"task_id": task_id, "transition": f"{old_status} -> {status}"}})
        
        return {"message": "updated", "task": task.model_dump()}

    def defer_task(self, task_id: str) -> Dict[str, Any]:
        """Defer a task temporarily"""
        task = self._get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        task.deferred = True
        log_agent_action(logger, "DEFER", {"task_id": task_id})
        return {"message": "deferred", "task": task.model_dump()}

    def escalate_task(self, task_id: str) -> Dict[str, Any]:
        """Escalate a task priority"""
        task = self._get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        task.escalated = True
        log_agent_action(logger, "ESCALATE", {"task_id": task_id})
        return {"message": "escalated", "task": task.model_dump()}

    def get_state(self) -> Dict[str, Any]:
        """Returns the full Global Simulation State"""
        # Since SimulationService now owns the real state, we should ideally fetch from there, 
        # but for now we fallback to the mock state if used manually.
        try:
            from app.services.simulation_service import SimulationService
            sim = SimulationService()
            return sim.get_state()
        except:
            return self.state.model_dump()

    # Private Helpers
    def _get_task(self, task_id: str) -> Optional[Task]:
        return next((t for t in self.state.tasks if t.id == task_id), None)
        
    def _get_worker(self, worker_id: str) -> Optional[Worker]:
        return next((w for w in self.state.workers if w.id == worker_id), None)

