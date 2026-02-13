from typing import List, Dict, Any
from app.db.models import Task, Worker, SimulationState, TaskStatus
from app.db.mock_db import global_state

class SimulationService:
    def __init__(self):
        self.state = global_state

    def step(self, hours: float = 1.0) -> Dict[str, Any]:
        """
        Advances the simulation by 'hours'.
        Updates task progress, worker fatigue, and checks deadlines.
        """
        self.state.current_time += hours
        events = []

        # 1. Process active tasks
        for worker in self.state.workers:
            if worker.current_task_id:
                task = self._get_task(worker.current_task_id)
                if task and task.status == TaskStatus.IN_PROGRESS:
                    # Calculate progress
                    # efficienty * skill_factor * time
                    skill_factor = min(worker.skill_level / task.required_skill_level, 2.0)
                    fatigue_factor = 1.0 - worker.fatigue
                    
                    progress = hours * worker.efficiency * skill_factor * fatigue_factor
                    
                    task.remaining_work -= progress
                    
                    # Update fatigue (simple linear increase)
                    worker.fatigue = min(worker.fatigue + (0.05 * hours), 1.0)
                    
                    # Check completion
                    if task.remaining_work <= 0:
                        task.remaining_work = 0
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = self.state.current_time
                        worker.current_task_id = None
                        worker.fatigue = max(worker.fatigue - 0.2, 0.0) # Rest bonus
                        events.append(f"Task {task.title} completed by {worker.name}")
            else:
                # Resting
                worker.fatigue = max(worker.fatigue - (0.1 * hours), 0.0)

        # 2. Check deadlines
        for task in self.state.tasks:
            if task.status in [TaskStatus.TODO, TaskStatus.IN_PROGRESS]:
                if self.state.current_time > task.deadline:
                    task.status = TaskStatus.FAILED
                    # If assigned, free the worker
                    if task.assigned_to:
                        worker = self._get_worker(task.assigned_to)
                        if worker:
                            worker.current_task_id = None
                    events.append(f"Task {task.title} failed due to deadline")

        return {
            "current_time": self.state.current_time,
            "events": events,
            "active_tasks_count": len([t for t in self.state.tasks if t.status == TaskStatus.IN_PROGRESS])
        }

    def get_state(self) -> Dict[str, Any]:
        return self.state.dict()

    def _get_task(self, task_id: str) -> Task | None:
        return next((t for t in self.state.tasks if t.id == task_id), None)

    def _get_worker(self, worker_id: str) -> Worker | None:
        return next((w for w in self.state.workers if w.id == worker_id), None)
