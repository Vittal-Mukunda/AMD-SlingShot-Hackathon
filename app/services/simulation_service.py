import sys
import os
from typing import List, Dict, Any
from app.db.models import Task, Worker, SimulationState, TaskStatus, TaskPriority
from app.core.logging import setup_logger

# Import the core RL environment
try:
    from slingshot.environment.project_env import ProjectEnv
    from slingshot.core.settings import config
except ImportError:
    ProjectEnv = None
    config = None

logger = setup_logger("SimulationService")

class SimulationService:
    """
    Adapter bridging the FastAPI backend to the RL ProjectEnv stochastic simulation.
    """
    _instance = None
    
    def __new__(cls):
        # Singleton pattern to ensure the simulation state persists across API calls
        if cls._instance is None:
            cls._instance = super(SimulationService, cls).__new__(cls)
            num_w = config.NUM_WORKERS if config else 5
            num_t = config.TOTAL_TASKS if config else 200
            cls._instance.env = ProjectEnv(num_workers=num_w, total_tasks=num_t, seed=42) if ProjectEnv else None
            if cls._instance.env:
                cls._instance.env.reset()
                logger.info("Initialized RL ProjectEnv Simulation")
            else:
                logger.warning("ProjectEnv not found - running in fallback mode")
        return cls._instance

    def step(self, action_index: int = None, hours: float = 1.0) -> Dict[str, Any]:
        """
        Advances the simulation. If action_index is provided, executes an RL action.
        Otherwise (fallback), just ticks the time.
        """
        events = []
        reward = 0.0
        done = False
        info = {}
        
        if self.env:
            if action_index is not None:
                _, reward, done, info = self.env.step(action_index)
                events.append(f"Executed RL action {action_index}")
            else:
                # Fallback tick without specific action (e.g. idle step)
                # We defer by default if no action provided but forcing a step
                valid = self.env.get_valid_actions()
                if valid:
                    # pick a defer action (100+) manually or random valid
                    defer_actions = [a for a in valid if a >= self.env.num_tasks * self.env.num_workers]
                    idle_act = defer_actions[0] if defer_actions else valid[0]
                    _, reward, done, info = self.env.step(idle_act)
                    events.append("Executed idle/defer tick")
            
            logger.info("Simulation Step", extra={"extra_data": {
                "time": self.env.clock.tick * config.SLOT_HOURS if config else 0, 
                "reward": reward,
                "done": done
            }})
            return {
                "current_time": self.env.clock.tick * config.SLOT_HOURS if config else 0,
                "events": events,
                "reward": reward,
                "done": done,
                "info": info
            }
        else:
            return {"error": "ProjectEnv not initialized"}

    def get_state(self) -> Dict[str, Any]:
        """
        Serializes the RL environment into the Pydantic API models
        """
        if not self.env:
            return {"error": "ProjectEnv not loaded"}
            
        api_workers = []
        for w in self.env.workers:
            api_workers.append(Worker(
                id=f"w{w.worker_id}",
                name=f"Worker {w.worker_id}",
                true_skill=w.true_skill,
                fatigue=w.fatigue,
                availability=w.availability,
                current_task_id=f"t{w.assigned_tasks[0]}" if w.assigned_tasks else None
            ))
            
        api_tasks = []
        for t in self.env.tasks:
            status = TaskStatus.TODO
            if t.is_completed: status = TaskStatus.COMPLETED
            elif t.is_failed: status = TaskStatus.FAILED
            elif t.assigned_worker is not None: status = TaskStatus.IN_PROGRESS
            
            # Map RL priority (0-3) to TaskPriority
            pri = TaskPriority(t.priority) if isinstance(t.priority, int) else t.priority
            
            api_tasks.append(Task(
                id=f"t{t.task_id}",
                title=f"Task {t.task_id}",
                complexity=t.complexity,
                deadline=t.deadline_slot * config.SLOT_HOURS if config else 0,
                priority=pri,
                status=status,
                assigned_to=f"w{t.assigned_worker}" if t.assigned_worker is not None else None
            ))
            
        return SimulationState(
            tasks=api_tasks,
            workers=api_workers,
            current_time=self.env.clock.tick * config.SLOT_HOURS if config else 0
        ).model_dump()
