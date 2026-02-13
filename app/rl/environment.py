from typing import Dict, Any, List, Tuple
import numpy as np
from app.services.simulation_service import SimulationService
from app.services.task_service import TaskService
from app.db.models import TaskStatus

class RLEnvironment:
    def __init__(self):
        self.simulation = SimulationService()
        self.task_service = TaskService()
        
    def reset(self) -> Dict[str, Any]:
        """Resets the environment (in this hackathon context, just returns current state)"""
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        Returns a simplified state vector/dict for the agent.
        Includes:
        - Worker states (skill, fatigue, is_busy)
        - Task states (remaining_work, deadline, priority)
        - Global time
        """
        state = self.simulation.get_state()
        
        # Flattened observation builder
        obs = {
            "time": state["current_time"],
            "workers": [],
            "tasks": []
        }
        
        for w in state["workers"]:
            obs["workers"].append({
                "id": w["id"],
                "skill": w["skill_level"],
                "fatigue": w["fatigue"],
                "is_busy": 1 if w["current_task_id"] else 0
            })
            
        for t in state["tasks"]:
            if t["status"] in [TaskStatus.TODO, TaskStatus.IN_PROGRESS]:
                 obs["tasks"].append({
                    "id": t["id"],
                    "priority": t["priority"],
                    "remaining": t["remaining_work"],
                    "deadline_gap": max(t["deadline"] - state["current_time"], 0)
                })
        
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes an action in the environment.
        Action format: {"type": "assign", "worker_id": "w1", "task_id": "t1"}
        OR {"type": "wait", "hours": 1.0}
        """
        reward = 0.0
        done = False
        info = {}
        
        if action["type"] == "assign":
            # Attempt assignment
            result = self.task_service.assign_task(
                worker_id=action.get("worker_id"), 
                task_id=action.get("task_id"), 
                automated=True
            )
            
            if "error" in result:
                reward = -10.0 # Penalty for invalid action
                info["error"] = result["error"]
            else:
                reward = 10.0 # Reward for successful assignment
                
        elif action["type"] == "wait":
             # Advance simulation
             hours = action.get("hours", 1.0)
             sim_result = self.simulation.step(hours)
             
             # Calculate reward based on simulation events
             # Example: -1 per failed task, +5 per completed task
             for event in sim_result["events"]:
                 if "completed" in event:
                     reward += 5.0
                 if "failed" in event:
                     reward -= 20.0
                     
             # Small time penalty to encourage speed
             reward -= (0.1 * hours)
             
             info["events"] = sim_result["events"]

        # Check termination (all tasks done or failed)
        active_tasks = [t for t in self.simulation.state.tasks if t.status in [TaskStatus.TODO, TaskStatus.IN_PROGRESS]]
        if not active_tasks:
            done = True
            
        next_state = self.get_observation()
        return next_state, reward, done, info

    def action_space_sample(self):
        """Helper to return a random valid action (stub)"""
        pass
