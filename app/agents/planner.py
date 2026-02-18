from typing import Any, Dict
from app.services.task_service import TaskService
from app.services.simulation_service import SimulationService
from app.agents.context import ContextAgent


class PlannerAgent:
    """Agent that observes state via a `ContextAgent`, uses a policy to decide,
    and executes actions through the provided services.

    The `policy` object is expected to implement `select_action(observation)`.

    Expected action shapes returned by `select_action`:
      - Assign: {"type": "assign", "worker_id": str, "task_id": str}
      - Wait:   {"type": "wait", "hours": float}
      - No-op:  {"type": "noop"} (or any other unknown type will be reported)

    Example:
        policy = MyPolicy()
        planner = PlannerAgent(policy, task_service, simulation, context)
        obs = planner.observe()
        action = planner.decide()
        result = planner.act(action)
    """

    def __init__(self, policy: Any, task_service: TaskService, simulation: SimulationService, context: ContextAgent):
        self.policy = policy
        self.task_service = task_service
        self.simulation = simulation
        self.context = context

    def observe(self) -> Dict[str, Any]:
        """Return the current observation (simulation state) from the context."""
        return self.context.get_state()

    def decide(self) -> Dict[str, Any]:
        """Use the policy to select an action given the current observation.

        Returns the action dict produced by `policy.select_action(obs)`.
        If the policy raises an exception, returns {"error": str(e)}.
        """
        obs = self.observe()
        try:
            action = self.policy.select_action(obs)
            return action
        except Exception as e:
            return {"error": str(e)}

    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action dict.

        - Assign actions call `TaskService.assign_task(..., automated=True)`
        - Wait actions call `SimulationService.step(hours)`
        - Unknown actions return {"error": "unknown_action"}

        The method is exception-safe and will return {"error": str(e)}
        on unexpected failures.
        """
        try:
            if not isinstance(action, dict):
                return {"error": "action_must_be_dict"}

            action_type = action.get("type")

            if action_type == "assign":
                worker_id = action.get("worker_id")
                task_id = action.get("task_id")
                if not worker_id or not task_id:
                    return {"error": "missing_worker_or_task_id"}
                return self.task_service.assign_task(worker_id, task_id, automated=True)

            if action_type == "wait":
                hours = action.get("hours", 1.0)
                try:
                    hours = float(hours)
                except Exception:
                    return {"error": "invalid_hours"}
                return self.simulation.step(hours=hours)

            return {"error": "unknown_action"}

        except Exception as e:
            return {"error": str(e)}
