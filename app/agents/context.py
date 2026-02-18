from typing import List, Dict, Any
from app.services.simulation_service import SimulationService
from app.services.task_service import TaskService


class ContextAgent:
	"""Provides a lightweight context interface to the simulation and task services.

	Designed for planners/workers to query state, inspect backlog, and record events.
	"""

	def __init__(self, simulation: SimulationService, task_service: TaskService):
		"""Initialize with the running `SimulationService` and `TaskService`.

		Args:
			simulation: SimulationService instance providing `get_state()`.
			task_service: TaskService instance (kept for potential extensions).
		"""
		self.simulation = simulation
		self.task_service = task_service
		self.history: List[Dict[str, Any]] = []

	def get_state(self) -> Dict[str, Any]:
		"""Return the current simulation state as a dict, or an empty dict if unavailable."""
		state = self.simulation.get_state()
		if state is None:
			return {}
		if not isinstance(state, dict):
			try:
				return dict(state)
			except Exception:
				return {}
		return state

	def get_backlog(self) -> List[Dict[str, Any]]:
		"""Return tasks whose status is TODO (uses same casing as services: 'todo')."""
		state = self.get_state()
		tasks = state.get("tasks") if isinstance(state, dict) else []
		if not isinstance(tasks, list):
			return []
		return [t for t in tasks if isinstance(t, dict) and t.get("status") == "todo"]

	def record_event(self, event: Dict[str, Any]) -> None:
		"""Record an event to the internal history. Expects a dict."""
		if not isinstance(event, dict):
			raise TypeError("event must be a dict")
		self.history.append(event)

	def get_history(self) -> List[Dict[str, Any]]:
		"""Return a shallow copy of the recorded history."""
		return self.history.copy()

	def summary(self) -> Dict[str, Any]:
		"""Return a small convenience summary of the current context.

		Includes backlog length, active task count, recent events, and current time.
		"""
		state = self.get_state()
		tasks = state.get("tasks", []) if isinstance(state, dict) else []
		backlog = [t for t in tasks if isinstance(t, dict) and t.get("status") == "todo"]
		active = [t for t in tasks if isinstance(t, dict) and t.get("status") == "in_progress"]
		return {
			"backlog_length": len(backlog),
			"active_tasks": len(active),
			"last_events": self.history[-5:],
			"current_time": state.get("current_time") if isinstance(state, dict) else None,
		}

