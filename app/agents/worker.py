import typing
from typing import Dict, Any
from app.db.mock_db import global_state


class WorkerAgent:
    """Lightweight convenience wrapper around a worker in the repo's `global_state`.

    This class provides read-only accessors to mirror the important fields from
    `app.db.models.Worker` without duplicating state-management logic handled
    by `TaskService` or `SimulationService`.
    """

    def __init__(self, worker_id: str):
        """Create a WorkerAgent for the worker with the given `worker_id`."""
        self.worker_id = worker_id

    def _worker(self) -> typing.Optional[Dict[str, Any]]:
        """Return the worker object (as a dict) from `global_state.workers` matching id.

        Returns None if the worker is not found.
        """
        w = next((w for w in global_state.workers if w.id == self.worker_id), None)
        if w is None:
            return None
        # Return a dict representation to keep callers decoupled from pydantic models
        try:
            return w.dict()
        except Exception:
            # Fallback: expose attributes manually
            return {
                "id": getattr(w, "id", None),
                "name": getattr(w, "name", None),
                "skill_level": getattr(w, "skill_level", None),
                "efficiency": getattr(w, "efficiency", None),
                "fatigue": getattr(w, "fatigue", None),
                "current_task_id": getattr(w, "current_task_id", None),
            }

    def profile(self) -> Dict[str, Any]:
        """Return a simple profile dict mirroring the `Worker` fields.

        Returned keys: `id`, `name`, `skill_level`, `efficiency`, `fatigue`, `current_task_id`.
        If the worker does not exist, returns an empty dict.
        """
        w = self._worker()
        if not w:
            return {}
        return {
            "id": w.get("id"),
            "name": w.get("name"),
            "skill_level": w.get("skill_level"),
            "efficiency": w.get("efficiency"),
            "fatigue": w.get("fatigue"),
            "current_task_id": w.get("current_task_id"),
        }

    def is_busy(self) -> bool:
        """Return True if the worker currently has a `current_task_id` assigned."""
        w = self._worker()
        if not w:
            return False
        return bool(w.get("current_task_id"))


# Unit-test suggestion (to place in tests/test_workers.py):
#
# def test_worker_profile_and_busy(fake_global_state_fixture):
#     # Setup: create a fake worker in global_state.workers with current_task_id set
#     agent = WorkerAgent(worker_id)
#     assert agent.profile()["id"] == worker_id
#     assert agent.is_busy() is True
