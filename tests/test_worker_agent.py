from app.agents.worker import WorkerAgent
from app.db.mock_db import global_state
from app.db.models import Worker


def test_worker_profile_and_busy():
    # create a temporary worker in the global_state
    w = Worker(id="tw", name="Test Worker", skill_level=5.0, efficiency=1.0)
    global_state.workers.append(w)
    try:
        agent = WorkerAgent("tw")
        p = agent.profile()
        assert p["id"] == "tw"
        assert p["name"] == "Test Worker"
        assert agent.is_busy() is False

        # mark busy
        w.current_task_id = "task1"
        assert agent.is_busy() is True
    finally:
        # cleanup
        global_state.workers[:] = [x for x in global_state.workers if x.id != "tw"]
