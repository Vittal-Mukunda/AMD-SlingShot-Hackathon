import pytest

from app.agents.planner import PlannerAgent


class FakePolicy:
    def __init__(self, action):
        self.action = action

    def select_action(self, obs):
        return self.action


class FakeTaskService:
    def __init__(self):
        self.calls = []

    def assign_task(self, worker_id, task_id, automated=False):
        self.calls.append((worker_id, task_id, automated))
        return {"message": "assigned", "worker_id": worker_id, "task_id": task_id, "automated": automated}


class FakeSimulation:
    def __init__(self):
        self.called = []

    def step(self, hours=1.0):
        self.called.append(float(hours))
        return {"current_time": float(hours), "events": [], "active_tasks_count": 0}


class FakeContext:
    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state


def test_planner_assign_action():
    policy = FakePolicy({"type": "assign", "worker_id": "w1", "task_id": "t1"})
    task_service = FakeTaskService()
    sim = FakeSimulation()
    ctx = FakeContext({"tasks": []})
    planner = PlannerAgent(policy, task_service, sim, ctx)

    action = planner.decide()
    assert action["type"] == "assign"

    res = planner.act(action)
    assert res.get("message") == "assigned"
    assert task_service.calls == [("w1", "t1", True)]


def test_planner_wait_action():
    policy = FakePolicy({"type": "wait", "hours": 2})
    task_service = FakeTaskService()
    sim = FakeSimulation()
    ctx = FakeContext({"tasks": []})
    planner = PlannerAgent(policy, task_service, sim, ctx)

    action = planner.decide()
    res = planner.act(action)
    assert res.get("current_time") == 2.0
    assert sim.called == [2.0]


def test_planner_unknown_action():
    policy = FakePolicy({"type": "dance"})
    planner = PlannerAgent(policy, FakeTaskService(), FakeSimulation(), FakeContext({}))
    res = planner.act({"type": "dance"})
    assert res.get("error") == "unknown_action"


def test_policy_exception_decide():
    class BadPolicy:
        def select_action(self, obs):
            raise RuntimeError("boom")

    planner = PlannerAgent(BadPolicy(), FakeTaskService(), FakeSimulation(), FakeContext({}))
    res = planner.decide()
    assert "error" in res
