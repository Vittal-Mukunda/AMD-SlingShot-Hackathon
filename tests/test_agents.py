import pytest

from app.agents.context import ContextAgent


class FakeSim:
    def __init__(self, state):
        self._state = state

    def get_state(self):
        return self._state


class FakeTaskService:
    pass


def test_record_event_stores_event():
    ctx = ContextAgent(FakeSim({"tasks": []}), FakeTaskService())
    event = {"type": "note", "msg": "hello"}
    ctx.record_event(event)
    assert ctx.get_history() == [event]

    # ensure get_history returns a shallow copy
    hist = ctx.get_history()
    hist.append({"extra": True})
    assert len(ctx.get_history()) == 1


def test_get_backlog_filters_todo():
    tasks = [
        {"id": "1", "status": "todo"},
        {"id": "2", "status": "in_progress"},
        {"id": "3", "status": "todo"},
        {"id": "4", "status": "completed"},
    ]
    ctx = ContextAgent(FakeSim({"tasks": tasks}), FakeTaskService())
    backlog = ctx.get_backlog()
    assert isinstance(backlog, list)
    assert len(backlog) == 2
    assert {t["id"] for t in backlog} == {"1", "3"}
