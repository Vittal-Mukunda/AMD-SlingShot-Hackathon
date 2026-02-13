from pydantic import BaseModel
from typing import Optional, List

class AssignTaskRequest(BaseModel):
    worker_id: str
    task_id: str

class TaskResponse(BaseModel):
    id: str
    title: str
    status: str
    assigned_to: Optional[str]
    priority: str
    deadline: float
    remaining_work: float

class WorkerResponse(BaseModel):
    id: str
    name: str
    skill_level: float
    fatigue: float
    current_task_id: Optional[str]

class StateResponse(BaseModel):
    tasks: List[TaskResponse]
    workers: List[WorkerResponse]
    current_time: float
