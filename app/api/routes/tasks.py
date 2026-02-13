from fastapi import APIRouter
from app.services.task_service import TaskService

router = APIRouter()
service = TaskService()

@router.get("/tasks")
def get_tasks():
    return service.get_all_tasks()
