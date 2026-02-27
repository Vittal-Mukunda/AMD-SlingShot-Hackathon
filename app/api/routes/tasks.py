from fastapi import APIRouter
from app.services.task_service import TaskService

router = APIRouter()
service = TaskService()

@router.get("/tasks")
def get_tasks():
    return service.get_all_tasks()

@router.get("/state")
def get_state():
    return service.get_state()

@router.post("/simulate_step")
def simulate_step(hours: float = 1.0):
    from app.services.simulation_service import SimulationService
    sim_service = SimulationService()
    return sim_service.step(hours)

@router.get("/predict_risk")
def predict_risk():
    return service.predict_deadline_risk()

@router.get("/optimize")
def optimize():
    return service.optimize_resource_allocation()
