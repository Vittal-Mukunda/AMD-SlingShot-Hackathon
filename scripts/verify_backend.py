from app.services.simulation_service import SimulationService
from app.db.mock_db import global_state
from app.rl.environment import RLEnvironment
from app.services.task_service import TaskService
import json

def run_checks():
    print('--- Starting Verification ---')
    
    # 1. Check Initial State
    print(f'Initial Tasks: {len(global_state.tasks)}')
    print(f'Initial Workers: {len(global_state.workers)}')
    
    # 2. Assign Task
    task_service = TaskService()
    t1 = global_state.tasks[0]
    w1 = global_state.workers[0]
    
    print(f'Assigning {t1.id} to {w1.id}...')
    res = task_service.assign_task(w1.id, t1.id)
    print(f'Assignment result: {res.get("message")}')
    
    if t1.assigned_to != w1.id:
        print('ERROR: Assignment failed in model')
        return

    # 3. Step Simulation
    sim = SimulationService()
    print('Stepping simulation by 2 hours...')
    step_res = sim.step(2.0)
    print(f'Events: {step_res["events"]}')
    
    # Check progress
    t1_after = task_service._get_task(t1.id)
    print(f'Task Remaining Work: {t1_after.remaining_work} (Starts at 4.0)')
    if t1_after.remaining_work >= 4.0:
        print('ERROR: Work did not decrease')

    # 4. Check RL Env
    env = RLEnvironment()
    obs = env.get_observation()
    print(f'RL Observation keys: {list(obs.keys())}')
    
    if len(obs['workers']) != 3:
        print('ERROR: Incorrect worker count in observation')
        
    print('--- Verification Complete ---')

if __name__ == '__main__':
    run_checks()
