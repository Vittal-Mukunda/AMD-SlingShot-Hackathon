import socketio
import requests
import time

sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")
    
@sio.event
def tick_update(data):
    if data["phase"] == 3:
        if data.get("last_assignment") and data["last_assignment"]["worker_id"] != "—":
            print("DQN Assign:", data["last_assignment"])
        else:
            pass # print("DQN Tick no assignment")

@sio.event
def phase_transition(data):
    print("Phase transition:", data["new_phase"])

@sio.event
def phase2_ready(data):
    print("Phase 2 ready!", len(data.get("baseline_results_snapshot", {})))
    
@sio.event
def gantt_block(data):
    if data["policy"] == "DQN":
        print("DQN Gantt:", data)

@sio.event
def simulation_complete(data):
    print("Simulation complete! DQN makespan delta:", data["final_metrics"]["dqn_vs_best_makespan_delta"])
    print("DQN throughput:", data["final_metrics"]["dqn_throughput"])
    sio.disconnect()

@sio.event
def simulation_error(data):
    print("Error:", data)

sio.connect("http://localhost:8000")

resp = requests.post("http://localhost:8000/api/initialize", json={
    "days_phase1": 5,
    "days_phase2": 5,
    "worker_mode": "auto",
    "worker_seed": 42,
    "num_workers": 3,
    "task_count": 20,
    "seed": 42
})
print("Init response:", resp.json())
sio.wait()
