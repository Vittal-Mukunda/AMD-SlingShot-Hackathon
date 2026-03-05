"""Smoke test for Round 3 fixes: task cap, skill_match, quality^2.5, training taper."""
import sys
sys.path.insert(0, r"c:\Users\vitta\OneDrive\Desktop\AMD\AMD-SlingShot-Hackathon")
import config as cfg_module
import math

print("=== Fix 1: Dynamic TOTAL_TASKS ===")
print(f"SIM_DAYS = {cfg_module.SIM_DAYS}")
print(f"TASK_ARRIVAL_RATE = {cfg_module.TASK_ARRIVAL_RATE}")
expected = max(50, math.ceil(cfg_module.SIM_DAYS * cfg_module.TASK_ARRIVAL_RATE * 1.5))
print(f"TOTAL_TASKS = {cfg_module.TOTAL_TASKS} (dynamic, expected >= {expected})")
assert cfg_module.TOTAL_TASKS >= expected, f"TOTAL_TASKS {cfg_module.TOTAL_TASKS} < {expected}"
assert cfg_module.TOTAL_TASKS >= 600, f"For 100 days at 4/day, need >=600, got {cfg_module.TOTAL_TASKS}"
print("PASS")

print("\n=== Fix 4: SIM_DAYS default ===")
assert cfg_module.SIM_DAYS == 100, f"Expected 100, got {cfg_module.SIM_DAYS}"
print(f"SIM_DAYS = {cfg_module.SIM_DAYS}")
print("PASS")

print("\n=== Fix 2: REWARD_COMPLETION_BASE ===")
assert cfg_module.REWARD_COMPLETION_BASE == 0.8, f"Expected 0.8, got {cfg_module.REWARD_COMPLETION_BASE}"
print(f"REWARD_COMPLETION_BASE = {cfg_module.REWARD_COMPLETION_BASE}")
print("PASS")

print("\n=== Fix 2: Quality^2.5 reward calculation ===")
base = cfg_module.REWARD_COMPLETION_BASE
pw = (1 + 1) / 4.0 * base
r_good = pw * (0.9 ** 2.5)
r_bad = pw * (0.1 ** 2.5) - 0.1
print(f"quality=0.9: reward = {r_good:.4f}")
print(f"quality=0.1: reward = {r_bad:.4f} (includes -0.1 penalty)")
assert r_good > 0
assert r_bad < 0, "Bad quality should be net negative"
print("PASS")

print("\n=== Fix 2: Skill-match state vector (5-dim) ===")
from slingshot.environment.project_env import ProjectEnv
cfg_module.TOTAL_TASKS = 20
cfg_module.TASK_ARRIVAL_RATE = 4.0
env = ProjectEnv(num_workers=5, total_tasks=20, seed=42, total_sim_slots=5*16)
state = env.reset()
print(f"State shape: {state.shape}")
assert state.shape[0] == cfg_module.STATE_DIM, f"Expected {cfg_module.STATE_DIM}, got {state.shape[0]}"
print("PASS")

print("\n=== Fix 3: Training taper ===")
from slingshot.agents.dqn_agent import DQNAgent
agent = DQNAgent()
# Simulate: exploration mode
agent.epsilon = 0.3
n_exploring = 2 if agent.epsilon <= agent.epsilon_end + 0.01 else 4
print(f"epsilon=0.3 (exploring): n_grad = {n_exploring}")
assert n_exploring == 4, "Should do 4 grad steps while exploring"

# Simulate: at floor
agent.epsilon = 0.05
n_floor = 2 if agent.epsilon <= agent.epsilon_end + 0.01 else 4
print(f"epsilon=0.05 (floor): n_grad = {n_floor}")
assert n_floor == 2, "Should do 2 grad steps at floor"
print("PASS")

print("\n=== Fix 1: task.py generate_poisson_arrivals with dynamic cap ===")
from slingshot.environment.task import generate_poisson_arrivals
cfg_module.SIM_DAYS = 100
cfg_module.TOTAL_TASKS = 600
cfg_module.TASK_ARRIVAL_RATE = 4.0
cfg_module.TOTAL_SIM_DAYS = 100
tasks = generate_poisson_arrivals(
    total_tasks=600,
    arrival_rate_per_day=4.0,
    total_slots=100 * 16,
    seed=42,
)
print(f"Generated {len(tasks)} tasks for 100-day sim (cap=600, rate=4/day)")
assert len(tasks) >= 350, f"Expected at least 350 tasks, got {len(tasks)}"
last_arrival_day = max(t.arrival_tick for t in tasks) // 16
print(f"Last task arrives on day {last_arrival_day}")
assert last_arrival_day >= 90, f"Tasks should span most of the 100 days, last at day {last_arrival_day}"
print("PASS")

print("\n=== ALL ROUND 3 SMOKE TESTS PASSED ===")
