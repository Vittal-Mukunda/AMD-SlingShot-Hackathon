"""Smoke test: verify the 500-task cap is truly killed and all fixes work end-to-end."""
import sys
sys.path.insert(0, r"c:\Users\vitta\OneDrive\Desktop\AMD\AMD-SlingShot-Hackathon")
import math
import config as cfg_module

print("=" * 60)
print("SMOKE TEST: Verifying the real task cap fix (v10b)")
print("=" * 60)

# 1. Verify config defaults
print(f"\n[1] Config defaults:")
print(f"  SIM_DAYS = {cfg_module.SIM_DAYS}")
print(f"  TASK_ARRIVAL_RATE = {cfg_module.TASK_ARRIVAL_RATE}")
print(f"  TOTAL_TASKS = {cfg_module.TOTAL_TASKS}")
assert cfg_module.SIM_DAYS == 100, f"SIM_DAYS should be 100, got {cfg_module.SIM_DAYS}"
assert cfg_module.TOTAL_TASKS >= 600, f"TOTAL_TASKS should be >=600, got {cfg_module.TOTAL_TASKS}"
print("  PASS")

# 2. Generate tasks for 100-day sim
print(f"\n[2] generate_poisson_arrivals (100-day sim, rate=4, cap=600):")
from slingshot.environment.task import generate_poisson_arrivals
tasks = generate_poisson_arrivals(
    total_tasks=600,
    arrival_rate_per_day=4.0,
    total_slots=100 * 16,
    seed=42,
)
n_tasks = len(tasks)
last_day = max(t.arrival_tick for t in tasks) // 16
min_expected = int(100 * 4.0 * 0.8)  # Poisson variance allows 0.8x
print(f"  Generated {n_tasks} tasks (min expected={min_expected})")
print(f"  Last arrival day: {last_day}")
assert n_tasks >= min_expected, f"Need >={min_expected} tasks, got {n_tasks}"
assert last_day >= 90, f"Tasks should span most of 100 days, last at {last_day}"
print("  PASS")

# 3. Test that 2x generation buffer works
print(f"\n[3] Generation with 2x headroom:")
tasks2 = generate_poisson_arrivals(
    total_tasks=600,
    arrival_rate_per_day=4.0,
    total_slots=100 * 16,
    seed=99,  # different seed
)
print(f"  Seed 99: {len(tasks2)} tasks, last day: {max(t.arrival_tick for t in tasks2) // 16}")
assert len(tasks2) >= min_expected
# Test caps work 
tasks3 = generate_poisson_arrivals(
    total_tasks=600,
    arrival_rate_per_day=4.0,
    total_slots=100 * 16,
    seed=7,
)
print(f"  Seed 7:  {len(tasks3)} tasks, last day: {max(t.arrival_tick for t in tasks3) // 16}")
assert len(tasks3) >= min_expected
assert len(tasks3) <= 600, f"Should not exceed cap of 600, got {len(tasks3)}"
print("  PASS")

# 4. ProjectEnv with dynamic cap
print(f"\n[4] ProjectEnv with dynamic cap:")
cfg_module.TOTAL_TASKS = 600
cfg_module.SIM_DAYS = 100
cfg_module.TOTAL_SIM_DAYS = 100
from slingshot.environment.project_env import ProjectEnv
env = ProjectEnv(num_workers=5, total_tasks=600, seed=42, total_sim_slots=100*16)
state = env.reset()
print(f"  env.tasks = {len(env.tasks)}")
assert len(env.tasks) >= min_expected
print("  PASS")

# 5. Verify _make_env would use cfg_module.TOTAL_TASKS
print(f"\n[5] _make_env parameter check:")
total_tasks_param = getattr(cfg_module, 'TOTAL_TASKS', 500)
print(f"  getattr(cfg_module, 'TOTAL_TASKS', 500) = {total_tasks_param}")
assert total_tasks_param == 600, f"Would use {total_tasks_param}, not 600!"
print("  PASS")

# 6. Check main.py default
print(f"\n[6] SimConfig defaults:")
import re
with open(r"c:\Users\vitta\OneDrive\Desktop\AMD\AMD-SlingShot-Hackathon\backend\main.py", "r") as f:
    content = f.read()
match = re.search(r'task_count:\s*int\s*=\s*(\d+)', content)
default_val = int(match.group(1)) if match else 0
print(f"  task_count default = {default_val}")
assert default_val >= 600, f"Expected >=600, got {default_val}"
print("  PASS")

# 7. Adaptive quality tracker
print(f"\n[7] Adaptive quality tracking:")
assert hasattr(env, '_quality_window')
assert hasattr(env, '_quality_boost_remaining')
assert hasattr(env, '_decision_count')
print("  All state vars present")
print("  PASS")

# 8. End-to-end 10-day sim
print(f"\n[8] E2E sim (10 days):")
cfg_module.SIM_DAYS = 10
cfg_module.TOTAL_SIM_DAYS = 10
cfg_module.TOTAL_TASKS = max(50, math.ceil(10 * 4.0 * 1.5))
env2 = ProjectEnv(num_workers=5, total_tasks=cfg_module.TOTAL_TASKS, seed=42, total_sim_slots=10*16)
state = env2.reset()
print(f"  Tasks: {len(env2.tasks)}")
steps = 0
done = False
while not done and steps < 2000:
    valid = env2.get_valid_actions()
    if valid:
        _, _, done, _ = env2.step(valid[0])
    else:
        env2.advance_to_next_event()
    steps += 1
final_day = env2.clock.tick // 16
print(f"  Final day: {final_day}, completed: {len(env2.completed_tasks)}")
assert final_day >= 9, f"Should run ~10 days, only ran {final_day}"
print("  PASS")

print("\n" + "=" * 60)
print("ALL TASK-CAP V10B SMOKE TESTS PASSED")
print("=" * 60)
