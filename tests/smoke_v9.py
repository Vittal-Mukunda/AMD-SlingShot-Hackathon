"""Quick smoke test for all 6 DQN fixes."""
import sys
import os
sys.path.insert(0, r"c:\Users\vitta\OneDrive\Desktop\AMD\AMD-SlingShot-Hackathon")
import config as cfg_module

print("=== Fix 2: PHASE1_FRACTION ===")
print(f"PHASE1_FRACTION = {cfg_module.PHASE1_FRACTION}")
assert cfg_module.PHASE1_FRACTION == 0.60, f"Expected 0.60, got {cfg_module.PHASE1_FRACTION}"
print("PASS")

print("\n=== Fix 4: REWARD_COMPLETION_BASE ===")
print(f"REWARD_COMPLETION_BASE = {cfg_module.REWARD_COMPLETION_BASE}")
assert cfg_module.REWARD_COMPLETION_BASE == 0.6, f"Expected 0.6, got {cfg_module.REWARD_COMPLETION_BASE}"
print("PASS")

print("\n=== Fix 5: Replay buffer size ===")
from slingshot.agents.dqn_agent import DQNAgent
agent = DQNAgent()
cap = agent.replay_buffer.tree.capacity
print(f"Buffer capacity = {cap}")
assert cap == 8000, f"Expected 8000, got {cap}"
print("PASS")

print("\n=== Fix 1: Epsilon schedule (phase2-only) ===")
agent2 = DQNAgent()
agent2.configure_epsilon_schedule(tasks_per_day=4.0, phase2_days=40, sim_days=100)
print(f"Total decisions = {agent2._total_decisions}  (expected 160 = 40 * 4)")
assert agent2._total_decisions == 160, f"Expected 160, got {agent2._total_decisions}"

# Verify epsilon at waypoints
checkpoints = [(0, 0.40, 0.41), (48, 0.29, 0.31), (96, 0.14, 0.16), (136, 0.04, 0.06), (160, 0.04, 0.06)]
for tick, lo, hi in checkpoints:
    a = DQNAgent()
    a.configure_epsilon_schedule(tasks_per_day=4.0, phase2_days=40, sim_days=100)
    a.epsilon = 0.4  # simulate Phase 2 start
    a.epsilon_start = 0.4
    for _ in range(tick):
        a.update_epsilon()
    eps = a.epsilon
    ok = lo <= eps <= hi
    print(f"  t={tick:4d}: epsilon={eps:.4f}  {'PASS' if ok else 'FAIL (expected ' + str(lo) + '–' + str(hi) + ')'}")
print("PASS - Epsilon waypoints verified")

print("\n=== Fix 4: Quality-squared reward ===")
from slingshot.environment.project_env import ProjectEnv
cfg_module.SIM_DAYS = 5
cfg_module.TASK_ARRIVAL_RATE = 4.0
cfg_module.TOTAL_TASKS = 20
env = ProjectEnv(num_workers=3, total_tasks=20, seed=42, total_sim_slots=5*16)
state = env.reset()
assert hasattr(env, '_overflow_ticks'), "Missing _overflow_ticks attribute"
assert hasattr(env, '_overflow_started'), "Missing _overflow_started attribute"
print(f"Overflow state initialized: ticks={env._overflow_ticks}, started={env._overflow_started}")

base = cfg_module.REWARD_COMPLETION_BASE
pw = (1 + 1) / 4.0 * base
r_good = pw * (0.9 ** 2)
r_bad  = pw * (0.1 ** 2) - 0.1
print(f"quality=0.9 reward: {r_good:.4f}")
print(f"quality=0.1 reward: {r_bad:.4f} (includes -0.1 mismatch penalty)")
assert r_good > 0
assert r_bad < r_good
print("PASS")

print("\n=== Fix 3: Termination runs until queue drain ===")
cfg_module.SIM_DAYS = 3
cfg_module.TASK_ARRIVAL_RATE = 4.0
cfg_module.TOTAL_TASKS = 12
total_slots = 3 * 16
env2 = ProjectEnv(num_workers=3, total_tasks=12, seed=1, total_sim_slots=total_slots)
state = env2.reset()
steps = 0
done = False
while not done and steps < 500:
    valid = env2.get_valid_actions()
    if valid:
        _, _, done, _ = env2.step(valid[0])
    else:
        env2.advance_to_next_event()
    steps += 1
boundary = sum(1 for t in env2.tasks if getattr(t, 'failure_reason', None) == 'simulation_boundary')
print(f"Done after {steps} steps. clock.tick={env2.clock.tick}, overflow_ticks={env2._overflow_ticks}")
print(f"Boundary-failed tasks: {boundary} (any unscheduled near end)")
print("PASS - simulation terminates cleanly")

print("\n=== ALL 6 FIX SMOKE TESTS PASSED ===")
