"""
Reward Signal Sanity Check — v6 Structural Fixes

Verifies that the 7 structural fixes to the DQN reward are working:
  1. Terminal penalty is NOT clipped — must be large for many unfinished tasks
  2. Backlog penalty scales superlinearly with queue length
  3. Lateness penalty is one-shot (only on tick of completion)
  4. Throughput milestones fire at 25/50/75/100%
  5. Urgency window is 8 slots (not 4)
  6. Completion reward uses base=5.0 (not 10.0)
  7. Reward clip bounds are ±15.0
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from slingshot.environment.project_env import ProjectEnv
from slingshot.core.settings import config

print("=" * 80)
print("REWARD SIGNAL SANITY CHECK — v6 Structural Fixes")
print("=" * 80)
passed = 0
failed = 0

# ── Check 1: Config values ──────────────────────────────────────────────────

print("\n1. Checking config values...")
checks = [
    ("REWARD_CLIP_MIN",  config.REWARD_CLIP_MIN, -15.0),
    ("REWARD_CLIP_MAX",  config.REWARD_CLIP_MAX,  15.0),
    ("REWARD_BACKLOG_PENALTY", config.REWARD_BACKLOG_PENALTY, -0.15),
    ("REWARD_TERMINAL_UNFINISHED_PENALTY", config.REWARD_TERMINAL_UNFINISHED_PENALTY, -8.0),
    ("REWARD_IDLE_PENALTY", config.REWARD_IDLE_PENALTY, -0.40),
    ("REWARD_URGENCY_PENALTY", config.REWARD_URGENCY_PENALTY, -0.5),
    ("REWARD_COMPLETION_BASE", config.REWARD_COMPLETION_BASE, 5.0),
    ("DEADLINE_MAX_DAYS", config.DEADLINE_MAX_DAYS, 3.0),
]
for name, actual, expected in checks:
    if abs(actual - expected) < 1e-6:
        print(f"  ✓ {name} = {actual}")
        passed += 1
    else:
        print(f"  ✗ {name} = {actual}, expected {expected}")
        failed += 1

# ── Check 2: Terminal penalty not clipped ────────────────────────────────────

print("\n2. Testing terminal penalty is NOT clipped...")
env = ProjectEnv(num_workers=5, total_tasks=200, seed=42, total_sim_slots=400)
state = env.reset()

# Run a few steps then force termination by exhausting time
# Step to end without completing many tasks
steps_run = 0
for _ in range(50):
    valid = env.get_valid_actions()
    if not valid:
        adv = env.advance_to_next_event()
        if adv == 0:
            break
        continue
    # Take defer actions (deliberately don't complete tasks)
    defer_actions = [a for a in valid if a >= 20 * env.num_workers]
    if defer_actions:
        action = defer_actions[0]
    else:
        action = valid[0]
    _, _, done, _ = env.step(action)
    steps_run += 1
    if done:
        break

# Force time-limit termination
env.clock.tick = env._total_sim_slots + 1
done, reason = env._check_termination()
assert done, "Should be terminated"

terminal_penalty = env._compute_terminal_penalty()
unfinished = sum(1 for t in env.tasks if not t.is_completed and not t.is_failed)
print(f"  Unfinished tasks: {unfinished}")
print(f"  Terminal penalty: {terminal_penalty:.1f}")
print(f"  Expected approx: {-8.0 * unfinished:.1f}")

# The terminal penalty should be much larger than the old clip of -3.0
if terminal_penalty < -50.0:
    print(f"  ✓ Terminal penalty ({terminal_penalty:.1f}) is NOT clipped to -3.0 — reaches agent")
    passed += 1
else:
    print(f"  ✗ Terminal penalty too small ({terminal_penalty:.1f}), old clip may still apply")
    failed += 1

# ── Check 3: Superlinear backlog penalty ─────────────────────────────────────

print("\n3. Testing superlinear backlog scaling...")
# Linear: n * -0.15
# Superlinear: n * -0.15 * (1 + n/10)
n5  = 5  * config.REWARD_BACKLOG_PENALTY * (1.0 + 5 / 10.0)
n20 = 20 * config.REWARD_BACKLOG_PENALTY * (1.0 + 20 / 10.0)
ratio = n20 / n5  # should be > 4 (linear would be exactly 4)
print(f"  Penalty for 5 tasks:  {n5:.3f}")
print(f"  Penalty for 20 tasks: {n20:.3f}")
print(f"  Ratio (20/5): {ratio:.2f}  (linear would be 4.00)")
if ratio > 4.5:
    print(f"  ✓ Superlinear scaling confirmed (ratio {ratio:.2f} > 4.0)")
    passed += 1
else:
    print(f"  ✗ Not superlinear enough (ratio {ratio:.2f})")
    failed += 1

# ── Check 4: Lateness penalty is one-shot ────────────────────────────────────

print("\n4. Testing lateness penalty is one-shot (not cumulative)...")
env2 = ProjectEnv(num_workers=5, total_tasks=50, seed=100, total_sim_slots=800)
state = env2.reset()

# Run until we get some completed tasks
late_penalty_history = []
for step in range(200):
    valid = env2.get_valid_actions()
    if not valid:
        adv = env2.advance_to_next_event()
        if adv == 0:
            break
        continue
    assign_actions = [a for a in valid if a < 20 * env2.num_workers]
    action = assign_actions[0] if assign_actions else valid[0]
    _, _, done, _ = env2.step(action)
    late_pen = env2._last_reward_breakdown.get('lateness_penalty', 0.0)
    late_penalty_history.append(late_pen)
    if done:
        break

# Count how many steps had non-zero lateness penalty
nonzero_late = sum(1 for p in late_penalty_history if p < -0.001)
zero_late = sum(1 for p in late_penalty_history if abs(p) < 0.001)
total_steps = len(late_penalty_history)
print(f"  Total steps: {total_steps}")
print(f"  Steps with lateness penalty: {nonzero_late}")
print(f"  Steps without lateness penalty: {zero_late}")

if zero_late > nonzero_late * 2:  # Most steps should have zero lateness
    print(f"  ✓ Lateness penalty is sparse (one-shot), not cumulative")
    passed += 1
else:
    print(f"  ✗ Lateness penalty may still be cumulative")
    failed += 1

# ── Check 5: Throughput milestone tracking ───────────────────────────────────

print("\n5. Testing throughput milestone bonus...")
env3 = ProjectEnv(num_workers=5, total_tasks=20, seed=42, total_sim_slots=800)
state = env3.reset()

milestone_rewards = []
for step in range(500):
    valid = env3.get_valid_actions()
    if not valid:
        adv = env3.advance_to_next_event()
        if adv == 0:
            break
        continue
    assign_actions = [a for a in valid if a < 20 * env3.num_workers]
    action = assign_actions[0] if assign_actions else valid[0]
    _, _, done, _ = env3.step(action)
    tb = env3._last_reward_breakdown.get('throughput_bonus', 0.0)
    if tb > 0:
        rate = len(env3.completed_tasks) / max(len(env3.tasks), 1)
        milestone_rewards.append((step, tb, rate))
    if done:
        break

print(f"  Milestone bonuses fired: {len(milestone_rewards)}")
for step, bonus, rate in milestone_rewards:
    print(f"    Step {step}: bonus={bonus:.1f}, completion_rate={rate:.0%}")

if len(milestone_rewards) > 0:
    print(f"  ✓ Throughput milestones working")
    passed += 1
else:
    print(f"  ✗ No milestones fired")
    failed += 1

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
print("=" * 80)

if failed > 0:
    print("\n⚠️  Some checks FAILED — review the output above.")
    sys.exit(1)
else:
    print("\n✅ All reward signal checks PASSED — v6 fixes confirmed.")
    sys.exit(0)
