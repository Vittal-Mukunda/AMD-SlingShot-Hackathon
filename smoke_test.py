"""Smoke test script to verify the v4 environment, agent, and scheduler."""
import sys, os
sys.path.insert(0, '.')
import numpy as np

print("=" * 60)
print("SMOKE TEST — v4 Continual Online Learning System")
print("=" * 60)

import config

# ── Test 1: Worker ────────────────────────────────────────────────────────────
print("\n[1] Worker v4...")
from environment.worker import Worker
w = Worker(worker_id=0, skill=1.0)
w.assign_task(10)
t, q = w.complete_task(10, complexity=3)
sv = w.get_state_vector()
assert len(sv) == 5, f"Worker state should be 5-dim, got {len(sv)}"
w.daily_reset()
mu, sig = w.get_skill_estimate()
print(f"  ✓ state_dim={len(sv)}, time={t:.2f}h, quality={q:.3f}, skill_est={mu:.3f}")

# ── Test 2: Task + Poisson arrivals ──────────────────────────────────────────
print("\n[2] Task v4 + Poisson arrivals...")
from environment.task import Task, generate_poisson_arrivals
tasks = generate_poisson_arrivals(total_tasks=30, seed=42)
assert len(tasks) == 30, f"Expected 30 tasks, got {len(tasks)}"
t0 = tasks[0]
sv_t = t0.get_state_vector(current_tick=t0.arrival_tick, completed_ids=[])
assert len(sv_t) == 5, f"Task state should be 5-dim, got {len(sv_t)}"
assert t0.is_available(t0.arrival_tick), "Task should be available at arrival tick"
assert not t0.is_available(t0.arrival_tick - 1), "Task should NOT be available before arrival"
print(f"  ✓ {len(tasks)} tasks generated, state_dim={len(sv_t)}, arrival={t0.arrival_tick}")

# ── Test 3: ProjectEnv ────────────────────────────────────────────────────────
print("\n[3] ProjectEnv v4...")
from environment.project_env import ProjectEnv
env = ProjectEnv(num_workers=5, total_tasks=30, seed=42)
state = env.reset()
assert len(state) == config.STATE_DIM, f"State should be {config.STATE_DIM}-dim, got {len(state)}"
# Run 100 steps
total_r = 0.0
for step in range(100):
    valid = env.get_valid_actions()
    if not valid:
        action = 20 * env.num_workers  # defer
    else:
        action = valid[0]
    ns, r, done, info = env.step(action)
    total_r += r
    if done:
        break
metrics = env.compute_metrics()
print(f"  ✓ state_dim={len(state)}, steps={env.clock.tick}, reward={total_r:.1f}")
print(f"  ✓ metrics: throughput={metrics['throughput']}, completion={metrics['completion_rate']:.2%}")

# ── Test 4: DQN Agent ──────────────────────────────────────────────────────────
print("\n[4] DQN Agent v4 (online)...")
from agents.dqn_agent import DQNAgent
agent = DQNAgent()
assert agent.state_dim == config.STATE_DIM
# fill buffer
for _ in range(300):
    s  = np.random.rand(config.STATE_DIM).astype(np.float32)
    ns = np.random.rand(config.STATE_DIM).astype(np.float32)
    agent.store_transition(s, np.random.randint(140), np.random.randn(), ns, False)
loss, q_mean, td = agent.train_step()
print(f"  ✓ train_step: loss={loss:.4f}, Q={q_mean:.3f}, TD={td:.4f}")
eps_before = agent.epsilon
agent.update_epsilon()
print(f"  ✓ epsilon decay: {eps_before:.4f} → {agent.epsilon:.4f}")

# ── Test 5: Skill Baseline ────────────────────────────────────────────────────
print("\n[5] SkillBaseline v4...")
from baselines.skill_baseline import SkillBaseline
env2 = ProjectEnv(num_workers=5, total_tasks=20, seed=99)
bl = SkillBaseline(env2, observation_episodes=2, debug=False)
state = env2.reset()
for _ in range(60):
    valid = env2.get_valid_actions()
    act = bl.select_action(state)
    if valid and act not in valid:
        act = valid[0]
    state, _, done, _ = env2.step(act)
    if done:
        break
bl.observe_episode(env2)
print(f"  ✓ SkillBaseline OK, estimates={dict(bl._skill_means)}")

# ── Test 6: Greedy Baseline ───────────────────────────────────────────────────
print("\n[6] GreedyBaseline v4...")
from baselines.greedy_baseline import GreedyBaseline
env3 = ProjectEnv(num_workers=5, total_tasks=20, seed=7)
bl2 = GreedyBaseline(env3)
state = env3.reset()
for _ in range(60):
    valid = env3.get_valid_actions()
    act = bl2.select_action(state)
    if valid and act not in valid:
        act = valid[0]
    state, _, done, _ = env3.step(act)
    if done:
        break
print(f"  ✓ GreedyBaseline OK, throughput={env3.compute_metrics()['throughput']}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
