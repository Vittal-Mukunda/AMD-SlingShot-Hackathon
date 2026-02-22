"""
demo_run.py — Interactive Demo: Live Task Allocation Visualization
==================================================================
Prompts for a seed, displays all hidden worker parameters (observer only),
then runs each baseline and the DQN agent with a live task-allocation grid.

Usage:
    python demo_run.py

Flow:
    1. Prompt for seed (auto or custom)
    2. Show environment seed
    3. Display hidden worker profile table (NOT accessible by agents)
    4. For each baseline: reset env → run episode → live grid update
    5. DQN agent: reset env → run episode → live grid update
    6. Final comparison table
"""

import os
import sys
import time
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from environment.project_env import ProjectEnv
from baselines.random_baseline import RandomBaseline
from baselines.greedy_baseline import GreedyBaseline
from baselines.stf_baseline import STFBaseline
from baselines.skill_baseline import SkillBaseline
from baselines.hybrid_baseline import HybridBaseline
from visualization.task_grid_viz import TaskGridVisualizer


# ─── Console helpers ──────────────────────────────────────────────────────────

def _banner(text: str, width: int = 72, char: str = '═'):
    print('\n' + char * width)
    print(f'  {text}')
    print(char * width)


def _box_line(text: str, width: int = 72):
    print(f'  {text}')


def _section(title: str):
    print(f'\n  {"─" * 60}')
    print(f'  ▶  {title}')
    print(f'  {"─" * 60}')


# ─── Step 1: Seed selection ───────────────────────────────────────────────────

def prompt_seed() -> int:
    """
    Ask the user to enter a seed or press Enter for an auto-generated one.

    Returns:
        Integer seed to use for ALL environment resets.
    """
    _banner('RL-DRIVEN AGENTIC PROJECT MANAGER — DEMO')
    print()
    print('  This demo runs 5 baselines + DQN agent on the SAME environment seed.')
    print('  All hidden worker parameters are revealed to YOU (the observer) only.')
    print('  Neither the baselines nor the DQN agent can see these values.')
    print()
    print('  ┌─────────────────────────────────────────────────────────┐')
    print('  │  SEED SELECTION                                         │')
    print('  │  • Press  Enter  for an auto-generated random seed      │')
    print('  │  • Type a number (e.g. 42) for a custom seed            │')
    print('  └─────────────────────────────────────────────────────────┘')
    print()

    raw = input('  Your choice: ').strip()

    if raw == '':
        seed = random.randint(1000, 99999)
        print(f'\n  [auto] Generated seed: {seed}')
    else:
        try:
            seed = int(raw)
            print(f'\n  [custom] Using seed: {seed}')
        except ValueError:
            print(f'  [warn] Invalid input "{raw}" — using auto-generated seed.')
            seed = random.randint(1000, 99999)
            print(f'  [auto] Generated seed: {seed}')

    _banner(f'ENVIRONMENT SEED:  {seed}', char='═')
    print(f'\n  Reproducibility guaranteed: run with seed {seed} again for identical results.\n')
    return seed


# ─── Step 2: Worker profile display ──────────────────────────────────────────

def display_worker_profiles(env: ProjectEnv):
    """
    Print a rich table of all hidden worker parameters.
    This is called ONCE after env init, before any agent runs.
    Agents never call this — it is purely observer information.

    Args:
        env: Initialized ProjectEnv whose workers have been seeded.
    """
    _section('HIDDEN WORKER PROFILES  [Observer View Only — Agents Cannot See These]')
    print()
    print('  ┌──────────┬───────────┬──────────────┬───────────────┬──────────────────┬──────────────────┐')
    print('  │  Worker  │  Skill    │  Fatigue     │  Recovery     │  Speed           │  Burnout         │')
    print('  │    ID    │  Level    │   Rate ↑     │   Rate ↓      │  Multiplier      │  Resilience      │')
    print('  ├──────────┼───────────┼──────────────┼───────────────┼──────────────────┼──────────────────┤')

    for worker in env.workers:
        p = worker.get_hidden_profile()
        skill_bar = '█' * int(p['true_skill'] / 1.4 * 8)  # 8-char bar
        speed_bar = '▶' * int(p['speed_multiplier'] / 1.5 * 5)

        # Rating labels
        skill_label = 'Expert' if p['true_skill'] > 1.2 else ('Good' if p['true_skill'] > 0.9 else 'Weak')
        speed_label = 'Fast' if p['speed_multiplier'] > 1.1 else ('Avg' if p['speed_multiplier'] > 0.85 else 'Slow')
        resilience_label = 'High' if p['burnout_resilience'] > 2.6 else ('Med' if p['burnout_resilience'] > 2.2 else 'Low')

        print(
            f"  │  W-{p['worker_id']}      │"
            f"  {p['true_skill']:.3f}   │"
            f"  {p['fatigue_rate']:.3f} ({'+' if p['fatigue_rate'] > config.FATIGUE_ACCUMULATION_RATE else '-'})  │"
            f"  {p['recovery_rate']:.3f} ({'+' if p['recovery_rate'] > config.FATIGUE_RECOVERY_RATE else '-'})   │"
            f"  {p['speed_multiplier']:.3f} ({speed_label:<4})  │"
            f"  {p['burnout_resilience']:.3f} ({resilience_label:<4})  │"
        )

    print('  └──────────┴───────────┴──────────────┴───────────────┴──────────────────┴──────────────────┘')
    print()
    print('  Legend:')
    print('    Skill Level       : 0.6 (weak) → 1.4 (expert) — affects task quality output')
    print('    Fatigue Rate  ↑   : higher = tires faster when overloaded (+ means above average)')
    print('    Recovery Rate ↓   : higher = recovers faster when idle')
    print('    Speed Multiplier  : higher = finishes tasks faster (affects completion time)')
    print('    Burnout Resilience: threshold before worker burns out; lower = more fragile')
    print()
    print('  ⚠️   These values are INVISIBLE to all agents during allocation.')
    print('  Agents only observe: worker load, observable fatigue level, and availability.\n')


# ─── Step 3: Run one agent + live grid ───────────────────────────────────────

def run_agent_with_viz(
    env: ProjectEnv,
    policy,
    agent_name: str,
    viz: TaskGridVisualizer,
    is_dqn: bool = False,
    step_delay: float = 0.05
) -> dict:
    """
    Run a single episode for the given policy and update the grid live.

    Args:
        env: Pre-initialized, already-reset ProjectEnv
        policy: A BasePolicy or DQNAgent (must have select_action(state, valid_actions) or select_action(state))
        agent_name: Display name
        viz: TaskGridVisualizer instance
        is_dqn: True if agent is the DQN agent
        step_delay: Seconds to pause after each step

    Returns:
        Final metrics dict from env.compute_metrics()
    """
    _section(f'Running: {agent_name}')

    # Reset the visualization for this agent
    viz.reset(agent_name, is_dqn=is_dqn)

    state = env.reset()  # Uses the stored seed for reproducibility
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # Select action (DQN has different signature: needs valid_actions)
        if is_dqn:
            action = policy.select_action(state, valid_actions, greedy=True)
        else:
            action = policy.select_action(state)

        # Decode action for visualization
        task_id, worker_id, action_type = env._decode_action(action)

        # Get task info for cell label (before step, while task is accessible)
        task_info = None
        if 0 <= task_id < len(env.tasks):
            t = env.tasks[task_id]
            task_info = {'complexity': t.complexity, 'priority': t.priority}

        # Step environment
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Update visualization
        if action_type in ('assign', 'defer'):
            viz.update(task_id, worker_id, action_type, step, task_info=task_info)

        # Console log for assign actions
        if action_type == 'assign' and 0 <= worker_id < env.num_workers:
            complexity_str = f"C{task_info['complexity']}" if task_info else ''
            print(
                f'    Step {step:>3} │ Task {task_id:>2} ({complexity_str}) ──► Worker W-{worker_id} '
                f'│  r={reward:+.3f}  │  completed: {len(env.completed_tasks)}/{env.num_tasks}'
            )

        if step_delay > 0:
            time.sleep(step_delay)

    metrics = env.compute_metrics()

    # Show completion summary in viz
    viz.finalize(agent_name, metrics)

    # Console summary
    print()
    print(f'  ✅  {agent_name} finished in {step} steps')
    print(f'     Return          : {total_reward:.3f}')
    print(f'     Throughput      : {metrics["throughput"]}/{env.num_tasks} tasks completed')
    print(f'     Deadline Hit    : {metrics["deadline_hit_rate"]:.1%}')
    print(f'     Avg Delay       : {metrics["avg_delay"]:.1f} steps')
    print(f'     Load Balance σ  : {metrics["load_balance"]:.3f}')
    print(f'     Quality Score   : {metrics["quality_score"]:.3f}')
    print(f'     Overload Events : {metrics["overload_events"]}')

    return {
        'agent': agent_name,
        'return': total_reward,
        **metrics
    }


# ─── Step 4: Final comparison table ──────────────────────────────────────────

def print_comparison_table(all_results: list):
    """Print a rich side-by-side comparison of all agents."""
    _banner('FINAL COMPARISON ─ All Agents (same seed, identical environment)')
    print()
    print(f'  {"Agent":<18} {"Return":>10} {"Throughput":>11} {"Deadline%":>10} {"Avg Delay":>10} {"Quality":>9} {"Overload":>9}')
    print(f'  {"─"*18} {"─"*10} {"─"*11} {"─"*10} {"─"*10} {"─"*9} {"─"*9}')

    for r in all_results:
        name = r['agent']
        marker = '🤖' if 'DQN' in name else '  '
        print(
            f'  {marker}{name:<16} '
            f'{r["return"]:>10.2f} '
            f'{r["throughput"]:>8}/{config.NUM_TASKS:<2} '
            f'{r["deadline_hit_rate"]:>9.1%} '
            f'{r["avg_delay"]:>10.1f} '
            f'{r["quality_score"]:>9.3f} '
            f'{r["overload_events"]:>9}'
        )

    print()
    # Highlight the best by return
    best = max(all_results, key=lambda x: x['return'])
    print(f'  🏆  Best return: {best["agent"]} ({best["return"]:.2f})')

    best_deadline = max(all_results, key=lambda x: x['deadline_hit_rate'])
    print(f'  ⏰  Best deadline hit rate: {best_deadline["agent"]} ({best_deadline["deadline_hit_rate"]:.1%})')
    print()


# ─── Step 5: Load DQN agent ──────────────────────────────────────────────────

def load_dqn_agent():
    """
    Load the trained DQN agent from checkpoints/best_model.pth.

    Returns:
        DQNAgent instance ready for greedy evaluation, or None if not found.
    """
    from agents.dqn_agent import DQNAgent

    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')

    if not os.path.exists(model_path):
        print(f'\n  ⚠️   DQN checkpoint not found at: {model_path}')
        print('  Train the agent first:  python run_pipeline.py --train')
        print('  Skipping DQN section of the demo.\n')
        return None

    try:
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        agent.load(model_path)
        agent.epsilon = 0.0          # Pure greedy — no random exploration during demo
        agent.policy_net.eval()
        print(f'\n  ✅  DQN agent loaded from {model_path}')
        return agent
    except Exception as e:
        print(f'\n  ⚠️   Failed to load DQN checkpoint: {e}')
        print('  The checkpoint may be from a different architecture or PyTorch version.')
        print('  Retrain with:  python run_pipeline.py --train\n')
        return None






# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Seed selection ──────────────────────────────────────────────────────
    env_seed = prompt_seed()

    # ── 2. Initialize environment and display hidden worker profiles ───────────
    env = ProjectEnv(seed=env_seed, reward_scale=0.1)
    _ = env.reset()  # Trigger worker init with the seeded RNG

    display_worker_profiles(env)

    input('  Press Enter to begin live visualization of all agents...\n')

    # ── 3. Initialize shared visualizer ────────────────────────────────────────
    viz = TaskGridVisualizer(num_workers=env.num_workers, num_tasks=env.num_tasks)

    # ── 4. Define baselines using the same env reference ──────────────────────
    # NOTE: baselines are created with the env but do NOT receive the seed.
    #       They only interact with env via select_action(state).
    baselines = [
        ('B1 - Random',  RandomBaseline(env)),
        ('B2 - Greedy',  GreedyBaseline(env)),
        ('B3 - STF',     STFBaseline(env)),
        ('B4 - Skill',   SkillBaseline(env)),
        ('B5 - Hybrid',  HybridBaseline(env)),
    ]

    all_results = []

    # ── Skill baseline observation phase (before visualization loop) ───────────
    print('\n  Running Skill baseline observation phase (10 silent episodes)...')
    skill_policy = baselines[3][1]  # B4 - Skill
    if hasattr(skill_policy, 'is_observing') and skill_policy.is_observing:
        obs_env = ProjectEnv(seed=env_seed + 10000, reward_scale=0.1)
        for obs_ep in range(skill_policy.observation_episodes):
            obs_env_ep = ProjectEnv(seed=env_seed + 10000 + obs_ep, reward_scale=0.1)
            obs_env_ep_state = obs_env_ep.reset()
            skill_policy.env = obs_env_ep
            obs_done = False
            while not obs_done:
                obs_valid = obs_env_ep.get_valid_actions()
                if not obs_valid:
                    break
                obs_action = skill_policy.select_action(obs_env_ep_state)
                obs_env_ep_state, _, obs_done, _ = obs_env_ep.step(obs_action)
            skill_policy.observe_episode(obs_env_ep)
        skill_policy.env = env  # Restore to main env ref
        print(f'  Skill baseline ready with {obs_ep+1} observation episodes.')

    # ── 5. Run each baseline with live visualization ───────────────────────────
    for baseline_name, policy in baselines:
        policy.env = env   # Ensure env reference is current
        policy.reset()     # Reset per-episode state (keeps learned skills for Skill/Hybrid)
        result = run_agent_with_viz(
            env=env,
            policy=policy,
            agent_name=baseline_name,
            viz=viz,
            is_dqn=False,
            step_delay=0.03,
        )
        all_results.append(result)
        print(f'\n  Waiting 2s before next agent...')
        time.sleep(2.0)

    # ── 6. DQN Agent ──────────────────────────────────────────────────────────
    dqn_agent = load_dqn_agent()
    if dqn_agent is not None:
        dqn_result = run_agent_with_viz(
            env=env,
            policy=dqn_agent,
            agent_name='DQN Agent',
            viz=viz,
            is_dqn=True,
            step_delay=0.03,
        )
        all_results.append(dqn_result)

    # ── 7. Final comparison ────────────────────────────────────────────────────
    print_comparison_table(all_results)

    _banner(f'Demo Complete  —  Seed used: {env_seed}')
    print('  The visualization window will stay open until you close it.')
    print()

    # Keep matplotlib open
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
