"""
demo_run.py -- Interactive Demo: Live Task Allocation Visualization
==================================================================
Prompts for configuration (seed, workers, goal, tasks), displays all hidden
worker parameters (observer only), then runs each baseline and the DQN agent
with a live task-allocation grid restricted to the Phase 2 (1-week) window.

Usage:
    python demo_run.py

Flow:
    1. Interactive configuration prompt (seed, workers, goal, tasks)
    2. Show environment seed
    3. Display hidden worker profile table (NOT accessible by agents)
    4. For each baseline: reset env -> run episode -> live grid update (Phase 2 only)
    5. DQN agent: reset env -> run episode -> live grid update (Phase 2 only)
    6. Final comparison pop-up table
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
from interactive_config import prompt_for_config


# --- Console helpers ---------------------------------------------------------

def _banner(text: str, width: int = 72, char: str = '='):
    print('\n' + char * width)
    print(f'  {text}')
    print(char * width)


def _section(title: str):
    print(f'\n  {"~" * 60}')
    print(f'  >>  {title}')
    print(f'  {"~" * 60}')


# --- Worker profile display --------------------------------------------------

def display_worker_profiles(env: ProjectEnv):
    """
    Print a table of all hidden worker parameters.
    Called ONCE after env init, before any agent runs.
    Agents never call this -- purely observer information.
    """
    _section('HIDDEN WORKER PROFILES  [Observer View Only -- Agents Cannot See These]')
    print()
    header = (
        f'  {"Worker":>8} | {"Skill":>7} | {"Fatigue Rate":>13} | '
        f'{"Recovery":>10} | {"Speed":>7} | {"Burnout Resil":>14}'
    )
    print(header)
    print(f'  {"-"*8}-+-{"-"*7}-+-{"-"*13}-+-{"-"*10}-+-{"-"*7}-+-{"-"*14}')

    for worker in env.workers:
        p = worker.get_hidden_profile()
        print(
            f"  W-{p['worker_id']:>4} | "
            f"{p['true_skill']:>7.3f} | "
            f"{p['fatigue_rate']:>13.3f} | "
            f"{p['recovery_rate']:>10.3f} | "
            f"{p['speed_multiplier']:>7.3f} | "
            f"{p['burnout_resilience']:>14.3f}"
        )

    print()
    print('  Legend:')
    print('    Skill Level       : 0.5 (weak) to 1.5 (expert) -- affects task quality output')
    print('    Fatigue Rate      : higher = tires faster when overloaded')
    print('    Recovery Rate     : higher = recovers faster when idle')
    print('    Speed Multiplier  : higher = finishes tasks faster')
    print('    Burnout Resilience: fatigue threshold before worker burns out; lower = fragile')
    print()
    print('  NOTE: These values are INVISIBLE to all agents during allocation.')
    print('  Agents only observe: worker load, observable fatigue level, and availability.\n')


# --- Run one agent + live grid -----------------------------------------------

def run_agent_with_viz(
    env: ProjectEnv,
    policy,
    agent_name: str,
    viz: TaskGridVisualizer,
    is_dqn: bool = False,
    step_delay: float = 0.05
) -> dict:
    """
    Run a single episode for the given policy and update the grid live
    (only during Phase 2 -- the 1-week operational window).

    Returns:
        Final metrics dict from env.compute_metrics()
    """
    _section(f'Running: {agent_name}')

    # Reset the visualization for this agent
    viz.reset(agent_name, is_dqn=is_dqn)

    state = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # Select action
        if is_dqn:
            action = policy.select_action(state, valid_actions, greedy=True)
        else:
            action = policy.select_action(state)

        # Decode action for visualization
        task_id, worker_id, action_type = env._decode_action(action)

        # Get task info for cell label
        task_info = None
        if 0 <= task_id < len(env.tasks):
            t = env.tasks[task_id]
            task_info = {'complexity': t.complexity, 'priority': t.priority}

        # Step environment
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Update visualization ONLY for Phase 2 (1-week testing window)
        is_phase2 = env.clock.day >= config.PHASE1_DAYS

        if is_phase2:
            if action_type in ('assign', 'defer'):
                viz.update(task_id, worker_id, action_type, step, task_info=task_info)

            # Console log for assign actions
            if action_type == 'assign' and 0 <= worker_id < env.num_workers:
                complexity_str = f"C{task_info['complexity']}" if task_info else ''
                print(
                    f'    Step {step:>3} | Task {task_id:>2} ({complexity_str}) --> Worker W-{worker_id} '
                    f'|  r={reward:+.3f}  |  completed: {len(env.completed_tasks)}/{env.num_tasks}'
                )

            if step_delay > 0:
                time.sleep(step_delay)

    metrics = env.compute_metrics()

    # Show completion summary in viz
    viz.finalize(agent_name, metrics)

    # Console summary
    print()
    print(f'  [Done] {agent_name} finished in {step} steps')
    print(f'     Return          : {total_reward:.3f}')
    print(f'     Throughput      : {metrics["throughput"]}/{env.num_tasks} tasks completed')
    print(f'     Lateness Rate   : {metrics["lateness_rate"]:.1%}')
    print(f'     Avg Lateness    : {metrics["avg_lateness_h"]:.1f} hours')
    print(f'     Load Balance s  : {metrics["load_balance"]:.3f}')
    print(f'     Quality Score   : {metrics["quality_score"]:.3f}')
    print(f'     Overload Events : {metrics["overload_events"]}')

    return {
        'agent': agent_name,
        'return': total_reward,
        **metrics
    }


# --- Final comparison table ---------------------------------------------------

def print_comparison_table(all_results: list):
    """Print a side-by-side comparison of all agents."""
    _banner('FINAL COMPARISON -- All Agents (same seed, identical environment)')
    print()
    print(f'  {"Agent":<18} {"Return":>10} {"Throughput":>11} {"Late%":>8} {"Quality":>9} {"Overload":>9}')
    print(f'  {"-"*18} {"-"*10} {"-"*11} {"-"*8} {"-"*9} {"-"*9}')

    for r in all_results:
        name = r['agent']
        marker = '[DQN]' if 'DQN' in name else '     '
        print(
            f'  {marker}{name:<13} '
            f'{r["return"]:>10.2f} '
            f'{r["throughput"]:>8}/{config.TOTAL_TASKS:<3} '
            f'{r["lateness_rate"]:>7.1%} '
            f'{r["quality_score"]:>9.3f} '
            f'{r["overload_events"]:>9}'
        )

    print()
    best = max(all_results, key=lambda x: x['return'])
    print(f'  Best return: {best["agent"]} ({best["return"]:.2f})')

    least_overload = min(all_results, key=lambda x: x['overload_events'])
    print(f'  Fewest overload events: {least_overload["agent"]} ({least_overload["overload_events"]})')
    print()


def show_comparison_popup(all_results: list):
    """Display a matplotlib pop-up bar chart comparing all agents on key metrics."""
    import matplotlib.pyplot as plt

    agents = [r['agent'] for r in all_results]
    metrics_to_plot = {
        'Throughput':       [r['throughput'] for r in all_results],
        'Quality Score':    [r['quality_score'] for r in all_results],
        'Overload Events':  [r['overload_events'] for r in all_results],
        'Load Balance (s)': [r['load_balance'] for r in all_results],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Agent Comparison -- Phase 2 (1-Week Operational Window)', fontsize=14, fontweight='bold')

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

    for ax, (metric_name, values) in zip(axes.flat, metrics_to_plot.items()):
        bars = ax.bar(agents, values, color=colors[:len(agents)], edgecolor='black', linewidth=0.5)
        ax.set_title(metric_name, fontsize=11)
        ax.set_ylabel(metric_name)
        ax.tick_params(axis='x', rotation=30)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}' if isinstance(val, float) else str(val),
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)


# --- Load DQN agent ----------------------------------------------------------

def load_dqn_agent():
    """
    Load the trained DQN agent from checkpoints/best_model.pth.
    Returns DQNAgent instance ready for greedy evaluation, or None if not found.
    """
    from agents.dqn_agent import DQNAgent

    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')

    if not os.path.exists(model_path):
        print(f'\n  [warn] DQN checkpoint not found at: {model_path}')
        print('  Train the agent first:  python run_pipeline.py --train')
        print('  Skipping DQN section of the demo.\n')
        return None

    try:
        agent = DQNAgent(state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM)
        agent.load(model_path)
        agent.epsilon = 0.0
        agent.policy_net.eval()
        print(f'\n  [ok] DQN agent loaded from {model_path}')
        return agent
    except Exception as e:
        print(f'\n  [warn] Failed to load DQN checkpoint: {e}')
        print('  The checkpoint may be from a different architecture or PyTorch version.')
        print('  Retrain with:  python run_pipeline.py --train\n')
        return None


# --- Main ---------------------------------------------------------------------

def main():
    # -- 1. Interactive setup --------------------------------------------------
    setup = prompt_for_config()
    env_seed = setup['seed']
    if setup['project_goal']:
        print(f"\n  [Setup] Primary Goal: {setup['project_goal']}")

    # -- 2. Initialize environment and display hidden worker profiles ----------
    env = ProjectEnv(seed=env_seed, reward_scale=0.1)

    # Apply manual worker overrides if specified
    if setup['manual_workers']:
        for override in setup['worker_overrides']:
            w_id = override['id']
            if 0 <= w_id < env.num_workers:
                worker = env.workers[w_id]
                worker.true_skill = override['skill']
                worker.speed_multiplier = override['speed']
                worker.fatigue_rate = override['fatigue_rate']
                worker.burnout_resilience = override['resilience']

    _ = env.reset()

    display_worker_profiles(env)

    input('  Press Enter to begin live visualization of all agents...\n')

    # -- 3. Initialize shared visualizer ---------------------------------------
    viz = TaskGridVisualizer(num_workers=env.num_workers, num_tasks=env.num_tasks)

    # -- 4. Define baselines ---------------------------------------------------
    baselines = [
        ('B1 - Random',  RandomBaseline(env)),
        ('B2 - Greedy',  GreedyBaseline(env)),
        ('B3 - STF',     STFBaseline(env)),
        ('B4 - Skill',   SkillBaseline(env)),
        ('B5 - Hybrid',  HybridBaseline(env)),
    ]

    all_results = []

    # -- Skill baseline observation phase --------------------------------------
    print('\n  Running Skill baseline observation phase (10 silent episodes)...')
    skill_policy = baselines[3][1]
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
        skill_policy.env = env
        print(f'  Skill baseline ready with {obs_ep+1} observation episodes.')

    # -- 5. Run each baseline with live visualization --------------------------
    for baseline_name, policy in baselines:
        policy.env = env
        policy.reset()
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

    # -- 6. DQN Agent ----------------------------------------------------------
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

    # -- 7. Final comparison ---------------------------------------------------
    print_comparison_table(all_results)

    _banner(f'Demo Complete  --  Seed used: {env_seed}')
    print('  The comparison chart will appear. Close it when done.')
    print()

    # Pop-up chart comparing all agents
    show_comparison_popup(all_results)

    # Keep matplotlib open
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
