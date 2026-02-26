"""
Interactive Configuration Prompt
================================
Run before starting the core simulation in demo_run.py to configure global settings.
"""

import config
import random

def prompt_for_config() -> dict:
    print("\n" + "="*80)
    print(" PROJECT SETUP AND CONFIGURATION ")
    print("="*80)
    
    # 1. Seed Selection
    raw_seed = input("  Enter a random seed (or press Enter for auto-generated): ").strip()
    if raw_seed:
        try:
            seed = int(raw_seed)
        except ValueError:
            print("  [warn] Invalid seed, using auto-generated.")
            seed = random.randint(1000, 99999)
    else:
        seed = random.randint(1000, 99999)
    print(f"  > Using Seed: {seed}")
        
    # 2. Project Goal
    print("\n  Describe the overarching project goal (e.g. 'Build a scalable generic web backend'):")
    project_goal = input("  > ").strip()
    if not project_goal:
        project_goal = "Complete randomly generated generic framework tasks."
        
    # 3. Tasks configuration
    print("\n  Number of Tasks for the simulation (Default 200, press Enter to keep default):")
    raw_tasks = input("  > ").strip()
    if raw_tasks:
        try:
            config.TOTAL_TASKS = int(raw_tasks)
        except ValueError:
            print(f"  [warn] Invalid input, keeping default {config.TOTAL_TASKS}.")
            
    # 4. Worker manual overrides
    print("\n  Do you want to manually configure worker properties (skill, speed, fatigue resilience)? (y/n)")
    manual_workers = input("  > ").strip().lower() == 'y'
    worker_overrides = []
    
    if manual_workers:
        print(f"\n  We generated a default of {config.NUM_WORKERS} workers. Please provide parameters for each.")
        for i in range(config.NUM_WORKERS):
            print(f"\n  --- Worker {i} ---")
            skill = prompt_float("Skill level (0.5 to 1.5):", default=1.0)
            speed = prompt_float("Speed multiplier (0.6 to 1.5):", default=1.0)
            fatigue_rate = prompt_float("Fatigue accumulation rate (0.05 to 0.5):", default=0.20)
            resilience = prompt_float("Burnout resilience threshold (1.8 to 3.2):", default=2.6)
            
            worker_overrides.append({
                'id': i,
                'skill': skill,
                'speed': speed,
                'fatigue_rate': fatigue_rate,
                'resilience': resilience
            })
            
    print("\n" + "="*80)
    print(f" CONFIGURATION SAVED ")
    print("="*80 + "\n")
    
    return {
        'seed': seed,
        'project_goal': project_goal,
        'manual_workers': manual_workers,
        'worker_overrides': worker_overrides
    }

def prompt_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  [warn] Invalid input, using default {default}.")
        return default
