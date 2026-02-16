"""
Task class with priorities, complexities, deadlines, and dependencies
Tracks assignment status and completion progress
"""

import numpy as np
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Task:
    """
    Represents a task with dependencies and deadline constraints
    
    Attributes:
        task_id (int): Unique identifier
        priority (int): Priority level [0-3]: 0=low, 1=medium, 2=high, 3=critical
        complexity (int): Difficulty level [1-5]
        deadline (int): Timesteps until deadline (decreases each step)
        dependencies (List[int]): List of task IDs that must complete first
        assigned_worker (int): Worker ID assigned to, or None
        start_time (int): Timestep when assigned
        arrival_time (int): Timestep when task entered queue
        completion_progress (float): Fraction complete [0, 1]
        is_completed (bool): Whether task is done
        is_failed (bool): Whether task missed deadline or failed quality
        quality_score (float): Quality of completion [0, 1]
    """
    
    def __init__(
        self,
        task_id: int,
        priority: int,
        complexity: int,
        deadline: int,
        dependencies: List[int] = None,
        arrival_time: int = 0
    ):
        """
        Initialize task
        
        Args:
            task_id: Unique task identifier
            priority: Priority level [0-3]
            complexity: Difficulty [1-5]
            deadline: Timesteps until deadline
            dependencies: List of prerequisite task IDs
            arrival_time: When task entered queue
        """
        self.task_id = task_id
        self.priority = priority
        self.complexity = complexity
        self.deadline = deadline
        self.dependencies = dependencies if dependencies is not None else []
        self.arrival_time = arrival_time
        
        # State
        self.assigned_worker = None
        self.start_time = None
        self.completion_progress = 0.0
        self.is_completed = False
        self.is_failed = False
        self.quality_score = 0.0
        
        # Completion time (sampled when assigned)
        self.expected_completion_time = None
        self.actual_completion_time = None
    
    def assign_to_worker(self, worker_id: int, current_time: int, skill: float):
        """
        Assign task to a worker and sample completion time
        
        Args:
            worker_id: ID of worker being assigned
            current_time: Current timestep
            skill: Worker's skill level (for completion time sampling)
        """
        if self.assigned_worker is not None:
            raise ValueError(f"Task {self.task_id} already assigned to worker {self.assigned_worker}")
        
        self.assigned_worker = worker_id
        self.start_time = current_time
        
        # Sample stochastic completion time
        expected_time = self.complexity / skill
        noise_std = config.COMPLETION_TIME_NOISE * expected_time
        sampled_time = np.random.normal(expected_time, noise_std)
        self.expected_completion_time = int(np.clip(sampled_time, 0.5 * expected_time, 2.0 * expected_time))
    
    def update_progress(self, current_time: int) -> bool:
        """
        Update task progress (called every timestep if assigned)
        
        Args:
            current_time: Current timestep
            
        Returns:
            True if task just completed this step
        """
        if self.assigned_worker is None or self.is_completed:
            return False
        
        # Progress based on time elapsed
        time_elapsed = current_time - self.start_time
        # Safety check: prevent division by zero
        if self.expected_completion_time is None or self.expected_completion_time <= 0:
            self.expected_completion_time = 1  # Default minimum
        self.completion_progress = min(1.0, time_elapsed / self.expected_completion_time)
        
        # Check if completed
        if self.completion_progress >= 1.0:
            self.is_completed = True
            self.actual_completion_time = current_time
            return True
        
        return False
    
    def update_deadline(self, current_time: int):
        """
        Update deadline (decreases each timestep)
        
        Args:
            current_time: Current timestep
        """
        time_remaining = self.deadline - (current_time - self.arrival_time)
        if time_remaining <= 0 and not self.is_completed:
            self.is_failed = True
    
    def apply_deadline_shock(self, shock_amount: int = None):
        """
        Apply sudden deadline reduction (environment stochasticity)
        
        Args:
            shock_amount: Timesteps to reduce deadline by (default from config)
        """
        if shock_amount is None:
            shock_amount = config.DEADLINE_SHOCK_AMOUNT
        
        self.deadline = max(5, self.deadline - shock_amount)  # Never reduce below 5
    
    def check_dependencies_met(self, completed_task_ids: List[int]) -> bool:
        """
        Check if all prerequisite tasks are completed
        
        Args:
            completed_task_ids: List of IDs of completed tasks
            
        Returns:
            True if all dependencies satisfied
        """
        for dep_id in self.dependencies:
            if dep_id not in completed_task_ids:
                return False
        return True
    
    def get_state_vector(self, current_time: int) -> np.ndarray:
        """
        Get observable state representation for this task
        
        Args:
            current_time: Current timestep
            
        Returns:
            4-dim vector: [priority_normalized, complexity_normalized, 
                          deadline_urgency, dependencies_met_flag]
        """
        priority_normalized = self.priority / 3.0  # [0, 1]
        complexity_normalized = (self.complexity - 1) / 4.0  # [0, 1]
        
        # Deadline urgency: higher when deadline is near
        time_remaining = max(0, self.deadline - (current_time - self.arrival_time))
        deadline_urgency = 1.0 - (time_remaining / config.DEADLINE_MAX)
        deadline_urgency = np.clip(deadline_urgency, 0.0, 1.0)
        
        # Dependencies: 0 if unmet, 1 if met (requires external check)
        # This is a placeholder; actual value set by environment
        deps_met_flag = 0.0 if len(self.dependencies) > 0 else 1.0
        
        return np.array([priority_normalized, complexity_normalized, deadline_urgency, deps_met_flag])
    
    def get_deadline_urgency(self, current_time: int) -> float:
        """
        Compute urgency score for priority sorting
        
        Args:
            current_time: Current timestep
            
        Returns:
            Urgency score (higher = more urgent)
        """
        time_remaining = max(1, self.deadline - (current_time - self.arrival_time))
        # Urgency = priority * (1 / time_remaining)
        urgency = (self.priority + 1) * 10.0 / time_remaining
        return urgency
    
    def reset(self):
        """
        Reset task state for new assignment attempt
        """
        self.assigned_worker = None
        self.start_time = None
        self.completion_progress = 0.0
        self.is_completed = False
        self.is_failed = False
        self.quality_score = 0.0
        self.expected_completion_time = None
        self.actual_completion_time = None
    
    def __repr__(self):
        status = "completed" if self.is_completed else ("failed" if self.is_failed else "pending")
        return (f"Task({self.task_id}, p={self.priority}, c={self.complexity}, "
                f"deadline={self.deadline}, deps={len(self.dependencies)}, {status})")


def generate_task_dependency_graph(num_tasks: int, complexity: int = 3) -> List[List[int]]:
    """
    Generate random task dependency graph (DAG)
    
    Args:
        num_tasks: Total number of tasks
        complexity: Number of dependency chains
        
    Returns:
        List of dependency lists (one per task)
    """
    dependencies = [[] for _ in range(num_tasks)]
    
    # Create dependency chains
    for chain_idx in range(complexity):
        # Random chain length
        chain_length = np.random.randint(2, min(5, num_tasks // complexity + 1))
        
        # Select random tasks for this chain (ensure they're in order)
        chain_tasks = np.random.choice(num_tasks, chain_length, replace=False)
        chain_tasks = sorted(chain_tasks)
        
        # Create dependencies: task[i] depends on task[i-1]
        for i in range(1, len(chain_tasks)):
            task_id = chain_tasks[i]
            dep_id = chain_tasks[i-1]
            if dep_id not in dependencies[task_id]:
                dependencies[task_id].append(dep_id)
    
    return dependencies


if __name__ == "__main__":
    # Unit test
    print("Testing Task class...")
    
    # Test 1: Initialize task
    task = Task(task_id=0, priority=2, complexity=3, deadline=50, dependencies=[])
    print(f"✓ Initialized: {task}")
    
    # Test 2: Assign to worker
    task.assign_to_worker(worker_id=1, current_time=0, skill=1.0)
    assert task.assigned_worker == 1
    print(f"✓ Assigned to worker {task.assigned_worker}, expected time={task.expected_completion_time}")
    
    # Test 3: Update progress
    for t in range(task.expected_completion_time + 1):
        completed = task.update_progress(current_time=t)
        if completed:
            print(f"✓ Completed at timestep {t}, progress={task.completion_progress:.2f}")
            break
    
    # Test 4: Deadline shock
    task2 = Task(task_id=1, priority=3, complexity=2, deadline=30)
    task2.apply_deadline_shock(10)
    assert task2.deadline == 20
    print(f"✓ Deadline shock applied: {task2.deadline}")
    
    # Test 5: Dependency graph
    deps = generate_task_dependency_graph(10, complexity=3)
    has_deps = sum(1 for d in deps if len(d) > 0)
    print(f"✓ Generated dependency graph: {has_deps}/10 tasks have dependencies")
    
    # Test 6: State vector
    state = task.get_state_vector(current_time=0)
    assert len(state) == 4
    print(f"✓ State vector: {state}")
    
    print("\nAll Task tests passed!")
