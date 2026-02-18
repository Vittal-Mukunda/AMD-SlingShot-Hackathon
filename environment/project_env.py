"""
Main ProjectEnv class implementing the POMDP for RL-driven task allocation
Integrates workers, tasks, belief state, reward function, and environment dynamics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.worker import Worker
from environment.task import Task, generate_task_dependency_graph
from environment.belief_state import BeliefState


class ProjectEnv:
    """
    POMDP environment for project task allocation
    
    State space: 88-dim vector (workers + tasks + beliefs + global context)
    Action space: 140 discrete actions (assign/defer/escalate)
    Reward: Composite function balancing completion, delay, overload, throughput, deadlines
    """
    
    def __init__(self, num_workers: int = None, num_tasks: int = None, seed: int = None,
                 enable_diagnostics: bool = False, reward_scale: float = 0.1,
                 config_overrides: Dict = None):
        """
        Initialize project management environment
        
        Args:
            num_workers: Number of workers (default from config)
            num_tasks: Number of tasks (default from config)
            seed: Random seed for reproducibility
            enable_diagnostics: Enable diagnostic logging for DQN stability analysis
            reward_scale: Reward scaling factor (default 0.1 brings max rewards to ~10 range)
            config_overrides: Dictionary to override default behaviors (for ablation studies)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.num_workers = num_workers or config.NUM_WORKERS
        self.num_tasks = num_tasks or config.NUM_TASKS
        
        # Stability improvements for DQN
        self.enable_diagnostics = enable_diagnostics
        self.reward_scale = reward_scale
        
        # Configuration overrides for Ablation Studies
        self.config_overrides = config_overrides or {}
        self.enable_fatigue = self.config_overrides.get('enable_fatigue', True)
        self.enable_deadline_shocks = self.config_overrides.get('enable_deadline_shocks', True)
        self.fully_observable = self.config_overrides.get('fully_observable', False)
        
        # Diagnostic statistics
        self.diagnostics = {
            'state_ranges': [],
            'reward_ranges': [],
            'valid_action_counts': [],
            'reward_components': []
        } if enable_diagnostics else None
        
        # Initialize workers
        self.workers = [Worker(worker_id=i) for i in range(self.num_workers)]
        
        # Initialize tasks (will be regenerated each episode)
        self.tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Belief state for skill tracking
        self.belief_state = BeliefState(num_workers=self.num_workers)
        
        # Episode state
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.episode_history = []
        
        # Metrics tracking
        self.metrics = {
            'throughput': 0,
            'deadline_hit_rate': 0.0,
            'avg_delay': 0.0,
            'load_balance': 0.0,
            'quality_score': 0.0,
            'overload_events': 0
        }
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode
        
        Returns:
            Initial state observation (88-dim vector)
        """
        # Reset workers
        for worker in self.workers:
            worker.reset()
        
        # Generate new tasks with dependencies
        self.tasks = self._generate_tasks()
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Reset belief state
        self.belief_state.reset()
        
        # Reset episode state
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.episode_history = []
        
        # Reset metrics
        self.metrics = {
            'throughput': 0,
            'deadline_hit_rate': 0.0,
            'avg_delay': 0.0,
            'load_balance': 0.0,
            'quality_score': 0.0,
            'overload_events': 0
        }
        
        return self._get_state()
    
    def _generate_tasks(self) -> List[Task]:
        """
        Generate random tasks with dependencies for new episode
        
        Returns:
            List of Task objects
        """
        tasks = []
        
        # Generate dependency graph
        dependencies = generate_task_dependency_graph(
            self.num_tasks, 
            complexity=config.DEPENDENCY_GRAPH_COMPLEXITY
        )
        
        for task_id in range(self.num_tasks):
            # Random priority and complexity
            priority = np.random.choice(config.TASK_PRIORITIES)
            complexity = np.random.choice(config.TASK_COMPLEXITY_LEVELS)
            
            # Deadline based on complexity (harder tasks get more time)
            base_deadline = config.DEADLINE_MIN + (complexity - 1) * 5
            deadline = np.random.randint(base_deadline, config.DEADLINE_MAX)
            
            task = Task(
                task_id=task_id,
                priority=priority,
                complexity=complexity,
                deadline=deadline,
                dependencies=dependencies[task_id],
                arrival_time=0  # All tasks arrive at start
            )
            tasks.append(task)
        
        return tasks
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one timestep with given action
        
        Args:
            action: Action index [0, 139]
                   0-99: assign task (task_id, worker_id) pairs
                   100-119: defer task
                   120-139: escalate task
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Decode action
        task_id, worker_id, action_type = self._decode_action(action)
        
        # Execute action
        action_reward = self._execute_action(task_id, worker_id, action_type)
        
        # Update environment dynamics
        self._update_dynamics()
        
        # Update task progress and check completions
        completion_reward = self._update_task_progress()
        
        # Compute reward components
        delay_penalty = self._compute_delay_penalty()
        overload_penalty = self._compute_overload_penalty()
        throughput_bonus = completion_reward  # Already computed
        deadline_penalty = self._compute_deadline_penalty()
        
        # Total reward (unscaled components)
        reward_unscaled = (action_reward + completion_reward + delay_penalty + 
                          overload_penalty + throughput_bonus + deadline_penalty)
        
        # Apply reward scaling for DQN stability
        reward = reward_unscaled * self.reward_scale
        
        # Log diagnostics if enabled
        if self.enable_diagnostics:
            self.diagnostics['reward_components'].append({
                'action': action_reward,
                'completion': completion_reward,
                'delay': delay_penalty,
                'overload': overload_penalty,
                'deadline': deadline_penalty,
                'total_unscaled': reward_unscaled,
                'total_scaled': reward
            })
        
        self.episode_reward += reward
        
        # Check termination conditions
        done, termination_reason = self._check_termination()
        
        # Increment timestep
        self.current_timestep += 1
        
        # Random deadline shocks
        if np.random.rand() < config.DEADLINE_SHOCK_PROB:
            self._apply_deadline_shock()
        
        # Get next state
        next_state = self._get_state()
        
        # Log diagnostics if enabled
        if self.enable_diagnostics:
            valid_actions = self.get_valid_actions()
            self.diagnostics['state_ranges'].append((np.min(next_state), np.max(next_state)))
            self.diagnostics['reward_ranges'].append(reward)
            self.diagnostics['valid_action_counts'].append(len(valid_actions))
        
        # Info for logging
        info = {
            'timestep': self.current_timestep,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'termination_reason': termination_reason if done else None
        }
        
        return next_state, reward, done, info
    
    def _decode_action(self, action: int) -> Tuple[int, int, str]:
        """
        Decode action index into (task_id, worker_id, action_type)
        
        Args:
            action: Action index [0, 139]
        
        Returns:
            Tuple of (task_id, worker_id, action_type)
        """
        if action < self.num_tasks * self.num_workers:
            # Assign action
            task_id = action // self.num_workers
            worker_id = action % self.num_workers
            return task_id, worker_id, 'assign'
        elif action < self.num_tasks * self.num_workers + self.num_tasks:
            # Defer action
            task_id = action - (self.num_tasks * self.num_workers)
            return task_id, -1, 'defer'
        else:
            # Escalate action
            task_id = action - (self.num_tasks * self.num_workers + self.num_tasks)
            return task_id, -1, 'escalate'
    
    def _execute_action(self, task_id: int, worker_id: int, action_type: str) -> float:
        """
        Execute the selected action
        
        Returns:
            Immediate reward for this action
        """
        reward = 0.0
        
        if action_type == 'assign':
            # Check if action is valid
            if not self._is_valid_assign(task_id, worker_id):
                return -1.0  # Small penalty for invalid action
            
            task = self.tasks[task_id]
            worker = self.workers[worker_id]
            
            # Assign task to worker
            worker.assign_task(task_id)
            task.assign_to_worker(worker_id, self.current_timestep, worker.true_skill)
            
            # Small reward for matching skill to complexity
            skill_match_quality = worker.true_skill / task.complexity
            if skill_match_quality > 1.0:
                reward += config.REWARD_EXPLORATION_BONUS * 0.5
        
        elif action_type == 'defer':
            # Strategic defer: check if this is wise
            task = self.tasks[task_id]
            if task.assigned_worker is None:
                # Check if deferring makes sense (no good worker available)
                available_skills = [w.true_skill for w in self.workers if w.availability == 1]
                if len(available_skills) == 0 or max(available_skills) < task.complexity * 0.5:
                    reward += config.REWARD_STRATEGIC_DEFER
        
        elif action_type == 'escalate':
            # Escalate task priority (costs resources but speeds completion)
            task = self.tasks[task_id]
            if task.assigned_worker is not None and task.priority < 3:
                task.priority = min(3, task.priority + 1)
                task.expected_completion_time = int(task.expected_completion_time * 0.8)
                reward -= 2.0  # Cost of escalation
        
        return reward
    
    def _is_valid_assign(self, task_id: int, worker_id: int) -> bool:
        """
        Check if assigning task to worker is valid
        
        Returns:
            True if valid, False otherwise
        """
        if task_id >= len(self.tasks) or worker_id >= self.num_workers:
            return False
        
        task = self.tasks[task_id]
        worker = self.workers[worker_id]
        
        # Check if task already assigned
        if task.assigned_worker is not None or task.is_completed or task.is_failed:
            return False
        
        # Check if worker available
        if worker.availability == 0:
            return False
        
        # Check if dependencies met
        completed_ids = [t.task_id for t in self.completed_tasks]
        if not task.check_dependencies_met(completed_ids):
            return False
        
        return True
    
    def _update_dynamics(self):
        """
        Update worker fatigue and other dynamics
        """
        for worker in self.workers:
            if self.enable_fatigue:
                worker.update_fatigue()
            
            # Track overload events
            if worker.load > 4:
                self.metrics['overload_events'] += 1
    
    def _update_task_progress(self) -> float:
        """
        Update all in-progress tasks and handle completions
        
        Returns:
            Reward from task completions this step
        """
        completion_reward = 0.0
        
        for task in self.tasks:
            if task.assigned_worker is not None and not task.is_completed:
                # Update progress
                just_completed = task.update_progress(self.current_timestep)
                
                if just_completed:
                    # Task completed!
                    worker = self.workers[task.assigned_worker]
                    
                    # Get completion time and quality from worker
                    time, quality = worker.complete_task(task.task_id, task.complexity)
                    task.quality_score = quality
                    
                    # Update belief state
                    self.belief_state.update(task.assigned_worker, quality)
                    
                    # Compute completion reward
                    priority_weight = (task.priority + 1) * config.REWARD_COMPLETION_BASE
                    completion_reward += priority_weight * quality
                    
                    # Move to completed list
                    self.completed_tasks.append(task)
                    self.metrics['throughput'] += 1
            
            # Update deadlines
            task.update_deadline(self.current_timestep)
    
        return completion_reward
    
    def _compute_delay_penalty(self) -> float:
        """
        Penalty for tasks waiting in queue
        
        Returns:
            Negative reward based on queue delays
        """
        penalty = 0.0
        for task in self.tasks:
            if not task.is_completed and not task.is_failed:
                time_in_queue = self.current_timestep - task.arrival_time
                normalized_delay = time_in_queue / max(1, task.deadline)
                penalty += config.REWARD_DELAY_WEIGHT * normalized_delay
        
        return penalty
    
    def _compute_overload_penalty(self) -> float:
        """
        Quadratic penalty for overloading workers
        
        Returns:
            Negative reward based on worker overload
        """
        penalty = 0.0
        for worker in self.workers:
            if worker.load > config.OVERLOAD_THRESHOLD:
                overload = worker.load - config.OVERLOAD_THRESHOLD
                penalty += config.REWARD_OVERLOAD_WEIGHT * (overload ** 2)
        
        return penalty
    
    def _compute_deadline_penalty(self) -> float:
        """
        Catastrophic penalty for missing deadlines
        
        Returns:
            Large negative reward for deadline misses
        """
        penalty = 0.0
        new_failures = 0
        
        for task in self.tasks:
            if task.is_failed and task not in self.failed_tasks:
                penalty += config.REWARD_DEADLINE_MISS_PENALTY
                self.failed_tasks.append(task)
                new_failures += 1
        
        return penalty
    
    def _apply_deadline_shock(self):
        """
        Random deadline shock: suddenly reduce deadline for a pending task
        """
        if not self.enable_deadline_shocks:
            return
            
        pending_tasks = [t for t in self.tasks if not t.is_completed and not t.is_failed]
        if len(pending_tasks) > 0:
            shocked_task = np.random.choice(pending_tasks)
            shocked_task.apply_deadline_shock()
    
    def _check_termination(self) -> Tuple[bool, Optional[str]]:
        """
        Check if episode should terminate
        
        Returns:
            Tuple of (done, reason)
        """
        # Success: all tasks completed
        if len(self.completed_tasks) == self.num_tasks:
            return True, 'success'
        
        # Timeout: max timesteps reached
        if self.current_timestep >= config.EPISODE_HORIZON:
            return True, 'timeout'
        
        # Failure: too many deadline misses
        failure_rate = len(self.failed_tasks) / self.num_tasks
        if failure_rate >= config.FAILURE_THRESHOLD:
            return True, 'failure'
        
        return False, None
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation (88-dim vector)
        
        Returns:
            State vector for DQN input
        """
        # Worker features: 5 workers × 3 features = 15 dim
        worker_features = np.concatenate([w.get_state_vector() for w in self.workers])
        
        # Task features: top-10 pending tasks × 4 features = 40 dim
        pending_tasks = [t for t in self.tasks if not t.is_completed and not t.is_failed]
        pending_tasks = sorted(pending_tasks, key=lambda t: -t.get_deadline_urgency(self.current_timestep))
        pending_tasks = pending_tasks[:10]  # Top 10 most urgent
        
        task_features = []
        for task in pending_tasks:
            state_vec = task.get_state_vector(self.current_timestep)
            # Update dependencies met flag
            completed_ids = [t.task_id for t in self.completed_tasks]
            deps_met = 1.0 if task.check_dependencies_met(completed_ids) else 0.0
            state_vec[3] = deps_met
            task_features.append(state_vec)
        
        # Pad if fewer than 10 tasks
        while len(task_features) < 10:
            task_features.append(np.zeros(4))
        
        task_features = np.concatenate(task_features)
        
        # Belief state: 5 means + 5 variances = 10 dim
        # Or True Skills if fully_observable is True (Ablation Study)
        if self.fully_observable:
            # Use true skills for mean, 0.0 for variance
            # This maintains the 88-dim state vector structure
            skill_means = [w.true_skill for w in self.workers]
            skill_vars = [0.0] * self.num_workers
            belief_features = np.array(skill_means + skill_vars, dtype=np.float32)
        else:
            belief_features = self.belief_state.get_state_vector()
        
        # Global context: 3 dim
        time_progress = self.current_timestep / config.EPISODE_HORIZON
        completion_rate = len(self.completed_tasks) / self.num_tasks
        failure_rate = len(self.failed_tasks) / self.num_tasks
        global_features = np.array([time_progress, completion_rate, failure_rate])
        
        # Concatenate all features: 15 + 40 + 10 + 3 = 68 dim (not 88 as originally planned, but close)
        state = np.concatenate([worker_features, task_features, belief_features, global_features])
        
        # Pad to 88 if needed
        if len(state) < 88:
            state = np.pad(state, (0, 88 - len(state)), 'constant')
        
        return state[:88]  # Ensure exactly 88 dim
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid action indices given current state
        
        Returns:
            List of valid action indices
        """
        valid_actions = []
        completed_ids = [t.task_id for t in self.completed_tasks]
        
        # Check assign actions
        for task_id, task in enumerate(self.tasks):
            if task.is_completed or task.is_failed or task.assigned_worker is not None:
                continue
            
            if not task.check_dependencies_met(completed_ids):
                continue
            
            for worker_id, worker in enumerate(self.workers):
                if worker.availability == 1:
                    action_idx = task_id * self.num_workers + worker_id
                    valid_actions.append(action_idx)
        
        # Defer actions (always valid for unassigned pending tasks)
        for task_id, task in enumerate(self.tasks):
            if not task.is_completed and not task.is_failed and task.assigned_worker is None:
                action_idx = self.num_tasks * self.num_workers + task_id
                valid_actions.append(action_idx)
        
        return valid_actions
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute final episode metrics
        
        Returns:
            Dictionary of metric values
        """
        # Throughput
        self.metrics['throughput'] = len(self.completed_tasks)
        
        # Deadline hit rate
        total_finished = len(self.completed_tasks) + len(self.failed_tasks)
        if total_finished > 0:
            self.metrics['deadline_hit_rate'] = len(self.completed_tasks) / total_finished
        
        # Average delay
        if len(self.completed_tasks) > 0:
            delays = [t.actual_completion_time - t.arrival_time for t in self.completed_tasks]
            self.metrics['avg_delay'] = np.mean(delays)
        
        # Load balance
        loads = [w.load for w in self.workers]
        self.metrics['load_balance'] = np.std(loads)
        
        # Quality score
        if len(self.completed_tasks) > 0:
            qualities = [t.quality_score for t in self.completed_tasks]
            self.metrics['quality_score'] = np.mean(qualities)
        
        return self.metrics
    
    def __repr__(self):
        return (f"ProjectEnv(t={self.current_timestep}, completed={len(self.completed_tasks)}/{self.num_tasks}, "
                f"failed={len(self.failed_tasks)}, reward={self.episode_reward:.1f})")


if __name__ == "__main__":
    # Unit test
    print("Testing ProjectEnv class...")
    
    # Test 1: Initialize
    env = ProjectEnv(num_workers=5, num_tasks=20, seed=42)
    print(f"✓ Initialized: {env}")
    
    # Test 2: Reset
    state = env.reset()
    assert len(state) == 88
    print(f"✓ Reset: state shape {state.shape}")
    
    # Test 3: Valid actions
    valid_actions = env.get_valid_actions()
    print(f"✓ Valid actions: {len(valid_actions)} available")
    
    # Test 4: Step with random valid action
    action = np.random.choice(valid_actions)
    next_state, reward, done, info = env.step(action)
    print(f"✓ Step: reward={reward:.2f}, done={done}, info={info}")
    
    # Test 5: Run full episode
    state = env.reset()
    episode_rewards = []
    for t in range(100):
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        if done:
            print(f"✓ Episode terminated at t={t}, reason={info['termination_reason']}")
            break
    
    # Test 6: Compute metrics
    metrics = env.compute_metrics()
    print(f"✓ Metrics: {metrics}")
    
    print(f"\n✓ Episode return: {sum(episode_rewards):.2f}")
    print("All ProjectEnv tests passed!")
