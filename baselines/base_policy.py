"""
Abstract base class for all baseline policies
Provides common interface and action masking utilities
"""

from abc import ABC, abstractmethod
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv


class BasePolicy(ABC):
    """
    Abstract base policy class
    
    All baseline heuristics inherit from this and implement select_action()
    """
    
    def __init__(self, env: ProjectEnv):
        """
        Initialize policy with environment
        
        Args:
            env: ProjectEnv instance
        """
        self.env = env
        self.name = "BasePolicy"
    
    @abstractmethod
    def select_action(self, state) -> int:
        """
        Select action given current state
        
        Args:
            state: Current state observation (may not be used by simple heuristics)
        
        Returns:
            Action index [0, 139]
        """
        pass
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of currently valid actions from environment
        
        Returns:
            List of valid action indices
        """
        return self.env.get_valid_actions()
    
    def decode_action(self, action: int):
        """
        Decode action index into readable format
        
        Args:
            action: Action index
        
        Returns:
            Tuple of (task_id, worker_id, action_type)
        """
        return self.env._decode_action(action)
    
    def encode_action(self, task_id: int, worker_id: int = -1, action_type: str = 'assign') -> int:
        """
        Encode (task, worker, type) into action index
        
        Args:
            task_id: Task ID
            worker_id: Worker ID (or -1 for defer/escalate)
            action_type: 'assign', 'defer', or 'escalate'
        
        Returns:
            Action index
        """
        num_tasks = self.env.num_tasks
        num_workers = self.env.num_workers
        
        if action_type == 'assign':
            return task_id * num_workers + worker_id
        elif action_type == 'defer':
            return num_tasks * num_workers + task_id
        elif action_type == 'escalate':
            return num_tasks * num_workers + num_tasks + task_id
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def reset(self):
        """
        Reset policy state (if stateful). Default: no-op
        """
        pass
    
    def __repr__(self):
        return f"{self.name}Policy()"


if __name__ == "__main__":
    print("BasePolicy is abstract - cannot be instantiated directly")
    print("Use concrete implementations like RandomBaseline, GreedyBaseline, etc.")
