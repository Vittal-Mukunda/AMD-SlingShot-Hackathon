"""
Deep Q-Network (DQN) Agent for Project Task Allocation
Architecture: 88 → 128 ReLU → 128 ReLU → 140
Implements: Experience replay, target network, epsilon-greedy with action masking
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class QNetwork(nn.Module):
    """
    Q-Network: approximates Q(s,a) for all actions
    Architecture: 88 → 128 ReLU → 128 ReLU → 140
    """
    
    def __init__(self, state_dim: int = 88, action_dim: int = 140, hidden_layers: List[int] = [128, 128]):
        """
        Initialize Q-network
        
        Args:
            state_dim: Dimension of state space (88)
            action_dim: Dimension of action space (140)
            hidden_layers: List of hidden layer sizes [128, 128]
        """
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier initialization)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, 88)
        
        Returns:
            Q-values for all actions (batch_size, 140)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning
    Stores (state, action, reward, next_state, done) transitions
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add transition to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of numpy arrays (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with target network and experience replay
    Epsilon-greedy exploration with action masking
    """
    
    def __init__(self, state_dim: int = 88, action_dim: int = 140,
                 learning_rate: float = 0.0005, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995, replay_capacity: int = 10000,
                 batch_size: int = 64, target_update_freq: int = 100,
                 device: str = None):
        """
        Initialize DQN agent
        
        Args:
            state_dim: State dimension (88)
            action_dim: Action dimension (140)
            learning_rate: Adam learning rate (0.0005)
            gamma: Discount factor (0.95)
            epsilon_start: Initial exploration rate (1.0)
            epsilon_end: Minimum exploration rate (0.05)
            epsilon_decay: Per-episode epsilon decay (0.995)
            replay_capacity: Replay buffer size (10000)
            batch_size: Training batch size (64)
            target_update_freq: Steps between target network updates (100)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network always in eval mode
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Huber loss (more stable than MSE)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        
        # Training stats
        self.steps_done = 0
        self.train_steps = 0
        self.last_loss = 0.0
        self.last_q_mean = 0.0
        self.last_td_error = 0.0
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], 
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection with action masking
        
        Args:
            state: Current state (88-dim)
            valid_actions: List of valid action indices
            greedy: If True, use greedy policy (epsilon=0)
        
        Returns:
            Selected action index
        """
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")
        
        # Epsilon-greedy
        if not greedy and np.random.rand() < self.epsilon:
            # Explore: random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: best valid action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                # Mask invalid actions with large negative value
                masked_q = np.full(self.action_dim, -np.inf)
                masked_q[valid_actions] = q_values[valid_actions]
                
                return int(np.argmax(masked_q))
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        Perform one gradient descent step on a batch from replay buffer
        
        Returns:
            Tuple of (loss, mean_q_value, mean_td_error)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Huber loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at step {self.train_steps}")
            return float('nan'), float('nan'), float('nan')
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in self.policy_net.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"WARNING: NaN gradients detected at step {self.train_steps}")
            self.optimizer.zero_grad()  # Clear bad gradients
            return float('nan'), float('nan'), float('nan')
        
        self.optimizer.step()
        
        # Update stats
        self.train_steps += 1
        self.last_loss = loss.item()
        self.last_q_mean = current_q_values.mean().item()
        self.last_td_error = (target_q_values - current_q_values).abs().mean().item()
        
        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return self.last_loss, self.last_q_mean, self.last_td_error
    
    def update_epsilon(self):
        """
        Decay epsilon after each episode
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'train_steps': self.train_steps
        }, path)
    
    def load(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.train_steps = checkpoint['train_steps']


if __name__ == "__main__":
    # Unit test
    print("Testing DQN Agent...")
    
    # Test 1: Initialize agent
    agent = DQNAgent()
    print(f"✓ Initialized: device={agent.device}, epsilon={agent.epsilon:.2f}")
    
    # Test 2: Network forward pass
    state = np.random.rand(88)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor)
        assert q_values.shape == (1, 140)
    print(f"✓ Forward pass: Q-values shape {q_values.shape}")
    
    # Test 3: Action selection
    valid_actions = list(range(50))
    action = agent.select_action(state, valid_actions)
    assert action in valid_actions
    print(f"✓ Action selection: action={action}")
    
    # Test 4: Store and train
    for _ in range(100):
        s = np.random.rand(88)
        a = np.random.choice(140)
        r = np.random.randn()
        s_next = np.random.rand(88)
        done = False
        agent.store_transition(s, a, r, s_next, done)
    
    loss, q_mean, td_error = agent.train_step()
    print(f"✓ Training: loss={loss:.4f}, Q={q_mean:.4f}, TD={td_error:.4f}")
    
    # Test 5: Checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        agent.save(f.name)
        agent.load(f.name)
        print(f"✓ Checkpoint save/load successful")
    
    print("\nAll DQN Agent tests passed!")
