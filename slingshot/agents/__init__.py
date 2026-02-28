# Agents package  (v3 — Dueling DQN + PER)
from .dqn_agent import DQNAgent, DuelingQNetwork, PrioritizedReplayBuffer, SumTree

# Backward-compatible aliases (prevents ImportError for code that imported old names)
QNetwork = DuelingQNetwork
ReplayBuffer = PrioritizedReplayBuffer

__all__ = [
    'DQNAgent',
    'DuelingQNetwork', 'PrioritizedReplayBuffer', 'SumTree',
    # legacy aliases
    'QNetwork', 'ReplayBuffer',
]
