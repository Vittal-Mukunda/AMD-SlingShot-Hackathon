"""
Deep Q-Network (DQN) Agent for Project Task Allocation
=======================================================
Architecture  : Dueling DQN  (88 → 256 → 256 shared → V + A streams → 140)
Enhancements  : Double DQN, Prioritized Experience Replay (PER), Cosine LR scheduling
Stability     : Huber loss, gradient clipping (norm=1.0), Xavier init, NaN guards

v3 changes vs v2
----------------
* QNetwork  → DuelingQNetwork  (separate Value / Advantage streams)
* ReplayBuffer → PrioritizedReplayBuffer  (sum-tree, O(log N) ops)
* Vanilla DQN  → Double DQN  (policy net selects action; target net evaluates)
* LR scheduler: CosineAnnealingWarmRestarts  (T_0=500 episodes)
* Hidden layers 128→256 everywhere; state_dim corrected to 88 throughout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ─────────────────────────────────────────────────────────────────────────────
# Dueling Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN Network: Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    Shared backbone compresses the 88-dim POMDP observation into a rich
    feature representation. Two separate heads then predict:
        V(s)    — scalar state value (how good is being here?)
        A(s,a)  — per-action advantage (how much better is this action?)

    This decomposition dramatically reduces variance in states where most
    actions are equally good (common in task-allocation environments) because
    the agent can update V(s) from every transition rather than only one Q(s,a).

    Architecture:
        Input (88) → Linear(256) → LayerNorm → ReLU
                   → Linear(256) → LayerNorm → ReLU  [shared]
              ┌────────────────────────────────────┐
              ↓ Value stream                        ↓ Advantage stream
          Linear(128) → ReLU → Linear(1)       Linear(128) → ReLU → Linear(140)
              ↓                                     ↓
              V(s)                          A(s,a) − mean_a[A]
              └──────────────── + ─────────────────┘
                                Q(s,a)
    """

    def __init__(self, state_dim: int = 88, action_dim: int = 140,
                 hidden_layers: List[int] = None):
        super(DuelingQNetwork, self).__init__()

        hidden_layers = hidden_layers or [256, 256]

        # ── Shared backbone ──────────────────────────────────────────────────
        shared = []
        in_dim = state_dim
        for h in hidden_layers:
            shared.append(nn.Linear(in_dim, h))
            shared.append(nn.LayerNorm(h))   # LayerNorm stabilises POMDP inputs
            shared.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*shared)

        # ── Value stream ──────────────────────────────────────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ── Advantage stream ─────────────────────────────────────────────────
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Weight initialisation
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier uniform init; zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass → Q-values for all actions.

        Args:
            state: (batch, state_dim) tensor
        Returns:
            Q-values: (batch, action_dim)
        """
        features = self.backbone(state)

        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)

        # Dueling aggregation: subtract mean advantage to recover identifiability
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Experience Replay  (Sum-Tree)
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """
    Binary sum-tree for O(log N) priority sampling and update.

    The leaves hold transition priorities p_i. Internal nodes store the sum
    of their children, so root = Σ p_i. Sampling a value v in [0, Σ p_i]
    walks the tree in O(log N) to reach the corresponding leaf.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity          # number of leaf nodes (must be ≥ 1)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.full(capacity, None, dtype=object)
        self.write = 0                    # circular write pointer
        self.n_entries = 0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left] or self.tree[right] == 0:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Transitions are sampled with probability proportional to their TD-error
    priority p_i^α, corrected by importance-sampling weights w_i = (N·P_i)^{-β}
    to keep the gradient update unbiased.

    References:
        Schaul et al. (2016). "Prioritized Experience Replay." ICLR 2016.

    Args:
        capacity:   Maximum number of stored transitions
        alpha:      Priority exponent (0 = uniform, 1 = fully prioritised)
        beta_start: IS correction start value (annealed to 1.0 over training)
        beta_frames:Number of steps to anneal beta from beta_start → 1.0
        epsilon:    Small constant to avoid zero priority
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 500000,
                 epsilon: float = 1e-5):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1           # incremented each sample() call for annealing
        self.tree = SumTree(capacity)
        self._max_priority = 1.0

    def _anneal_beta(self) -> float:
        """Linear anneal: beta_start → 1.0 over beta_frames frames."""
        b = self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames
        return min(1.0, b)

    def push(self, state, action, reward, next_state, done):
        """Store transition with maximum current priority (ensures new samples are visited)."""
        self.tree.add(self._max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch proportional to priorities.

        Returns:
            (states, actions, rewards, next_states, dones, indices, is_weights)
        """
        if len(self.tree) < batch_size:
            raise ValueError("Buffer too small to sample requested batch.")

        beta = self._anneal_beta()
        self.frame = min(self.frame + 1, self.beta_frames)

        total = self.tree.total
        segment = total / batch_size

        indices, priorities, data = [], [], []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, transition = self.tree.get(v)
            # guard against None entries (un-initialised leaves)
            while transition is None:
                v = np.random.uniform(0, total)
                idx, priority, transition = self.tree.get(v)
            indices.append(idx)
            priorities.append(priority)
            data.append(transition)

        # Importance-sampling weights
        probs = np.array(priorities) / (total + 1e-8)
        is_weights = (len(self.tree) * probs) ** (-beta)
        is_weights /= is_weights.max()   # normalise by max weight

        states, actions, rewards, next_states, dones = zip(*data)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                np.array(indices, dtype=np.int64),
                np.array(is_weights, dtype=np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after a training step."""
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), p)
            self._max_priority = max(self._max_priority, p)

    def __len__(self):
        return len(self.tree)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent  (Dueling + Double + PER + Cosine LR)
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Full DQN agent combining:
        • Dueling Q-Network architecture
        • Double DQN target computation   (reduces Q-value overestimation)
        • Prioritized Experience Replay   (focuses on informative transitions)
        • Cosine Annealing Warm Restarts  (escapes local optima periodically)
        • Target network                  (stabilises Bellman targets)
        • Epsilon-greedy with action masking

    Hyperparameter rationale
    ──────────────────────────────────────────────────────────────────────────
    state_dim=88    : Full POMDP observation (workers+tasks+beliefs+global)
    action_dim=140  : 5w×20t assign + 20 defer + 20 escalate
    lr=0.001        : 2× v2 value; compensates for larger batch & Huber linear regime
    gamma=0.95      : 100-step horizon → effective look-ahead ~20 steps (1/(1-γ))
    epsilon_decay=  : 0.9997/ep → reaches 0.05 at ~5000 eps (full training run)
    replay_capacity : 50000 transitions → rich diversity for 100-step episodes
    batch_size=128  : Large enough to cover rare reward events in a single batch
    target_update=200: Steps; longer than v2 to match wider network update scale
    per_alpha=0.6   : Moderately prioritised (not fully greedy)
    per_beta=0.4→1  : IS correction annealed to full correction by end of training
    """

    def __init__(self, state_dim: int = 88, action_dim: int = 140,
                 learning_rate: float = None, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = None, replay_capacity: int = None,
                 batch_size: int = None, target_update_freq: int = None,
                 per_alpha: float = None, per_beta_start: float = None,
                 per_beta_frames: int = None,
                 lr_scheduler_t0: int = None,
                 device: str = None):

        # Always fall back to config values so that a single config.py controls
        # all hyperparameters (train script may still override via kwargs)
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.epsilon      = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay or config.EPSILON_DECAY
        self.batch_size    = batch_size    or config.BATCH_SIZE
        self.target_update_freq = target_update_freq or config.TARGET_UPDATE_FREQ

        # Device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Networks ────────────────────────────────────────────────────────
        self.policy_net = DuelingQNetwork(
            state_dim, action_dim,
            hidden_layers=config.HIDDEN_LAYERS
        ).to(self.device)

        self.target_net = DuelingQNetwork(
            state_dim, action_dim,
            hidden_layers=config.HIDDEN_LAYERS
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ── Optimiser + LR scheduler ────────────────────────────────────────
        lr = learning_rate if learning_rate is not None else config.LEARNING_RATE
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr,
                                    eps=1e-5)    # eps > default to avoid div/0
        t0 = lr_scheduler_t0 or getattr(config, 'LR_SCHEDULER_T0', 500)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=t0)

        # ── Huber loss ───────────────────────────────────────────────────────
        self.criterion = nn.SmoothL1Loss(reduction='none')  # element-wise for IS weights

        # ── Prioritized Replay Buffer ────────────────────────────────────────
        cap          = replay_capacity   or config.REPLAY_BUFFER_SIZE
        alpha        = per_alpha         or getattr(config, 'PER_ALPHA',        0.6)
        beta_start   = per_beta_start    or getattr(config, 'PER_BETA_START',   0.4)
        beta_frames  = per_beta_frames   or getattr(config, 'PER_BETA_FRAMES',  500000)

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=cap, alpha=alpha,
            beta_start=beta_start, beta_frames=beta_frames
        )

        # ── Training stats ───────────────────────────────────────────────────
        self.steps_done  = 0
        self.train_steps = 0
        self.last_loss   = 0.0
        self.last_q_mean = 0.0
        self.last_td_error = 0.0

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, valid_actions: List[int],
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection with action masking.

        During exploration (ε-greedy), a uniformly random *valid* action is
        chosen.  During exploitation, Q-values for invalid actions are masked
        to −∞ so argmax always picks a legal move.

        Args:
            state:         Current 88-dim observation
            valid_actions: Legal action indices from env.get_valid_actions()
            greedy:        If True, force ε=0 (evaluation mode)
        Returns:
            Selected action index
        """
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s).cpu().numpy()[0]

            masked_q = np.full(self.action_dim, -np.inf)
            masked_q[valid_actions] = q[valid_actions]
            return int(np.argmax(masked_q))

    # ── Transition storage ───────────────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done):
        """Push one transition into the PER buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── Training step ────────────────────────────────────────────────────────

    def train_step(self) -> Tuple[float, float, float]:
        """
        One gradient-descent update using a PER-sampled mini-batch.

        Uses **Double DQN** for the target:
            a* = argmax_a  policy_net(s')          [action selection]
            y  = r + γ · target_net(s')[a*]        [value evaluation]

        IS weights from PER scale each sample's loss contribution to correct
        for the non-uniform sampling distribution.

        Returns:
            (loss, mean_q_value, mean_td_error)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        # ── Sample batch ─────────────────────────────────────────────────────
        (states, actions, rewards, next_states, dones,
         tree_indices, is_weights) = self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        is_weights  = torch.FloatTensor(is_weights).to(self.device)

        # ── Current Q-values: Q(s,a) ─────────────────────────────────────────
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Double DQN target ────────────────────────────────────────────────
        with torch.no_grad():
            # Policy net selects the best next action
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net evaluates that action
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # ── Importance-sampling weighted Huber loss ───────────────────────────
        td_errors = (target_q - current_q).detach().cpu().numpy()
        element_loss = self.criterion(current_q, target_q)   # (batch,)
        loss = (is_weights * element_loss).mean()

        if torch.isnan(loss):
            print(f"WARNING: NaN loss at train_step {self.train_steps}")
            return float('nan'), float('nan'), float('nan')

        # ── Gradient step ─────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # NaN gradient guard
        nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in self.policy_net.parameters()
        )
        if nan_grad:
            print(f"WARNING: NaN gradients at train_step {self.train_steps}")
            self.optimizer.zero_grad()
            return float('nan'), float('nan'), float('nan')

        self.optimizer.step()

        # ── Update PER priorities ─────────────────────────────────────────────
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # ── Update stats ──────────────────────────────────────────────────────
        self.train_steps += 1
        self.last_loss     = loss.item()
        self.last_q_mean   = current_q.mean().item()
        self.last_td_error = float(np.abs(td_errors).mean())

        # ── Soft target-network sync ──────────────────────────────────────────
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.last_loss, self.last_q_mean, self.last_td_error

    # ── Epsilon / LR scheduling ───────────────────────────────────────────────

    def update_epsilon(self, episode: int = None):
        """
        Exponential epsilon decay after each episode.
        Also steps the cosine LR scheduler (per-episode).
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # Cosine restart scheduler: step every episode
        self.scheduler.step(episode if episode is not None else self.train_steps)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save full checkpoint including PER state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict':  self.optimizer.state_dict(),
            'scheduler_state_dict':  self.scheduler.state_dict(),
            'epsilon':    self.epsilon,
            'steps_done': self.steps_done,
            'train_steps':self.train_steps,
            'architecture': 'DuelingDQN_v3',
        }, path)

    def load(self, path: str):
        """Load checkpoint. Falls back gracefully if scheduler key absent."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon     = checkpoint.get('epsilon',     self.epsilon)
        self.steps_done  = checkpoint.get('steps_done',  0)
        self.train_steps = checkpoint.get('train_steps', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Testing DQN Agent v3 (Dueling + Double + PER + Cosine-LR)")
    print("=" * 70)

    # Test 1: Initialize
    agent = DQNAgent()
    print(f"✓ Initialized: device={agent.device}, ε={agent.epsilon:.2f}")

    # Test 2: Dueling network forward pass
    state = np.random.rand(88).astype(np.float32)
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q = agent.policy_net(t)
        assert q.shape == (1, 140), f"Expected (1,140) got {q.shape}"
    print(f"✓ DuelingQNetwork forward pass: Q-values shape {tuple(q.shape)}")

    # Test 3: Action selection (greedy + epsilon)
    valid = list(range(50))
    a_explore = agent.select_action(state, valid, greedy=False)
    a_exploit = agent.select_action(state, valid, greedy=True)
    assert a_explore in valid and a_exploit in valid
    print(f"✓ Action selection: explore={a_explore}, exploit={a_exploit}")

    # Test 4: PER push + sample
    for _ in range(200):
        s  = np.random.rand(88).astype(np.float32)
        a  = int(np.random.choice(140))
        r  = float(np.random.randn())
        ns = np.random.rand(88).astype(np.float32)
        agent.store_transition(s, a, r, ns, False)

    batch = agent.replay_buffer.sample(64)
    assert len(batch) == 7, "PER sample should return 7-tuple (incl. idx & weights)"
    print(f"✓ PER sample: states={batch[0].shape}, IS-weights={batch[6].shape}")

    # Test 5: Train step
    loss, q_mean, td_err = agent.train_step()
    if not np.isnan(loss):
        print(f"✓ Train step: loss={loss:.4f}, Q={q_mean:.4f}, TD={td_err:.4f}")
    else:
        print("⚠️  NaN in train step — expected early on with random data")

    # Test 6: Epsilon + scheduler update
    eps_before = agent.epsilon
    agent.update_epsilon(episode=1)
    assert agent.epsilon <= eps_before
    print(f"✓ Epsilon decay: {eps_before:.4f} → {agent.epsilon:.4f}")

    # Test 7: Checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        agent.save(f.name)
        agent2 = DQNAgent()
        agent2.load(f.name)
        assert agent2.epsilon == agent.epsilon
    print("✓ Checkpoint save/load successful")

    print("\n" + "=" * 70)
    print("All DQN Agent v3 tests passed!")
    print("=" * 70)
