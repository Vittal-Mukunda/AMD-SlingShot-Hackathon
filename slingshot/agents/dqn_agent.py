"""
DQN Agent — v4: Continual Online Learning.

Architecture PRESERVED from v3 (Dueling DQN + PER + Double DQN + Cosine LR).

New in v4:
  - state_dim updated to 96 (was 88; driven by config.STATE_DIM)
  - online_step(): one-shot method that selects, stores, and trains in-line
  - update_epsilon() operates per-DECISION (not per-episode) for online decay
  - Epsilon continues from wherever it left off — seamless Phase1→Phase2
  - Comprehensive logging per step: loss, Q-mean, TD-error, epsilon
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import List, Tuple, Optional
import sys
import os
from slingshot.core.settings import config


# ─────────────────────────────────────────────────────────────────────────────
# Dueling Q-Network  (PRESERVED from v3 — architecture unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN:  Q(s,a) = V(s) + A(s,a) − mean_a'[A(s,a')]

    Shared backbone → Value stream + Advantage stream → Q-values.

    Architecture (v4, state_dim=96):
        Input (96) → Linear(256) → LayerNorm → ReLU
                   → Linear(256) → LayerNorm → ReLU  [shared]
          ┌──────────────────────────────────────────┐
          ↓ Value stream           ↓ Advantage stream
      Linear(128)→ReLU→Linear(1)  Linear(128)→ReLU→Linear(140)
    """

    def __init__(self, state_dim: int = None, action_dim: int = None,
                 hidden_layers: List[int] = None):
        super().__init__()
        state_dim    = state_dim  or config.STATE_DIM
        action_dim   = action_dim or config.ACTION_DIM
        hidden_layers = hidden_layers or config.HIDDEN_LAYERS

        # Shared backbone
        shared, in_dim = [], state_dim
        for h in hidden_layers:
            shared += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*shared)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat  = self.backbone(state)
        V     = self.value_stream(feat)
        A     = self.advantage_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer  (PRESERVED from v3)
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    """Binary sum-tree for O(log N) priority sampling and update."""

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.full(capacity, None, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left] or self.tree[right] == 0:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> Tuple[int, float, object]:
        idx      = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (Schaul et al. 2016)."""

    def __init__(self, capacity: int = None, alpha: float = None,
                 beta_start: float = None, beta_frames: int = None,
                 epsilon: float = 1e-5):
        self.alpha       = alpha      or config.PER_ALPHA
        self.beta_start  = beta_start or config.PER_BETA_START
        self.beta_frames = beta_frames or config.PER_BETA_FRAMES
        self.epsilon     = epsilon
        self.frame       = 1
        self.tree        = SumTree(capacity or config.REPLAY_BUFFER_SIZE)
        self._max_priority = 1.0

    def _anneal_beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self._max_priority ** self.alpha,
                      (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        if len(self.tree) < batch_size:
            raise ValueError("Buffer too small to sample.")
        beta    = self._anneal_beta()
        self.frame = min(self.frame + 1, self.beta_frames)
        total   = self.tree.total
        segment = total / batch_size

        indices, priorities, data = [], [], []
        for i in range(batch_size):
            v = np.random.uniform(segment * i, segment * (i + 1))
            idx, pri, trans = self.tree.get(v)
            while trans is None:
                v = np.random.uniform(0, total)
                idx, pri, trans = self.tree.get(v)
            indices.append(idx); priorities.append(pri); data.append(trans)

        probs      = np.array(priorities) / (total + 1e-8)
        is_weights = (len(self.tree) * probs) ** (-beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*data)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                np.array(indices, dtype=np.int64),
                np.array(is_weights, dtype=np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), p)
            self._max_priority = max(self._max_priority, p)

    def __len__(self):
        return len(self.tree)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent — v4 Online Learning
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Continual online DQN agent.

    Architecture: Dueling DQN + Double DQN + PER + Cosine LR  (PRESERVED from v3)
    New:  online_step(), per-decision epsilon decay, passive observation mode.

    Epsilon:
      - Decays every call to update_epsilon() (called per DECISION, not per episode)
      - Phase 1: passive observation → stores transitions but doesn't control actions
      - Phase 2: takes control; epsilon starts at EPSILON_PHASE2_START
    """

    def __init__(
        self,
        state_dim: int              = None,
        action_dim: int             = None,
        learning_rate: float        = None,
        gamma: float                = None,
        epsilon_start: float        = None,
        epsilon_end: float          = None,
        epsilon_decay: float        = None,
        replay_capacity: int        = None,
        batch_size: int             = None,
        target_update_freq: int     = None,
        per_alpha: float            = None,
        per_beta_start: float       = None,
        per_beta_frames: int        = None,
        lr_scheduler_t0: int        = None,
        min_replay_size: int        = None,
        device: str                 = None,
    ):
        self.state_dim  = state_dim  or config.STATE_DIM
        self.action_dim = action_dim or config.ACTION_DIM
        self.gamma      = gamma      or config.GAMMA

        self.epsilon       = epsilon_start if epsilon_start is not None else config.EPSILON_START
        self.epsilon_start = self.epsilon
        self.epsilon_end   = epsilon_end   or config.EPSILON_END
        self.epsilon_decay = epsilon_decay or config.EPSILON_DECAY

        self.batch_size         = batch_size         or config.BATCH_SIZE
        self.target_update_freq = target_update_freq or config.TARGET_UPDATE_FREQ
        self.min_replay_size    = min_replay_size    or config.MIN_REPLAY_SIZE

        self.device = torch.device(device if device else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

        # ── Networks ─────────────────────────────────────────────────────────
        self.policy_net = DuelingQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ── Optimiser + LR scheduler ─────────────────────────────────────────
        lr = learning_rate if learning_rate is not None else config.LEARNING_RATE
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)
        t0 = lr_scheduler_t0 or getattr(config, 'LR_SCHEDULER_T0', 2000)
        # v12 fix: add eta_min to prevent Q-value collapse from LR decaying to near-zero
        eta_min_frac = getattr(config, 'LR_SCHEDULER_ETA_MIN_FRACTION', 0.15)
        eta_min = max(lr * eta_min_frac, 1e-5)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t0, T_mult=getattr(config, 'LR_SCHEDULER_T_MULT', 1),
            eta_min=eta_min
        )

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = nn.SmoothL1Loss(reduction='none')

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity     = replay_capacity or config.REPLAY_BUFFER_SIZE,
            alpha        = per_alpha       or config.PER_ALPHA,
            beta_start   = per_beta_start  or config.PER_BETA_START,
            beta_frames  = per_beta_frames or config.PER_BETA_FRAMES,
        )

        # ── Stats ─────────────────────────────────────────────────────────────
        self.steps_done    = 0    # Total decisions taken (including passive)
        self.train_steps   = 0    # Gradient updates applied
        self.train_skipped = 0    # Times training was skipped (buffer not warm)
        self.last_loss     = 0.0
        self.last_q_mean   = 0.0
        self.last_td_error = 0.0
        self.debug_training = False  # Set True for verbose per-step training logs

        # v7 Fix 4: time-aware epsilon schedule
        self._total_decisions = 5000  # default, reconfigured by runner
        self._decision_count = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, valid_actions: List[int],
                      greedy: bool = False) -> int:
        """
        Epsilon-greedy with action masking.

        Args:
            state        : Current state vector (state_dim,)
            valid_actions: List of legal action indices
            greedy       : If True, forces epsilon=0 (evaluation mode)
        """
        if not valid_actions:
            raise ValueError("No valid actions available")

        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s).cpu().numpy()[0]

        masked = np.full(self.action_dim, -np.inf)
        masked[valid_actions] = q[valid_actions]
        return int(np.argmax(masked))

    # ── Transition storage ────────────────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done):
        """Push one (s, a, r, s', done) transition into the PER buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ── Online step (Phase 2 convenience wrapper) ─────────────────────────────

    def online_step(self, state: np.ndarray, valid_actions: List[int],
                    env, greedy: bool = False,
                    train_every: int = 1) -> Tuple[int, float, np.ndarray, bool, float, float]:
        """
        Full online learning step:
          1. Select action (eps-greedy with masking)
          2. Execute in env → get (next_state, reward, done)
          3. Store transition in PER buffer
          4. Train every `train_every` decisions if buffer is warm
          5. Decay epsilon per decision
          6. Return (action, reward, next_state, done, loss, q_mean)

        Debug: set agent.debug_training=True for per-step console logs.
        """
        action                       = self.select_action(state, valid_actions, greedy=greedy)
        next_state, reward, done, _  = env.step(action)

        # Defensive copy — ensure state and next_state differ (not aliased)
        state_copy      = np.array(state,      dtype=np.float32)
        next_state_copy = np.array(next_state, dtype=np.float32)
        self.store_transition(state_copy, action, reward, next_state_copy, float(done))
        self.steps_done += 1

        buf_size = len(self.replay_buffer)
        loss, q_mean, td_err = 0.0, 0.0, 0.0

        if buf_size >= self.min_replay_size and (self.steps_done % train_every == 0):
            # v10 Fix 3: Training taper — 4 grad steps while exploring, 2 after eps floor
            # Prevents overfitting on narrow replay data when greedy
            n_grad = 2 if self.epsilon <= self.epsilon_end + 0.01 else 4
            for _ in range(n_grad):
                _loss, _qm, _td = self.train_step()
                if _loss != 0.0:
                    loss, q_mean, td_err = _loss, _qm, _td
            if self.debug_training and self.train_steps % 10 == 0:
                print(f"    [DQN-TRAIN×{n_grad}] step={self.steps_done}, buf={buf_size}, "
                      f"loss={loss:.4f}, Q={q_mean:.3f}, "
                      f"eps={self.epsilon:.4f}, train_steps={self.train_steps}")
        else:
            self.train_skipped += 1
            if self.debug_training and self.train_skipped <= 5:
                print(f"    [DQN-SKIP] buf={buf_size} < min={self.min_replay_size} "
                      f"(need {self.min_replay_size - buf_size} more transitions)")

        self.update_epsilon()
        return action, reward, next_state_copy, done, loss, q_mean

    # ── Gradient update ───────────────────────────────────────────────────────

    def train_step(self) -> Tuple[float, float, float]:
        """
        One mini-batch gradient update (Double DQN + PER IS-weights + Huber loss).

        Always call after len(replay_buffer) >= batch_size.
        Increments self.train_steps on success.
        Returns: (loss, mean_q_pred, mean_td_error)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0

        (states, actions, rewards, next_states, dones,
         tree_idx, is_weights) = self.replay_buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        is_weights  = torch.FloatTensor(is_weights).to(self.device)

        # Current Q-values: Q(s, a) from policy net
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: action selected by policy net, value from target net
        with torch.no_grad():
            self.target_net.eval()
            next_a    = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q    = self.target_net(next_states).gather(1, next_a).squeeze(1)
            target_q  = rewards + (1.0 - dones) * self.gamma * next_q

        # Huber loss weighted by PER importance-sampling weights
        td_errors  = (target_q - current_q).detach().cpu().numpy()
        loss_elem  = self.criterion(current_q, target_q)   # element-wise Huber
        loss       = (is_weights * loss_elem).mean()

        if torch.isnan(loss):
            if self.debug_training:
                print(f"    [DQN-WARN] NaN loss detected at train_step={self.train_steps}")
            return float('nan'), float('nan'), float('nan')

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # NaN gradient guard
        if any(p.grad is not None and torch.isnan(p.grad).any()
               for p in self.policy_net.parameters()):
            if self.debug_training:
                print(f"    [DQN-WARN] NaN gradient at train_step={self.train_steps}")
            self.optimizer.zero_grad()
            return float('nan'), float('nan'), float('nan')

        self.optimizer.step()
        self.replay_buffer.update_priorities(tree_idx, td_errors)

        self.train_steps  += 1
        self.last_loss     = float(loss.item())
        self.last_q_mean   = float(current_q.mean().item())
        self.last_q_target = float(target_q.mean().item())
        self.last_td_error = float(np.abs(td_errors).mean())

        # Periodic target network sync (NOT every step — prevents Bellman collapse)
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.debug_training:
                print(f"    [DQN-TARGET] Synced target net at train_step={self.train_steps}")

        return self.last_loss, self.last_q_mean, self.last_td_error

    # ── Epsilon decay (called per DECISION in online mode) ────────────────────

    def configure_epsilon_schedule(self, total_expected_decisions: int = 0,
                                    tasks_per_day: float = 4.0,
                                    num_workers: int = 5,
                                    sim_days: int = 25,
                                    phase2_days: int = 0):
        """v9: Phase-2-only epsilon schedule.

        Decision estimate is scoped to PHASE 2 decisions only:
          expected_decisions = phase2_days × tasks_per_day

        Waypoints (fraction of Phase-2 expected decisions):
          30% → eps = 0.3      (rapid initial exploration reduction)
          60% → eps = 0.15     (agent mostly exploiting by mid-phase)
          85% → eps = 0.05     (floor, near-greedy)
        """
        if phase2_days <= 0:
            phase2_days = max(1, int(sim_days * 0.40))  # default 40% Phase 2
        # Pure Phase 2 decision estimate: one decision per arriving task per day
        estimated = int(phase2_days * tasks_per_day)
        self._total_decisions = max(estimated, 50)  # floor prevents div-by-zero
        self._decision_count  = 0
        print(f"  [DQN-v9] Epsilon schedule: phase2_days={phase2_days}, "
              f"tasks_per_day={tasks_per_day}, "
              f"total_expected={self._total_decisions} decisions, "
              f"eps=0.3 at {int(self._total_decisions*0.30)}, "
              f"eps=0.15 at {int(self._total_decisions*0.60)}, "
              f"eps=0.05 at {int(self._total_decisions*0.85)}")

    def update_epsilon(self, step: Optional[int] = None, **kwargs):
        """v9: Piecewise exponential decay scoped to Phase-2 decisions.

        Waypoints (fraction of total Phase-2 expected decisions):
          0%  → eps = epsilon_start (inherits from Phase 1, typically 0.4)
          30% → eps = 0.3
          60% → eps = 0.15
          85% → eps = 0.05 (floor)
        """
        # Handle legacy 'episode' keyword if passed
        if step is None and 'episode' in kwargs:
            step = kwargs['episode']

        self._decision_count = getattr(self, '_decision_count', 0) + 1
        total = getattr(self, '_total_decisions', 100)
        frac  = self._decision_count / max(total, 1)

        # Piecewise linear between waypoints
        if frac < 0.30:
            # epsilon_start → 0.3  (first 30%)
            self.epsilon = self.epsilon_start - (self.epsilon_start - 0.3) * (frac / 0.30)
        elif frac < 0.60:
            # 0.3 → 0.15           (next 30%)
            self.epsilon = 0.3 - (0.3 - 0.15) * ((frac - 0.30) / 0.30)
        elif frac < 0.85:
            # 0.15 → floor         (remaining 25%)
            self.epsilon = 0.15 - (0.15 - self.epsilon_end) * ((frac - 0.60) / 0.25)
        else:
            self.epsilon = self.epsilon_end

        self.scheduler.step(step if step is not None else self.train_steps)

    def set_epsilon(self, eps: float):
        """Manually set epsilon (e.g. when transitioning from Phase 1 to Phase 2)."""
        self.epsilon = float(np.clip(eps, self.epsilon_end, 1.0))

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict':  self.optimizer.state_dict(),
            'scheduler_state_dict':  self.scheduler.state_dict(),
            'epsilon':      self.epsilon,
            'steps_done':   self.steps_done,
            'train_steps':  self.train_steps,
            'architecture': 'DuelingDQN_v4_online',
            'state_dim':    self.state_dim,
            'action_dim':   self.action_dim,
        }, path)
        print(f"  [ckpt] Saved → {path}  (eps={self.epsilon:.4f}, "
              f"steps={self.steps_done}, train={self.train_steps})")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        self.target_net.load_state_dict(ckpt['target_net_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epsilon     = ckpt.get('epsilon',     self.epsilon)
        self.steps_done  = ckpt.get('steps_done',  0)
        self.train_steps = ckpt.get('train_steps', 0)
        print(f"  [ckpt] Loaded ← {path}  (eps={self.epsilon:.4f}, "
              f"steps={self.steps_done}, train={self.train_steps})")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DQNAgent v4 (Online Learning)")
    print("=" * 60)

    agent = DQNAgent()
    print(f"✓ Init: device={agent.device}, eps={agent.epsilon:.2f}, "
          f"state_dim={agent.state_dim}")

    # Forward pass
    state = np.random.rand(config.STATE_DIM).astype(np.float32)
    with torch.no_grad():
        t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q = agent.policy_net(t)
    assert q.shape == (1, config.ACTION_DIM), f"Got {q.shape}"
    print(f"✓ Forward pass: Q-values shape {tuple(q.shape)}")

    # Action selection
    valid = list(range(50))
    a = agent.select_action(state, valid)
    assert a in valid
    print(f"✓ Action selection: a={a}")

    # Fill buffer and train
    for _ in range(200):
        s  = np.random.rand(config.STATE_DIM).astype(np.float32)
        ns = np.random.rand(config.STATE_DIM).astype(np.float32)
        agent.store_transition(s, np.random.randint(140), np.random.randn(), ns, False)

    if len(agent.replay_buffer) >= agent.batch_size:
        loss, q_mean, td = agent.train_step()
        print(f"✓ Train step: loss={loss:.4f}, Q={q_mean:.4f}, TD={td:.4f}")

    # Epsilon decay (per-decision mode)
    eps_before = agent.epsilon
    agent.update_epsilon()
    print(f"✓ Epsilon decay (per-decision): {eps_before:.4f} → {agent.epsilon:.4f}")

    # Checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        agent.save(f.name)
        agent2 = DQNAgent()
        agent2.load(f.name)
        assert abs(agent2.epsilon - agent.epsilon) < 1e-6
    print("✓ Checkpoint save/load OK")

    print("=" * 60)
    print("DQNAgent v4 tests passed!")
    print("=" * 60)
