# Reinforcement Learning Theory & Mathematical Foundations
### RL-Driven Agentic Project Manager — Complete Technical Reference

> This document provides a **complete first-principles derivation** of every reinforcement learning concept used in this project, along with the full mathematical modelling of the task-allocation environment and a detailed rationale for every hyperparameter choice. It is intended as a standalone reference for anyone wishing to understand or reproduce the system from scratch.

---

## Table of Contents

1. [Markov Decision Processes](#1-markov-decision-processes)
2. [Partial Observability & POMDPs](#2-partial-observability--pomdps)
3. [Bellman Equations](#3-bellman-equations)
4. [Q-Learning & Temporal Difference Learning](#4-q-learning--temporal-difference-learning)
5. [Function Approximation with Neural Networks](#5-function-approximation-with-neural-networks)
6. [Loss Function & Backpropagation](#6-loss-function--backpropagation)
7. [Target Networks](#7-target-networks)
8. [Experience Replay & Prioritized Experience Replay](#8-experience-replay--prioritized-experience-replay)
9. [Epsilon-Greedy Exploration](#9-epsilon-greedy-exploration)
10. [Action Masking](#10-action-masking)
11. [Double DQN](#11-double-dqn)
12. [Dueling DQN Architecture](#12-dueling-dqn-architecture)
13. [Convergence & Stability](#13-convergence--stability)
14. [Environment Mathematical Model](#14-environment-mathematical-model)
15. [Complete Hyperparameter Reference](#15-complete-hyperparameter-reference)

---

## 1. Markov Decision Processes

### 1.1 Formal Definition

A **Markov Decision Process (MDP)** is defined by the tuple *(S, A, T, R, γ)*:

| Symbol | Meaning |
|--------|---------|
| **S** | State space — all possible configurations of the world |
| **A** | Action space — all decisions the agent can make |
| **T(s, a, s')** | Transition probability — P(s' \| s, a) |
| **R(s, a, s')** | Reward function — scalar feedback signal |
| **γ ∈ [0, 1)** | Discount factor — importance of future rewards |

The **Markov property** is the cornerstone: the future is conditionally independent of the past given the present state:
```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

This makes the problem tractable — we only need to represent the current state, not the full history.

### 1.2 Agent Objective

The agent seeks a **policy** π: S → A (or a distribution over actions) that maximises the expected **discounted return**:

```
G_t = Σ_{k=0}^{∞}  γ^k · R_{t+k+1}

J(π) = E_π [ G_0 ]
```

The discount factor γ serves two purposes:
1. **Mathematical convergence**: ensures G_t is finite for bounded rewards
2. **Preference for early rewards**: a reward r steps in the future is worth γ^r today

**In this project:** γ = 0.95, EPISODE_HORIZON = 100 steps. Effective look-ahead ≈ 1/(1−0.95) = 20 timesteps, which is enough to plan across task completion cycles without future rewards becoming negligibly small.

---

## 2. Partial Observability & POMDPs

### 2.1 Why Standard MDP is Insufficient

In this environment, **worker skill levels are hidden** — the agent cannot directly observe them. This violates the Markov property for raw observations alone.

A **Partially Observable MDP (POMDP)** extends the MDP tuple with:
- **O**: Observation space (what the agent actually sees)
- **Ω(o \| s, a)**: Observation probability given state and action
- **b(s)**: Belief state — posterior distribution over true states given history

### 2.2 Belief State Update (Bayesian Inference)

We model each worker's skill as a **Beta distribution** (conjugate prior for bounded continuous values):

```
Prior:         skill_w ~ Beta(α_0, β_0)   (α_0 = β_0 = 2.0 → centered at 0.5)

Observation:   quality_score  ∈ [0, 1]

Posterior update (after observing quality q_w):
               α_w ← α_w + q_w
               β_w ← β_w + (1 − q_w)

Posterior mean:  μ_w = α_w / (α_w + β_w)
Posterior var:   σ²_w = α_w β_w / [(α_w + β_w)² (α_w + β_w + 1)]
```

Both μ_w and σ²_w enter the 88-dim state vector, allowing the DQN to reason about both **estimated skill** and **estimation uncertainty**.

### 2.3 POMDP → Approximate MDP

Rather than tracking the full POMDP belief (exponential state space), we compress the belief into the 10-dim belief state vector [μ_1,...,μ_5, σ²_1,...,σ²_5] and feed it directly to the DQN. This converts the POMDP into an **approximate MDP** over belief states, a widely-used approach in deep RL.

---

## 3. Bellman Equations

### 3.1 State Value Function

The **state value function** V^π(s) gives the expected return when starting from state s and following policy π:

```
V^π(s) = E_π [ G_t | S_t = s ]
        = E_π [ R_{t+1} + γ · V^π(S_{t+1}) | S_t = s ]
```

The second form is the **Bellman expectation equation** — it expresses V^π recursively.

### 3.2 Action-Value (Q) Function

The **Q-function** Q^π(s, a) gives the expected return when taking action a from state s, *then* following π:

```
Q^π(s, a) = E_π [ G_t | S_t = s, A_t = a ]
           = E_π [ R_{t+1} + γ · Q^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a ]
```

The relationship between V and Q:
```
V^π(s) = Σ_a π(a|s) · Q^π(s, a)
```

### 3.3 Bellman Optimality Equations

For the **optimal policy** π*, the optimal Q-function satisfies:

```
Q*(s, a) = E [ R_{t+1} + γ · max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a ]
```

This is the **Bellman optimality equation**. If we can solve it exactly, the optimal policy is simply:
```
π*(s) = argmax_a Q*(s, a)
```

Q-learning and DQN are both approximation methods for solving this equation at scale.

---

## 4. Q-Learning & Temporal Difference Learning

### 4.1 Temporal Difference (TD) Error

Instead of waiting for a full episode return (Monte Carlo), TD methods use **bootstrapping** — updating estimates using other estimates:

```
δ_t  =  R_{t+1} + γ · max_{a'} Q(S_{t+1}, a')  −  Q(S_t, A_t)
        ──────────────────────────────────────────    ──────────
             TD target (what we want Q to be)         current Q
```

δ_t is the **TD error** — a measure of prediction surprise. When δ_t → 0 everywhere, Q has converged to Q*.

### 4.2 Tabular Q-Learning Update

For a finite state/action space, we update Q directly:

```
Q(S_t, A_t) ←  Q(S_t, A_t)  +  α · δ_t
```

where α ∈ (0, 1] is the learning rate. This is guaranteed to converge to Q* under:
- All (s, a) pairs visited infinitely often
- Learning rate satisfies Robbins-Monro conditions: Σα_t = ∞, Σα_t² < ∞

With 88-dim states and 140 actions, a tabular Q-table would require a lookup table of size |S|^88 × 140 — infeasible. We need function approximation.

---

## 5. Function Approximation with Neural Networks

### 5.1 Q-Network Parameterization

We approximate Q*(s, a) with a neural network Q(s, a; θ):

```
Q(s, a; θ) ≈ Q*(s, a)    for all  (s, a) ∈ S × A
```

The network parameters θ are updated via gradient descent to minimize the Bellman error.

**Why a neural network?**
- **Generalisation**: similar states produce similar Q-values, enabling transfer
- **Continuous input**: can process the 88-dim real-valued POMDP observation
- **Compact representation**: millions fewer parameters than a tabular approach

### 5.2 Network Architecture (v3 — Dueling DQN)

See Section 12 for the full Dueling DQN derivation. The high-level architecture:

```
Input: s ∈ ℝ^88  (POMDP observation)
  ↓
Linear(88→256) → LayerNorm → ReLU
Linear(256→256) → LayerNorm → ReLU    [shared backbone]
  ↓ splits into:
Value stream: Linear(256→128) → ReLU → Linear(128→1)       → V(s; θ_V)
Advantage stream: Linear(256→128) → ReLU → Linear(128→140) → A(s,a; θ_A)
  ↓
Q(s, a; θ) = V(s; θ_V) + A(s,a; θ_A) − mean_{a'} A(s,a'; θ_A)
```

**Why LayerNorm over BatchNorm?**
LayerNorm normalises over the feature dimension (not the batch dimension), making it compatible with single-sample inference during action selection. BatchNorm's running statistics become unreliable with non-stationary distributions like RL observations.

### 5.3 Weight Initialisation (Xavier Uniform)

For a linear layer mapping from n_in to n_out neurons:

```
W_{ij}  ~  Uniform( −√(6 / (n_in + n_out)),  +√(6 / (n_in + n_out)) )
b        =  0
```

**Why Xavier?** It maintains the variance of activations and gradients across layers, preventing vanishing/exploding gradient pathologies at initialisation. Designed for ReLU-family activations when combined with the Kaiming (He) variant, though Xavier uniform is also effective here.

---

## 6. Loss Function & Backpropagation

### 6.1 Huber Loss (Smooth L1)

Given a predicted Q-value q̂ = Q(s, a; θ) and a target y, the **Huber loss** is:

```
L_δ(y, q̂) = {  ½ (y − q̂)²           if |y − q̂| ≤ δ   [quadratic]
              {  δ · |y − q̂| − ½ δ²   if |y − q̂| > δ   [linear]
```

In PyTorch's `SmoothL1Loss`, δ = 1.0 (the default).

**Why not MSE?** With rewards of ±20 (scaled to ±2), large TD-errors produce huge MSE gradients (proportional to error²), causing parameter oscillations. The Huber loss clips the gradient magnitude to δ=1.0 for large errors, providing **robust gradient estimation**.

### 6.2 Importance-Sampling Weighted Loss (PER Correction)

With Prioritized Experience Replay, transitions are sampled with probability P(i) ∝ p_i^α. This creates a **biased gradient** compared to uniform sampling. The IS weight corrects for this:

```
w_i = (N · P(i))^{−β}
                                  (normalised so max_i(w_i) = 1)
```

The effective loss per transition becomes:
```
ℓ_i = w_i · L_δ(y_i, q̂_i)

Total loss: L(θ) = (1/B) Σ_{i∈batch} ℓ_i
```

As β → 1, the IS weights fully correct the bias. We anneal β from 0.4 → 1.0 over training, meaning:
- **Early training**: some bias is *acceptable* (more aggressive learning from high-priority transitions)
- **Late training**: fully unbiased gradient estimation for convergence guarantees

### 6.3 Backpropagation

Given the batch loss L(θ), parameters are updated via:

```
∇_θ L(θ) = (1/B) Σ_i  w_i · ∂L_δ(y_i, Q(s_i, a_i; θ)) / ∂θ

θ ← θ − α · ∇_θ L(θ)          [basic gradient descent]
```

**Adam optimiser** maintains per-parameter adaptive learning rates using first (m) and second (v) moment estimates of the gradient:

```
m_t = β_1 m_{t-1} + (1 − β_1) ∇_θ L        (momentum,  β_1 = 0.9)
v_t = β_2 v_{t-1} + (1 − β_2) ∇_θ L²       (RMSProp,   β_2 = 0.999)

m̂_t = m_t / (1 − β_1^t)                    (bias correction)
v̂_t = v_t / (1 − β_2^t)

θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```

Adam's adaptive learning rates mean that rarely-updated parameters (e.g., advantage head neurons for rarely-chosen actions) receive larger effective steps, beneficial in sparse-reward settings.

**Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)` rescales the gradient vector if ‖∇‖ > 1.0, preventing destructive updates in non-stationary reward landscapes.

---

## 7. Target Networks

### 7.1 Instability Without Target Networks

If we used a single network for both Q prediction and target computation:

```
y_t = R + γ · max_a' Q(S', a'; θ)
L   = (y_t − Q(S, A; θ))²
```

Updating θ changes both Q(S,A;θ) *and* the target y_t simultaneously. This creates a **moving target** — like chasing a shadow that moves with every step. The resulting training is highly unstable and often diverges.

### 7.2 Frozen Target Network

We introduce a second copy θ⁻ that is **frozen** for C steps:

```
y_t = R + γ · max_a' Q(S', a'; θ⁻)    [target net, frozen]
L   = (y_t − Q(S, A; θ))²             [policy net, updated each step]
```

Every C steps: θ⁻ ← θ (hard update).

This provides a **stable learning signal** for C steps, dramatically reducing oscillations.

**Choice of C = 200 steps (v3):** with a batch size of 128 and wider network, the policy net changes more significantly per update. A longer freeze period (200 vs 100) provides more stability. The downside is slower propagation of learned values, which is acceptable given the 5000-episode training budget.

---

## 8. Experience Replay & Prioritized Experience Replay

### 8.1 Vanilla Experience Replay

At each step, the agent stores transition (s, a, r, s', done) in a circular buffer of capacity N=50,000. During training, mini-batches of size B=128 are sampled uniformly.

**Benefits:**
1. **Decorrelates consecutive samples**: consecutive environment steps are highly correlated; uniform random sampling breaks this
2. **Data efficiency**: each transition can be replayed multiple times

**Limitation for sparse rewards:** in a 100-step episode with rewards mainly from terminal events, only ~1-2 transitions per episode carry the +20/-20 signal. In a 50,000-capacity buffer, these transitions have probability 2/50,000 = 0.004% of being sampled — effectively invisible.

### 8.2 Prioritized Experience Replay (PER)

**Key idea:** sample transitions proportionally to their Bellman error magnitude — large errors indicate "surprising" transitions where the agent has most to learn.

**Priority:** `p_i = (|δ_i| + ε)^α`

where:
- |δ_i| = |y_i − Q(s_i, a_i; θ)| is the TD-error
- ε = 1e-5 is a small floor to ensure non-zero probability for all transitions
- α = 0.6 interpolates between uniform (α=0) and fully prioritised (α=1)

**Sampling probability:** `P(i) = p_i / Σ_j p_j`

### 8.3 Sum-Tree Implementation

Naively computing P(i) ∝ p_i requires O(N) to sample and O(N) to update. The **sum-tree** reduces both to **O(log N)**:

```
                 [100]               ← root = Σ all priorities
                /     \
           [60]          [40]
          /    \         /   \
       [40]   [20]    [10]  [30]    ← internal sums
       / \   / \     / \   / \
      [p₁][p₂][p₃][p₄][p₅][p₆][p₇][p₈]  ← leaf priorities
```

To sample: draw v ~ Uniform(0, root), traverse tree left/right until leaf.
To update: change leaf priority, propagate difference up to root.

### 8.4 Importance Sampling Correction

Since P(i) ≠ 1/N, the gradient estimator is biased. The IS weight w_i corrects for this:

```
w_i = (N · P(i))^{−β}    normalized by max_j(w_j)
```

β anneals from 0.4 → 1.0 over 500,000 frames (≈ 5,000 episodes × 100 steps), ensuring that convergence guarantees are satisfied as training matures.

---

## 9. Epsilon-Greedy Exploration

### 9.1 The Exploration-Exploitation Dilemma

The agent must balance:
- **Exploration**: trying new actions to discover high-reward trajectories
- **Exploitation**: using current knowledge to maximise reward

With 140 actions and a POMDP, under-exploration means many action-state pairs are never visited, and the Q-function over them remains a poor extrapolation.

### 9.2 ε-Greedy Policy

At each step:
```
a_t = { random valid action        with probability ε    [explore]
      { argmax_a Q(s_t, a; θ)      with probability 1-ε  [exploit]
```

### 9.3 Exponential Epsilon Decay

```
ε_t = max(ε_min, ε_0 · ε_decay^t)

ε_0     = 1.0     (fully random initially)
ε_min   = 0.05    (5% exploration maintained forever)
ε_decay = 0.9997  (per episode)
```

**Derivation of decay schedule:**
ε reaches the floor ε_min when:
```
ε_0 · ε_decay^T = ε_min
T = ln(ε_min / ε_0) / ln(ε_decay)
  = ln(0.05) / ln(0.9997)
  ≈ 2996 / 0.0003
  ≈ 9,987 episodes    (theoretical — floor kicks in at ~5000 ep)
```

So ε ≈ 0.22 at episode 5000, meaning the agent is still exploring 22% of the time at the end of training. For a 140-action POMDP with sparse rewards, this extended exploration is crucial.

**Why ε_min = 0.05?** Prevents the agent from becoming completely deterministic — small random perturbations help escape local optima even in exploitation mode.

---

## 10. Action Masking

The environment can forbid certain actions at certain states (e.g., assigning an already-assigned task, or a worker in burnout). Naively, the agent could waste updates learning to assign −∞ Q-values to invalid actions.

**Action masking solution**: at inference, set Q-values of invalid actions to −∞ *before* the argmax:

```python
masked_q = full(140, −∞)
masked_q[valid_actions] = Q(s; θ)[valid_actions]
a* = argmax(masked_q)
```

This ensures the argmax always selects a legal move without requiring the network to explicitly learn which actions are invalid (which would be wasteful and slow).

---

## 11. Double DQN

### 11.1 Overestimation Bias in Vanilla DQN

The max operation in the Bellman target introduces **positive bias**:

```
E[ max_a' Q(s', a'; θ⁻) ]  ≥  max_a' E[ Q(s', a'; θ⁻) ]
```

(Jensen's inequality applied to the convex max function)

This means the target network systematically overestimates Q*, causing the agent to be overoptimistic and sometimes diverge.

### 11.2 Double DQN Correction

Decouple action **selection** and action **evaluation**:

```
Vanilla DQN:   y = r + γ · Q(s', argmax_a' Q(s', a'; θ⁻); θ⁻)
                                  ─────────────────────────
                                  same network selects AND evaluates

Double DQN:    a* = argmax_a' Q(s', a'; θ)         [policy net selects]
               y  = r + γ · Q(s', a*; θ⁻)          [target net evaluates]
```

This reduces the overestimation bias substantially because the errors in selection and evaluation are decoupled — a positive error in selection is not automatically amplified by evaluation.

---

## 12. Dueling DQN Architecture

### 12.1 Motivation

In task allocation, many states have similar value regardless of which valid task is assigned next (the agent just needs to keep assigning tasks). In such states, accurately estimating the advantage of each specific action is unnecessary — what matters more is V(s), the value of being in that state.

Vanilla DQN conflates V and A into a single Q-value, requiring all 140 action neurons to update simultaneously for every transition (even though only one action was taken).

### 12.2 Mathematical Decomposition

Any Q-function can be decomposed as:

```
Q^π(s, a) = V^π(s) + A^π(s, a)

where:
  V^π(s)   = E_π [Q^π(s, A)]                    (state value)
  A^π(s,a) = Q^π(s, a) − V^π(s)                 (advantage of action a over average)

Note: E_π [A^π(s, A)] = 0   by definition
```

### 12.3 Identifiability Problem & Solution

Naively, V and A are not separately identifiable from Q alone (any constant c can be shifted from V to A). The **mean subtraction trick** resolves this:

```
Q(s, a; θ) = V(s; θ_V) + [ A(s, a; θ_A) − mean_{a'} A(s, a'; θ_A) ]
```

This forces `mean_a A(s, a; θ_A) = 0`, making the decomposition unique.

### 12.4 Practical Benefits

| Property | Benefit |
|----------|---------|
| V(s) updated on every step | Faster value function learning |
| A(s,a) captures fine-grained action differences | Better policy in similar-value states |
| Separate streams | Reduced gradient interference between V and A |

In task allocation: when all 5 workers are available and 10 tasks are pending, V(s) captures "this is a healthy state", while A(s,a) captures "this specific worker-task pair is 15% better than the rest."

---

## 13. Convergence & Stability

### 13.1 Convergence Conditions for DQN

Theoretical guarantees require:
1. **Ergodicity**: all (s,a) pairs visited with positive probability infinitely often
2. **Decaying learning rate**: Σ α_t = ∞, Σ α_t² < ∞
3. **Contraction**: the Bellman operator is a contraction in L_∞ norm (guaranteed for γ < 1)

In practice with deep networks, we use a **fixed learning rate** with the Adam optimiser and rely on the empirical stability of the training process.

### 13.2 Cosine Annealing Warm Restarts (SGDR)

The LR scheduler uses cosine annealing:

```
α_t = α_min + ½(α_max − α_min)(1 + cos(π · t_cur / T_i))
```

where t_cur is steps since last restart, T_i is the current cycle length.

**After warm restart** (every T_0=500 episodes), α resets to α_max. This serves two functions:
1. **Fine-grained convergence** during the cosine decay phase
2. **Local optima escape** at warm restarts (higher LR temporarily)

This is particularly valuable in non-stationary RL loss landscapes where the loss surface shifts as the policy improves.

### 13.3 Stability Monitors

| Monitor | Trigger | Meaning |
|---------|---------|---------|
| NaN loss check | `torch.isnan(loss)` | Gradient explosion or numerical error |
| NaN gradient check | `torch.isnan(param.grad)` | Same, at gradient level |
| Q-value explosion | `|Q| > 1000` | Divergence — target network desynchronised |
| Early stopping | 1000 eps no improvement | Convergence or stuck in local optimum |

---

## 14. Environment Mathematical Model

### 14.1 State Space Decomposition

```
s ∈ ℝ^88  =  [w_features | t_features | b_features | g_features | padding]
               ───15────    ────40────    ───10────    ───3────     ──20──
```

**Worker features** (w_features ∈ ℝ^15): for each worker w ∈ {1,...,5}:
```
w_i = [ load_i / MAX_WORKER_LOAD,    ∈ [0,1]
         fatigue_i / FATIGUE_THRESHOLD, ∈ [0,1]
         availability_i ]              ∈ {0,1}
```

**Task features** (t_features ∈ ℝ^40): for the 10 most urgent pending tasks:
```
t_j = [ priority_j / 3,                        ∈ [0,1]
         complexity_j / 5,                       ∈ [0,1]
         deadline_urgency_j,                     ∈ [0,1]  (normalised time-to-deadline)
         is_dependencies_met_j ]                 ∈ {0,1}
```

**Belief features** (b_features ∈ ℝ^10):
```
b = [ μ_1, ..., μ_5,       (posterior skill means)
      σ²_1, ..., σ²_5 ]    (posterior skill variances)
```

**Global features** (g_features ∈ ℝ^3):
```
g = [ t / T,                    ∈ [0,1]  (time progress)
      |completed| / N_tasks,    ∈ [0,1]  (completion rate)
      |failed| / N_tasks ]      ∈ [0,1]  (failure rate)
```

### 14.2 Action Space

The 140 actions encode:

```
a ∈ {0, ..., 139}

a ∈ [0,   99] → a = 5·task_id + worker_id    (assign task_id to worker_id)
a ∈ [100,119] → a = 100 + task_id            (defer task_id)
a ∈ [120,139] → a = 120 + task_id            (escalate task_id)
```

**Constraints (action masking):**
- Assign is valid iff: task unassigned, not complete/failed, worker available, all dependencies met
- Defer is valid iff: task unassigned, not complete/failed
- Escalate: task must be assigned and priority < 3

### 14.3 Reward Function

At each timestep t:

```
R_t =  Σ_{j: completed} (priority_j + 1) · REWARD_COMPLETION_BASE · quality_j
     + REWARD_DELAY_WEIGHT
     + REWARD_OVERLOAD_WEIGHT · σ(loads)
     + Σ_{j: newly_missed} REWARD_DEADLINE_MISS_PENALTY
     + terminal_bonus · 𝟙[all tasks completed]
```

With reward_scale = 0.1 applied uniformly:
- Max completion per step: ~(4 * 20.0 * 1.0) * 0.1 = 8.0 (all tasks priority 3, quality 1)
- Deadline miss penalty: 20.0 * 0.1 = 2.0 per missed task
- Delay penalty: constant 0.01 per step (negligible)

### 14.4 Transition Dynamics

**Stochastic completion:**
```
actual_completion_time ~ N(expected_completion_time, (0.3 · expected)²)
```

**Fatigue update:**
```
If load_w > OVERLOAD_THRESHOLD:
    fatigue_w ← fatigue_w + FATIGUE_ACCUMULATION_RATE  (0.2)
Else:
    fatigue_w ← max(0, fatigue_w − FATIGUE_RECOVERY_RATE)  (0.1)

If fatigue_w > FATIGUE_THRESHOLD (2.5):
    worker enters burnout for BURNOUT_RECOVERY_TIME (5) steps
```

**Deadline shock:**
```
With probability p_shock = 0.15 per step:
    randomly select pending task j
    deadline_j ← max(current_t + 1, deadline_j − DEADLINE_SHOCK_AMOUNT)
```

### 14.5 Objective Function

```
π* = argmax_π E_π [ Σ_{t=0}^{T} γ^t · R_t ]

subject to:
    (1)  a_t ∈ Valid(s_t)         [action masking constraint]
    (2)  load_w ≤ MAX_WORKER_LOAD   [parallel task capacity]
    (3)  fatigue_w < FATIGUE_THRESHOLD  [burnout prevention]
```

---

## 15. Complete Hyperparameter Reference

### 15.1 Network Architecture

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `STATE_DIM` | 88 | Full POMDP observation including belief state |
| `ACTION_DIM` | 140 | 20×5 assign + 20 defer + 20 escalate |
| `HIDDEN_LAYERS` | [256, 256] | **v3 upgrade from [128,128]**: 128-unit layers had ~36K parameters for the shared backbone; 256-unit layers provide ~200K, sufficient to represent belief-state correlations. Empirically, doubling width vs adding depth reduces optimization difficulty |
| V-stream | [256→128→1] | Compact — V(s) is a scalar, needs less capacity than Q |
| A-stream | [256→128→140] | Moderate — advantage differences are fine-grained |
| LayerNorm | after each linear | Stabilises non-stationary POMDP input distributions |
| Activation | ReLU | Simple, fast; avoids vanishing gradients (no saturation for positive inputs) |
| Init | Xavier uniform | Variance-preserving for bounded ReLU activations |

### 15.2 Optimisation

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `LEARNING_RATE` | 0.001 | **v3 upgrade from 0.0005**: with batch size 128 and IS-weighted Huber loss, the per-update signal strength is larger. 0.001 provides faster convergence while cosine annealing prevents overshooting |
| Optimiser | Adam | Adaptive per-parameter LR handles sparse gradients well (advantage neurons for rarely-chosen actions get larger effective updates) |
| Adam ε | 1e-5 | Slightly larger than default (1e-8) to avoid numerical issues with very small squared gradient estimates early in training |
| `BATCH_SIZE` | 128 | **v3 upgrade from 64**: with a 50k PER buffer and episodes generating ~1-2 high-reward transitions each, batch 128 provides ~2-3x higher probability of including a completion event per batch |
| Gradient clip | norm ≤ 1.0 | Prevents destructive updates when TD-errors are large (e.g., at transition from reward_scale=0.1 × ±20 = ±2) |

### 15.3 LR Scheduler

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `LR_SCHEDULER_T0` | 500 episodes | One cosine cycle per 500 episodes; agent completes 10 cycles in 5000 episodes, allowing periodic escapes from local optima |
| T_mult | 1 | Constant period (not exponentially growing) — simpler to reason about in a fixed training budget |

### 15.4 Exploration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `EPSILON_START` | 1.0 | Fully random at start — critical to fill the PER buffer with diverse high-priority transitions |
| `EPSILON_END` | 0.05 | 5% permanent randomness — prevents complete determinism, helps escape local optima |
| `EPSILON_DECAY` | 0.9997 | **v3 update from 0.9994**: with 140 actions and POMDP, more exploration is needed. ε ≈ 0.22 at episode 5000 — agent still explores 22% of actions by end |

### 15.5 Replay Buffer

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `REPLAY_BUFFER_SIZE` | 50,000 | At 100 steps/episode × 5000 episodes = 500k total steps, the buffer stores 10% of all transitions (prioritising the most informative ones via PER) |
| `MIN_REPLAY_SIZE` | 1,000 | ~10 full episodes of warmup ensures the buffer contains diverse task/worker combinations before gradient updates begin |
| `PER_ALPHA` | 0.6 | Moderate prioritisation. α=1 would be too aggressive (over-replaying high-error early explorations before the Q-function is calibrated). α=0.6 balances priority with diversity |
| `PER_BETA_START` | 0.4 | Weak IS correction early on (agent needs to learn quickly from high-error transitions). Anneals to full correction (β=1.0) by end of training |
| `PER_BETA_FRAMES` | 500,000 | ≈ 100 steps × 5000 episodes — matches full training length so IS correction is fully applied by the time the agent converges |

### 15.6 Target Network

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `TARGET_UPDATE_FREQ` | 200 steps | **v3 update from 100**: wider network + larger batch size means the policy net changes more per step. Longer freeze (200 vs 100) provides more stable Bellman targets. Trade-off: slightly slower value propagation, acceptable with 5000-episode budget |
| Update type | Hard copy (θ⁻ ← θ) | Simpler than soft updates (τ polyak averaging). For discrete action spaces, hard updates at lower frequency perform comparably to soft updates |

### 15.7 Discount & Environment

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `GAMMA` | 0.95 | Effective horizon ≈ 1/(1-0.95) = 20 steps. Since task completion typically takes 5-30 steps from assignment, γ=0.95 allows future completion rewards to propagate back to assignment decisions with 60-90% of their value intact |
| `EPISODE_HORIZON` | 100 | Enough for most tasks to complete (deadline range [20,60]) while keeping episodes short enough to maintain non-stationary training diversity |
| `reward_scale` | 0.1 | Scales raw rewards of [-20, +80] to [-2, +8]. This keeps the Huber loss in its quadratic regime (|error| ≤ 1) for most transitions, providing smooth gradients |

---

*Document generated for AMD SlingShot Hackathon — RL-Driven Agentic Project Manager v3*
