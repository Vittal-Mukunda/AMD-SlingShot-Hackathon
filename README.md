# RL-Driven Agentic Project Manager

> **AMD SlingShot Hackathon** — A full reinforcement learning pipeline for optimal dynamic project task allocation under partial observability, fatigue, deadline shocks, and stochastic task completion.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start](#3-quick-start)
4. [Installation](#4-installation)
5. [Running the Program](#5-running-the-program)
6. [POMDP Formulation](#6-pomdp-formulation)
7. [State Space (88-dim)](#7-state-space-88-dim)
8. [Action Space (140 actions)](#8-action-space-140-actions)
9. [Reward Function](#9-reward-function)
10. [DQN Architecture (v3 — Dueling DQN)](#10-dqn-architecture-v3--dueling-dqn)
11. [DQN Hyperparameter Tuning Guide](#11-dqn-hyperparameter-tuning-guide)
12. [Environment Dynamics](#12-environment-dynamics)
13. [Baseline Policies](#13-baseline-policies)
14. [Training Process](#14-training-process)
15. [Evaluation Methodology](#15-evaluation-methodology)
16. [Statistical Testing](#16-statistical-testing)
17. [Ablation Studies](#17-ablation-studies)
18. [Directory Structure](#18-directory-structure)
19. [Expected Output Files](#19-expected-output-files)
20. [Interpreting Results](#20-interpreting-results)
21. [Theoretical Reference](#21-theoretical-reference)
22. [Reproducibility](#22-reproducibility)
23. [Future Work](#23-future-work)

---

## 1. Project Overview

This system trains a **Deep Q-Network (DQN)** agent to manage a portfolio of software engineering tasks assigned to workers in a simulated project environment. The agent must:

- Assign tasks to workers with **partially observable skill levels** (learned via Bayesian belief updates)
- Manage **worker fatigue** to prevent quality degradation and burnout
- Respond to **sudden deadline shocks** (external crises reducing deadlines mid-episode)
- Maximise **throughput and quality** while minimising delays and overload

The system is benchmarked against 5 hand-crafted heuristic baselines and validated using Welch's t-tests with Bonferroni correction.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    DQN Agent (v3)                                 │
│                                                                    │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐  │
│  │  88-dim POMDP│    │         Dueling Q-Network               │  │
│  │  Observation │───▶│  Input(88) → 256 → 256 [LayerNorm+ReLU] │  │
│  └──────────────┘    │       ↓               ↓                 │  │
│                       │  Value(1)      Advantage(140)           │  │
│  ┌──────────────┐    │       └────── Q(s,a) ─────┘            │  │
│  │   PER Buffer │◀───│    Q = V + A - mean(A)                  │  │
│  │  (Sum-Tree)  │    └─────────────────────────────────────────┘  │
│  └──────┬───────┘                                                  │
│         │ IS-weighted      Double DQN targets                      │
│         ▼                 policy selects, target evaluates         │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────────────┐  │
│  │ Mini-batch   │───▶│ Huber Loss   │──▶│ Adam + CosineWarmLR  │  │
│  │  B=128       │    │ + IS weights │   │ LR=0.001, T0=500 ep  │  │
│  └──────────────┘    └──────────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Key improvements in v3 over v2:**

| Component | v2 | v3 |
|-----------|----|----|
| Network | MLP [128,128] | Dueling DQN [256,256] + V/A streams |
| Replay | Uniform ReplayBuffer | Prioritized Experience Replay (sum-tree) |
| Target computation | Vanilla DQN | Double DQN (decoupled select/evaluate) |
| LR scheduling | Fixed | CosineAnnealingWarmRestarts (T₀=500 ep) |
| Batch size | 64 | 128 |
| LR | 0.0005 | 0.001 |
| ε decay | 0.9994/ep | 0.9997/ep (slower, for 140-action POMDP) |

---

## 3. Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train → baselines → evaluate → stats → plots)
python run_pipeline.py --full

# Or stage by stage:
python run_pipeline.py --train --episodes 5000
python run_pipeline.py --baselines
python run_pipeline.py --evaluate
python run_pipeline.py --stats
python run_pipeline.py --plots
```

---

## 4. Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)
- NumPy, SciPy, Matplotlib, Pandas

### Steps

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AMD-SlingShot-Hackathon.git
cd AMD-SlingShot-Hackathon

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install via uv for faster resolution
pip install uv
uv pip install -r requirements.txt
```

### Verify Installation

```powershell
# Test environment
python environment/project_env.py

# Test DQN agent (unit tests)
python agents/dqn_agent.py

# Quick 10-episode training smoke test
python tests/test_dqn_training.py
```

---

## 5. Running the Program

### 5.1 Full Training Pipeline

```powershell
# Run everything end-to-end (recommended for first run)
python run_pipeline.py --full

# With custom seed and shock probability
python run_pipeline.py --full --seed 42 --shock-prob 0.15

# For a quick test (100 episodes)
python run_pipeline.py --train --episodes 100
```

### 5.2 Individual Stages

```powershell
# Stage 1: Train DQN agent (5000 episodes)
python run_pipeline.py --train --episodes 5000

# Stage 2: Evaluate baseline policies (200 episodes each)
python run_pipeline.py --baselines

# Stage 3: Evaluate trained DQN under 4 conditions
python run_pipeline.py --evaluate

# Stage 4: Run statistical tests (Welch's t-test + Bonferroni + Cohen's d)
python run_pipeline.py --stats

# Stage 5: Generate plots
python run_pipeline.py --plots
```

### 5.3 Training Script Directly

```powershell
python training/train_dqn.py
```

The training loop automatically uses all v3 hyperparameters from `config.py`.

### 5.4 Demo (DQN vs Baselines Visual Comparison)

```powershell
python demo_run.py
```

### 5.5 Web Server & Agent Loop

```powershell
# Start FastAPI backend (port 8000)
uv run python scripts/run_server.py

# Start discrete AI agent orchestration loop
uv run python scripts/run_agent_loop.py
```

### 5.6 Testing

```powershell
# Quick DQN smoke test (10 episodes)
python tests/test_dqn_training.py

# Full stability test suite
python tests/test_stability.py

# Agent unit tests
python tests/test_agents.py

# Ablation smoke test
python tests/test_ablation.py
```

---

## 6. POMDP Formulation

The task allocation problem is modelled as a **Partially Observable Markov Decision Process (POMDP)**:

| Component | Definition |
|-----------|-----------|
| **State** `S` | True world state: worker skills, task states, fatigue levels |
| **Observation** `O` | 88-dim vector: worker load/fatigue, task urgency, **belief** over skills |
| **Action** `A` | 140 discrete actions: assign / defer / escalate |
| **Transition** `T` | Stochastic: noisy completion times, fatigue dynamics, deadline shocks |
| **Reward** `R` | Composite function (see §9) |
| **Belief** `B` | Bayesian posterior over each worker's skill (Beta distribution) |

**Partial observability** is central: worker skill levels are **hidden**. The agent maintains a **Belief State** (5 posterior means + 5 variances) updated incrementally with each task completion observation via Bayesian inference.

For full mathematical details, see [`docs/RL_THEORY.md`](docs/RL_THEORY.md).

---

## 7. State Space (88-dim)

```
State = [Worker Features | Task Features | Belief State | Global Context | Padding ]
         ─────15──────    ──────40──────   ───10─────    ────3────────    ──20────
```

| Segment | Dimensions | Content |
|---------|-----------|---------|
| **Worker features** | 5 × 3 = 15 | `[load/max, fatigue/threshold, availability]` per worker |
| **Task features** | 10 × 4 = 40 | `[priority/3, complexity/5, deadline_urgency, deps_met]` for top-10 urgent tasks |
| **Belief state** | 5 + 5 = 10 | Posterior mean and variance of each worker's skill (Beta distribution) |
| **Global context** | 3 | `[time_progress, completion_rate, failure_rate]` |
| **Padding** | 20 | Zero-padded to guarantee 88-dim |

---

## 8. Action Space (140 actions)

```
Actions [0 … 139]:
  [0   …  99]  → Assign task t to worker w  (20 tasks × 5 workers = 100 actions)
  [100 … 119]  → Defer task t               (20 tasks)
  [120 … 139]  → Escalate task t            (20 tasks, ↑priority, ↓completion time, resource cost)
```

**Action masking** is applied at every step: invalid actions (assigning already-assigned tasks, busy workers, unmet dependencies) are excluded from the agent's choice set via `env.get_valid_actions()`.

---

## 9. Reward Function

At each timestep *t*:

```
R(t) = Σ_{completed j} (priority_j + 1) × 20 × quality_j     [+completion bonus]
     + (−0.1)                                                   [−constant delay tick]
     + (−0.1) × σ(worker loads)                                [−load imbalance]
     + Σ_{missed j} (−20)                                      [−deadline miss penalty]
     × reward_scale (0.1)
```

| Component | Raw magnitude | Scaled (×0.1) |
|-----------|--------------|---------------|
| Task completion (+20 × priority × quality) | Up to +80 | Up to +8.0 |
| Deadline miss | −20 per task | −2.0 per task |
| Step delay | −0.1 constant | −0.01 |
| Load imbalance | −0.1 × σ | −0.02 max |

**Sparse reward problem**: rewards are episodic and rare — the vast majority of steps have near-zero reward. This is precisely what Prioritized Experience Replay is designed to solve.

---

## 10. DQN Architecture (v3 — Dueling DQN)

```
Input: s ∈ ℝ^88  (POMDP observation)
        │
  ┌─────▼───────────────────────────────┐
  │  Linear(88→256) → LayerNorm → ReLU  │  ← Shared Backbone
  │  Linear(256→256) → LayerNorm → ReLU │
  └──────────┬──────────────────────────┘
             │ splits
     ┌───────┘        └───────────────────┐
     ▼ Value stream                        ▼ Advantage stream
     Linear(256→128) → ReLU              Linear(256→128) → ReLU
     Linear(128→1)                       Linear(128→140)
     V(s; θ_V)                           A(s,a; θ_A)
         └──────────────────────────────────┘
                    Q(s,a) = V(s) + A(s,a) − mean_{a'} A(s,a')
                                     │
                              Output: 140 Q-values
```

| Hyperparameter | Value | Why |
|---------------|-------|-----|
| Architecture | Dueling DQN | Separates state value from action advantage — better in states where most actions are similar |
| Hidden layers | [256, 256] | 2× wider than v2; needed for 88-dim POMDP representation |
| Normalization | LayerNorm | Handles non-stationary RL input distributions |
| Replay | PER (sum-tree) | Prioritizes rare ±20 reward events |
| Target | Double DQN | Eliminates Q-value overestimation bias |
| LR schedule | CosineWarmRestarts | Escapes local optima periodically |

---

## 11. DQN Hyperparameter Tuning Guide

This section documents **every hyperparameter**, why the v2 value was changed, and how the new value was determined.

### Why the Agent Was Flatlining (v2 Diagnosis)

| Root Cause | v2 Problem | v3 Fix |
|-----------|-----------|--------|
| **Under-capacity network** | [128,128] MLP: ~36K params cannot model POMDP belief correlations | Dueling [256,256]: ~200K params + V/A decomposition |
| **Uniform replay misses rare events** | +20/-20 rewards diluted to 0.004% sampling probability in 50k buffer | PER sum-tree: high-TD-error transitions sampled 10–100× more often |
| **Overoptimistic Q-values** | Vanilla DQN max operator introduces upward bias → instability | Double DQN: separate selection and evaluation networks |
| **Premature exploitation** | ε=0.9994 → 0.05 by ep 3500; only 3500 of 5000 eps explore sufficiently | ε=0.9997 → 0.22 at ep 5000; full exploration throughout |
| **Weak gradient signal** | LR=0.0005 + batch 64 with Huber in linear regime → slow learning | LR=0.001 + batch 128 + cosine restarts |

### Hyperparameter-by-Hyperparameter Rationale

**`LEARNING_RATE`: 0.0005 → 0.001**
- With IS-weighted Huber loss and batch=128, the effective gradient scale is larger and more stable
- Cosine annealing prevents overshooting at this higher rate
- Formula: LR_effective = LR_base / sqrt(batch_size) scaling suggests 0.0005 × sqrt(128/64) ≈ 0.0007; rounded up for faster convergence

**`BATCH_SIZE`: 64 → 128**
- Probability of a completion event appearing in a uniform batch: (2 events/episode) × (1/50000) × 64 ≈ 0.26%
- With PER, high-priority events appear far more often, but larger batch still captures more diversity
- 128 is GPU-memory efficient for the 256-unit network and provides stable gradient estimates

**`HIDDEN_LAYERS`: [128,128] → [256,256]**
- Parameters: [128,128] shared backbone ≈ 88×128 + 128×128 = 27,648 params
- Parameters: [256,256] shared backbone ≈ 88×256 + 256×256 = 88,064 params
- The 88-dim POMDP observation includes belief state variances that require cross-feature interactions the 128-dim bottleneck couldn't represent

**`EPSILON_DECAY`: 0.9994 → 0.9997**
- With 140 actions, the agent needs to visit each action ~O(log(1/δ)) times to form a reliable Q-estimate at each state
- Slowing from 0.9994 (ε=0.05 at ep 3500) to 0.9997 (ε=0.22 at ep 5000) extends meaningful exploration across the full training run

**`TARGET_UPDATE_FREQ`: 100 → 200**
- Wider network (256 units) with larger batch (128) produces larger parameter updates per step
- Longer freeze period (200 steps) prevents the moving-target instability that arises when targets track a fast-changing policy network

**`PER_ALPHA = 0.6`**
- α=0 is uniform sampling (standard replay)
- α=1 is fully greedy priority (only replays highest-TD-error transitions, leading to overfitting to outliers)
- α=0.6: empirically validated value from Schaul et al. 2016 as a good balance across most RL problems

**`PER_BETA_START = 0.4`**
- β=0: no IS correction (biased but fast)
- β=1: full IS correction (unbiased but slow early on)
- β=0.4 starting point allows aggressive learning from priority samples early, with bias reduction as training matures

---

## 12. Environment Dynamics

### Worker Fatigue
- Workers accumulate fatigue when `load > OVERLOAD_THRESHOLD` (3)
- `fatigue += 0.2` per step when overloaded; `fatigue -= 0.1` per step when idle
- Burnout (`fatigue > 2.5`) renders a worker unavailable for 5 steps
- Output quality degrades as: `quality = (1 − 0.3 × fatigue)`

### Deadline Shocks
- With probability 0.15 per step, a random pending task has its deadline reduced by 10 timesteps
- Simulates external crises; forces agent to learn adaptive re-prioritisation

### Stochastic Completion
- Task completion time ~ `N(expected_time, (0.3 × expected_time)²)`
- Prevents exact schedule memorisation

### Task Dependencies
- Tasks form a **DAG** of depth ≤ 3 with 3 dependency chains
- Dependent tasks cannot be assigned until all prerequisites are completed

---

## 13. Baseline Policies

| Baseline | Strategy |
|---------|---------|
| **Random** | Uniformly random valid action each step |
| **Greedy** | Always assign highest-priority task to any available worker |
| **Shortest-Task-First (STF)** | Assigns task with shortest expected completion time |
| **Skill Matcher** | 10-episode skill observation phase; then matches task complexity to best-skill worker |
| **Hybrid** | Combines STF + Skill Matching with urgency-weighted prioritisation (strongest heuristic) |

---

## 14. Training Process

1. **Replay Warm-up** (1,000 steps): random policy fills the PER buffer with initial transitions
2. **Training loop** (up to 5,000 episodes):
   - ε-greedy action selection with action masking
   - Transition stored in PER buffer with maximum current priority
   - PER samples mini-batch (B=128) proportional to TD-error priorities
   - Double DQN Huber loss × IS weights, minimised via Adam+CosLR
   - PER priorities updated with new TD-errors after each gradient step
   - Target network hard-synced every 200 gradient steps
   - ε decayed by ×0.9997 per episode; cosine LR scheduler stepped per episode
3. **Best model tracking**: saved whenever moving-50-avg return improves (only after ep ≥ 50)
4. **Early stopping**: halts if no improvement for 1,000 consecutive episodes
5. **Stability monitoring**: NaN loss, NaN gradient, Q explosion (|Q| > 1000)
6. **Logging**: 17 metrics per episode → `results/training_log.csv`

> ⚠️ **Breaking change from v2**: checkpoint format changed (DuelingQNetwork vs QNetwork). Delete `checkpoints/best_model.pth` before re-training.

---

## 15. Evaluation Methodology

The trained agent is evaluated under **4 conditions** (200 episodes each):

| Condition | Description |
|-----------|-------------|
| **Standard** | Default environment parameters |
| **High Variance** | Completion time noise × 1.5 |
| **Frequent Shocks** | Deadline shock probability = 0.30 (2× default) |
| **Fixed Seed** | Constant seed — tests for overfitting |

All evaluations use `epsilon=0` (greedy policy) and **unscaled rewards**.

---

## 16. Statistical Testing

For each of 5 RL vs Baseline comparisons:

1. **Welch's t-test** (one-tailed, `alternative='greater'`): tests if RL mean > Baseline mean
2. **Bonferroni correction**: adjusted threshold `α = 0.05 / 5 = 0.01`
3. **Cohen's d**: `d = (μ_RL − μ_base) / σ_pooled` — `d > 0.8` = large effect

Results: `results/statistical_summary.csv`

---

## 17. Ablation Studies

| Ablation | Override | Question |
|---------|---------|---------|
| **Standard** | None | Control group |
| **No Fatigue** | `enable_fatigue=False` | Does the agent exploit fatigue management? |
| **No Shocks** | `enable_deadline_shocks=False` | Does shock resilience contribute? |
| **Full Info** | `fully_observable=True` | Value of perfect skill knowledge? |

```powershell
python evaluation/ablation_studies.py
python visualization/plot_ablations.py
```

---

## 18. Directory Structure

```
AMD-SlingShot-Hackathon/
├── run_pipeline.py            ← End-to-end CLI orchestrator  ← START HERE
├── config.py                  ← All hyperparameters & paths  (v3 tuned values)
├── requirements.txt
│
├── agents/
│   └── dqn_agent.py           ← Dueling DQN + PER + Double DQN + Cosine LR  (v3)
│
├── environment/
│   ├── project_env.py         ← POMDP environment (88-dim state, 140 actions)
│   ├── worker.py              ← Worker model (fatigue, skills, belief)
│   ├── task.py                ← Task model (deadlines, dependencies)
│   └── belief_state.py        ← Bayesian Beta-distribution skill tracking
│
├── baselines/
│   ├── random_baseline.py
│   ├── greedy_baseline.py
│   ├── stf_baseline.py
│   ├── skill_baseline.py
│   └── hybrid_baseline.py
│
├── training/
│   ├── train_dqn.py           ← DQN training loop (v3 — headless, CSV logging)
│   ├── train_baselines.py     ← Baseline evaluation runner
│   └── visualize.py           ← Learning curve plots
│
├── evaluation/
│   ├── evaluate_agent.py      ← 4-condition RL evaluation
│   ├── statistical_tests.py   ← Welch's t-test + Bonferroni + Cohen's d
│   └── ablation_studies.py    ← Ablation study runner
│
├── visualization/
│   ├── plot_metrics.py        ← "Money Plot" (RL vs Baselines bar chart)
│   └── plot_ablations.py      ← Ablation study bar chart
│
├── docs/
│   └── RL_THEORY.md           ← Complete RL theory & math reference  ← NEW
│
├── utils/
│   └── metrics.py             ← Composite score computation
│
├── tests/
│   ├── test_dqn_training.py   ← DQN smoke test (10 episodes)
│   ├── test_stability.py      ← Environment stability tests
│   ├── test_agents.py         ← Agent unit tests
│   └── test_ablation.py       ← Ablation smoke test
│
├── checkpoints/               ← Model checkpoints (auto-created)
├── results/                   ← CSVs and plots (auto-created)
└── logs/                      ← Training logs (auto-created)
```

---

## 19. Expected Output Files

After `python run_pipeline.py --full`:

| File | Description |
|------|-------------|
| `results/training_log.csv` | Per-episode: return, epsilon, Q-values, TD error, task metrics, reward breakdown |
| `checkpoints/best_model.pth` | Best DQN checkpoint (DuelingQNetwork v3) |
| `checkpoints/checkpoint_ep*.pth` | Periodic checkpoints every 100 episodes |
| `results/baseline_performance.csv` | 5 baselines × 200 episodes |
| `results/rl_test_performance.csv` | 4 conditions × 200 episodes |
| `results/statistical_summary.csv` | T-statistic, p-value, Cohen's d for 5 comparisons |
| `results/learning_curve.png` | 4-panel: returns, epsilon, Q-values, task completion |
| `results/money_plot.png` | Grouped bar chart with significance stars |
| `results/ablation_results.csv` | 4 ablation conditions × 50 episodes |
| `results/ablation_study.png` | Ablation bar chart with error bars |

---

## 20. Interpreting Results

### Learning Curve
- **Episode Return**: look for upward trend in moving average (MA-50) — indicates learning
- **Epsilon**: should decay from 1.0 → ~0.2 over 5000 episodes
- **Q-values**: gradual monotonic increase → value function calibrating correctly
- **Tasks Completed**: rising mean over episodes confirms better task throughput

### Money Plot
- Each bar = mean composite score ± SEM
- Stars (`*`, `**`, `***`) = Bonferroni-corrected significance
- RL Agent should separate from Random/Greedy and approach/surpass Hybrid

### What "Breaking the Plateau" Looks Like
After v3 changes, within the first 500 episodes you should see:
- Loss decreasing from ~4-5 → below 1.0
- Q-values growing from ~0 → 2-5 monotonically
- `comp_reward` in the CSV increasing with episodes
- MA-50 trending upward WITHOUT triggering early stopping by episode 1000

---

## 21. Theoretical Reference

A complete mathematical derivation of all RL concepts is in [`docs/RL_THEORY.md`](docs/RL_THEORY.md), covering:

- Markov Decision Processes & POMDP formulation
- Bellman equations (optimality and expectation forms)
- Q-learning & temporal difference learning
- Neural network function approximation
- Huber loss, backpropagation, Adam optimiser
- Target networks (moving target problem)
- Prioritized Experience Replay (sum-tree, IS weights)
- Epsilon-greedy exploration schedule
- Double DQN (overestimation bias)
- Dueling DQN (V + A decomposition)
- Convergence & stability theory
- Full environment mathematical model
- **Every hyperparameter** with theoretical justification

---

## 22. Reproducibility

Full deterministic reproducibility via:

```python
random.seed(seed)        # Python stdlib
np.random.seed(seed)     # NumPy
torch.manual_seed(seed)  # PyTorch
torch.cuda.manual_seed(seed)  # CUDA (if available)
```

Default seed: `42`. To reproduce exactly:

```powershell
python run_pipeline.py --full --seed 42
```

For CPU-only exact reproducibility (eliminates GPU non-determinism):

```powershell
$env:CUDA_VISIBLE_DEVICES=""; python run_pipeline.py --full --seed 42
```

---

## 23. Future Work

### Algorithmic Extensions
- **PPO / A2C**: Policy gradient methods may handle POMDP variance better via natural gradients
- **LSTM Q-Network**: Replace belief state with LSTM memory across timesteps
- **Rainbow DQN**: Combine Dueling + Double + PER + Distributional + NoisyNet + n-step returns

### Environment Improvements
- **Particle Filter Belief State**: More expressive uncertainty than Beta posterior
- **Domain Randomisation**: Randomise shock/fatigue/noise params during training for generalisation
- **Continuous Action Space**: Fractional task assignments via Actor-Critic (SAC)

### Multi-Agent Extensions
- **QMIX / MAPPO**: Decentralised agent execution with central training
- **Competitive Multi-Project**: Multiple managers competing for a shared worker pool

### Explainability
- **LLM Overlay**: Feed agent action sequences to Gemini/GPT for natural-language explanations
- **SHAP Attribution**: Identify which belief state components drive Q-value predictions

---

## Git Commands to Commit & Push

After all modifications are complete, run:

```powershell
cd "c:\Users\vitta\OneDrive\Desktop\Python 3.10\AMD_Repo\AMD-SlingShot-Hackathon"

# Stage all changes
git add agents/dqn_agent.py config.py training/train_dqn.py docs/RL_THEORY.md README.md

# Commit with descriptive message
git commit -m "feat: upgrade DQN to v3 (Dueling+PER+Double+CosLR) + RL theory docs

- agents/dqn_agent.py: DuelingQNetwork [256,256] with LayerNorm, V+A streams
- agents/dqn_agent.py: PrioritizedReplayBuffer with sum-tree (O(log N))
- agents/dqn_agent.py: Double DQN target computation
- agents/dqn_agent.py: CosineAnnealingWarmRestarts LR scheduler (T0=500 ep)
- config.py: HIDDEN_LAYERS [128,128]->[256,256], LR 0.0005->0.001,
             BATCH_SIZE 64->128, EPSILON_DECAY 0.9994->0.9997,
             TARGET_UPDATE_FREQ 100->200, added PER_*/LR_SCHEDULER_* keys
- training/train_dqn.py: pass episode to update_epsilon for scheduler
- docs/RL_THEORY.md: complete RL theory, math foundations, hyperparameter guide
- README.md: full rewrite with v3 architecture, installation, step-by-step commands"

# Push to GitHub
git push origin main
```

---

*AMD SlingShot Hackathon — RL-Driven Agentic Project Manager v3*