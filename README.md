# RL-Driven Agentic Project Manager

> **AMD SlingShot Hackathon** — A full reinforcement learning pipeline for optimal dynamic project task allocation under partial observability, fatigue, deadline shocks, and stochastic task completion.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [POMDP Formulation](#3-pomdp-formulation)
4. [State Space (88-dim)](#4-state-space-88-dim)
5. [Action Space (140 actions)](#5-action-space-140-actions)
6. [Reward Function](#6-reward-function)
7. [Environment Dynamics](#7-environment-dynamics)
8. [DQN Architecture](#8-dqn-architecture)
9. [Baseline Policies](#9-baseline-policies)
10. [Training Process](#10-training-process)
11. [Evaluation Methodology](#11-evaluation-methodology)
12. [Statistical Testing](#12-statistical-testing)
13. [Ablation Studies](#13-ablation-studies)
14. [Directory Structure](#14-directory-structure)
15. [Example Commands](#15-example-commands)
16. [Expected Output Files](#16-expected-output-files)
17. [Interpreting Results](#17-interpreting-results)
18. [Reproducibility](#18-reproducibility)
19. [Future Work](#19-future-work)

---

## 1. Project Overview

This system trains a **Deep Q-Network (DQN)** agent to manage a portfolio of software engineering tasks assigned to workers in a simulated project environment. The agent must:

- Assign tasks to workers with partially observable skill levels
- Manage worker fatigue to prevent quality degradation
- Respond to sudden deadline shocks (external crises)
- Maximise throughput and quality while minimising delays and overload

The system is benchmarked against 5 hand-crafted heuristic baselines and validated using Welch's t-tests with Bonferroni correction, demonstrating statistically significant superiority.

---

## 2. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (train → baselines → evaluate → stats → plots)
python run_pipeline.py --full

# Or run individual stages:
python run_pipeline.py --train --episodes 2000
python run_pipeline.py --baselines
python run_pipeline.py --evaluate
python run_pipeline.py --stats
python run_pipeline.py --plots
```

---

## 3. POMDP Formulation

The task allocation problem is modelled as a **Partially Observable Markov Decision Process (POMDP)**:

| Component | Definition |
|---|---|
| **State** `S` | True world state: worker skills, task states, fatigue levels |
| **Observation** `O` | 88-dim vector: worker load/fatigue, task urgency, **belief** over skills |
| **Action** `A` | 140 discrete actions: assign / defer / escalate |
| **Transition** `T` | Stochastic: noisy completion times, fatigue dynamics, deadline shocks |
| **Reward** `R` | Composite function (see §6) |
| **Belief** `B` | Bayesian posterior over each worker's skill (Beta distribution) |

The **partial observability** is central: worker skill levels are **hidden**. The agent maintains a **Belief State** (5 posterior means + 5 variances) that is updated incrementally with each task completion observation. This forces the agent to learn an *exploration vs exploitation* strategy for skill estimation — a key differentiator from greedy baselines.

---

## 4. State Space (88-dim)

```
State = [Worker Features | Task Features | Belief State | Global Context]
         ─────15──────     ──────40──────   ──────10────   ─────3────────
         Total: 68 dims (zero-padded to 88)
```

| Segment | Dimensions | Content |
|---|---|---|
| **Worker features** | 5 × 3 = 15 | `[load, fatigue, availability]` per worker |
| **Task features** | 10 × 4 = 40 | `[priority, complexity, deadline_urgency, deps_met]` for top-10 urgent tasks |
| **Belief state** | 5 + 5 = 10 | Posterior mean and variance of each worker's skill |
| **Global context** | 3 | `[time_progress, completion_rate, failure_rate]` |

---

## 5. Action Space (140 actions)

```
Actions [0 … 139]:
  [0  … 99]   → Assign task t to worker w  (20 tasks × 5 workers)
  [100 … 119] → Defer task t               (20 tasks)
  [120 … 139] → Escalate task t            (20 tasks, ↑priority, ↑speed, ↓cost)
```

**Action masking** is applied at every step: invalid actions (e.g. assigning an already-assigned task, or a busy worker) are excluded from the agent's choice set via `get_valid_actions()`.

---

## 6. Reward Function

At each timestep *t*, the agent receives:

```
R(t) = R_completion + R_delay + R_overload + R_throughput + R_deadline
```

| Component | Formula | Weight |
|---|---|---|
| **Completion bonus** | `+10 × priority_weight × quality` | Per completed task |
| **Delay penalty** | `−0.5 × Σ (time_in_queue / deadline)` | Per pending task |
| **Overload penalty** | `−5 × Σ (load − threshold)²` | Per overloaded worker |
| **Throughput bonus** | `+2 × tasks_completed_this_step` | Per step |
| **Deadline miss** | `−50` | Catastrophic, per missed deadline |
| **Strategic defer bonus** | `+1` | If no skilled worker available |

All raw rewards are scaled by `reward_scale=0.1` during training for DQN stability.

**Composite Score** (used for evaluation and comparison):

```
Score = 10 × tasks_completed − 0.5 × avg_delay − 5 × overload_events
        + 2 × tasks_completed − 50 × deadline_misses
```

---

## 7. Environment Dynamics

### Worker Fatigue
- Workers accumulate fatigue when `load > OVERLOAD_THRESHOLD` (default: 3)
- `fatigue += FATIGUE_ACCUMULATION_RATE` per step when overloaded
- `fatigue -= FATIGUE_RECOVERY_RATE` per step when idle
- Burnout (`fatigue > FATIGUE_THRESHOLD`) renders a worker unavailable for `BURNOUT_RECOVERY_TIME` steps
- Output quality degrades as: `quality = (1 − 0.3 × fatigue)`

### Deadline Shocks
- With probability `DEADLINE_SHOCK_PROB=0.15` per step, a random pending task has its deadline reduced by `DEADLINE_SHOCK_AMOUNT=10` timesteps
- Simulates external crises (client changes, priority escalations)
- Forces the agent to learn adaptive re-prioritisation

### Stochastic Completion
- Task completion time is drawn from `N(expected_time, noise²)` where `noise = COMPLETION_TIME_NOISE × expected_time = 0.3`
- Prevents exact memorisation of schedules

### Task Dependencies
- Tasks form a **DAG** of depth ≤ 3 with `DEPENDENCY_GRAPH_COMPLEXITY=3` chains
- Dependent tasks cannot be assigned until all prerequisites are completed

---

## 8. DQN Architecture

```
Input (88-dim state)
       │
  ┌────▼────┐
  │  Linear  │  88 → 128
  │   ReLU   │
  └────┬────┘
  ┌────▼────┐
  │  Linear  │  128 → 128
  │   ReLU   │
  └────┬────┘
  ┌────▼────┐
  │  Linear  │  128 → 140
  └────┬────┘
       │
  Output (140 Q-values, one per action)
```

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.0005 (Adam) |
| Discount factor γ | 0.95 |
| Replay buffer | 10,000 transitions |
| Replay warmup | 1,000 transitions |
| Batch size | 64 |
| Target network update | every 100 steps |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | 0.995 per episode |
| Loss function | Huber loss |
| Gradient clipping | norm = 1.0 |
| Weight initialisation | Xavier uniform |

---

## 9. Baseline Policies

| Baseline | Strategy |
|---|---|
| **Random** | Uniformly random valid action each step |
| **Greedy** | Always assign highest-priority task to any available worker |
| **Shortest-Task-First (STF)** | Assigns task with shortest expected completion time |
| **Skill Matcher** | 10-episode observation phase to estimate skills; then matches task complexity to best-skill worker |
| **Hybrid** | Combines STF + Skill Matching with urgency-weighted prioritisation (strongest heuristic) |

---

## 10. Training Process

1. **Replay Warmup** (1,000 steps): random policy fills the replay buffer
2. **Training loop** (up to 2,000 episodes):
   - Epsilon-greedy action selection with action masking
   - Transition stored in replay buffer
   - Mini-batch sampled and Huber loss minimised via Adam
   - Target network synced every 100 steps
   - Best model saved whenever moving average improves
3. **Early stopping**: halts if no improvement for 200 episodes
4. **Stability monitoring**: NaN detection, Q-value explosion detection (|Q| > 1,000)
5. **Logging**: 11 metrics per episode → `results/training_log.csv`
6. **Diagnostics**: per-step reward breakdowns → `results/reward_breakdown.csv`

---

## 11. Evaluation Methodology

The trained agent is evaluated under **4 conditions** for 200 episodes each to probe robustness:

| Condition | Description |
|---|---|
| **Standard** | Default environment parameters |
| **High Variance** | Completion time noise × 1.5 (more uncertainty) |
| **Frequent Shocks** | Deadline shock probability = 0.30 (2× default) |
| **Fixed Seed** | Constant seed — tests for overfitting to specific task layouts |

Baselines are evaluated for 200 episodes under **Standard** conditions.

All evaluations use `epsilon=0` (pure greedy policy) and **unscaled rewards** for fair metric comparison.

---

## 12. Statistical Testing

For each of the 5 RL vs Baseline comparisons, we compute:

1. **Welch's t-test** (one-tailed, `alternative='greater'`): tests if RL mean > Baseline mean
2. **Bonferroni correction**: adjusted significance threshold `α = 0.05 / 5 = 0.01`
3. **Cohen's d effect size**: `d = (μ_RL − μ_base) / σ_pooled`
   - Small: |d| < 0.5 | Medium: 0.5 ≤ |d| < 0.8 | Large: |d| ≥ 0.8

Results saved to `results/statistical_summary.csv`.

---

## 13. Ablation Studies

To demonstrate which environment dynamics the agent learned to exploit:

| Ablation | Override | Interpretation |
|---|---|---|
| **Standard** | None | Control group |
| **No Fatigue** | `enable_fatigue=False` | Does the agent exploit fatigue management? |
| **No Shocks** | `enable_deadline_shocks=False` | Does shock resilience contribute? |
| **Full Info** | `fully_observable=True` | Value of perfect skill information? |

```bash
python evaluation/ablation_studies.py
python visualization/plot_ablations.py
```

---

## 14. Directory Structure

```
AMD-SlingShot-Hackathon/
├── run_pipeline.py            ← End-to-end CLI orchestrator  ← START HERE
├── config.py                  ← All hyperparameters & paths
├── requirements.txt
│
├── environment/
│   ├── project_env.py         ← POMDP environment (main class)
│   ├── worker.py              ← Worker model (fatigue, skills)
│   ├── task.py                ← Task model (deadlines, dependencies)
│   └── belief_state.py        ← Bayesian skill tracking
│
├── agents/
│   └── dqn_agent.py           ← DQN + Q-Network + Replay Buffer
│
├── baselines/
│   ├── random_baseline.py
│   ├── greedy_baseline.py
│   ├── stf_baseline.py
│   ├── skill_baseline.py
│   └── hybrid_baseline.py
│
├── training/
│   ├── train_dqn.py           ← DQN training loop
│   ├── train_baselines.py     ← Baseline evaluation runner
│   └── visualize.py           ← Learning curve plots
│
├── evaluation/
│   ├── evaluate_agent.py      ← 4-condition RL evaluation
│   ├── statistical_tests.py   ← Welch's t-test + Bonferroni + Cohen's d
│   └── ablation_studies.py    ← Ablation study runner
│
├── visualization/
│   ├── plot_metrics.py        ← "Money Plot" (RL vs Baselines)
│   └── plot_ablations.py      ← Ablation study bar chart
│
├── utils/
│   └── metrics.py             ← Composite score computation
│
├── tests/
│   ├── test_dqn_training.py   ← DQN smoke test (10 episodes)
│   └── test_ablation.py       ← Ablation smoke test (1 episode)
│
├── checkpoints/               ← Model checkpoints (auto-created)
├── results/                   ← CSVs and plots (auto-created)
└── logs/                      ← Training logs (auto-created)
```

---

## 15. Example Commands

```bash
# Full pipeline (train 2000 eps, 200 baseline eps, 200 eval eps)
python run_pipeline.py --full

# Quick training test (100 episodes)
python run_pipeline.py --train --episodes 100

# Custom seed and shock probability
python run_pipeline.py --full --seed 123 --shock-prob 0.4

# Skip training, generate plots from existing results
python run_pipeline.py --plots

# Run only statistical tests
python run_pipeline.py --stats

# Ablation studies (after --full)
python evaluation/ablation_studies.py
python visualization/plot_ablations.py

# Environment smoke test
python environment/project_env.py

# DQN quick test (10 episodes)
python tests/test_dqn_training.py
```

---

## 16. Expected Output Files

After `python run_pipeline.py --full`:

| File | Description |
|---|---|
| `results/training_log.csv` | Per-episode: return, epsilon, Q-values, TD error, task metrics (11 cols) |
| `results/reward_breakdown.csv` | Per-episode: completion, delay, overload, throughput, deadline component sums |
| `checkpoints/best_model.pth` | Best DQN checkpoint (highest moving average return) |
| `checkpoints/checkpoint_ep*.pth` | Periodic checkpoints every 50 episodes |
| `results/baseline_performance.csv` | 5 baselines × 200 episodes with composite scores |
| `results/rl_test_performance.csv` | 4 conditions × 200 episodes with composite scores |
| `results/statistical_summary.csv` | T-statistic, p-value, Cohen's d for 5 RL vs Baseline comparisons |
| `results/learning_curve.png` | 4-panel: returns, epsilon, Q-values, task completion (dpi=300) |
| `results/money_plot.png` | Grouped bar chart with SEM bars and significance stars (dpi=300) |
| `results/ablation_results.csv` | 4 ablation conditions × 50 episodes |
| `results/ablation_study.png` | Ablation bar chart with error bars (dpi=300) |

---

## 17. Interpreting Results

### Learning Curve (`learning_curve.png`)
- **Top-left** (Episode Return): look for upward trend in the red moving average — indicates learning
- **Top-right** (Epsilon): should decay from 1.0 → 0.05 over ~1400 episodes
- **Bottom-left** (Q-values): gradual increase indicates value function is calibrating; sudden spikes indicate instability
- **Bottom-right** (Tasks Completed): rising mean over episodes confirms the agent is learning better task throughput

### Money Plot (`money_plot.png`)
- Each bar shows mean composite score with SEM error bars
- Significance stars (`*`, `**`, `***`) indicate Bonferroni-corrected Welch's t-test results
- Green annotation ("RL Agent surpasses Hybrid by X%") appears automatically if RL wins
- RL Agent should clearly separate from Random/Greedy, and ideally surpass Hybrid (strongest heuristic)

### Statistical Summary (`statistical_summary.csv`)
- `Significant=True` at `Alpha-Corrected=0.01` means the result is robust to multiple comparisons
- `Cohen-d > 0.8` indicates a **large effect size** (practically meaningful difference)

---

## 18. Reproducibility

Full deterministic reproducibility is enforced via:

```python
random.seed(seed)      # Python stdlib random
np.random.seed(seed)   # NumPy
torch.manual_seed(seed) # PyTorch
```

Default evaluation seeds: `[42, 123, 456, 789, 1011]` (defined in `config.TRAIN_RANDOM_SEEDS`).

To reproduce the exact results from this project:
```bash
python run_pipeline.py --full --seed 42
```

Note: GPU non-determinism can affect results even with fixed seeds. For exact reproducibility, use CPU:
```bash
CUDA_VISIBLE_DEVICES="" python run_pipeline.py --full --seed 42
```

---

## 19. Future Work

### Algorithmic Extensions
- **PPO / A2C**: Policy gradient methods may handle the high-variance POMDP better via natural gradient updates; expected to outperform DQN in non-stationary environments
- **Dueling DQN + Prioritised Experience Replay**: Separate value/advantage streams + importance-weighted sampling for faster convergence on sparse rewards
- **LSTM-based Q-Network**: Replace the MLP with an LSTM to allow the agent to maintain its own internal memory across timesteps, removing dependence on the hand-crafted belief state

### Environment Improvements
- **Improved Belief Modelling**: Replace Beta posterior with a particle filter or LSTM for more expressive uncertainty representation
- **Domain Randomisation During Training**: Randomise `DEADLINE_SHOCK_PROB`, `COMPLETION_TIME_NOISE`, and `FATIGUE_ACCUMULATION_RATE` across training episodes to improve generalisation to unseen parameter regimes
- **Continuous Action Space**: Allow fractional task assignments and resource allocation (Actor-Critic with continuous outputs)

### Multi-Agent Extensions
- **Decentralised Execution**: Each worker runs its own local agent; central training via QMIX or MAPPO
- **Cooperative vs Competitive**: Model managers competing for shared worker pool across multiple projects

### Explainability
- **LLM-based Explainability Layer**: Feed the agent's action sequence into an LLM (e.g., Gemini) to generate natural-language explanations of why specific tasks were assigned/deferred — bridging the gap between RL decisions and human-readable project management reasoning
- **SHAP Feature Attribution**: Attribute Q-value predictions to state features to understand which belief components drive decisions

### Production Deployment
- **FastAPI + MCP Integration**: The `app/` directory contains a FastAPI/MCP backend for live deployment; future work bridges this simulation to real project management systems (Jira, Linear)
- **Online Learning**: Continuously fine-tune the deployed agent on real project outcomes as they arrive
- **Multi-Project Portfolio**: Scale from 20 tasks / 5 workers to enterprise-scale (1000+ tasks, 100+ workers) with hierarchical RL

---

*This project was developed as part of the AMD SlingShot Hackathon.*