# RL-Driven Agentic Project Manager

**AMD SlingShot Hackathon** - A full reinforcement learning pipeline for optimal dynamic project task allocation under partial observability, fatigue, deadline shocks, and stochastic task completion.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start](#3-quick-start)
4. [Installation](#4-installation)
5. [Running the Program](#5-running-the-program)
6. [POMDP Formulation](#6-pomdp-formulation)
7. [Environment Dynamics](#7-environment-dynamics)
8. [Directory Structure](#8-directory-structure)

---

## 1. Project Overview

This system trains an **Online Deep Q-Network (DQN)** agent to manage a portfolio of software engineering tasks assigned to workers in a simulated project environment. The agent must:

- Assign tasks to workers with **partially observable skill levels** (learned via Bayesian belief updates)
- Manage **worker fatigue** to prevent quality degradation and burnout
- Respond to **sudden deadline shocks**
- Maximize **throughput and quality** while minimizing delays and overload

Because the architecture has transitioned exclusively to an online DQN framework, previous references to offline reinforcement learning have been deprecated and removed. The system is benchmarked against 4 hand-crafted heuristic baselines.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Online DQN Agent (v4)                          │
│                                                                    │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐  │
│  │  96-dim POMDP│    │         Dueling Q-Network               │  │
│  │  Observation │───>│  Input(96) -> 256 -> 256 [LayerNorm+ReLU] │  │
│  └──────────────┘    │       |               |                 │  │
│                       │  Value(1)      Advantage(140)           │  │
│  ┌──────────────┐    │       └────── Q(s,a) ─────┘            │  │
│  │   PER Buffer │<───│    Q = V + A - mean(A)                  │  │
│  │  (Sum-Tree)  │    └─────────────────────────────────────────┘  │
│  └──────┬───────┘                                                  │
│         │ IS-weighted      Double DQN targets                      │
│         V                 policy selects, target evaluates         │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────────────┐  │
│  │ Mini-batch   │───>│ Huber Loss   │──>│ Adam + CosineWarmLR  │  │
│  │  B=32        │    │ + IS weights │   │ LR=0.0003, T0=2000 ep│  │
│  └──────────────┘    └──────────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Key improvements in v4 over v3:**

| Component | v3 | v4 |
|-----------|----|----|
| State Dim | 88-dim | 96-dim |
| Mode | Offline/Passive | Strictly Online learning directly mapping 96-dim observations |
| Target computation | Double DQN | Double DQN (decoupled select/evaluate) |
| Batch size | 128 | 32 (For fast online updates) |
| LR | 0.001 | 0.0003 |

---

## 3. Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run interactive demo
python demo_run.py
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
```

---

## 5. Running the Program

The system is executed through specific entry points. The primary runner for the demonstration is `demo_run.py`, and the entire pipeline is handled by `run_pipeline.py`.

### 5.1 Interactive Demo

To witness the agent perform comparative analysis dynamically against baselines:

```powershell
python demo_run.py
```

**What this command does:**
1. Triggers `interactive_config.py` which prompts you to define a random seed, project goal, task limit, and optionally define worker traits manually.
2. The interactive script parses your text and overrides the global variables in `config.py`.
3. It then initializes the `ProjectEnv` using the chosen specific seed.
4. The system then boots up the baselines one by one, scoring them behind the scenes (No visual plot is printed for Phase 1 - 1-Month Observational run).
5. Finally, the system plots the Phase 2 (1-Week Operational run) visualizer containing task assignments for both the DQN Agent and Baseline agents to allow direct competitive evaluation on tasks completed, overload events, and quality outputs.

### 5.2 Full Pipeline Execution

To train the DQN from scratch and review statistical differences against baselines:

```powershell
# Train Agent
python run_pipeline.py --train --episodes 5000

# Full Evaluation
python run_pipeline.py --full
```

**What these commands do:**
- `--train`: Bootstraps the `train_dqn.py` loop which orchestrates learning for the DQN agent directly in `ProjectEnv`. At the end of 5000 iterations, the system will save the `best_model.pth` in the `checkpoints` directory.
- `--full`: Chains multiple stages sequentially. It will (1) train the agent, (2) run through validation stages against default heuristics, and (3) dump graphical metrics like learning_curves and CSV logs to the `results/` folder for analysis.

---

## 6. POMDP Formulation

The task allocation problem is modelled as a **Partially Observable Markov Decision Process (POMDP)**. Partial observability is central: worker skill levels are hidden. The agent maintains a Belief State updated incrementally with each task completion observation via Bayesian inference.

For full theoretical details, navigate to the `docs/team_onboarding/` and `docs/` directories.

---

## 7. Environment Dynamics

### Worker Fatigue and Overload
- Workers accumulate fatigue dynamically. There is a penalty specifically configured for overloaded workers (when assigned assignments exceed internal limits). 
- Burnout renders a worker unavailable.
- Output quality drops off exponentially when a worker is continuously overloaded, defined implicitly by an internal `fatigue_rate` and `base_efficiency`. 

### Phases
- **Phase 1 (1 Month)**: The initial observational period.
- **Phase 2 (1 Week)**: 5 consecutive days of 8-hour shifts representing the strict evaluation testing phase.

---

## 8. Directory Structure

```
AMD-SlingShot-Hackathon/
├── interactive_config.py      <-- Setup UI before starting demo
├── run_pipeline.py            <-- End-to-end CLI orchestrator
├── demo_run.py                <-- Interactive demonstration UI
├── config.py                  <-- All hyperparameters & paths
├── requirements.txt
│
├── agents/                    <-- Agent configurations (DQN)
├── environment/               <-- POMDP environments (Worker, Task)
├── baselines/                 <-- Baseline testing logic
├── training/                  <-- Training loops
├── evaluation/                <-- Stats calculation tools
├── visualization/             <-- Generators for charts/grids
│
├── docs/                      <-- Mathematical and theoretical logic
│   └── team_onboarding/       <-- Parameter tweaking and Onboarding docs
│
├── tests/                     <-- Unit tests
├── checkpoints/               <-- Model checkpoints (auto-created)
├── results/                   <-- CSVs and plots (auto-created)
└── logs/                      <-- Training logs (auto-created)
```