"""
Configuration file for RL-Driven Agentic Project Manager
Contains all hyperparameters and environment settings

v5 — Unified 365-Day Simulation Framework
  * SIM_DAYS: single configurable horizon (1–365 working days)
  * Adaptive event-driven time stepping (no fixed tick-rate assumption)
  * Bounded replay buffer + controlled training/emission intervals
  * Backlog, idle, and terminal reward penalties for correct DQN incentives
  * Realistic deadline windows (1–5 working days from arrival)
"""

# ============================================================================
# WORKDAY SIMULATION
# ============================================================================

SLOT_HOURS         = 0.5     # Each time slot = 30 minutes
SLOTS_PER_DAY      = 16      # 8-hour workday (09:00–17:00)
WORK_DAYS_PER_WEEK = 5       # Mon–Fri
WORK_START_SLOT    = 0       # Slot 0 = 09:00 within each day
WORK_END_SLOT      = 15      # Slot 15 = last slot of 09:00–17:00 window

# ── UNIFIED SIMULATION HORIZON ────────────────────────────────────────────────
# Single source of truth. Override via API SimConfig.sim_days.
# PHASE1/PHASE2 are derived as fractions of SIM_DAYS at runtime by the runner.
SIM_DAYS           = 100     # Default total working days (overridden by API)
PHASE1_FRACTION    = 0.60    # 60% of SIM_DAYS → Phase 1 (baseline observation)
PHASE2_FRACTION    = 0.40    # 40% of SIM_DAYS → Phase 2 (DQN scheduling)

# Derived (set at runtime by SimulationRunner when user config is received)
PHASE1_DAYS        = int(SIM_DAYS * PHASE1_FRACTION)
PHASE2_DAYS        = SIM_DAYS - PHASE1_DAYS
TOTAL_SIM_DAYS     = SIM_DAYS
EPISODE_HORIZON    = TOTAL_SIM_DAYS * SLOTS_PER_DAY  # Max steps per episode

# v10: TOTAL_TASKS is DYNAMICALLY computed from SIM_DAYS × TASK_ARRIVAL_RATE × 1.5
import math
TOTAL_TASKS        = max(50, math.ceil(SIM_DAYS * 4.0 * 1.5))  # default ~600
NUM_TASKS          = TOTAL_TASKS  # Alias (legacy compat)

# ============================================================================
# WORKERS
# ============================================================================

NUM_WORKERS        = 5
MAX_WORKER_LOAD    = 5       # Max concurrent tasks per worker
FATIGUE_LEVELS     = 4       # 0:fresh 1:tired 2:exhausted 3:burnout
BURNOUT_RECOVERY_TIME = 8    # Slots unavailable after burnout (~4 hours)

# Worker skill range (hidden from agents)
SKILL_MIN = 0.5
SKILL_MAX = 1.5
SKILL_PRIOR_ALPHA = 2.0
SKILL_PRIOR_BETA  = 2.0

# Worker heterogeneity params (all hidden, per-worker, sampled at init)
SPEED_MULT_MEAN     = 1.0    # Processing speed multiplier around 1.0
SPEED_MULT_STD      = 0.20   # ±20% variation
FATIGUE_RATE_MEAN   = 0.20   # Fatigue accumulation rate
FATIGUE_RATE_STD    = 0.06
RECOVERY_RATE_MEAN  = 0.12   # Recovery rate when idle
RECOVERY_RATE_STD   = 0.04
FATIGUE_SENS_MEAN   = 0.18   # How much fatigue hurts productivity
FATIGUE_SENS_STD    = 0.05
BURNOUT_RESIL_MEAN  = 2.6    # Burnout threshold (fatigue ≥ this → burnout)
BURNOUT_RESIL_STD   = 0.25

# Intra-day performance decay: quality drops slightly as hours_worked increases
INTRADAY_DECAY_RATE = 0.03   # Quality × (1 - rate × hours_worked_today)
FATIGUE_CARRYOVER   = 0.10   # Fraction of fatigue carried over to next day

# ============================================================================
# TASKS & ARRIVAL PROCESS
# ============================================================================

TASK_ARRIVAL_RATE    = 4.0   # Mean tasks arriving per working day (Poisson)
# TOTAL_TASKS is set dynamically at top of file. The line below is kept for
# backward-compat reference but is overridden at runtime by the runner.
# TOTAL_TASKS      = dynamically computed above
MIN_TASK_DURATION_H  = 1.0   # Minimum task duration in hours
MAX_TASK_DURATION_H  = 12.0  # Maximum task duration in hours
TASK_COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]   # Difficulty levels
TASK_PRIORITIES      = [0, 1, 2, 3]         # low, medium, high, critical

# ── REALISTIC DEADLINE WINDOW ─────────────────────────────────────────────────
# Deadlines are set relative to arrival time, in working days.
# Range: MIN_DEADLINE_DAYS … MAX_DEADLINE_DAYS (from task arrival).
# This creates genuine scheduling pressure instead of distant, irrelevant deadlines.
DEADLINE_MIN_DAYS    = 0.5   # Minimum: half a working day from arrival
DEADLINE_MAX_DAYS    = 3.0   # Maximum: 3 working days from arrival (tighter scheduling pressure)

# Legacy deadline fields (kept for backward-compat usage in old code paths)
DEADLINE_MIN_H       = DEADLINE_MIN_DAYS * 8.0   # → 4h
DEADLINE_MAX_H       = DEADLINE_MAX_DAYS * 8.0   # → 40h (but now enforced via DAYS)

COMPLETION_TIME_NOISE = 0.25 # Std dev as fraction of expected time

# Dependencies
DEPENDENCY_GRAPH_COMPLEXITY = 4
MAX_DEPENDENCY_DEPTH        = 3

# Stochasticity
DEADLINE_SHOCK_PROB   = 0.02  # Lowered: shocks remain but less frequent for longer runs
DEADLINE_SHOCK_SLOTS  = 8     # Slots removed from deadline per shock

# ============================================================================
# DQN HYPERPARAMETERS  (v5 — Bounded Online Learning)
# ============================================================================

# Network Architecture (Dueling DQN — PRESERVED from v3)
STATE_DIM   = 96    # 5 workers × 5 + 10 tasks × 5 + 10 belief + 6 global + 15 pad
ACTION_DIM  = 140   # 20 tasks × 5 workers + 20 defer + 20 escalate (unchanged)
HIDDEN_LAYERS  = [256, 256]
ACTIVATION     = 'relu'
DUELING_DQN    = True

# ── BOUNDED REPLAY BUFFER ────────────────────────────────────────────────────
# Hard cap regardless of SIM_DAYS. FIFO eviction via SumTree.write pointer.
REPLAY_BUFFER_MAX_CAPACITY = 8000    # v9: denser PER sampling vs fewer transitions
REPLAY_BUFFER_SIZE         = 8000    # Alias used by DQNAgent constructor

# ── CONTROLLED TRAINING INTERVAL ─────────────────────────────────────────────
# Train every N env steps — prevents per-step gradient updates at long horizons.
TRAIN_EVERY_N_STEPS = 1              # v7: every decision triggers 3 gradient updates

# ── RATE-LIMITED FRONTEND EMISSIONS ──────────────────────────────────────────
# Emit tick_update every N ticks — prevents frontend flood at 365-day scale.
EMIT_EVERY_N_TICKS = 8              # One Socket.IO emit per 8 env ticks

# ── BOUNDED IN-MEMORY ACCUMULATORS ───────────────────────────────────────────
# Caps so that memory does not grow O(horizon) for long runs.
MAX_GANTT_BLOCKS_IN_MEMORY  = 500   # Per-baseline ring-buffer cap
MAX_DAILY_METRICS_IN_MEMORY = 730   # Room for 2 × 365 days (phase1 + phase2)
MAX_TASK_HISTORY_IN_MEMORY  = 2000  # Max task records retained in runner

# Training (tuned for online learning)
LEARNING_RATE        = 0.0002   # Adam LR
GAMMA                = 0.95     # Discount factor
BATCH_SIZE           = 64       # Mini-batch size
MIN_REPLAY_SIZE      = 32       # v7: start training earlier
TARGET_UPDATE_FREQ   = 200      # Sync target net every 200 training steps

# Exploration (per-DECISION decay, not per-episode)
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9998    # Slower decay for longer runs — reaches 0.5 in ~3400 decisions
EPSILON_PHASE2_START = 0.4   # Phase 2 starts with partial exploration

# Prioritized Experience Replay
PER_ALPHA       = 0.6
PER_BETA_START  = 0.4
PER_BETA_FRAMES = 100000   # Larger frame budget for 365-day runs

# LR Scheduler
LR_SCHEDULER_T0   = 5000   # Steps per cosine restart (longer for 365-day)
LR_SCHEDULER_T_MULT = 1

# Legacy training config (kept for backward compat)
MAX_EPISODES            = 5000
CHECKPOINT_FREQ         = 100
EARLY_STOPPING_PATIENCE = 1000
CONVERGENCE_THRESHOLD   = 1000

# ============================================================================
# REWARD FUNCTION  (v5 — Correct Incentives for 365-day Scheduling)
# ============================================================================

# Primary reward: task completion (scales with priority + quality)
REWARD_COMPLETION_BASE   = 0.8   # v10: raised to 0.8; completion reward = base × priority × quality²·⁵

# Early/on-time completion bonuses (v8)
REWARD_EARLY_COMPLETION_BONUS  = 0.2   # +0.2 if completed 20%+ ahead of deadline
REWARD_ONTIME_COMPLETION_BONUS = 0.1   # +0.1 for any on-time completion

# ── BACKLOG PENALTY ──────────────────────────────────────────────────────────
REWARD_BACKLOG_PENALTY   = -0.02  # v7: scaled for [-2,+1] range

# ── TERMINAL UNFINISHED PENALTY ──────────────────────────────────────────────
REWARD_TERMINAL_UNFINISHED_PENALTY = -0.5   # v7: per unfinished task (normalized)

# Idle penalty
REWARD_IDLE_PENALTY      = -0.10  # v7: normalized

# Per-step idle worker penalty
REWARD_IDLE_WORKER_PENALTY = -0.05

# Lateness penalty
REWARD_LATENESS_PENALTY  = -0.1   # v7: per slot late at completion

# Overload: per-worker penalty at/above capacity
REWARD_OVERLOAD_WEIGHT   = -5.0   # v7: FATAL per overloaded worker

# Deadline urgency
REWARD_URGENCY_PENALTY   = -0.1   # v7: normalized

# Terminal bonus: awarded when ALL tasks are completed
REWARD_MAKESPAN_BONUS    = 1.0    # v7: normalized

# Step delay
REWARD_DELAY_WEIGHT      = -0.001

# Deadline miss
REWARD_DEADLINE_MISS_PENALTY = -0.5  # v7: normalized

# Skill-match shaped reward
REWARD_SKILL_MATCH_WEIGHT    = 0.3   # v7: quality bonus
REWARD_FATIGUE_ASSIGN_PENALTY = -0.2
REWARD_OVERLOAD_ASSIGN_PENALTY = -5.0  # v7: hard block

# Reward clipping (v7: normalized range)
REWARD_CLIP_MIN = -2.0
REWARD_CLIP_MAX =  1.0

# Idle-waiting penalty
REWARD_IDLE_WAITING_PENALTY = -0.05

# Legacy
REWARD_THROUGHPUT_WEIGHT  = 2.0
REWARD_STRATEGIC_DEFER    = 0.3
REWARD_EXPLORATION_BONUS  = 0.3
REWARD_LOAD_BALANCE_BONUS = 0.1
REWARD_OVERLOAD_IMMEDIATE = -5.0   # v7: hard overload penalty
FATIGUE_QUALITY_PENALTY   = 0.25

# ============================================================================
# TEAM SYNERGY
# ============================================================================
TEAM_SYNERGY_ENABLED = False

# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

BASELINE_SKILL_ESTIMATION_EPISODES = 10
BASELINE_DEBUG_SKILL = False

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

TRAIN_EPISODES           = 5000
TRAIN_RANDOM_SEEDS       = [42, 123, 456, 789, 1011]
TEST_EPISODES            = 200
TEST_EPISODES_STANDARD   = 50
TEST_EPISODES_HIGH_VARIANCE = 50
TEST_EPISODES_FREQUENT_SHOCKS = 50
TEST_EPISODES_FIXED      = 50
TEST_VARIANCE_MULTIPLIER = 1.5
TEST_SHOCK_PROB_HIGH     = 0.3
SIGNIFICANCE_LEVEL       = 0.05
BONFERRONI_NUM_COMPARISONS = 5
COHEN_D_THRESHOLD        = 0.5

# ============================================================================
# METRICS WEIGHTS
# ============================================================================

METRIC_WEIGHT_THROUGHPUT   = 2.0
METRIC_WEIGHT_DEADLINE     = 3.0
METRIC_WEIGHT_DELAY        = -0.5
METRIC_WEIGHT_LOAD_BALANCE = -1.0
METRIC_WEIGHT_QUALITY      = 1.0
METRIC_WEIGHT_OVERLOAD     = -2.0

# ============================================================================
# PATHS
# ============================================================================

import os

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR        = os.path.join(PROJECT_ROOT, 'logs')
TESTS_DIR       = os.path.join(PROJECT_ROOT, 'tests')

for directory in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, TESTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEMO CONFIGURATION
# ============================================================================

DEMO_GREEDY          = True
DEMO_SHOWCASE_EPISODES = 3
DEMO_STEP_DELAY      = 1.0

# ============================================================================
# CONTEXTUAL BANDIT FALLBACK
# ============================================================================

LINUCB_ALPHA       = 1.0
LINUCB_FEATURE_DIM = 96   # Matches STATE_DIM
