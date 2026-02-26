"""
Configuration file for RL-Driven Agentic Project Manager
Contains all hyperparameters and environment settings

v4 — Continual Online Learning Framework
  * Workday simulation: 8h/day, 5-day week, slot-based clock
  * Dynamic Poisson task arrivals (no lookahead for any agent)
  * Heterogeneous workers with per-worker fatigue sensitivity
  * Two-phase adaptive framework: Phase 1 (baseline obs) → Phase 2 (DQN control)
  * State dim updated to 96 (was 88)
"""

# ============================================================================
# WORKDAY SIMULATION
# ============================================================================

SLOT_HOURS         = 0.5     # Each time slot = 30 minutes
SLOTS_PER_DAY      = 16      # 8-hour workday (09:00–17:00)
WORK_DAYS_PER_WEEK = 5       # Mon–Fri
WORK_START_SLOT    = 0       # Slot 0 = 09:00 within each day
WORK_END_SLOT      = 15      # Slot 15 = last slot of 09:00–17:00 window

# Two-phase framework durations
PHASE1_DAYS   = 20           # Phase 1: 4 working weeks / 1 month (baseline-driven + passive DQN)
PHASE2_DAYS   = 5            # Phase 2: 1 working week (DQN-controlled with online learning)
TOTAL_SIM_DAYS = PHASE1_DAYS + PHASE2_DAYS   # 25 working days total
EPISODE_HORIZON = TOTAL_SIM_DAYS * SLOTS_PER_DAY  # Max steps per episode
NUM_TASKS       = 200   # Alias for TOTAL_TASKS (legacy compat)

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

TASK_ARRIVAL_RATE    = 3.5   # Mean tasks arriving per working day (Poisson)
TOTAL_TASKS          = 200   # Total tasks across entire simulation
MIN_TASK_DURATION_H  = 1.0   # Minimum task duration in hours
MAX_TASK_DURATION_H  = 12.0  # Maximum task duration in hours
TASK_COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]   # Difficulty levels
TASK_PRIORITIES      = [0, 1, 2, 3]         # low, medium, high, critical
DEADLINE_MIN_H       = 4.0   # Minimum deadline in hours (8 slots)
DEADLINE_MAX_H       = 40.0  # Maximum deadline in hours (80 slots)
COMPLETION_TIME_NOISE = 0.25 # Std dev as fraction of expected time

# Dependencies
DEPENDENCY_GRAPH_COMPLEXITY = 4
MAX_DEPENDENCY_DEPTH        = 3

# Stochasticity
DEADLINE_SHOCK_PROB   = 0.08  # Probability of a deadline shock per slot
DEADLINE_SHOCK_SLOTS  = 8     # Slots removed from deadline per shock

# ============================================================================
# DQN HYPERPARAMETERS  (v4 — Online Learning)
# ============================================================================

# Network Architecture (Dueling DQN — PRESERVED from v3)
STATE_DIM   = 96    # 5 workers × 5 + 10 tasks × 5 + 10 belief + 6 global + 15 pad
ACTION_DIM  = 140   # 20 tasks × 5 workers + 20 defer + 20 escalate (unchanged)
HIDDEN_LAYERS  = [256, 256]
ACTIVATION     = 'relu'
DUELING_DQN    = True

# Training (tuned for online learning — fewer steps per decision, continuous updates)
LEARNING_RATE        = 0.0003   # Slightly lower for online stability
GAMMA                = 0.97     # Longer effective horizon for scheduling
BATCH_SIZE           = 32       # Small batch for fast online updates
REPLAY_BUFFER_SIZE   = 30000    # Adequate for online continual learning
MIN_REPLAY_SIZE      = 64       # === CRITICAL FIX: must equal BATCH_SIZE to start training immediately ===
                                 # was 512 — caused train_steps=0 (buffer never reached threshold)
TARGET_UPDATE_FREQ   = 100      # Sync target net every 100 decisions (was 150)

# Exploration (per-DECISION decay, not per-episode)
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.999     # Per-decision decay — reaches 0.5 in ~700 decisions, 0.1 in ~2300

# Phase 2 epsilon (DQN starts with partial exploration already done in Phase 1)
EPSILON_PHASE2_START = 0.35  # Start Phase 2 with partial exploration

# Prioritized Experience Replay
PER_ALPHA       = 0.6
PER_BETA_START  = 0.4
PER_BETA_FRAMES = 50000

# LR Scheduler
LR_SCHEDULER_T0   = 2000   # Steps per cosine restart (longer for online mode)
LR_SCHEDULER_T_MULT = 1

# Legacy training config (kept for backward compat with --full pipeline)
MAX_EPISODES            = 5000
CHECKPOINT_FREQ         = 100
EARLY_STOPPING_PATIENCE = 1000
CONVERGENCE_THRESHOLD   = 1000

# ============================================================================
# REWARD FUNCTION  (Makespan-centric)
# ============================================================================

# Primary reward: task completion
REWARD_COMPLETION_BASE   = 10.0  # × (priority+1) × quality per completed task

# Idle penalty: incentivise keeping workers busy
REWARD_IDLE_PENALTY      = -0.2  # per available-but-idle worker per slot

# Lateness penalty: tasks completed after their deadline
REWARD_LATENESS_PENALTY  = -1.5  # per slot late at completion

# Overload balance: std-dev of worker loads (stronger penalty → meaningful gradient)
REWARD_OVERLOAD_WEIGHT   = -0.5  # was -0.15; stronger signal against load imbalance

# Deadline urgency: unstarted tasks with very little time left
REWARD_URGENCY_PENALTY   = -0.3  # per unstarted task with ≤ 4 slots to deadline

# Terminal bonus: awarded when ALL tasks are completed
REWARD_MAKESPAN_BONUS    = 50.0  # reduced from 100 to keep gradient scale manageable

# Step delay: small constant nudge to act quickly
REWARD_DELAY_WEIGHT      = -0.02  # reduced to not dominate early learning

# Deadline miss: one-shot penalty per deadline miss
REWARD_DEADLINE_MISS_PENALTY = -10.0  # was -15; keep in proportion with completion reward

# Legacy (kept for compat)
REWARD_THROUGHPUT_WEIGHT  = 2.0
REWARD_STRATEGIC_DEFER    = 0.5
REWARD_EXPLORATION_BONUS  = 0.3
FATIGUE_QUALITY_PENALTY   = 0.25

# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

BASELINE_SKILL_ESTIMATION_EPISODES = 10
BASELINE_DEBUG_SKILL = False   # Set True to print per-assignment skill logs

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
LINUCB_FEATURE_DIM = 96   # Updated to match new STATE_DIM
