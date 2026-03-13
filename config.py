"""
Configuration file for RL-Driven Agentic Project Manager
Contains all hyperparameters and environment settings

v13 — End-of-simulation crash fixes:
  - MAX_GANTT_BLOCKS_IN_MEMORY  : 5000  → 50000  (prevents gantt ring-buffer from evicting
                                                    blocks needed for baseline_snapshot)
  - MAX_TASK_HISTORY_IN_MEMORY  : 20000 → 100000 (headroom for 365-day × 4 baselines)
  - MAX_DAILY_METRICS_IN_MEMORY : 5000  → 20000  (prevents deque from dropping early baselines)
"""

# ============================================================================
# WORKDAY SIMULATION
# ============================================================================

SLOT_HOURS         = 0.5     # Each time slot = 30 minutes
SLOTS_PER_DAY      = 16      # 8-hour workday (09:00-17:00)
WORK_DAYS_PER_WEEK = 5       # Mon-Fri
WORK_START_SLOT    = 0       # Slot 0 = 09:00 within each day
WORK_END_SLOT      = 15      # Slot 15 = last slot of 09:00-17:00 window

# ── UNIFIED SIMULATION HORIZON ────────────────────────────────────────────────
SIM_DAYS           = 100     # Default total working days (overridden by API)
PHASE1_FRACTION    = 0.55    # v13: 55% of SIM_DAYS → Phase 1 (gives DQN 45% of sim)
PHASE2_FRACTION    = 0.45    # v13: 45% of SIM_DAYS → Phase 2 (DQN scheduling)

# Derived (set at runtime by SimulationRunner when user config is received)
PHASE1_DAYS        = int(SIM_DAYS * PHASE1_FRACTION)
PHASE2_DAYS        = SIM_DAYS - PHASE1_DAYS
TOTAL_SIM_DAYS     = SIM_DAYS
EPISODE_HORIZON    = TOTAL_SIM_DAYS * SLOTS_PER_DAY

import math
TOTAL_TASKS        = max(50, math.ceil(SIM_DAYS * 4.0 * 1.5))
NUM_TASKS          = TOTAL_TASKS  # Alias (legacy compat)

# ============================================================================
# WORKERS
# ============================================================================

NUM_WORKERS        = 5
MAX_WORKER_LOAD    = 5
FATIGUE_LEVELS     = 4
BURNOUT_RECOVERY_TIME = 8

SKILL_MIN = 0.5
SKILL_MAX = 1.5
SKILL_PRIOR_ALPHA = 2.0
SKILL_PRIOR_BETA  = 2.0

SPEED_MULT_MEAN     = 1.0
SPEED_MULT_STD      = 0.20
FATIGUE_RATE_MEAN   = 0.20
FATIGUE_RATE_STD    = 0.06
RECOVERY_RATE_MEAN  = 0.12
RECOVERY_RATE_STD   = 0.04
FATIGUE_SENS_MEAN   = 0.18
FATIGUE_SENS_STD    = 0.05
BURNOUT_RESIL_MEAN  = 2.6
BURNOUT_RESIL_STD   = 0.25

INTRADAY_DECAY_RATE = 0.03
FATIGUE_CARRYOVER   = 0.10

# ============================================================================
# TASKS & ARRIVAL PROCESS
# ============================================================================

TASK_ARRIVAL_RATE    = 4.0
MIN_TASK_DURATION_H  = 1.0
MAX_TASK_DURATION_H  = 12.0
TASK_COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]
TASK_PRIORITIES      = [0, 1, 2, 3]

DEADLINE_MIN_DAYS    = 0.5
DEADLINE_MAX_DAYS    = 2.5   # v12: tighter deadlines = more urgency signal for DQN

DEADLINE_MIN_H       = DEADLINE_MIN_DAYS * 8.0
DEADLINE_MAX_H       = DEADLINE_MAX_DAYS * 8.0

COMPLETION_TIME_NOISE = 0.25

DEPENDENCY_GRAPH_COMPLEXITY = 4
MAX_DEPENDENCY_DEPTH        = 3

DEADLINE_SHOCK_PROB   = 0.02
DEADLINE_SHOCK_SLOTS  = 8

# ============================================================================
# DQN HYPERPARAMETERS  (v12 — corrected)
# ============================================================================

# Network Architecture
STATE_DIM      = 96
ACTION_DIM     = 140
HIDDEN_LAYERS  = [256, 256]
ACTIVATION     = 'relu'
DUELING_DQN    = True

# ── REPLAY BUFFER ─────────────────────────────────────────────────────────────
REPLAY_BUFFER_MAX_CAPACITY = 10000  # v12: was 30000, focus on quality over quantity
REPLAY_BUFFER_SIZE         = 10000

# ── TRAINING INTERVALS ────────────────────────────────────────────────────────
TRAIN_EVERY_N_STEPS = 1
EMIT_EVERY_N_TICKS  = 8

# ── BOUNDED IN-MEMORY ACCUMULATORS ───────────────────────────────────────────
# FIX v13: Raised all three caps substantially.
#   - MAX_GANTT_BLOCKS_IN_MEMORY  : 5000  → 50000
#       5 baselines × 365 days × ~20 assignments/day = ~36 500 blocks minimum.
#       The old 5000 cap silently evicted most gantt history before
#       baseline_snapshot was built, corrupting the final results payload.
#   - MAX_TASK_HISTORY_IN_MEMORY  : 20000 → 100000
#       Matches TOTAL_TASKS upper bound (SIM_DAYS * rate * 1.5 ≈ 2190 for 365 days,
#       but we use a generous ceiling for safety on user-configured longer runs).
#   - MAX_DAILY_METRICS_IN_MEMORY : 5000  → 20000
#       Handled dynamically in SimulationRunner but the config floor matters
#       when runner computes maxlen = max_days * n_baselines.
MAX_GANTT_BLOCKS_IN_MEMORY  = 50000
MAX_DAILY_METRICS_IN_MEMORY = 20000
MAX_TASK_HISTORY_IN_MEMORY  = 100000

# ── CORE TRAINING HYPERPARAMETERS ────────────────────────────────────────────
LEARNING_RATE        = 0.0005
GAMMA                = 0.90     # v12: back to sweep winner
BATCH_SIZE           = 64
MIN_REPLAY_SIZE      = 64       # v12: was 500 — proportional to smaller buffer
TARGET_UPDATE_FREQ   = 400      # v12: back to sweep winner

# ── EXPLORATION ───────────────────────────────────────────────────────────────
EPSILON_START        = 1.0
EPSILON_END          = 0.05
EPSILON_DECAY        = 0.9995   # v12: sweep winner
EPSILON_PHASE2_START = 0.4

# ── PRIORITIZED EXPERIENCE REPLAY ─────────────────────────────────────────────
PER_ALPHA       = 0.4           # v12: sweep winner (was being overridden to 0.6 by settings.py)
PER_BETA_START  = 0.4
PER_BETA_FRAMES = 100000

# LR Scheduler
LR_SCHEDULER_T0              = 5000
LR_SCHEDULER_T_MULT          = 1
LR_SCHEDULER_ETA_MIN_FRACTION = 0.15  # v12: floor = LR * 0.15 — prevents Q-value collapse

# Legacy
MAX_EPISODES            = 5000
CHECKPOINT_FREQ         = 100
EARLY_STOPPING_PATIENCE = 1000
CONVERGENCE_THRESHOLD   = 1000

# ============================================================================
# REWARD FUNCTION
# ============================================================================

REWARD_COMPLETION_BASE          = 1.0
REWARD_EARLY_COMPLETION_BONUS   = 0.2
REWARD_ONTIME_COMPLETION_BONUS  = 0.1
REWARD_BACKLOG_PENALTY          = -0.02
REWARD_TERMINAL_UNFINISHED_PENALTY = -0.5
REWARD_IDLE_PENALTY             = -0.10
REWARD_IDLE_WORKER_PENALTY      = -0.05
REWARD_LATENESS_PENALTY         = -0.1
REWARD_OVERLOAD_WEIGHT          = -5.0
REWARD_URGENCY_PENALTY          = -0.1
REWARD_MAKESPAN_BONUS           = 1.0
REWARD_DELAY_WEIGHT             = -0.001
REWARD_DEADLINE_MISS_PENALTY    = -0.5
REWARD_SKILL_MATCH_WEIGHT       = 0.4   # v12: was 0.3 — closes quality gap vs baselines
REWARD_FATIGUE_ASSIGN_PENALTY   = -0.2
REWARD_OVERLOAD_ASSIGN_PENALTY  = -5.0
REWARD_CLIP_MIN                 = -2.0
REWARD_CLIP_MAX                 =  1.0
REWARD_IDLE_WAITING_PENALTY     = -0.05

# Legacy reward fields
REWARD_THROUGHPUT_WEIGHT  = 2.0
REWARD_STRATEGIC_DEFER    = 0.3
REWARD_EXPLORATION_BONUS  = 0.3
REWARD_LOAD_BALANCE_BONUS = 0.1
REWARD_OVERLOAD_IMMEDIATE = -5.0
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
TRAIN_EPISODES              = 5000
TRAIN_RANDOM_SEEDS          = [42, 123, 456, 789, 1011]
TEST_EPISODES               = 200
TEST_EPISODES_STANDARD      = 50
TEST_EPISODES_HIGH_VARIANCE = 50
TEST_EPISODES_FREQUENT_SHOCKS = 50
TEST_EPISODES_FIXED         = 50
TEST_VARIANCE_MULTIPLIER    = 1.5
TEST_SHOCK_PROB_HIGH        = 0.3
SIGNIFICANCE_LEVEL          = 0.05
BONFERRONI_NUM_COMPARISONS  = 5
COHEN_D_THRESHOLD           = 0.5

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

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR    = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR       = os.path.join(PROJECT_ROOT, 'logs')
TESTS_DIR      = os.path.join(PROJECT_ROOT, 'tests')

for directory in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, TESTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEMO CONFIGURATION
# ============================================================================
DEMO_GREEDY            = True
DEMO_SHOWCASE_EPISODES = 3
DEMO_STEP_DELAY        = 1.0

# ============================================================================
# CONTEXTUAL BANDIT FALLBACK
# ============================================================================
LINUCB_ALPHA       = 1.0
LINUCB_FEATURE_DIM = 96