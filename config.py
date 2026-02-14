"""
Configuration file for RL-Driven Agentic Project Manager
Contains all hyperparameters and environment settings
"""

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Workers
NUM_WORKERS = 5
MAX_WORKER_LOAD = 5  # Maximum concurrent tasks per worker
FATIGUE_LEVELS = 4  # 0: fresh, 1: tired, 2: exhausted, 3: burnout
BURNOUT_RECOVERY_TIME = 5  # Timesteps worker unavailable after burnout

# Worker Skills (Hidden)
SKILL_MIN = 0.6
SKILL_MAX = 1.4
SKILL_PRIOR_ALPHA = 2.0  # Beta distribution prior for belief state
SKILL_PRIOR_BETA = 2.0

# Tasks
NUM_TASKS = 20
TASK_COMPLEXITY_LEVELS = [1, 2, 3, 4, 5]  # Difficulty levels
TASK_PRIORITIES = [0, 1, 2, 3]  # low, medium, high, critical
DEADLINE_MIN = 20
DEADLINE_MAX = 60

# Episode
EPISODE_HORIZON = 100  # Max timesteps per episode
FAILURE_THRESHOLD = 0.5  # 50% deadline miss rate triggers early termination

# Dynamics
FATIGUE_ACCUMULATION_RATE = 0.2  # Fatigue increase when overloaded
FATIGUE_RECOVERY_RATE = 0.1  # Fatigue decrease when idle
FATIGUE_THRESHOLD = 2.5  # Burnout threshold
OVERLOAD_THRESHOLD = 3  # Load > this triggers accelerated fatigue

# Stochasticity
COMPLETION_TIME_NOISE = 0.3  # Std dev as fraction of expected time
DEADLINE_SHOCK_PROB = 0.15  # Probability of deadline shock per episode
DEADLINE_SHOCK_AMOUNT = 10  # Timesteps reduced from deadline
TASK_FAILURE_PROB_BASE = 0.3  # Base failure probability when overloaded (load > 4)

# Dependencies
DEPENDENCY_GRAPH_COMPLEXITY = 3  # Number of dependency chains
MAX_DEPENDENCY_DEPTH = 3  # Max depth of task dependency tree

# ============================================================================
# DQN HYPERPARAMETERS
# ============================================================================

# Network Architecture
STATE_DIM = 88  # 5 workers × 3 + 10 tasks × 4 + 5 skill means + 5 skill vars + 3 global
ACTION_DIM = 140  # 20 tasks × 5 workers + 20 defer + 20 escalate
HIDDEN_LAYERS = [128, 128]
ACTIVATION = 'relu'

# Training
LEARNING_RATE = 0.0005
GAMMA = 0.95  # Discount factor
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000  # Start training after this many transitions
TARGET_UPDATE_FREQ = 100  # Steps between target network updates

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995  # Per episode

# Training Control
MAX_EPISODES = 2000
CHECKPOINT_FREQ = 50  # Save model every N episodes
EARLY_STOPPING_PATIENCE = 200  # Stop if no improvement for N episodes
CONVERGENCE_THRESHOLD = 1000  # Max Q-value magnitude (divergence check)

# ============================================================================
# REWARD FUNCTION WEIGHTS
# ============================================================================

REWARD_COMPLETION_BASE = 10.0  # Base reward per completed task
REWARD_DELAY_WEIGHT = -0.5  # Penalty for time in queue
REWARD_OVERLOAD_WEIGHT = -5.0  # Penalty for overloading workers
REWARD_THROUGHPUT_WEIGHT = 2.0  # Bonus for tasks completed this step
REWARD_DEADLINE_MISS_PENALTY = -50.0  # Catastrophic penalty for deadline miss

# Reward shaping
REWARD_STRATEGIC_DEFER = 1.0  # Bonus for deferring when no skilled worker
REWARD_EXPLORATION_BONUS = 0.5  # Bonus for exploring uncertain workers

# Quality calculation
FATIGUE_QUALITY_PENALTY = 0.3  # Quality multiplier: (1 - 0.3 * fatigue)

# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================

BASELINE_SKILL_ESTIMATION_EPISODES = 10  # Episodes to observe before using skill estimates

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Training
TRAIN_EPISODES = 2000
TRAIN_RANDOM_SEEDS = [42, 123, 456, 789, 1011]

# Testing
TEST_EPISODES = 200
TEST_EPISODES_STANDARD = 50
TEST_EPISODES_HIGH_VARIANCE = 50
TEST_EPISODES_FREQUENT_SHOCKS = 50
TEST_EPISODES_FIXED = 50

# High variance test
TEST_VARIANCE_MULTIPLIER = 1.5

# Frequent shocks test
TEST_SHOCK_PROB_HIGH = 0.3

# Statistical testing
SIGNIFICANCE_LEVEL = 0.05
BONFERRONI_NUM_COMPARISONS = 5  # 5 baselines
COHEN_D_THRESHOLD = 0.5  # Minimum effect size for "meaningful"

# ============================================================================
# METRICS WEIGHTS (for composite score)
# ============================================================================

METRIC_WEIGHT_THROUGHPUT = 2.0
METRIC_WEIGHT_DEADLINE = 3.0
METRIC_WEIGHT_DELAY = -0.5
METRIC_WEIGHT_LOAD_BALANCE = -1.0
METRIC_WEIGHT_QUALITY = 1.0
METRIC_WEIGHT_OVERLOAD = -2.0

# ============================================================================
# PATHS
# ============================================================================

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, TESTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DEMO CONFIGURATION
# ============================================================================

DEMO_GREEDY = True  # Use greedy policy (epsilon=0) for demo
DEMO_SHOWCASE_EPISODES = 3  # Number of pre-selected showcase episodes
DEMO_STEP_DELAY = 1.0  # Seconds between steps in auto-play mode

# ============================================================================
# CONTEXTUAL BANDIT (FALLBACK) CONFIGURATION
# ============================================================================

LINUCB_ALPHA = 1.0  # Exploration parameter
LINUCB_FEATURE_DIM = 88  # Same as DQN state dim
