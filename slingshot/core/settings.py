from pydantic_settings import BaseSettings
import os
import sys
from typing import List

# ── v12 fix: import config.py as single source of truth ──────────────────────
_cfg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _cfg_root not in sys.path:
    sys.path.insert(0, _cfg_root)
import config as _cfg

PROJECT_ROOT = _cfg_root


class Settings(BaseSettings):
    """
    Pydantic settings for runtime use.
    v12 fix: ALL hyperparameter defaults now source from config.py.
    This makes config.py the single source of truth — changing config.py
    will correctly propagate to all DQN agent/baseline/sweep code paths.
    """

    # App config
    APP_NAME: str = "MCP Project Manager"
    DEBUG: bool = True

    # WORKDAY SIMULATION
    SLOT_HOURS: float = 0.5
    SLOTS_PER_DAY: int = 16
    WORK_DAYS_PER_WEEK: int = 5
    WORK_START_SLOT: int = 0
    WORK_END_SLOT: int = 15

    PHASE1_DAYS: int = 20
    PHASE2_DAYS: int = 5
    SIM_DAYS: int = 25
    PHASE1_FRACTION: float = 0.60
    PHASE2_FRACTION: float = 0.40

    @property
    def TOTAL_SIM_DAYS(self) -> int:
        return self.SIM_DAYS

    @property
    def EPISODE_HORIZON(self) -> int:
        return self.SIM_DAYS * self.SLOTS_PER_DAY

    # Controlled training / emission intervals
    TRAIN_EVERY_N_STEPS: int = 4
    EMIT_EVERY_N_TICKS: int  = 8

    # Bounded accumulator caps
    MAX_GANTT_BLOCKS_IN_MEMORY: int  = 500
    MAX_DAILY_METRICS_IN_MEMORY: int = 730

    # WORKERS
    NUM_WORKERS: int = 5
    MAX_WORKER_LOAD: int = 5
    FATIGUE_LEVELS: int = 4
    BURNOUT_RECOVERY_TIME: int = 8

    SKILL_MIN: float = 0.5
    SKILL_MAX: float = 1.5
    SKILL_PRIOR_ALPHA: float = 2.0
    SKILL_PRIOR_BETA: float = 2.0

    SPEED_MULT_MEAN: float = 1.0
    SPEED_MULT_STD: float = 0.20
    FATIGUE_RATE_MEAN: float = 0.20
    FATIGUE_RATE_STD: float = 0.06
    RECOVERY_RATE_MEAN: float = 0.12
    RECOVERY_RATE_STD: float = 0.04
    FATIGUE_SENS_MEAN: float = 0.18
    FATIGUE_SENS_STD: float = 0.05
    BURNOUT_RESIL_MEAN: float = 2.6
    BURNOUT_RESIL_STD: float = 0.25

    INTRADAY_DECAY_RATE: float = 0.03
    FATIGUE_CARRYOVER: float = 0.10

    # TASKS
    TASK_ARRIVAL_RATE: float = 4.0
    TOTAL_TASKS: int = 200
    NUM_TASKS: int = 200
    MIN_TASK_DURATION_H: float = 1.0
    MAX_TASK_DURATION_H: float = 12.0
    TASK_COMPLEXITY_LEVELS: List[int] = [1, 2, 3, 4, 5]
    TASK_PRIORITIES: List[int] = [0, 1, 2, 3]
    TASK_TYPES: List[int] = [0, 1, 2, 3, 4]
    CONTEXT_SWITCH_PENALTY: float = 0.2
    TEAM_SYNERGY_ENABLED: bool = True
    DEADLINE_MIN_DAYS: float = _cfg.DEADLINE_MIN_DAYS
    DEADLINE_MAX_DAYS: float = _cfg.DEADLINE_MAX_DAYS  # v12: 2.5 sourced from config
    DEADLINE_MIN_H: float = _cfg.DEADLINE_MIN_DAYS * 8.0
    DEADLINE_MAX_H: float = _cfg.DEADLINE_MAX_DAYS * 8.0
    COMPLETION_TIME_NOISE: float = 0.25

    DEPENDENCY_GRAPH_COMPLEXITY: int = 4
    MAX_DEPENDENCY_DEPTH: int = 3

    DEADLINE_SHOCK_PROB: float = 0.08
    DEADLINE_SHOCK_SLOTS: int = 8

    # ── DQN HYPERPARAMETERS — READ FROM config.py ────────────────────────────
    # v12 fix: no more hardcoded values that silently override config.py
    STATE_DIM: int = 96
    ACTION_DIM: int = 140
    HIDDEN_LAYERS: List[int] = [256, 256]
    ACTIVATION: str = 'relu'
    DUELING_DQN: bool = True

    LEARNING_RATE: float = _cfg.LEARNING_RATE
    GAMMA: float = _cfg.GAMMA
    EPSILON_DECAY: float = _cfg.EPSILON_DECAY
    REWARD_COMPLETION_BASE: float = _cfg.REWARD_COMPLETION_BASE
    PER_ALPHA: float = _cfg.PER_ALPHA          # v12: 0.4 from config (was hardcoded 0.6)
    TARGET_UPDATE_FREQ: int = _cfg.TARGET_UPDATE_FREQ
    BATCH_SIZE: int = _cfg.BATCH_SIZE
    MIN_REPLAY_SIZE: int = _cfg.MIN_REPLAY_SIZE
    REPLAY_BUFFER_SIZE: int = _cfg.REPLAY_BUFFER_SIZE
    REPLAY_BUFFER_MAX_CAPACITY: int = _cfg.REPLAY_BUFFER_MAX_CAPACITY

    EPSILON_START: float = _cfg.EPSILON_START
    EPSILON_END: float = _cfg.EPSILON_END
    EPSILON_PHASE2_START: float = _cfg.EPSILON_PHASE2_START

    PER_BETA_START: float = _cfg.PER_BETA_START
    PER_BETA_FRAMES: int = _cfg.PER_BETA_FRAMES
    LR_SCHEDULER_ETA_MIN_FRACTION: float = _cfg.LR_SCHEDULER_ETA_MIN_FRACTION  # v12: 0.15

    LR_SCHEDULER_T0: int = _cfg.LR_SCHEDULER_T0
    LR_SCHEDULER_T_MULT: int = _cfg.LR_SCHEDULER_T_MULT

    MAX_EPISODES: int = 5000
    CHECKPOINT_FREQ: int = 100
    EARLY_STOPPING_PATIENCE: int = 1000
    CONVERGENCE_THRESHOLD: int = 1000

    # REWARD FUNCTION — sourced from config.py
    REWARD_IDLE_PENALTY: float = _cfg.REWARD_IDLE_PENALTY
    REWARD_LATENESS_PENALTY: float = _cfg.REWARD_LATENESS_PENALTY
    REWARD_OVERLOAD_WEIGHT: float = _cfg.REWARD_OVERLOAD_WEIGHT
    REWARD_URGENCY_PENALTY: float = _cfg.REWARD_URGENCY_PENALTY
    REWARD_MAKESPAN_BONUS: float = _cfg.REWARD_MAKESPAN_BONUS
    REWARD_DELAY_WEIGHT: float = _cfg.REWARD_DELAY_WEIGHT
    REWARD_DEADLINE_MISS_PENALTY: float = _cfg.REWARD_DEADLINE_MISS_PENALTY
    REWARD_BACKLOG_PENALTY: float = _cfg.REWARD_BACKLOG_PENALTY
    REWARD_TERMINAL_UNFINISHED_PENALTY: float = _cfg.REWARD_TERMINAL_UNFINISHED_PENALTY
    REWARD_THROUGHPUT_WEIGHT: float = _cfg.REWARD_THROUGHPUT_WEIGHT
    REWARD_STRATEGIC_DEFER: float = _cfg.REWARD_STRATEGIC_DEFER
    REWARD_EXPLORATION_BONUS: float = _cfg.REWARD_EXPLORATION_BONUS
    FATIGUE_QUALITY_PENALTY: float = _cfg.FATIGUE_QUALITY_PENALTY
    REWARD_SKILL_MATCH_WEIGHT: float = _cfg.REWARD_SKILL_MATCH_WEIGHT  # v12: 0.4
    REWARD_FATIGUE_ASSIGN_PENALTY: float = _cfg.REWARD_FATIGUE_ASSIGN_PENALTY
    REWARD_OVERLOAD_ASSIGN_PENALTY: float = _cfg.REWARD_OVERLOAD_ASSIGN_PENALTY
    REWARD_OVERLOAD_IMMEDIATE: float = _cfg.REWARD_OVERLOAD_IMMEDIATE
    REWARD_LOAD_BALANCE_BONUS: float = _cfg.REWARD_LOAD_BALANCE_BONUS
    REWARD_IDLE_WAITING_PENALTY: float = _cfg.REWARD_IDLE_WAITING_PENALTY
    REWARD_CLIP_MIN: float = _cfg.REWARD_CLIP_MIN
    REWARD_CLIP_MAX: float = _cfg.REWARD_CLIP_MAX

    # BASELINE CONFIGURATION
    BASELINE_SKILL_ESTIMATION_EPISODES: int = 10
    BASELINE_DEBUG_SKILL: bool = False

    # EVALUATION CONFIGURATION
    TRAIN_EPISODES: int = 5000
    TRAIN_RANDOM_SEEDS: List[int] = [42, 123, 456, 789, 1011]
    TEST_EPISODES: int = 200
    TEST_EPISODES_STANDARD: int = 50
    TEST_EPISODES_HIGH_VARIANCE: int = 50
    TEST_EPISODES_FREQUENT_SHOCKS: int = 50
    TEST_EPISODES_FIXED: int = 50
    TEST_VARIANCE_MULTIPLIER: float = 1.5
    TEST_SHOCK_PROB_HIGH: float = 0.3
    SIGNIFICANCE_LEVEL: float = 0.05
    BONFERRONI_NUM_COMPARISONS: int = 5
    COHEN_D_THRESHOLD: float = 0.5

    # METRICS WEIGHTS
    METRIC_WEIGHT_THROUGHPUT: float = 2.0
    METRIC_WEIGHT_DEADLINE: float = 3.0
    METRIC_WEIGHT_DELAY: float = -0.5
    METRIC_WEIGHT_LOAD_BALANCE: float = -1.0
    METRIC_WEIGHT_QUALITY: float = 1.0
    METRIC_WEIGHT_OVERLOAD: float = -2.0

    # PATHS
    PROJECT_ROOT: str = PROJECT_ROOT
    CHECKPOINT_DIR: str = os.path.join(PROJECT_ROOT, 'checkpoints')
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, 'results')
    LOGS_DIR: str = os.path.join(PROJECT_ROOT, 'logs')
    TESTS_DIR: str = os.path.join(PROJECT_ROOT, 'tests')

    # DEMO CONFIG
    DEMO_GREEDY: bool = True
    DEMO_SHOWCASE_EPISODES: int = 3
    DEMO_STEP_DELAY: float = 1.0

    # BANDIT
    LINUCB_ALPHA: float = 1.0
    LINUCB_FEATURE_DIM: int = 96

    class Config:
        env_prefix = "SLINGSHOT_"


# Create the global instance
config = Settings()

# Ensure directories exist
for directory in [config.CHECKPOINT_DIR, config.RESULTS_DIR, config.LOGS_DIR, config.TESTS_DIR]:
    os.makedirs(directory, exist_ok=True)
