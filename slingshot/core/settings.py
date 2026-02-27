from pydantic import Field
from pydantic_settings import BaseSettings
import os
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
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
    
    @property
    def TOTAL_SIM_DAYS(self) -> int:
        return self.PHASE1_DAYS + self.PHASE2_DAYS
        
    @property
    def EPISODE_HORIZON(self) -> int:
        return self.TOTAL_SIM_DAYS * self.SLOTS_PER_DAY

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
    TASK_ARRIVAL_RATE: float = 3.5
    TOTAL_TASKS: int = 200
    NUM_TASKS: int = 200 # Alias
    MIN_TASK_DURATION_H: float = 1.0
    MAX_TASK_DURATION_H: float = 12.0
    TASK_COMPLEXITY_LEVELS: List[int] = [1, 2, 3, 4, 5]
    TASK_PRIORITIES: List[int] = [0, 1, 2, 3]
    TASK_TYPES: List[int] = [0, 1, 2, 3, 4] # For context switches
    CONTEXT_SWITCH_PENALTY: float = 0.2
    TEAM_SYNERGY_ENABLED: bool = True
    DEADLINE_MIN_H: float = 4.0
    DEADLINE_MAX_H: float = 40.0
    COMPLETION_TIME_NOISE: float = 0.25

    DEPENDENCY_GRAPH_COMPLEXITY: int = 4
    MAX_DEPENDENCY_DEPTH: int = 3

    DEADLINE_SHOCK_PROB: float = 0.08
    DEADLINE_SHOCK_SLOTS: int = 8

    # DQN HYPERPARAMETERS
    STATE_DIM: int = 96
    ACTION_DIM: int = 140
    HIDDEN_LAYERS: List[int] = [256, 256]
    ACTIVATION: str = 'relu'
    DUELING_DQN: bool = True

    LEARNING_RATE: float = 0.0003
    GAMMA: float = 0.97
    BATCH_SIZE: int = 32
    REPLAY_BUFFER_SIZE: int = 30000
    MIN_REPLAY_SIZE: int = 32
    TARGET_UPDATE_FREQ: int = 100

    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY: float = 0.999
    EPSILON_PHASE2_START: float = 0.35

    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: int = 50000

    LR_SCHEDULER_T0: int = 2000
    LR_SCHEDULER_T_MULT: int = 1

    MAX_EPISODES: int = 5000
    CHECKPOINT_FREQ: int = 100
    EARLY_STOPPING_PATIENCE: int = 1000
    CONVERGENCE_THRESHOLD: int = 1000

    # REWARD FUNCTION
    REWARD_COMPLETION_BASE: float = 18.0
    REWARD_IDLE_PENALTY: float = -0.2
    REWARD_LATENESS_PENALTY: float = -1.5
    REWARD_OVERLOAD_WEIGHT: float = -0.5
    REWARD_URGENCY_PENALTY: float = -0.3
    REWARD_MAKESPAN_BONUS: float = 50.0
    REWARD_DELAY_WEIGHT: float = -0.005
    REWARD_DEADLINE_MISS_PENALTY: float = -20.0

    REWARD_THROUGHPUT_WEIGHT: float = 2.0
    REWARD_STRATEGIC_DEFER: float = 0.5
    REWARD_EXPLORATION_BONUS: float = 0.3
    FATIGUE_QUALITY_PENALTY: float = 0.25

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
