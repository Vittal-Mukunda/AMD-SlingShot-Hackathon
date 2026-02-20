from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import uuid

class TaskStatus(str, Enum):
    # Standard statuses
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskPriority(int, Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    status: TaskStatus = TaskStatus.TODO
    
    # Simulation properties
    required_skill_level: float = 1.0  # Legacy/API use
    complexity: int = 1                # RL env: [1, 2, 3, 4, 5]
    estimated_duration: float = 1.0    # In hours
    remaining_work: float = 1.0        # In hours
    deadline: float = 0.0              # Simulation time (hours from start)
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # RL Control Flags
    deferred: bool = False
    escalated: bool = False
    
    # Assignment
    assigned_to: Optional[str] = None  # Worker ID
    
    # Validation/Tracking
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class Worker(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    
    # Attributes
    skill_level: float = 1.0           # Legacy/API use
    true_skill: float = 1.0            # True unseen skill [0.6, 1.4]
    estimated_skill: float = 1.0       # Agent's belief
    efficiency: float = 1.0            # Speed multiplier (0.5 to 1.5)
    max_capacity: float = 1.0          # Max concurrent tasks
    
    # Dynamic state
    fatigue: float = 0.0               # 0.0 to 3.0 (burnout)
    availability: int = 1              # 1=available, 0=unavailable
    current_task_id: Optional[str] = None

class SimulationState(BaseModel):
    tasks: List[Task] = []
    workers: List[Worker] = []
    current_time: float = 0.0          # Global simulation time (hours)
