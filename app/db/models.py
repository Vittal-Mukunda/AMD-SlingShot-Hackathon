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

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    status: TaskStatus = TaskStatus.TODO
    
    # Simulation properties
    required_skill_level: float = 1.0  # 1.0 to 10.0
    estimated_duration: float = 1.0    # In hours
    remaining_work: float = 1.0        # In hours
    deadline: float = 0.0              # Simulation time (hours from start)
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Assignment
    assigned_to: Optional[str] = None  # Worker ID
    
    # Validation/Tracking
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class Worker(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    
    # Attributes
    skill_level: float = 1.0           # 1.0 to 10.0
    efficiency: float = 1.0            # Speed multiplier (0.5 to 1.5)
    max_capacity: float = 1.0          # Max concurrent tasks (usually 1.0)
    
    # Dynamic state
    fatigue: float = 0.0               # 0.0 to 1.0 (affects efficiency)
    current_task_id: Optional[str] = None

class SimulationState(BaseModel):
    tasks: List[Task] = []
    workers: List[Worker] = []
    current_time: float = 0.0          # Global simulation time (hours)
