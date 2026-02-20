"""
Metrics calculation utilities for Day 5 evaluation
"""

import numpy as np
from typing import Dict

def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Compute composite score based on Day 5 evaluation weights:
    +10 for completion
    -0.5 for delay
    -5 for overload
    +2 for throughput (per task)
    -50 for deadline misses
    
    Args:
        metrics: Dictionary containing:
            - tasks_completed (int)
            - avg_delay (float)
            - overload_events (int)
            - deadline_misses (int)
            
    Returns:
        float: Composite score
    """
    # Extract metrics (handle different naming conventions if needed)
    tasks_completed = metrics.get('tasks_completed', metrics.get('throughput', 0))
    avg_delay = metrics.get('avg_delay', 0.0)
    overload_events = metrics.get('overload_events', 0)
    deadline_misses = metrics.get('deadline_misses', 0)
    
    # Calculate score components
    score_completion = 10.0 * tasks_completed
    score_delay = -0.5 * avg_delay
    score_overload = -5.0 * overload_events
    score_throughput = 2.0 * tasks_completed
    score_deadline = -50.0 * deadline_misses
    
    total_score = (score_completion + score_delay + score_overload + 
                   score_throughput + score_deadline)
    
    return total_score
