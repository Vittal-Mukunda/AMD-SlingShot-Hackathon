import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs as JSON."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Add any extra attributes passed to the logger
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data) # type: ignore
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a JSON logger."""
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to prevent duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        
        logger.addHandler(handler)
        
    return logger

def log_agent_action(logger: logging.Logger, action_type: str, details: Dict[str, Any]):
    """Helper to log agent decisions."""
    logger.info(
        f"Agent Action: {action_type}", 
        extra={"extra_data": {"action_type": action_type, "details": details}}
    )

def log_state_transition(logger: logging.Logger, from_state: str, to_state: str, metrics: Dict[str, Any] = None):
    """Helper to log backend state transitions."""
    extra = {
        "transition": f"{from_state} -> {to_state}",
        "metrics": metrics or {}
    }
    logger.info(
        "State Transition", 
        extra={"extra_data": extra}
    )
