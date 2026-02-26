# Task Generation

The simulation environment dynamically streams task arrivals based on a Poisson process to enforce an online problem without any lookahead. 

## Dynamics Overview
Tasks arrive strictly based on `TASK_ARRIVAL_RATE` scaled mathematically across the configured days.
- Each `Task` object evaluates complexity on a scale from 1 to 5 (`TASK_COMPLEXITY_LEVELS`).
- They inherently include dynamic priority mappings linking urgency and minimum durations required to fulfill them.

## Dependencies Execution
Some generated tasks will depend recursively on previous ones finishing (`DEPENDENCY_GRAPH_COMPLEXITY`).
- A task can require up to 3 dependencies (`MAX_DEPENDENCY_DEPTH`). 
- When generating the environment, the DAG resolves logically by creating edges between sequentially generated task chunks to ensure correct simulation ordering.
- The `valid_actions` array restricts scheduling any task with unmet dependencies, forcing the DQN to evaluate complex time constraints strategically.
