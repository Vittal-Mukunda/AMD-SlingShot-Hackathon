"""
Live Task Allocation Grid Visualizer

Displays a real-time 5-row × 20-column matrix grid showing how tasks are assigned
to workers by each agent (baselines and DQN). Each cell in the grid lights up when
a task is assigned to a worker.

No training graphs — purely task allocation visualization.
"""

import matplotlib
try:
    matplotlib.use('TkAgg')  # Preferred for live interactive updates on Windows
except Exception:
    pass  # Fall back to default backend if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colour palette (one distinct colour per worker, DQN gets a blue set) ──────
WORKER_COLORS = [
    '#E74C3C',  # W0 - Crimson Red
    '#27AE60',  # W1 - Emerald Green
    '#F39C12',  # W2 - Amber Orange
    '#8E44AD',  # W3 - Deep Purple
    '#2980B9',  # W4 - Steel Blue
]
DQN_COLORS = [
    '#1ABC9C',  # W0 - Teal
    '#3498DB',  # W1 - Bright Blue
    '#5DADE2',  # W2 - Light Blue
    '#2874A6',  # W3 - Dark Blue
    '#AED6F1',  # W4 - Pale Blue
]
DEFER_ALPHA = 0.25   # Transparency for deferred tasks
ASSIGN_ALPHA = 0.85  # Transparency for assigned tasks


class TaskGridVisualizer:
    """
    Live 5-row × 20-column task allocation matrix.

    Rows = workers (W0–W4)
    Columns = task IDs (0–19)
    Each cell lights up when a task is assigned to a worker.
    """

    def __init__(self, num_workers: int = 5, num_tasks: int = 20):
        """
        Create the matplotlib figure and grid.

        Args:
            num_workers: Number of workers (rows in the matrix)
            num_tasks: Number of tasks (columns in the matrix)
        """
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self._is_dqn = False

        # Track assignments: grid[worker_id][task_id] = (action_type, step)
        self.grid = [[None] * num_tasks for _ in range(num_workers)]
        self.assign_count = 0
        self.defer_count = 0
        self.step_count = 0

        # ── Create figure ──────────────────────────────────────────────────────
        plt.ion()  # Interactive mode ON
        self.fig, self.ax = plt.subplots(figsize=(16, 5))
        self.fig.patch.set_facecolor('#1A1A2E')   # Dark navy background
        self.ax.set_facecolor('#16213E')

        # ── Render empty grid ──────────────────────────────────────────────────
        self._draw_empty_grid()
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.pause(0.001)

    def _draw_empty_grid(self):
        """Draw the empty grid skeleton."""
        self.ax.clear()
        self.ax.set_facecolor('#16213E')

        # Draw grid lines
        for w in range(self.num_workers + 1):
            self.ax.axhline(y=w, color='#2C3E50', linewidth=0.8)
        for t in range(self.num_tasks + 1):
            self.ax.axvline(x=t, color='#2C3E50', linewidth=0.8)

        # Worker row labels
        for w in range(self.num_workers):
            self.ax.text(
                -0.6, w + 0.5,
                f'W{w}', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold'
            )

        # Task column labels
        for t in range(self.num_tasks):
            self.ax.text(
                t + 0.5, -0.35,
                f'T{t}', ha='center', va='center',
                color='#BDC3C7', fontsize=7
            )

        self.ax.set_xlim(-0.9, self.num_tasks)
        self.ax.set_ylim(-0.6, self.num_workers)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.tick_params(left=False, bottom=False)
        for spine in self.ax.spines.values():
            spine.set_color('#2C3E50')

    def reset(self, agent_name: str, is_dqn: bool = False):
        """
        Clear the grid for a new agent run.

        Also removes all figure-level Text artists (metric overlays) left by
        the previous agent's finalize() call — fixing the text overlap bug.

        Args:
            agent_name: Name of agent about to run (e.g. 'Random', 'DQN Agent')
            is_dqn: True if the running agent is the DQN agent (uses blue palette)
        """
        self.grid = [[None] * self.num_tasks for _ in range(self.num_workers)]
        self.assign_count = 0
        self.defer_count = 0
        self.step_count = 0
        self._is_dqn = is_dqn
        self._agent_name = agent_name

        # ── FIX: remove all figure-level Text artists from previous finalize() ──
        # fig.text() appends to fig.texts without auto-clearing, causing overlap.
        for txt in self.fig.texts[:]:   # iterate over a copy — we modify the list
            txt.remove()

        self._draw_empty_grid()
        self._update_title()
        plt.pause(0.1)


    def _update_title(self):
        """Refresh the figure title with current stats."""
        palette = 'Blue Palette' if self._is_dqn else '5-Color Palette'
        self.fig.suptitle(
            f'🎯  Task Allocation Grid  ─  Agent: {self._agent_name}  '
            f'│  Step {self.step_count}  │  Assigned: {self.assign_count}  │  Deferred: {self.defer_count}  '
            f'│  [{palette}]',
            color='white', fontsize=11, fontweight='bold',
            y=0.98
        )

    def update(self, task_id: int, worker_id: int, action_type: str, step: int,
               task_info: dict = None):
        """
        Add a colored cell to the grid and redraw.

        Args:
            task_id: Task index (column)
            worker_id: Worker index (row), -1 for defer actions
            action_type: 'assign' | 'defer' | 'escalate'
            step: Current timestep
            task_info: Optional dict with 'complexity' and 'priority' for cell label
        """
        self.step_count = step

        if action_type == 'assign' and 0 <= worker_id < self.num_workers and 0 <= task_id < self.num_tasks:
            self._place_cell(task_id, worker_id, action_type='assign', task_info=task_info)
            self.assign_count += 1

        elif action_type == 'defer' and 0 <= task_id < self.num_tasks:
            # Show deferred tasks faintly in row 0 band (optional — only first defer per task)
            if all(self.grid[w][task_id] is None for w in range(self.num_workers)):
                self.defer_count += 1
                # Mark as deferred — we draw a faint cell at the bottom strip
                self._place_defer_strip(task_id)

        self._update_title()
        self.fig.canvas.draw_idle()
        plt.pause(0.05)  # Small pause for live update

    def _place_cell(self, task_id: int, worker_id: int, action_type: str, task_info: dict = None):
        """Draw a colored rectangle at grid[worker_id][task_id]."""
        colors = DQN_COLORS if self._is_dqn else WORKER_COLORS
        color = colors[worker_id % len(colors)]
        alpha = ASSIGN_ALPHA

        # Draw filled rectangle
        rect = mpatches.FancyBboxPatch(
            (task_id + 0.05, worker_id + 0.08),
            0.90, 0.84,
            boxstyle="round,pad=0.02",
            linewidth=0.5,
            edgecolor='white',
            facecolor=color,
            alpha=alpha,
            zorder=3
        )
        self.ax.add_patch(rect)

        # Cell label: complexity / priority
        label = ''
        if task_info:
            c = task_info.get('complexity', '?')
            p = task_info.get('priority', '?')
            label = f'C{c}\nP{p}'
        else:
            label = f'T{task_id}'

        self.ax.text(
            task_id + 0.5, worker_id + 0.5, label,
            ha='center', va='center',
            color='white', fontsize=6, fontweight='bold',
            zorder=4
        )

        # Store assignment
        self.grid[worker_id][task_id] = action_type

    def _place_defer_strip(self, task_id: int):
        """Draw a thin faint stripe at the bottom of column task_id to mark deferral."""
        rect = mpatches.FancyBboxPatch(
            (task_id + 0.05, 0.02),
            0.90, 0.12,
            boxstyle="round,pad=0.01",
            linewidth=0,
            edgecolor='none',
            facecolor='#F1C40F',  # Yellow for defer
            alpha=0.35,
            zorder=2
        )
        self.ax.add_patch(rect)

    def add_legend(self):
        """Show a legend panel for color-to-worker mapping."""
        colors = DQN_COLORS if self._is_dqn else WORKER_COLORS
        patches = [
            mpatches.Patch(color=colors[w], label=f'Worker {w}', alpha=ASSIGN_ALPHA)
            for w in range(self.num_workers)
        ]
        patches.append(mpatches.Patch(color='#F1C40F', label='Deferred', alpha=0.5))
        legend = self.ax.legend(
            handles=patches, loc='upper left',
            bbox_to_anchor=(1.01, 1), borderaxespad=0,
            framealpha=0.15, labelcolor='white',
            fontsize=8, title='Agent', title_fontsize=8,
        )
        plt.setp(legend.get_title(), color='white')

    def finalize(self, agent_name: str, metrics: dict = None):
        """
        Display final stats overlay after an agent finishes.

        Args:
            agent_name: Agent name for the summary
            metrics: Dict of final metrics to display
        """
        self.add_legend()
        summary = f'{agent_name} Complete — Assigned: {self.assign_count} | Deferred: {self.defer_count}'
        if metrics:
            throughput = metrics.get('throughput', '?')
            hit_rate = metrics.get('deadline_hit_rate', 0.0)
            quality = metrics.get('quality_score', 0.0)
            summary += f'\nThroughput: {throughput} | Deadline Hit: {hit_rate:.0%} | Quality: {quality:.2f}'

        self.fig.text(
            0.5, 0.02, summary,
            ha='center', va='bottom',
            color='#2ECC71', fontsize=9, fontweight='bold',
            transform=self.fig.transFigure
        )
        self.fig.canvas.draw_idle()
        plt.pause(1.5)  # Show the final state for a moment before next agent

    def close(self):
        """Close the visualization window."""
        plt.ioff()
        plt.show(block=False)
        plt.close(self.fig)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing TaskGridVisualizer...")
    viz = TaskGridVisualizer(num_workers=5, num_tasks=20)
    viz.reset("Test Agent", is_dqn=False)

    import random, time
    for step in range(30):
        task_id = random.randint(0, 19)
        worker_id = random.randint(0, 4)
        atype = 'assign' if random.random() > 0.2 else 'defer'
        viz.update(task_id, worker_id, atype, step,
                   task_info={'complexity': random.randint(1, 5), 'priority': random.randint(0, 3)})
        time.sleep(0.1)

    viz.finalize("Test Agent", {'throughput': 12, 'deadline_hit_rate': 0.8, 'quality_score': 0.75})
    print("Visualization test complete.")
