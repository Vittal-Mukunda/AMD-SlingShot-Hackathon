"""
live_training_visualizer.py — Non-blocking real-time training dashboard.

Creates a 4-panel Matplotlib window that updates every N episodes during training:
  Panel 1: Episode return + 50-ep moving average
  Panel 2: Epsilon decay curve
  Panel 3: Mean Q-value per episode
  Panel 4: Reward component breakdown (stacked area)

Usage:
    from slingshot.training.live_training_visualizer import LiveTrainingVisualizer
    viz = LiveTrainingVisualizer(output_path="results/learning_curve.png")
    viz.update(episode, return_, epsilon, q_mean, breakdown_dict)
    viz.save_final()   # call once at end of training
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from typing import Dict, Optional

# ── Use TkAgg if available, otherwise Agg (headless / no display) ────────────
_INTERACTIVE = True
try:
    matplotlib.use('TkAgg')
    import tkinter  # probe for display
    tkinter.Tk().destroy()
except Exception:
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        matplotlib.use('Agg')
        _INTERACTIVE = False


class LiveTrainingVisualizer:
    """
    Non-blocking 4-panel training dashboard.

    Parameters
    ----------
    output_path : str
        Where to save the final high-resolution PNG.
    update_freq : int
        Refresh the plot every this many episodes (default: 10).
    window_size : int
        Moving average window (default: 50).
    max_points : int
        Max data points kept in memory per series (default: 10000).
    """

    def __init__(
        self,
        output_path: str = "results/learning_curve.png",
        update_freq: int = 10,
        window_size: int = 50,
        max_points: int = 10000,
    ):
        self.output_path = output_path
        self.update_freq = update_freq
        self.window_size = window_size
        self.interactive = _INTERACTIVE

        # Data buffers (bounded to avoid memory growth over 5000 episodes)
        self.episodes:       list = []
        self.returns:        list = []
        self.moving_avgs:    list = []
        self.epsilons:       list = []
        self.q_means:        list = []

        # Reward breakdown
        self.comp_rewards:   list = []
        self.delay_penalties: list = []
        self.deadline_penalties: list = []
        self.overload_penalties: list = []
        self.throughput_bonuses: list = []

        self._return_window = deque(maxlen=window_size)
        self._fig = None
        self._axes = None
        self._initialized = False

    # ── Internal setup ────────────────────────────────────────────────────────

    def _setup(self):
        """Create the figure (called on first update)."""
        if self.interactive:
            plt.ion()

        self._fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
        self._fig.canvas.manager.set_window_title("DQN Live Training Dashboard")

        gs = gridspec.GridSpec(2, 2, figure=self._fig, hspace=0.45, wspace=0.35)
        self._ax_return   = self._fig.add_subplot(gs[0, 0])
        self._ax_epsilon  = self._fig.add_subplot(gs[0, 1])
        self._ax_q        = self._fig.add_subplot(gs[1, 0])
        self._ax_breakdown = self._fig.add_subplot(gs[1, 1])

        for ax in [self._ax_return, self._ax_epsilon, self._ax_q, self._ax_breakdown]:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#e0e0e0', labelsize=8)
            ax.spines[:].set_color('#444466')
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_color('#e0e0e0')

        self._initialized = True

    def _style_ax(self, ax, title: str, xlabel: str, ylabel: str):
        ax.set_title(title, color='#e0e0e0', fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel, color='#aaaacc', fontsize=8)
        ax.set_ylabel(ylabel, color='#aaaacc', fontsize=8)
        ax.grid(True, alpha=0.2, color='#555577', linestyle='--', linewidth=0.5)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        episode: int,
        episode_return: float,
        epsilon: float,
        q_mean: float,
        breakdown: Optional[Dict] = None,
    ):
        """
        Record one episode's data and (every update_freq eps) refresh the plot.

        Parameters
        ----------
        episode       : int   — current episode number
        episode_return: float — total scaled return for this episode
        epsilon       : float — current exploration rate
        q_mean        : float — mean Q-value this episode (0 if buffer not warm)
        breakdown     : dict  — reward component dict from env.get_episode_reward_breakdown()
                                Keys: completion_reward, delay_penalty, overload_penalty,
                                      throughput_bonus, deadline_penalty
        """
        self._return_window.append(episode_return)
        moving_avg = float(np.mean(self._return_window))

        self.episodes.append(episode)
        self.returns.append(episode_return)
        self.moving_avgs.append(moving_avg)
        self.epsilons.append(epsilon)
        self.q_means.append(q_mean)

        if breakdown:
            self.comp_rewards.append(breakdown.get('completion_reward', 0.0))
            self.delay_penalties.append(breakdown.get('delay_penalty', 0.0))
            self.deadline_penalties.append(breakdown.get('deadline_penalty', 0.0))
            self.overload_penalties.append(breakdown.get('overload_penalty', 0.0))
            self.throughput_bonuses.append(breakdown.get('throughput_bonus', 0.0))
        else:
            for lst in [self.comp_rewards, self.delay_penalties,
                        self.deadline_penalties, self.overload_penalties,
                        self.throughput_bonuses]:
                lst.append(0.0)

        # Only redraw every update_freq episodes
        if episode % self.update_freq != 0:
            return

        if not self._initialized:
            try:
                self._setup()
            except Exception as e:
                print(f"  [viz] Could not init display: {e}. Falling back to file-only mode.")
                self.interactive = False
                matplotlib.use('Agg')
                self._setup()

        self._draw()

    def _draw(self):
        """Redraw all 4 panels."""
        eps = self.episodes
        if not eps:
            return

        # ── Panel 1: Episode Return ──────────────────────────────────────────
        ax = self._ax_return
        ax.clear()
        ax.plot(eps, self.returns, color='#6688cc', alpha=0.35, linewidth=0.7, label='Episode return')
        ax.plot(eps, self.moving_avgs, color='#ff6b6b', linewidth=1.8, label=f'MA-{self.window_size}')
        ax.axhline(0, color='#888888', linestyle='--', linewidth=0.6, alpha=0.6)
        ax.legend(fontsize=7, loc='lower right', facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#e0e0e0')
        self._style_ax(ax, 'Episode Return', 'Episode', 'Return (scaled)')

        # ── Panel 2: Epsilon ─────────────────────────────────────────────────
        ax = self._ax_epsilon
        ax.clear()
        ax.plot(eps, self.epsilons, color='#56cc9d', linewidth=1.5)
        ax.axhline(0.05, color='#ffcc66', linestyle=':', linewidth=1.0, alpha=0.8, label='eps floor (0.05)')
        ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#555577', labelcolor='#e0e0e0')
        ax.set_ylim(0, 1.05)
        self._style_ax(ax, 'Exploration Rate (eps)', 'Episode', 'Epsilon')

        # ── Panel 3: Q-values ────────────────────────────────────────────────
        ax = self._ax_q
        ax.clear()
        ax.plot(eps, self.q_means, color='#cc88ff', linewidth=1.2)
        self._style_ax(ax, 'Mean Q-Value', 'Episode', 'Q-value')

        # ── Panel 4: Reward Breakdown ────────────────────────────────────────
        ax = self._ax_breakdown
        ax.clear()
        n = len(eps)
        if n > 0 and len(self.comp_rewards) == n:
            # Smooth with a simple running mean for readability
            w = min(20, n)
            def smooth(lst):
                arr = np.array(lst, dtype=float)
                if len(arr) < w:
                    return arr
                kernel = np.ones(w) / w
                return np.convolve(arr, kernel, mode='same')

            comp   = smooth(self.comp_rewards)
            delay  = smooth(self.delay_penalties)
            dead   = smooth(self.deadline_penalties)
            over   = smooth(self.overload_penalties)
            thru   = smooth(self.throughput_bonuses)

            ax.plot(eps, comp,  color='#56cc9d', linewidth=1.2, label='Completion')
            ax.plot(eps, thru,  color='#66aaff', linewidth=1.0, label='Throughput')
            ax.plot(eps, delay, color='#ffcc66', linewidth=1.0, label='Delay pen.')
            ax.plot(eps, dead,  color='#ff6b6b', linewidth=1.2, label='Deadline pen.')
            ax.plot(eps, over,  color='#ff9966', linewidth=0.8, label='Overload pen.')
            ax.axhline(0, color='#888888', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.legend(fontsize=6, loc='lower right', facecolor='#1a1a2e',
                      edgecolor='#555577', labelcolor='#e0e0e0', ncol=2)
        self._style_ax(ax, 'Reward Components (smoothed)', 'Episode', 'Raw reward')

        # ── Main title ───────────────────────────────────────────────────────
        ep_now = eps[-1]
        eps_now = self.epsilons[-1]
        ma_now = self.moving_avgs[-1]
        self._fig.suptitle(
            f'DQN Training  |  Episode {ep_now}  |  eps={eps_now:.3f}  |  MA-50={ma_now:.1f}',
            color='#ffffff', fontsize=12, fontweight='bold', y=0.98
        )

        if self.interactive:
            try:
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass  # headless or window closed — continue training silently

    def save_final(self):
        """
        Save the final high-resolution version to output_path.
        Call once at end of training.
        """
        if self._fig is None:
            # Nothing was ever drawn — generate a static plot from data
            self._setup()
            self._draw()

        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        # Switch to Agg for saving even if Tk was active
        self._fig.savefig(self.output_path, dpi=300, bbox_inches='tight',
                          facecolor=self._fig.get_facecolor())
        print(f"  [viz] Final learning curve saved → {self.output_path}")

        if self.interactive:
            plt.ioff()


# ── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == '__main__':
    import math
    print("Smoke-testing LiveTrainingVisualizer...")
    viz = LiveTrainingVisualizer(output_path='results/learning_curve_test.png', update_freq=5)

    for ep in range(100):
        ret = -200 + ep * 2 + np.random.randn() * 20
        eps = max(0.05, 1.0 * (0.995 ** ep))
        q   = min(40.0, ep * 0.3)
        bd  = {
            'completion_reward': 10 + ep * 0.1,
            'delay_penalty':    -5 - ep * 0.02,
            'deadline_penalty': -30 + ep * 0.2,
            'overload_penalty': -2.0,
            'throughput_bonus':  5.0,
        }
        viz.update(ep, ret, eps, q, bd)

    viz.save_final()
    print("Done.")
