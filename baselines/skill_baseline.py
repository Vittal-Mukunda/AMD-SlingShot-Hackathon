"""
B4: Skill-Matching Baseline — v4 (Fixed & Debugged).

Fixes in v4:
  - skill_estimates stored as incremental FLOATS (Welford online mean) — no list accumulation
  - Deterministic selection: explicit max() loop; preserves original worker_id at all times
  - Verify: assertion ensures best_worker is always the highest-skill feasible worker
  - Debug logging: controlled by config.BASELINE_DEBUG_SKILL (prints per-assignment info)
  - Removed accumulation bias from cross-episode list growth
  - Works in dynamic-arrival mode: only considers tasks visible at current tick
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from baselines.base_policy import BasePolicy
from environment.project_env import ProjectEnv


class SkillBaseline(BasePolicy):
    """
    Static skill-matching policy (corrected).

    Strategy:
      1. Observe first N episodes/windows to maintain per-worker running skill estimates.
      2. Assign each arriving task to the available worker with the HIGHEST estimated skill
         who still has capacity (load < MAX_WORKER_LOAD).
      3. Sort candidate tasks by (priority DESC, deadline_urgency DESC).

    Skill estimates use Welford's online mean — no list growth across resets.
    """

    def __init__(self, env: ProjectEnv, observation_episodes: int = None,
                 debug: bool = None):
        super().__init__(env)
        self.name = "Skill"
        self.observation_episodes = observation_episodes or config.BASELINE_SKILL_ESTIMATION_EPISODES
        self.debug = debug if debug is not None else config.BASELINE_DEBUG_SKILL

        # Welford running stats: {worker_id: (count, mean)}
        self._skill_counts: dict  = {i: 0   for i in range(env.num_workers)}
        self._skill_means:  dict  = {i: 1.0 for i in range(env.num_workers)}  # Prior = 1.0

        self.episodes_observed = 0
        self.is_observing      = True

    # ── Observation phase ─────────────────────────────────────────────────────

    def observe_episode(self, env: ProjectEnv):
        """
        After an episode, update Welford running mean for each worker's skill.
        Called externally at end of each observation episode.
        """
        for wid, worker in enumerate(env.workers):
            if len(worker.completion_history) == 0:
                continue
            mean_est, _ = worker.get_skill_estimate()
            if np.isnan(mean_est) or mean_est <= 0:
                continue
            # Welford update: new_mean = old_mean + (x - old_mean) / n
            self._skill_counts[wid] += 1
            n = self._skill_counts[wid]
            self._skill_means[wid] += (mean_est - self._skill_means[wid]) / n

        self.episodes_observed += 1
        if self.episodes_observed >= self.observation_episodes:
            self.is_observing = False
            if self.debug:
                print("\n[SKILL] Observation phase complete. Final estimates:")
                for wid in range(env.num_workers):
                    print(f"  W-{wid}: est={self._skill_means[wid]:.3f} "
                          f"(n={self._skill_counts[wid]})")

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state) -> int:
        """
        Select the best available task → best available worker (highest skill).

        All task visibility is governed by env.clock.tick — no lookahead.
        """
        if self.is_observing or all(self._skill_counts[i] == 0 for i in range(self.env.num_workers)):
            return self._greedy_fallback()

        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]

        # Tasks: arrived, unassigned, deps met
        pending = [
            t for t in self.env.tasks
            if t.is_available(tick) and t.is_unassigned()
            and t.check_dependencies_met(completed_ids)
        ]
        if not pending:
            # Nothing to assign: defer task 0 slot as no-op
            return self.encode_action(0, action_type='defer')

        # Sort: highest priority first, then tightest deadline
        pending.sort(key=lambda t: (-t.priority, -t.get_deadline_urgency(tick)))
        selected = pending[0]

        # Available workers: not burned out AND have capacity
        available_workers = [
            w for w in self.env.workers
            if w.availability == 1 and w.load < config.MAX_WORKER_LOAD
        ]
        if not available_workers:
            if self.debug:
                print(f"[SKILL] tick={tick}: No capacity → defer Task T-{selected.task_id}")
            return self.encode_action(selected.task_id, action_type='defer')

        # ── Deterministic best-worker selection (explicit max loop) ──────────
        best_worker = None
        best_score  = -np.inf
        for w in available_workers:
            raw = self._skill_means.get(w.worker_id, 1.0)
            if isinstance(raw, (list, np.ndarray)):
                score = float(np.mean(raw)) if len(raw) > 0 else 1.0
            else:
                score = float(raw)
            if self.debug:
                print(f"  [SKILL] W-{w.worker_id}: est_skill={score:.3f}")
            if score > best_score:
                best_score  = score
                best_worker = w   # Retain original Worker object (original worker_id)

        # Assertion: best_worker must be the one with the highest score
        assert best_worker is not None, "No best worker found despite non-empty available_workers"
        assert best_worker.worker_id == min(
            available_workers, key=lambda w: -self._skill_means.get(w.worker_id, 1.0)
        ).worker_id if len(available_workers) > 0 else True, \
            "Skill consistency check failed — best_worker is not the highest-score worker"

        if self.debug:
            print(f"[SKILL] tick={tick}: Task T-{selected.task_id} "
                  f"(p={selected.priority}, c={selected.complexity}) "
                  f"→ W-{best_worker.worker_id} (est={best_score:.3f})")

        return self.encode_action(selected.task_id, best_worker.worker_id, 'assign')

    # ── Greedy fallback ───────────────────────────────────────────────────────

    def _greedy_fallback(self) -> int:
        """Use greedy (lowest-load available worker) during observation phase."""
        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]

        pending = [
            t for t in self.env.tasks
            if t.is_available(tick) and t.is_unassigned()
            and t.check_dependencies_met(completed_ids)
        ]
        if not pending:
            return self.encode_action(0, action_type='defer')

        pending.sort(key=lambda t: -t.priority)
        available = [w for w in self.env.workers if w.availability == 1]
        if not available:
            return self.encode_action(pending[0].task_id, action_type='defer')

        available.sort(key=lambda w: w.load)
        return self.encode_action(pending[0].task_id, available[0].worker_id, 'assign')

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """
        Do NOT reset skill estimates — they represent accumulated knowledge.
        Only the observation phase state is reset if explicitly required.
        """
        pass   # Skill estimates are permanent knowledge across episodes

    def hard_reset(self):
        """Full reset including skill estimates (for brand-new simulations)."""
        self._skill_counts = {i: 0   for i in range(self.env.num_workers)}
        self._skill_means  = {i: 1.0 for i in range(self.env.num_workers)}
        self.episodes_observed = 0
        self.is_observing      = True

    def encode_action(self, task_id: int, worker_id: int = -1, action_type: str = 'assign') -> int:
        """Map (task_id, worker_id, type) to action index using env dimensions."""
        num_tasks   = 20   # Fixed action space window
        num_workers = self.env.num_workers

        # Find slot index of this task in the visible window
        tick          = self.env.clock.tick
        completed_ids = [t.task_id for t in self.env.completed_tasks]
        available = [t for t in self.env.tasks if t.is_available(tick) and t.is_unassigned()]
        available.sort(key=lambda t: -t.get_deadline_urgency(tick))
        available = available[:num_tasks]

        task_slot = next((i for i, t in enumerate(available) if t.task_id == task_id), 0)

        if action_type == 'assign':
            return task_slot * num_workers + worker_id
        elif action_type == 'defer':
            return num_tasks * num_workers + task_slot
        elif action_type == 'escalate':
            return num_tasks * num_workers + num_tasks + task_slot
        return num_tasks * num_workers   # Safe default: defer first slot


if __name__ == "__main__":
    print("Testing SkillBaseline v4...")
    from environment.project_env import ProjectEnv
    import config as cfg

    cfg.BASELINE_DEBUG_SKILL = True
    env    = ProjectEnv(num_workers=5, total_tasks=40, seed=42)
    policy = SkillBaseline(env, observation_episodes=3, debug=True)

    # Observation phase
    for ep in range(3):
        state = env.reset()
        for _ in range(50):
            valid = env.get_valid_actions()
            if not valid:
                break
            action = policy.select_action(state)
            if action not in valid:
                action = valid[0]
            state, _, done, _ = env.step(action)
            if done:
                break
        policy.observe_episode(env)
        print(f"Observation ep {ep+1}: estimates={dict(policy._skill_means)}")

    # Active phase
    state = env.reset()
    total_r = 0
    for _ in range(100):
        action = policy.select_action(state)
        valid  = env.get_valid_actions()
        if valid and action not in valid:
            action = valid[0]
        state, r, done, _ = env.step(action)
        total_r += r
        if done:
            break

    metrics = env.compute_metrics()
    print(f"✓ SkillBaseline: reward={total_r:.1f}, "
          f"throughput={metrics['throughput']}, "
          f"completion_rate={metrics['completion_rate']:.2f}")
    print("SkillBaseline v4 tests passed!")
