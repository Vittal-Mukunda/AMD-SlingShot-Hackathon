/* ── Types: simulation.ts ─────────────────────────────────────────────────────
   All event payloads matching the WebSocket contract exactly.
──────────────────────────────────────────────────────────────────────────── */

export type FatigueLevel = 'fresh' | 'tired' | 'exhausted' | 'burnout';
export type UrgencyLevel = 'low' | 'medium' | 'high' | 'critical';
// Phase type is defined in simulationStore.ts to avoid circular imports

// ── Per-entity states ────────────────────────────────────────────────────────

export interface WorkerState {
    id: string;             // "w0", "w1", ...
    name: string;
    fatigue: number;        // 0.0–3.0
    fatigue_level: FatigueLevel;
    availability: number;   // 1 = available, 0 = burnout/unavailable
    assigned_tasks: number[];
    skill_level: number;    // 0.5–1.5 (true_skill, hidden in RL env)
}

export interface QueueItem {
    task_id: string;
    priority: number;         // 0-3
    urgency_label: UrgencyLevel;
    duration_slots: number;
    required_skill: number;
    deadline_tick: number;
    slots_remaining: number;
}

export interface GanttBlock {
    task_id: string;
    worker_id: string;
    start_tick: number;
    end_tick: number;
    urgency: number;          // 0-3 maps to color
    policy: string;           // "Greedy" | "Skill" | "Hybrid" | "Random" | "DQN"
}

// ── Tick update ──────────────────────────────────────────────────────────────

export interface LastAssignment {
    task_id: string;
    worker_id: string;
    policy: string;
}

export interface TickUpdatePayload {
    tick: number;
    day: number;
    phase: number | string;
    worker_states: WorkerState[];
    queue_state: QueueItem[];
    last_assignment: LastAssignment | null;
    active_policy?: string;
}

// ── Task completed ────────────────────────────────────────────────────────────

export interface TaskCompletedPayload {
    task_id: string;
    worker_id: string;
    completion_tick: number;
    lateness: number;          // negative = early, positive = late
    quality: number;           // 0.0–1.0
}

// ── Phase transition ──────────────────────────────────────────────────────────

export interface BaselinePolicyMetrics {
    throughput: number;
    completion_rate: number;
    lateness_rate: number;
    quality_score: number;
    overload_events: number;
    gantt_blocks: GanttBlock[];
}

export interface PhaseTransitionPayload {
    new_phase: number | string;
    baseline_results_snapshot: Record<string, BaselinePolicyMetrics>;
}

// ── Daily summary ─────────────────────────────────────────────────────────────

export interface DailyMetrics {
    baseline: string;
    day: number;
    phase: number;
    throughput_per_day: number;
    completion_rate: number;
    lateness_rate: number;
    quality_score: number;
    load_balance: number;
    overload_events: number;
    decisions: number;
}

export interface DailySummaryPayload {
    day: number;
    phase: number;
    metrics_per_policy: Record<string, DailyMetrics>;
}

// ── Simulation complete ───────────────────────────────────────────────────────

export interface FinalMetrics {
    best_policy: string;
    dqn_vs_best_makespan_delta: number;
    total_tasks_completed: number;
    overall_lateness_rate: number;
    peak_overload_events: number;
    avg_quality_score: number;
    dqn_throughput: number;
    baseline_results: Record<string, BaselinePolicyMetrics>;
    phase1_daily: DailyMetrics[];
    phase2_daily: DailyMetrics[];
    overall: Record<string, number>;
}

export interface SimulationCompletePayload {
    final_metrics: FinalMetrics;
}

// ── README progress ───────────────────────────────────────────────────────────

export interface ReadmeProgressPayload {
    file_path: string;           // "__complete__" signals done
    status: 'generating' | 'done' | 'unchanged' | 'complete';
    preview_snippet: string;
}
