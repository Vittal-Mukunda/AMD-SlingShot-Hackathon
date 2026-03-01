/* ── Types: metrics.ts ────────────────────────────────────────────────────────
   Derived/computed metric types used in analytics views.
──────────────────────────────────────────────────────────────────────────── */

export interface PolicyMetrics {
    policy: string;
    throughput: number;
    completion_rate: number;
    lateness_rate: number;
    quality_score: number;
    overload_events: number;
    makespan_hours?: number;
}

export interface RadarDataPoint {
    axis: string;
    [policyName: string]: number | string;
}

// 5-axis radar: Throughput, CompletionRate, QualityScore, (inverted) LatenessRate, (inverted) OverloadEvents
export function buildRadarData(
    policies: Record<string, PolicyMetrics>
): RadarDataPoint[] {
    const axes = [
        { key: 'throughput', label: 'Throughput', invert: false, scale: 5 },
        { key: 'completion_rate', label: 'Completion', invert: false, scale: 1 },
        { key: 'quality_score', label: 'Quality', invert: false, scale: 1 },
        { key: 'lateness_rate', label: 'Timeliness', invert: true, scale: 1 },
        { key: 'overload_events', label: 'Load Balance', invert: true, scale: 20 },
    ] as const;

    return axes.map(axis => {
        const row: RadarDataPoint = { axis: axis.label };
        Object.entries(policies).forEach(([name, m]) => {
            const raw = m[axis.key as keyof PolicyMetrics] as number ?? 0;
            const normalized = Math.min(1, raw / axis.scale);
            row[name] = axis.invert ? Math.max(0, 1 - normalized) : normalized;
        });
        return row;
    });
}

// For head-to-head comparison strip
export interface HeadToHeadMetrics {
    dqn_throughput: number;          // live-updating
    baseline_throughputs: Record<string, number>;
    dqn_makespan_proj: number;       // projected remaining ticks
    baseline_makespans: Record<string, number>;
    dqn_lateness_rate: number;
    baseline_lateness_rates: Record<string, number>;
    dqn_worker_fatigue: number[];    // per-worker 0–3
    queue_depth_history: number[];   // rolling last 20 ticks
}

export interface SummaryCardData {
    label: string;
    value: string | number;
    unit?: string;
    delta?: string;
    deltaPositive?: boolean;
    highlight?: boolean;
}

// Color map for policies
export const POLICY_COLORS: Record<string, string> = {
    Greedy: '#60A5FA',    // blue
    Hybrid: '#A78BFA',    // violet
    Skill: '#34D399',    // emerald
    Random: '#94A3B8',    // slate
    FIFO: '#F97316',    // orange
    DQN: '#F59E0B',    // amber (primary)
};

// Urgency color map
export const URGENCY_COLORS: Record<number, string> = {
    0: '#64748B',   // low — slate
    1: '#3B82F6',   // medium — blue
    2: '#F97316',   // high — orange
    3: '#EF4444',   // critical — red
};

export const URGENCY_LABELS: Record<number, string> = {
    0: 'low',
    1: 'medium',
    2: 'high',
    3: 'critical',
};
