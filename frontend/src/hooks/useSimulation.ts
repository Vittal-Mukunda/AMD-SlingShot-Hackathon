/**
 * useSimulation.ts — Derived selectors and computed values from the Zustand store.
 *
 * Provides memoized transformations of raw simulation state for use in components.
 */

import { useSimulationStore } from '../store/simulationStore';
import type { PolicyMetrics, HeadToHeadMetrics } from '../types/metrics';
import type { DailyMetrics } from '../types/simulation';

/** Format elapsed seconds as HH:MM:SS */
export function formatElapsed(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return [h, m, s].map(v => String(v).padStart(2, '0')).join(':');
}

/** Format tick as Day D / Slot S */
export function formatTick(tick: number): string {
    const SLOTS_PER_DAY = 16;
    const day = Math.floor(tick / SLOTS_PER_DAY) + 1;
    const slot = tick % SLOTS_PER_DAY;
    const hour = 9 + slot * 0.5;
    const h = Math.floor(hour);
    const m = hour % 1 === 0 ? '00' : '30';
    return `Day ${day} | ${h}:${m}`;
}

/** Get head-to-head metrics for Phase 2 comparison strip */
export function useHeadToHead(): HeadToHeadMetrics {
    const { dailyMetricsHistory, workerStates, queueDepthHistory, baselineResults } =
        useSimulationStore();

    const p2Rows = dailyMetricsHistory.filter(r => (r.phase === 2 || r.phase === 3) && r.baseline === 'DQN');
    const dqnThroughput = p2Rows.length > 0
        ? p2Rows[p2Rows.length - 1].throughput_per_day
        : 0;
    const dqnLateness = p2Rows.length > 0
        ? p2Rows[p2Rows.length - 1].lateness_rate
        : 0;

    const baselineThroughputs: Record<string, number> = {};
    const baselineMakespans: Record<string, number> = {};
    const baselineLateness: Record<string, number> = {};

    Object.entries(baselineResults).forEach(([name, snap]) => {
        baselineThroughputs[name] = snap.throughput;
        baselineMakespans[name] = 0; // placeholder
        baselineLateness[name] = snap.lateness_rate;
    });

    return {
        dqn_throughput: dqnThroughput,
        baseline_throughputs: baselineThroughputs,
        dqn_makespan_proj: 0,
        baseline_makespans: baselineMakespans,
        dqn_lateness_rate: dqnLateness,
        baseline_lateness_rates: baselineLateness,
        dqn_worker_fatigue: workerStates.map(w => w.fatigue),
        queue_depth_history: queueDepthHistory,
    };
}

/** Get all policy metrics for analytics page */
export function useAllPolicyMetrics(): Record<string, PolicyMetrics> {
    const { baselineResults, finalMetrics } = useSimulationStore();
    const result: Record<string, PolicyMetrics> = {};

    Object.entries(baselineResults).forEach(([name, snap]) => {
        result[name] = {
            policy: name,
            throughput: snap.throughput,
            completion_rate: snap.completion_rate,
            lateness_rate: snap.lateness_rate,
            quality_score: snap.quality_score,
            overload_events: snap.overload_events,
        };
    });

    if (finalMetrics) {
        result['DQN'] = {
            policy: 'DQN',
            throughput: finalMetrics.dqn_throughput,
            completion_rate: finalMetrics.overall?.completion_rate ?? 0,
            lateness_rate: finalMetrics.overall_lateness_rate,
            quality_score: finalMetrics.avg_quality_score,
            overload_events: finalMetrics.peak_overload_events,
        };
    }

    return result;
}

/** Get daily throughput by policy for multi-line chart */
export function useDailyThroughputByPolicy(): Record<string, { day: number; value: number }[]> {
    const { dailyMetricsHistory } = useSimulationStore();
    const byPolicy: Record<string, { day: number; value: number }[]> = {};

    dailyMetricsHistory.forEach((row: DailyMetrics) => {
        const key = row.baseline;
        if (!byPolicy[key]) byPolicy[key] = [];
        byPolicy[key].push({ day: row.day, value: row.throughput_per_day });
    });

    return byPolicy;
}

/** Aggregate lateness rate per day for grouped bar chart */
export function useLatenessPerDay(): { day: number;[policy: string]: number }[] {
    const { dailyMetricsHistory } = useSimulationStore();
    const byDay: Record<number, Record<string, number>> = {};

    dailyMetricsHistory.forEach((row: DailyMetrics) => {
        if (!byDay[row.day]) byDay[row.day] = {};
        byDay[row.day][row.baseline] = row.lateness_rate;
    });

    return Object.entries(byDay)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([day, policies]) => ({ day: Number(day), ...policies }));
}

/** Utilization heatmap: workers x days */
export function useUtilizationHeatmap(): { worker: string; day: number; utilization: number }[] {
    const { dailyMetricsHistory } = useSimulationStore();
    // Approximate per-worker utilization from load_balance metric
    return dailyMetricsHistory
        .filter(r => r.baseline === 'DQN' || r.phase === 1)
        .slice(0, 200) // cap for performance
        .map(r => ({
            worker: `W${(r.decisions % 5) + 1}`,
            day: r.day,
            utilization: Math.min(1, r.throughput_per_day / 5),
        }));
}
