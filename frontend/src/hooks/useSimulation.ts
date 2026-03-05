/**
 * useSimulation.ts — Derived selectors and computed values from the Zustand store.
 *
 * IMPORTANT: Every function that returns an object or array MUST use useShallow()
 * or Zustand's shallow equality so that object identity is stable across renders.
 * Returning a plain `{ ... }` literal from a custom hook causes a new reference
 * every render, which breaks Recharts memoization and triggers the infinite loop.
 */

import { useMemo } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useShallow } from 'zustand/react/shallow';
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

/**
 * Get head-to-head metrics for Phase 2 comparison strip.
 *
 * FIX: Use useShallow to pick individual primitive/array fields from the store
 * and useMemo to build the HeadToHeadMetrics object. This means the object
 * reference only changes when the underlying store values change — NOT every render.
 */
export function useHeadToHead(): HeadToHeadMetrics {
    const {
        dqnThroughput,
        dqnLateness,
        workerStates,
        queueDepthHistory,
        baselineResults,
        finalMetrics,
    } = useSimulationStore(
        useShallow(s => ({
            dqnThroughput: s.finalMetrics?.dqn_throughput ??
                (s.dailyMetricsHistory.filter(r => (r.phase === 2 || r.phase === 3) && r.baseline === 'DQN').slice(-1)[0]?.throughput_per_day ?? 0),
            dqnLateness: s.finalMetrics?.overall_lateness_rate ??
                (s.dailyMetricsHistory.filter(r => (r.phase === 2 || r.phase === 3) && r.baseline === 'DQN').slice(-1)[0]?.lateness_rate ?? 0),
            workerStates: s.workerStates,
            queueDepthHistory: s.queueDepthHistory,
            baselineResults: s.baselineResults,
            finalMetrics: s.finalMetrics,
        }))
    );

    return useMemo<HeadToHeadMetrics>(() => {
        const baseline_throughputs: Record<string, number> = {};
        const baseline_makespans: Record<string, number> = {};
        const baseline_lateness: Record<string, number> = {};

        Object.entries(baselineResults).forEach(([name, snap]) => {
            baseline_throughputs[name] = snap.throughput;
            baseline_makespans[name] = 0;
            baseline_lateness[name] = snap.lateness_rate;
        });

        // Derive fatigue array HERE inside useMemo (not in the selector)
        // so we don't create a new array ref that defeats useShallow.
        const workerFatigue = workerStates.map(w => w.fatigue);

        return {
            dqn_throughput: dqnThroughput,
            baseline_throughputs,
            dqn_makespan_proj: 0,
            baseline_makespans,
            dqn_lateness_rate: dqnLateness,
            baseline_lateness_rates: baseline_lateness,
            dqn_worker_fatigue: workerFatigue,
            queue_depth_history: queueDepthHistory,
        };
    }, [dqnThroughput, dqnLateness, workerStates, queueDepthHistory, baselineResults, finalMetrics]);
}

/** Get all policy metrics for analytics page */
export function useAllPolicyMetrics(): Record<string, PolicyMetrics> {
    const { baselineResults, finalMetrics } = useSimulationStore(
        useShallow(s => ({ baselineResults: s.baselineResults, finalMetrics: s.finalMetrics }))
    );

    return useMemo(() => {
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
                completion_rate: (finalMetrics as any).overall?.completion_rate ?? 0,
                lateness_rate: finalMetrics.overall_lateness_rate,
                quality_score: finalMetrics.avg_quality_score,
                overload_events: finalMetrics.peak_overload_events,
            };
        }
        return result;
    }, [baselineResults, finalMetrics]);
}

/** Get daily throughput by policy for multi-line chart */
export function useDailyThroughputByPolicy(): Record<string, { day: number; value: number }[]> {
    const dailyMetricsHistory = useSimulationStore(s => s.dailyMetricsHistory);

    return useMemo(() => {
        const byPolicy: Record<string, { day: number; value: number }[]> = {};
        dailyMetricsHistory.forEach((row: DailyMetrics) => {
            const key = row.baseline;
            if (!byPolicy[key]) byPolicy[key] = [];
            byPolicy[key].push({ day: row.day, value: row.throughput_per_day });
        });
        return byPolicy;
    }, [dailyMetricsHistory]);
}

/** Aggregate lateness rate per day for grouped bar chart */
export function useLatenessPerDay(): { day: number;[policy: string]: number }[] {
    const dailyMetricsHistory = useSimulationStore(s => s.dailyMetricsHistory);

    return useMemo(() => {
        const byDay: Record<number, Record<string, number>> = {};
        dailyMetricsHistory.forEach((row: DailyMetrics) => {
            if (!byDay[row.day]) byDay[row.day] = {};
            byDay[row.day][row.baseline] = row.lateness_rate;
        });
        return Object.entries(byDay)
            .sort(([a], [b]) => Number(a) - Number(b))
            .map(([day, policies]) => ({ day: Number(day), ...policies }));
    }, [dailyMetricsHistory]);
}
