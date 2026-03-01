/**
 * simulationStore.ts — Zustand global state v2
 *
 * Phase protocol:
 *   0 = not started
 *   1 = Phase 1 baseline running
 *   "training" = DQN training in background (waiting screen)
 *   3 = Phase 2 DQN scheduling live
 *
 * New events handled:
 *   training_progress  → { percent, steps }
 *   phase2_ready       → { baseline_results_snapshot }
 *   simulation_error   → { message, traceback }
 */

import { create } from 'zustand';
import type {
    WorkerState,
    QueueItem,
    GanttBlock,
    DailyMetrics,
    FinalMetrics,
    BaselinePolicyMetrics,
    ReadmeProgressPayload,
} from '../types/simulation';
import type { SimConfig } from '../types/config';
import { DEFAULT_SIM_CONFIG } from '../types/config';

export type Phase = 0 | 1 | 'training' | 3;

// ── Store Shape ───────────────────────────────────────────────────────────────

export interface SimulationState {
    simConfig: SimConfig;
    setSimConfig: (cfg: SimConfig) => void;

    // Runtime
    phase: Phase;
    currentTick: number;
    currentDay: number;
    elapsedSeconds: number;
    startTime: number | null;
    isRunning: boolean;
    isPaused: boolean;
    activePolicy: string;

    // Live states
    workerStates: WorkerState[];
    queueState: QueueItem[];

    // Gantt blocks per policy key
    ganttBlocks: Record<string, GanttBlock[]>;

    // Phase 1 baseline snapshot (cached for analytics)
    baselineResults: Record<string, BaselinePolicyMetrics>;

    // Daily metrics history
    dailyMetricsHistory: DailyMetrics[];

    // Queue depth sparkline
    queueDepthHistory: number[];

    // DQN training progress (0–100)
    trainingPercent: number;

    // Simulation error
    simulationError: string | null;

    // Final metrics (set on simulation_complete)
    finalMetrics: FinalMetrics | null;

    // README generator
    readmeProgress: ReadmeProgressPayload[];
    readmeComplete: boolean;

    // UI state
    selectedBaseline: string;

    // ── Actions ──────────────────────────────────────────────────────────────

    startSimulation: (cfg: SimConfig) => void;

    applyTickUpdate: (payload: {
        tick: number;
        day: number;
        phase: number | string;
        worker_states: WorkerState[];
        queue_state: QueueItem[];
        last_assignment: { task_id: string; worker_id: string; policy: string } | null;
        active_policy?: string;
    }) => void;

    applyPhaseTransition: (payload: {
        new_phase: Phase | string;
        baseline_results_snapshot: Record<string, BaselinePolicyMetrics>;
    }) => void;

    applyPhase2Ready: (payload: {
        baseline_results_snapshot: Record<string, BaselinePolicyMetrics>;
    }) => void;

    applyDailySummary: (payload: {
        day: number;
        phase: number;
        metrics_per_policy: Record<string, DailyMetrics>;
    }) => void;

    applySimulationComplete: (payload: { final_metrics: FinalMetrics }) => void;
    applyReadmeProgress: (payload: ReadmeProgressPayload) => void;
    applyTrainingProgress: (payload: { percent: number; steps: number }) => void;
    applySimulationError: (payload: { message: string; traceback: string }) => void;

    addGanttBlock: (block: GanttBlock) => void;
    setSelectedBaseline: (name: string) => void;
    setPaused: (v: boolean) => void;
    tickElapsed: () => void;
    reset: () => void;
}

// ── Initial state ─────────────────────────────────────────────────────────────

const INITIAL = {
    simConfig: DEFAULT_SIM_CONFIG,
    phase: 0 as Phase,
    currentTick: 0,
    currentDay: 0,
    elapsedSeconds: 0,
    startTime: null as number | null,
    isRunning: false,
    isPaused: false,
    activePolicy: 'Greedy',
    workerStates: [] as WorkerState[],
    queueState: [] as QueueItem[],
    ganttBlocks: {} as Record<string, GanttBlock[]>,
    baselineResults: {} as Record<string, BaselinePolicyMetrics>,
    dailyMetricsHistory: [] as DailyMetrics[],
    queueDepthHistory: [] as number[],
    trainingPercent: 0,
    simulationError: null as string | null,
    finalMetrics: null as FinalMetrics | null,
    readmeProgress: [] as ReadmeProgressPayload[],
    readmeComplete: false,
    selectedBaseline: 'Greedy',
};

// ── Store ─────────────────────────────────────────────────────────────────────

export const useSimulationStore = create<SimulationState>((set, get) => ({
    ...INITIAL,

    setSimConfig: (cfg) => set({ simConfig: cfg }),

    startSimulation: (cfg) => set({
        ...INITIAL,
        simConfig: cfg,
        isRunning: true,
        phase: 0,
        startTime: Date.now(),
        simulationError: null,
    }),

    applyTickUpdate: (payload) =>
        set((state) => {
            const queueState = payload.queue_state ?? [];
            const newDepth = queueState.length;
            const depthHistory = [...state.queueDepthHistory, newDepth].slice(-60);

            // Remap backend phase (1, 2, 3) to store phase
            // Bug 4: When in 'training', never let tick_update override to phase 1 (prevents premature dismissal)
            // Bug 8: Handle reconnect — backend phase 2 = frontend 'training'
            let newPhase: Phase = state.phase;
            if (payload.phase === 1) {
                newPhase = state.phase === 'training' ? 'training' : 1;
            } else if (payload.phase === 2) {
                newPhase = 'training';
            } else if (payload.phase === 3) {
                newPhase = 3;
            }

            return {
                currentTick: payload.tick,
                currentDay: payload.day,
                phase: newPhase,
                workerStates: payload.worker_states ?? state.workerStates,
                queueState,
                activePolicy: payload.active_policy ?? state.activePolicy,
                queueDepthHistory: depthHistory,
            };
        }),

    applyPhaseTransition: (payload) =>
        set((state) => {
            // new_phase="training" or 2 means we enter the spinner screen (Bug 8: handle numeric phase)
            const raw = payload.new_phase;
            const newPhase: Phase =
                raw === 'training' || raw === 2 ? 'training' : 1;

            const bSnap = payload.baseline_results_snapshot ?? {};
            const mergedGantt: Record<string, GanttBlock[]> = { ...state.ganttBlocks };
            Object.entries(bSnap).forEach(([name, snap]) => {
                mergedGantt[name] = snap.gantt_blocks ?? [];
            });

            return {
                phase: newPhase,
                baselineResults: bSnap,
                ganttBlocks: mergedGantt,
                selectedBaseline: Object.keys(bSnap)[0] ?? 'Greedy',
            };
        }),

    applyPhase2Ready: (payload) =>
        set((state) => {
            const bSnap = payload?.baseline_results_snapshot ?? {};
            const mergedGantt: Record<string, GanttBlock[]> = { ...state.ganttBlocks };
            Object.entries(bSnap).forEach(([name, snap]) => {
                mergedGantt[name] = snap?.gantt_blocks ?? [];
            });
            return {
                phase: 3,
                trainingPercent: 100,
                baselineResults: bSnap,
                ganttBlocks: mergedGantt,
            };
        }),

    applyDailySummary: (payload) =>
        set((state) => {
            const newEntries = Object.values(payload.metrics_per_policy);
            return { dailyMetricsHistory: [...state.dailyMetricsHistory, ...newEntries] };
        }),

    applySimulationComplete: (payload) =>
        set({
            finalMetrics: payload.final_metrics,
            isRunning: false,
            phase: 3,
        }),

    applyReadmeProgress: (payload) =>
        set((state) => ({
            readmeProgress: [...state.readmeProgress, payload],
            readmeComplete: payload.status === 'complete',
        })),

    applyTrainingProgress: (payload) =>
        set({ trainingPercent: payload.percent }),

    applySimulationError: (payload) =>
        set({ simulationError: payload.message, isRunning: false }),

    addGanttBlock: (block) =>
        set((state) => {
            const existing = state.ganttBlocks[block.policy] ?? [];
            return {
                ganttBlocks: {
                    ...state.ganttBlocks,
                    [block.policy]: [...existing, block],
                },
            };
        }),

    setSelectedBaseline: (name) => set({ selectedBaseline: name }),
    setPaused: (v) => set({ isPaused: v }),

    tickElapsed: () =>
        set((state) => ({
            elapsedSeconds: state.startTime
                ? Math.floor((Date.now() - state.startTime) / 1000)
                : 0,
        })),

    reset: () => set(INITIAL),
}));
