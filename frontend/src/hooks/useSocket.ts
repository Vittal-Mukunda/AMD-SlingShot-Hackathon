/**
 * useSocket.ts — Socket.IO client hook (v2)
 *
 * Handles all events from backend:
 *   tick_update, gantt_block, daily_summary, task_completed
 *   phase_transition, phase2_ready, training_progress
 *   simulation_complete, simulation_error
 *   readme_progress
 */

import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { useSimulationStore } from '../store/simulationStore';
import type {
    TaskCompletedPayload,
    ReadmeProgressPayload,
    GanttBlock,
} from '../types/simulation';

// Use same-origin when dev server proxies /socket.io to backend (avoids CORS/connection drops)
const BACKEND_URL = (import.meta as any).env?.DEV ? '' : 'http://localhost:8000';

// Singleton socket — survives React 18 Strict Mode double-mounts
const socket = io(BACKEND_URL, {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 15,
    reconnectionDelay: 1500,
    autoConnect: true,
});

// Setup global listeners ONCE outside of any React lifecycle
const store = useSimulationStore.getState;

socket.on('tick_update', (p) => store().applyTickUpdate(p));
socket.on('gantt_block', (block: GanttBlock) => store().addGanttBlock(block));
socket.on('task_completed', (_p: TaskCompletedPayload) => {
    // Future: store completed task list; for now just log
});
socket.on('phase_transition', (p) => store().applyPhaseTransition(p));
socket.on('phase2_ready', (p) => store().applyPhase2Ready(p));
socket.on('training_progress', (p: { percent: number; steps: number }) => store().applyTrainingProgress(p));
socket.on('daily_summary', (p) => store().applyDailySummary(p));
socket.on('simulation_complete', (p) => store().applySimulationComplete(p));
socket.on('simulation_error', (p: { message: string; traceback: string }) => store().applySimulationError(p));
socket.on('readme_progress', (p: ReadmeProgressPayload) => store().applyReadmeProgress(p));

socket.on('connect', () => console.log('[Socket] Connected:', socket.id));
socket.on('disconnect', (reason) => console.warn('[Socket] Disconnected:', reason));
socket.on('connect_error', (err) => console.error('[Socket] Connection error:', err.message));

export function useSocket() {
    const socketRef = useRef<Socket>(socket);

    // ── Emit helpers ──────────────────────────────────────────────────────────

    const emit = useCallback(<T>(event: string, data?: T) => {
        socketRef.current?.emit(event, data);
    }, []);

    const pauseSimulation = useCallback(() => {
        emit('pause_simulation');
        useSimulationStore.getState().setPaused(true);
    }, [emit]);

    const resumeSimulation = useCallback(() => {
        emit('resume_simulation');
        useSimulationStore.getState().setPaused(false);
    }, [emit]);

    const injectTask = useCallback((task: {
        task_id: string;
        duration: number;
        urgency: number;
        required_skill: number;
        arrival_tick: number;
    }) => {
        emit('inject_task', task);
    }, [emit]);

    const generateReadme = useCallback(() => {
        emit('generate_readme');
    }, [emit]);

    return {
        socket: socketRef.current,
        emit,
        pauseSimulation,
        resumeSimulation,
        injectTask,
        generateReadme,
    };
}
