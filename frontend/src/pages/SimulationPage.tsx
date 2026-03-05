/**
 * SimulationPage.tsx — Live Operations Center (Redesigned)
 * Premium dark ops aesthetic: glowing phase badge, live dot, cinematic training screen
 */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useShallow } from 'zustand/react/shallow';
import { useSimulationStore } from '../store/simulationStore';
import { formatElapsed, useHeadToHead } from '../hooks/useSimulation';
import { useSocket } from '../hooks/useSocket';
import GanttChart from '../components/simulation/GanttChart';
import WorkerSidebar from '../components/simulation/WorkerSidebar';
import TaskQueue from '../components/simulation/TaskQueue';
import ComparisonStrip from '../components/simulation/ComparisonStrip';
import PriorityInjectionPanel from '../components/config/PriorityInjectionPanel';
import type { InjectedTask } from '../types/config';

const BASELINE_NAMES = ['Greedy', 'Skill', 'FIFO', 'Hybrid', 'Random'];

// ── Phase badge with glow effect ─────────────────────────────────────────────
function PhaseBadge({ phase }: { phase: 0 | 1 | 'training' | 3 }) {
    if (phase === 1) return (
        <span className="badge badge-phase1" style={{ fontSize: 11 }}>
            <span className="live-dot" />
            PHASE 1 — BASELINE ALLOCATION
        </span>
    );
    if (phase === 'training') return (
        <span className="badge badge-amber" style={{ fontSize: 11 }}>
            <span className="live-dot live-dot-amber" />
            DQN TRAINING
        </span>
    );
    if (phase === 3) return (
        <span className="badge badge-amber" style={{ fontSize: 11 }}>
            <span className="live-dot live-dot-amber" />
            PHASE 2 — DQN SCHEDULING
        </span>
    );
    return <span className="badge badge-slate" style={{ fontSize: 11 }}>INITIALIZING</span>;
}

// ── DQN observing indicator ───────────────────────────────────────────────────
function DQNPassiveIndicator() {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '4px 12px',
            background: 'rgba(37,99,235,0.08)',
            border: '1px solid rgba(37,99,235,0.25)',
            borderRadius: 100,
        }}>
            <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'var(--color-electric)',
                boxShadow: '0 0 6px var(--color-electric)',
                display: 'inline-block',
                animation: 'pulse-amber 2s ease-in-out infinite',
            }} />
            <span className="font-mono" style={{ fontSize: 10, color: '#60A5FA', letterSpacing: '0.08em' }}>
                DQN OBSERVING
            </span>
        </div>
    );
}

// ── Training waiting screen — premium AI loading ──────────────────────────────
function TrainingScreen({ percent }: { percent: number }) {
    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: '#0A0E1A',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 40,
            animation: 'page-enter 0.5s ease forwards',
        }}>
            {/* Ambient rotating glow */}
            <div className="ambient-glow" style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }} />

            {/* Card */}
            <div style={{
                position: 'relative', zIndex: 1,
                background: 'var(--color-surface)',
                border: '1px solid rgba(245,158,11,0.2)',
                borderRadius: 20,
                padding: '48px 56px',
                textAlign: 'center',
                maxWidth: 520,
                boxShadow: '0 0 60px rgba(245,158,11,0.1), 0 32px 80px rgba(0,0,0,0.6)',
            }}>
                {/* Icon */}
                <div style={{ fontSize: 48, marginBottom: 24 }}>🧠</div>

                {/* Label */}
                <div className="font-mono" style={{
                    fontSize: 'var(--text-label)', color: 'var(--color-amber)',
                    letterSpacing: '0.20em', textTransform: 'uppercase', marginBottom: 16,
                }}>
                    PHASE 2 — DQN TRAINING
                </div>

                {/* Headline with typing cursor */}
                <h2 className="typing-cursor" style={{
                    fontFamily: 'var(--font-ui)', fontSize: 22, fontWeight: 700,
                    color: 'var(--color-text)', lineHeight: 1.4, marginBottom: 12,
                }}>
                    DQN Agent Analyzing Patterns
                </h2>

                {/* Status message */}
                <p style={{
                    color: 'var(--color-slate-text)', fontSize: 'var(--text-body)',
                    lineHeight: 1.7, marginBottom: 32,
                }}>
                    {percent < 100
                        ? 'Replaying workforce transitions and computing Bellman updates…'
                        : 'Training complete — transitioning to live scheduling.'}
                </p>

                {/* Amber progress bar — no raw percentage label */}
                <div style={{
                    width: '100%', height: 4,
                    background: 'rgba(245,158,11,0.12)',
                    borderRadius: 2, overflow: 'hidden', marginBottom: 16,
                }}>
                    <div style={{
                        height: '100%',
                        width: `${percent}%`,
                        background: 'linear-gradient(90deg, var(--color-amber), #FBBF24)',
                        borderRadius: 2,
                        transition: 'width 0.6s ease',
                        boxShadow: '0 0 12px rgba(245,158,11,0.5)',
                    }} />
                </div>

                {/* Subtle status text */}
                <div className="font-mono" style={{
                    fontSize: 'var(--text-label)', color: 'var(--color-slate-dim)',
                    letterSpacing: '0.06em',
                }}>
                    {percent < 100 ? 'Agent learning from Phase 1 observations' : 'Preparing live scheduler…'}
                </div>
            </div>

            {/* Footer note */}
            <div className="font-mono" style={{
                position: 'relative', zIndex: 1,
                fontSize: 'var(--text-label)', color: 'var(--color-slate-dim)',
                letterSpacing: '0.1em', textTransform: 'uppercase',
            }}>
                Phase 1 complete — All baseline policies benchmarked
            </div>
        </div>
    );
}

// ── Error banner ──────────────────────────────────────────────────────────────
function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
    return (
        <div style={{
            position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
            zIndex: 300,
            background: 'rgba(239,68,68,0.12)',
            border: '1px solid rgba(239,68,68,0.4)',
            color: 'var(--color-danger)',
            padding: '14px 24px',
            borderRadius: 'var(--radius-lg)',
            fontFamily: 'var(--font-mono)', fontSize: 12, maxWidth: 640,
            boxShadow: '0 8px 40px rgba(239,68,68,0.3)',
            display: 'flex', alignItems: 'center', gap: 16,
            backdropFilter: 'blur(8px)',
        }}>
            <span>⚠ Simulation Error: {message}</span>
            {onDismiss && (
                <button onClick={onDismiss} style={{
                    background: 'rgba(239,68,68,0.15)',
                    border: '1px solid rgba(239,68,68,0.4)',
                    color: 'var(--color-danger)', borderRadius: 'var(--radius-md)',
                    padding: '4px 12px', cursor: 'pointer',
                    fontFamily: 'var(--font-mono)', fontSize: 11, flexShrink: 0,
                }}>
                    ← Return to Config
                </button>
            )}
        </div>
    );
}

// ── Main SimulationPage ───────────────────────────────────────────────────────
export default function SimulationPage() {
    const navigate = useNavigate();
    const { pauseSimulation, resumeSimulation, injectTask } = useSocket();

    const {
        phase, currentTick, currentDay, elapsedSeconds,
        workerStates, queueState, ganttBlocks, finalMetrics,
        isPaused, selectedBaseline, setSelectedBaseline, simConfig,
        trainingPercent, simulationError, activePolicy,
    } = useSimulationStore(
        useShallow(s => ({
            phase: s.phase, currentTick: s.currentTick,
            currentDay: s.currentDay, elapsedSeconds: s.elapsedSeconds,
            workerStates: s.workerStates, queueState: s.queueState,
            ganttBlocks: s.ganttBlocks, finalMetrics: s.finalMetrics,
            isPaused: s.isPaused, selectedBaseline: s.selectedBaseline,
            setSelectedBaseline: s.setSelectedBaseline, simConfig: s.simConfig,
            trainingPercent: s.trainingPercent, simulationError: s.simulationError,
            activePolicy: s.activePolicy,
        }))
    );

    const [showInjectPanel, setShowInjectPanel] = useState(false);
    const [injectedTasks, setInjectedTasks] = useState<InjectedTask[]>([]);

    useEffect(() => {
        if (finalMetrics) {
            const t = setTimeout(() => navigate('/analytics'), 3500);
            return () => clearTimeout(t);
        }
    }, [finalMetrics, navigate]);

    const hhMetrics = useHeadToHead();

    const handleInjectTask = (task: InjectedTask) => {
        injectTask({
            task_id: task.task_id, duration: task.duration,
            urgency: task.urgency, required_skill: task.required_skill,
            arrival_tick: task.arrival_tick,
        });
    };

    const currentGanttBlocks =
        phase === 3 ? (ganttBlocks['DQN'] ?? []) : (ganttBlocks[selectedBaseline] ?? []);

    const isActiveBaseline = (name: string) =>
        phase === 1 && activePolicy.includes(name);

    return (
        <div className="page-enter" style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)' }}>

            {/* ── Training waiting screen ── */}
            {phase === 'training' && <TrainingScreen percent={trainingPercent} />}

            {/* ── Error banner ── */}
            {simulationError && (
                <ErrorBanner message={simulationError} onDismiss={() => {
                    useSimulationStore.getState().reset();
                    navigate('/');
                }} />
            )}

            {/* ── TOP NAVBAR ── */}
            <nav style={{
                height: 56, flexShrink: 0,
                background: 'var(--color-surface)',
                borderBottom: '1px solid var(--color-border)',
                display: 'flex', alignItems: 'center',
                padding: '0 20px', gap: 16,
                boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            }}>
                {/* Logo */}
                <span className="font-mono" style={{
                    color: 'var(--color-electric)', fontWeight: 700,
                    fontSize: 13, letterSpacing: '0.1em', flexShrink: 0,
                }}>
                    DQN SCHEDULER
                </span>
                <div style={{ width: 1, height: 24, background: 'var(--color-border-bright)', flexShrink: 0 }} />

                {/* Phase badge */}
                <PhaseBadge phase={phase} />

                {/* Day / Tick counters */}
                <div className="num font-mono" style={{ color: 'var(--color-text-dim)', fontSize: 13 }}>
                    <span style={{ color: 'var(--color-slate-dim)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>DAY </span>
                    <span style={{ color: 'var(--color-text)', fontWeight: 700, marginRight: 16 }}>{currentDay}</span>
                    <span style={{ color: 'var(--color-slate-dim)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>TICK </span>
                    <span style={{ color: 'var(--color-cyan)', fontWeight: 700 }}>{currentTick.toLocaleString()}</span>
                </div>

                {phase === 1 && <DQNPassiveIndicator />}

                <div style={{ flex: 1 }} />

                {/* Active policy */}
                {(phase === 1 || phase === 3) && (
                    <div className="font-mono" style={{
                        fontSize: 11, color: 'var(--color-slate-text)',
                        background: 'rgba(255,255,255,0.04)',
                        border: '1px solid var(--color-border)',
                        padding: '3px 10px', borderRadius: 100,
                    }}>
                        {activePolicy}
                    </div>
                )}

                {/* Elapsed */}
                <div className="num font-mono" style={{ color: 'var(--color-slate-dim)', fontSize: 12 }}>
                    ⏱ {formatElapsed(elapsedSeconds)}
                </div>

                {/* Inject task */}
                <button className="btn-ghost" style={{ fontSize: 11, padding: '5px 12px' }}
                    onClick={() => setShowInjectPanel(p => !p)}>
                    + Inject Task
                </button>

                {/* Pause/Resume */}
                <button
                    style={{
                        fontSize: 11, padding: '5px 14px',
                        background: isPaused ? 'var(--color-electric)' : 'transparent',
                        color: isPaused ? '#fff' : 'var(--color-slate-text)',
                        border: `1px solid ${isPaused ? 'var(--color-electric)' : 'var(--color-border-bright)'}`,
                        borderRadius: 'var(--radius-md)', cursor: 'pointer',
                        fontFamily: 'var(--font-ui)', fontWeight: 500,
                        transition: 'all 0.15s ease', minWidth: 80,
                        boxShadow: isPaused ? '0 0 12px rgba(37,99,235,0.4)' : 'none',
                    }}
                    onClick={() => isPaused ? resumeSimulation() : pauseSimulation()}
                >
                    {isPaused ? '▶ Resume' : '⏸ Pause'}
                </button>

                {/* Analytics link */}
                {finalMetrics && (
                    <button className="btn-amber" style={{ fontSize: 11, padding: '5px 14px' }}
                        onClick={() => navigate('/analytics')}>
                        Analytics →
                    </button>
                )}
            </nav>

            {/* ── Inject Panel ── */}
            {showInjectPanel && (
                <div style={{
                    background: 'var(--color-surface)',
                    borderBottom: '1px solid var(--color-border)',
                    padding: '16px 20px',
                }}>
                    <PriorityInjectionPanel
                        pendingTasks={injectedTasks}
                        onAdd={t => setInjectedTasks(p => [...p, t])}
                        onRemove={id => setInjectedTasks(p => p.filter(t => t.task_id !== id))}
                        onEmit={handleInjectTask}
                        currentTick={currentTick}
                    />
                </div>
            )}

            {/* ── Body ── */}
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

                {/* ── Left sidebar: Workers ── */}
                <WorkerSidebar workers={workerStates} simConfig={simConfig} currentTick={currentTick} />

                {/* ── Main: Gantt + Queue ── */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

                    {/* Phase 1: Baseline tabs */}
                    {phase === 1 && (
                        <div style={{
                            display: 'flex', gap: 0,
                            background: 'var(--color-surface)',
                            borderBottom: '1px solid var(--color-border)',
                            padding: '0 16px', flexShrink: 0,
                        }}>
                            {BASELINE_NAMES.map(name => {
                                const hasData = (ganttBlocks[name]?.length ?? 0) > 0;
                                const isActive = isActiveBaseline(name);
                                const isSelected = selectedBaseline === name;
                                return (
                                    <button key={name} onClick={() => setSelectedBaseline(name)} style={{
                                        padding: '10px 18px',
                                        background: 'none', border: 'none',
                                        borderBottom: `2px solid ${isSelected ? 'var(--color-electric)' : 'transparent'}`,
                                        color: isSelected ? 'var(--color-electric)' : hasData ? 'var(--color-text-dim)' : 'var(--color-slate-dim)',
                                        fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600,
                                        letterSpacing: '0.08em', textTransform: 'uppercase',
                                        cursor: 'pointer',
                                        transition: 'all 0.15s ease',
                                        display: 'flex', alignItems: 'center', gap: 6,
                                    }}>
                                        {name}
                                        {isActive && (
                                            <span style={{
                                                width: 5, height: 5, borderRadius: '50%',
                                                background: 'var(--color-electric)',
                                                boxShadow: '0 0 6px var(--color-electric)',
                                                animation: 'pulse-amber 1.2s ease-in-out infinite',
                                                display: 'inline-block',
                                            }} />
                                        )}
                                        {!isActive && hasData && (
                                            <span style={{
                                                width: 5, height: 5, borderRadius: '50%',
                                                background: 'var(--color-success)',
                                                display: 'inline-block',
                                            }} />
                                        )}
                                    </button>
                                );
                            })}
                        </div>
                    )}

                    {/* Phase 3: DQN header bar */}
                    {phase === 3 && (
                        <div style={{
                            padding: '7px 16px',
                            background: 'rgba(245,158,11,0.04)',
                            borderBottom: '1px solid rgba(245,158,11,0.15)',
                            flexShrink: 0, display: 'flex', alignItems: 'center', gap: 10,
                        }}>
                            <span className="live-dot live-dot-amber" />
                            <span className="font-mono" style={{
                                fontSize: 11, color: 'var(--color-amber)', letterSpacing: '0.08em',
                            }}>
                                DQN AGENT — ONLINE SCHEDULING ACTIVE
                            </span>
                        </div>
                    )}

                    {/* Phase 0: Initializing spinner */}
                    {phase === 0 && (
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    width: 48, height: 48, borderRadius: '50%',
                                    border: '3px solid rgba(37,99,235,0.12)',
                                    borderTopColor: 'var(--color-electric)',
                                    animation: 'spin 1s linear infinite',
                                    margin: '0 auto 16px',
                                    boxShadow: '0 0 20px rgba(37,99,235,0.3)',
                                }} />
                                <div className="font-mono" style={{
                                    fontSize: 12, color: 'var(--color-electric)',
                                    letterSpacing: '0.15em', textTransform: 'uppercase',
                                }}>
                                    CONNECTING TO BACKEND…
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Gantt chart */}
                    {(phase === 1 || phase === 3) && (
                        <div style={{ flex: 1, overflow: 'hidden', position: 'relative', minHeight: 0 }}>
                            <GanttChart
                                blocks={currentGanttBlocks}
                                workers={workerStates}
                                currentTick={currentTick}
                                phase={phase === 3 ? 2 : 1}
                                slotsPerDay={16}
                                numWorkers={simConfig.num_workers}
                                totalDays={simConfig.days_phase1 + simConfig.days_phase2}
                            />
                        </div>
                    )}

                    {/* Task queue */}
                    {(phase === 1 || phase === 3) && (
                        <div style={{
                            flexShrink: 0, height: 180,
                            borderTop: '1px solid var(--color-border)',
                        }}>
                            <TaskQueue queue={queueState} currentTick={currentTick} />
                        </div>
                    )}
                </div>

                {/* ── Right panel: ComparisonStrip ── */}
                <div style={{
                    width: 300, flexShrink: 0,
                    borderLeft: '1px solid var(--color-border)',
                    background: 'var(--color-surface)', overflow: 'auto',
                }}>
                    <ComparisonStrip
                        phase={phase === 3 ? 2 : phase === 1 ? 1 : 0}
                        hhMetrics={hhMetrics}
                        workerCount={simConfig.num_workers}
                    />
                </div>
            </div>
        </div>
    );
}
