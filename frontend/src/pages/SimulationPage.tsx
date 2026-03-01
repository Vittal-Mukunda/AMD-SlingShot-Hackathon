/**
 * SimulationPage.tsx — Real-Time Simulation View (v2)
 *
 * Phase protocol:
 *   0          → Initializing (spinner)
 *   1          → Phase 1: All 5 baselines running live with Gantt + fatigue gauges
 *   'training' → Phase 2a: DQN training in background — spinner + progress bar
 *   3          → Phase 2b: DQN scheduling live Gantt
 *
 * Events:
 *   phase_transition { new_phase: 'training' } → show training screen
 *   phase2_ready                               → switch to phase 3 realtime view
 *   simulation_complete                        → navigate to /analytics after 2s
 *   simulation_error                           → show error banner
 */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
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

// ── Phase badge ───────────────────────────────────────────────────────────────
function PhaseBadge({ phase }: { phase: 0 | 1 | 'training' | 3 }) {
    if (phase === 1) return (
        <span className="badge badge-phase1 font-mono" style={{ fontSize: 11 }}>
            PHASE 1 — BASELINE ALLOCATION
        </span>
    );
    if (phase === 'training') return (
        <span className="badge badge-phase2 font-mono" style={{ fontSize: 11 }}>
            DQN TRAINING
        </span>
    );
    if (phase === 3) return (
        <span className="badge badge-phase2 font-mono" style={{ fontSize: 11 }}>
            PHASE 2 — DQN SCHEDULING
        </span>
    );
    return <span className="badge badge-slate font-mono" style={{ fontSize: 11 }}>INITIALIZING</span>;
}

// ── DQN passive indicator (phase 1) ──────────────────────────────────────────
function DQNPassiveIndicator() {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '4px 10px', background: 'rgba(59,130,246,0.08)',
            border: '1px solid rgba(59,130,246,0.25)', borderRadius: 4,
        }}>
            <span style={{
                width: 7, height: 7, borderRadius: '50%',
                background: '#60A5FA', display: 'inline-block',
                animation: 'pulse-amber 2s ease-in-out infinite',
            }} />
            <span className="font-mono" style={{ fontSize: 10, color: '#60A5FA', letterSpacing: '0.07em' }}>
                DQN OBSERVING
            </span>
        </div>
    );
}

// ── Training waiting screen ───────────────────────────────────────────────────
function TrainingScreen({ percent, elapsed }: { percent: number; elapsed: number }) {
    const ringStyle: React.CSSProperties = {
        width: 120, height: 120, borderRadius: '50%',
        border: '3px solid rgba(245,158,11,0.12)',
        borderTopColor: '#F59E0B',
        animation: 'spin 1.4s linear infinite',
        position: 'absolute',
    };
    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(13,15,18,0.97)',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 32,
        }}>
            {/* Spinner */}
            <div style={{ position: 'relative', width: 120, height: 120 }}>
                <div style={ringStyle} />
                <div style={{
                    ...ringStyle, width: 88, height: 88, top: 16, left: 16,
                    borderTopColor: 'transparent',
                    borderRightColor: 'rgba(245,158,11,0.4)',
                    animationDuration: '2.2s',
                    animationDirection: 'reverse',
                }} />
                <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <span style={{ fontSize: 28 }}>🧠</span>
                </div>
            </div>

            {/* Message */}
            <div style={{ textAlign: 'center', maxWidth: 480 }}>
                <div className="font-mono" style={{
                    fontSize: 10, color: 'var(--color-amber)',
                    letterSpacing: '0.18em', textTransform: 'uppercase', marginBottom: 14,
                }}>
                    PHASE 2 — DQN TRAINING
                </div>
                <h2 className="font-display" style={{
                    fontSize: '1.45rem', fontWeight: 700, color: 'var(--color-text)',
                    marginBottom: 12, lineHeight: 1.4,
                }}>
                    DQN Agent is learning scheduling patterns
                </h2>
                <p style={{
                    color: 'var(--color-slate-text)', fontSize: '0.88rem',
                    lineHeight: 1.7, marginBottom: 24,
                }}>
                    Replaying {percent < 100 ? 'workforce transitions' : 'complete!'} from Phase 1 baselines
                    and computing Bellman updates. Agent takes control once training completes.
                </p>

                {/* Progress bar */}
                <div style={{
                    width: 340, height: 6, background: 'rgba(245,158,11,0.12)',
                    borderRadius: 3, overflow: 'hidden', margin: '0 auto 16px',
                }}>
                    <div style={{
                        height: '100%', width: `${percent}%`,
                        background: 'var(--color-amber)',
                        borderRadius: 3,
                        transition: 'width 0.4s ease',
                    }} />
                </div>
                <div className="font-mono" style={{ fontSize: 13, color: 'var(--color-amber)', marginBottom: 8 }}>
                    {percent}%
                </div>
                <div className="font-mono" style={{ fontSize: 10, color: 'var(--color-slate-dim)' }}>
                    Elapsed: {formatElapsed(elapsed)}
                </div>
            </div>

            <div className="font-mono" style={{
                fontSize: 10, color: 'var(--color-slate-dim)',
                letterSpacing: '0.1em', textTransform: 'uppercase',
            }}>
                Phase 1 complete — All 5 baseline policies benchmarked
            </div>
        </div>
    );
}

// ── Error banner ──────────────────────────────────────────────────────────────
function ErrorBanner({ message }: { message: string }) {
    return (
        <div style={{
            position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
            zIndex: 300, background: '#EF4444', color: '#fff',
            padding: '12px 24px', borderRadius: 6,
            fontFamily: 'var(--font-mono)', fontSize: 12, maxWidth: 600,
            boxShadow: '0 4px 24px rgba(239,68,68,0.4)',
        }}>
            ⚠ Simulation Error: {message}
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
    } = useSimulationStore();

    // Training screen elapsed timer
    const [trainingElapsed, setTrainingElapsed] = useState(0);
    const trainingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const [showInjectPanel, setShowInjectPanel] = useState(false);
    const [injectedTasks, setInjectedTasks] = useState<InjectedTask[]>([]);

    // Start/stop training timer based on phase
    useEffect(() => {
        if (phase === 'training') {
            setTrainingElapsed(0);
            trainingTimerRef.current = setInterval(() =>
                setTrainingElapsed(v => v + 1), 1000);
        } else {
            if (trainingTimerRef.current) {
                clearInterval(trainingTimerRef.current);
                trainingTimerRef.current = null;
            }
        }
        return () => {
            if (trainingTimerRef.current) clearInterval(trainingTimerRef.current);
        };
    }, [phase]);

    // Navigate to analytics when simulation completes (Bug 6: 3.5s delay so user sees final Phase 2 view)
    useEffect(() => {
        if (finalMetrics) {
            const t = setTimeout(() => navigate('/analytics'), 3500);
            return () => clearTimeout(t);
        }
    }, [finalMetrics, navigate]);

    const hhMetrics = useHeadToHead();

    const handleInjectTask = (task: InjectedTask) => {
        injectTask({
            task_id: task.task_id,
            duration: task.duration,
            urgency: task.urgency,
            required_skill: task.required_skill,
            arrival_tick: task.arrival_tick,
        });
    };

    // Gantt data: Phase 1 → selected baseline tab; Phase 3 → DQN
    const currentGanttBlocks =
        phase === 3
            ? (ganttBlocks['DQN'] ?? [])
            : (ganttBlocks[selectedBaseline] ?? []);

    // Which tab to highlight as "live" during Phase 1
    const isActiveBaseline = (name: string) =>
        phase === 1 && activePolicy.includes(name);

    return (
        <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)' }}>

            {/* ── Training waiting screen (blocks the whole UI) ── */}
            {phase === 'training' && (
                <TrainingScreen percent={trainingPercent} elapsed={trainingElapsed} />
            )}

            {/* ── Error banner ── */}
            {simulationError && <ErrorBanner message={simulationError} />}

            {/* ── TOP NAVBAR ── */}
            <nav style={{
                height: 56, flexShrink: 0,
                background: 'var(--color-panel)',
                borderBottom: '1px solid var(--color-border)',
                display: 'flex', alignItems: 'center',
                padding: '0 1.5rem', gap: '1.2rem',
            }}>
                <span className="font-mono" style={{
                    color: 'var(--color-amber)', fontWeight: 700,
                    fontSize: 13, letterSpacing: '0.1em', flexShrink: 0,
                }}>
                    DQN SCHEDULER
                </span>
                <div style={{ width: 1, height: 28, background: 'var(--color-border)', flexShrink: 0 }} />

                <PhaseBadge phase={phase} />

                {/* Tick / day */}
                <div className="num font-mono" style={{ color: 'var(--color-text)', fontSize: 13 }}>
                    <span style={{ color: 'var(--color-slate-text)' }}>TICK</span>
                    <span style={{ color: 'var(--color-amber)', marginLeft: 6, marginRight: 16 }}>
                        {currentTick.toLocaleString()}
                    </span>
                    <span style={{ color: 'var(--color-slate-text)' }}>DAY</span>
                    <span style={{ color: 'var(--color-text)', marginLeft: 6 }}>{currentDay}</span>
                </div>

                {phase === 1 && <DQNPassiveIndicator />}

                <div style={{ flex: 1 }} />

                {/* Active policy label */}
                {(phase === 1 || phase === 3) && (
                    <div className="font-mono" style={{
                        fontSize: 11, color: 'var(--color-slate-text)',
                        background: 'rgba(255,255,255,0.04)',
                        padding: '3px 8px', borderRadius: 3,
                    }}>
                        {activePolicy}
                    </div>
                )}

                {/* Elapsed */}
                <div className="num font-mono" style={{ color: 'var(--color-slate-text)', fontSize: 13 }}>
                    ⏱ {formatElapsed(elapsedSeconds)}
                </div>

                {/* Inject task */}
                <button className="btn-ghost" style={{ fontSize: 11 }}
                    onClick={() => setShowInjectPanel(p => !p)}>
                    + Inject Task
                </button>

                {/* Pause / Resume */}
                <button
                    className={isPaused ? 'btn-primary' : 'btn-ghost'}
                    style={{ fontSize: 11, minWidth: 80 }}
                    onClick={() => isPaused ? resumeSimulation() : pauseSimulation()}
                >
                    {isPaused ? '▶ Resume' : '⏸ Pause'}
                </button>

                {/* Analytics link when done */}
                {finalMetrics && (
                    <button className="btn-primary" style={{ fontSize: 11 }}
                        onClick={() => navigate('/analytics')}>
                        Analytics →
                    </button>
                )}
            </nav>

            {/* ── INJECT PANEL ── */}
            {showInjectPanel && (
                <div style={{
                    background: 'var(--color-panel)',
                    borderBottom: '1px solid var(--color-border)',
                    padding: '1rem 1.5rem',
                }}>
                    <PriorityInjectionPanel
                        pendingTasks={injectedTasks}
                        onAdd={(t) => setInjectedTasks(p => [...p, t])}
                        onRemove={(id) => setInjectedTasks(p => p.filter(t => t.task_id !== id))}
                        onEmit={handleInjectTask}
                        currentTick={currentTick}
                    />
                </div>
            )}

            {/* ── BODY ── */}
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

                {/* ── LEFT SIDEBAR: Workers ── */}
                <WorkerSidebar
                    workers={workerStates}
                    simConfig={simConfig}
                    currentTick={currentTick}
                />

                {/* ── MAIN: Gantt + Queue ── */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

                    {/* Phase 1: 5 baseline tabs */}
                    {phase === 1 && (
                        <div style={{
                            display: 'flex', gap: 0,
                            background: 'var(--color-panel)',
                            borderBottom: '1px solid var(--color-border)',
                            padding: '0 1rem', flexShrink: 0,
                        }}>
                            {BASELINE_NAMES.map(name => {
                                const hasData = (ganttBlocks[name]?.length ?? 0) > 0;
                                const isActive = isActiveBaseline(name);
                                return (
                                    <button key={name}
                                        onClick={() => setSelectedBaseline(name)}
                                        style={{
                                            padding: '8px 16px', background: 'none', border: 'none',
                                            borderBottom: `2px solid ${selectedBaseline === name
                                                ? 'var(--color-amber)' : 'transparent'}`,
                                            color: selectedBaseline === name
                                                ? 'var(--color-amber)'
                                                : hasData ? 'var(--color-text)' : 'var(--color-slate-dim)',
                                            fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600,
                                            letterSpacing: '0.08em', textTransform: 'uppercase',
                                            cursor: 'pointer',
                                            transition: 'all 0.15s ease',
                                            display: 'flex', alignItems: 'center', gap: 6,
                                        }}>
                                        {name}
                                        {isActive && (
                                            <span style={{
                                                width: 6, height: 6, borderRadius: '50%',
                                                background: 'var(--color-amber)',
                                                animation: 'pulse-amber 1.2s ease-in-out infinite',
                                                display: 'inline-block',
                                            }} />
                                        )}
                                        {!isActive && hasData && (
                                            <span style={{
                                                width: 6, height: 6, borderRadius: '50%',
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
                            padding: '6px 1rem',
                            background: 'rgba(245,158,11,0.06)',
                            borderBottom: '1px solid rgba(245,158,11,0.2)',
                            flexShrink: 0,
                            display: 'flex', alignItems: 'center', gap: 12,
                        }}>
                            <span style={{
                                width: 8, height: 8, borderRadius: '50%',
                                background: 'var(--color-amber)', display: 'inline-block',
                                animation: 'pulse-amber 1.5s ease-in-out infinite',
                            }} />
                            <span className="font-mono" style={{
                                fontSize: 11, color: 'var(--color-amber)', letterSpacing: '0.08em',
                            }}>
                                DQN AGENT — ONLINE SCHEDULING ACTIVE
                            </span>
                        </div>
                    )}

                    {/* Phase 0: Initializing */}
                    {phase === 0 && (
                        <div style={{
                            flex: 1, display: 'flex', alignItems: 'center',
                            justifyContent: 'center',
                        }}>
                            <div style={{ textAlign: 'center' }}>
                                <div className="font-mono" style={{
                                    fontSize: 12, color: 'var(--color-amber)',
                                    letterSpacing: '0.15em', marginBottom: 16,
                                }}>
                                    CONNECTING TO BACKEND…
                                </div>
                                <div style={{
                                    width: 40, height: 40, borderRadius: '50%',
                                    border: '3px solid rgba(245,158,11,0.15)',
                                    borderTopColor: '#F59E0B',
                                    animation: 'spin 1s linear infinite',
                                    margin: '0 auto',
                                }} />
                            </div>
                        </div>
                    )}

                    {/* Gantt chart (shown in phase 1 and 3) */}
                    {(phase === 1 || phase === 3) && (
                        <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
                            <GanttChart
                                blocks={currentGanttBlocks}
                                workers={workerStates}
                                currentTick={currentTick}
                                phase={phase === 3 ? 2 : 1}
                                slotsPerDay={16}
                            />
                        </div>
                    )}

                    {/* Task queue (shown in phase 1 and 3) */}
                    {(phase === 1 || phase === 3) && (
                        <div style={{ flexShrink: 0, height: 180, borderTop: '1px solid var(--color-border)' }}>
                            <TaskQueue queue={queueState} currentTick={currentTick} />
                        </div>
                    )}
                </div>

                {/* ── RIGHT PANEL: ComparisonStrip ── */}
                <div style={{
                    width: 320, flexShrink: 0,
                    borderLeft: '1px solid var(--color-border)',
                    background: 'var(--color-panel)', overflow: 'auto',
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
