/**
 * SimulationPage.tsx — Live Operations Center
 *
 * Visual overhaul: premium dark command-center aesthetic.
 * All socket/store bindings, handlers, and data flows UNCHANGED.
 * All special characters use SVG or plain ASCII — no emoji encoding issues.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useShallow } from 'zustand/react/shallow';
import { motion, AnimatePresence } from 'framer-motion';
import { useSimulationStore } from '../store/simulationStore';
import { formatElapsed, useHeadToHead } from '../hooks/useSimulation';
import { useSocket } from '../hooks/useSocket';
import GanttChart from '../components/simulation/GanttChart';
import WorkerSidebar from '../components/simulation/WorkerSidebar';
import ComparisonStrip from '../components/simulation/ComparisonStrip';
import PriorityInjectionPanel from '../components/config/PriorityInjectionPanel';
import PhaseTransitionOverlay from '../components/PhaseTransitionOverlay';
import AnimatedTaskQueue from '../components/TaskQueueCard';
import { PageTransition } from '../components/PageTransition';
import type { InjectedTask } from '../types/config';
import type { Phase } from '../store/simulationStore';

// ── Constants ─────────────────────────────────────────────────────────────────
const BASELINE_NAMES = ['Greedy', 'Skill', 'FIFO', 'Hybrid', 'Random'];

const POLICY_COLORS: Record<string, string> = {
    Greedy: '#4493f8',
    Skill: '#2dd4bf',
    FIFO: '#3fb950',
    Hybrid: '#a78bfa',
    Random: '#f85149',
    DQN: '#e3b341',
};

// ── Page-scoped styles ────────────────────────────────────────────────────────
const SIM_STYLES = `
/* ── Scan line overlay on the entire page ── */
.sim-root {
  position: relative;
}
.sim-root::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.03) 2px,
    rgba(0,0,0,0.03) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Phase progress bar ── */
.phase-bar-root {
  display: flex;
  align-items: stretch;
  height: 48px;
  flex-shrink: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--rim);
  position: relative;
  overflow: hidden;
}
.phase-bar-root::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(68,147,248,0.03) 0%, transparent 60%);
  pointer-events: none;
}
.phase-seg {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 18px;
  position: relative;
  transition: background 0.3s;
  border-right: 1px solid var(--rim);
}
.phase-seg:last-child { border-right: none; }
.phase-seg.active   { background: rgba(255,255,255,0.02); }
.phase-seg.done     { background: rgba(63,185,80,0.03); }
.phase-seg-fill {
  position: absolute;
  bottom: 0; left: 0;
  height: 2px;
  border-radius: 0;
  transition: width 0.6s cubic-bezier(0.22,1,0.36,1);
}
.phase-seg-num {
  width: 20px; height: 20px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-mono);
  font-size: 10px; font-weight: 500;
  flex-shrink: 0;
  border: 1.5px solid var(--muted);
  color: var(--dim);
  transition: all 0.3s;
}
.phase-seg-num.active {
  border-color: currentColor;
  color: inherit;
  box-shadow: 0 0 10px currentColor;
}
.phase-seg-num.done {
  border-color: var(--emerald);
  color: var(--emerald);
  background: rgba(63,185,80,0.1);
}
.phase-seg-label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  transition: color 0.3s;
}

/* ── Subheader ── */
.sim-subheader {
  height: 48px;
  flex-shrink: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--rim);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 0;
  backdrop-filter: blur(4px);
}
.sim-stat {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 0 18px;
  border-right: 1px solid var(--rim);
}
.sim-stat:first-child { padding-left: 0; }
.sim-stat-label {
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  line-height: 1;
  margin-bottom: 3px;
}
.sim-stat-val {
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 500;
  line-height: 1;
}

/* ── Header action buttons ── */
.sim-btn {
  padding: 5px 14px;
  border-radius: 6px;
  font-family: var(--font-body);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  border: 1px solid var(--rim-hi);
  background: transparent;
  color: var(--dim);
  white-space: nowrap;
}
.sim-btn:hover {
  border-color: var(--blue);
  color: var(--blue);
  background: rgba(68,147,248,0.06);
}
.sim-btn.active {
  background: var(--blue);
  color: #fff;
  border-color: var(--blue);
  box-shadow: 0 0 14px rgba(68,147,248,0.35);
}
.sim-btn.cta {
  background: var(--amber);
  color: #000;
  border-color: var(--amber);
  font-weight: 700;
  box-shadow: 0 0 14px rgba(227,179,65,0.35);
}

/* ── Baseline tabs ── */
.baseline-tabs {
  display: flex;
  gap: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--rim);
  padding: 0 12px;
  flex-shrink: 0;
  overflow-x: auto;
}
.baseline-tab {
  padding: 9px 16px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 0.15s;
  display: flex;
  align-items: center;
  gap: 7px;
  white-space: nowrap;
  flex-shrink: 0;
}
.baseline-tab:hover { color: var(--sol) !important; }

/* ── DQN status header ── */
.dqn-header {
  padding: 0 18px;
  height: 38px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  background: rgba(227,179,65,0.04);
  border-bottom: 1px solid rgba(227,179,65,0.15);
  position: relative;
  overflow: hidden;
}
.dqn-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--amber);
  box-shadow: 0 0 12px var(--amber);
}

/* ── DQN metrics panel ── */
.dqn-panel {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 50;
  width: 272px;
  background: var(--surface);
  border: 1px solid rgba(227,179,65,0.25);
  border-radius: 14px;
  padding: 18px;
  backdrop-filter: blur(12px);
}
.dqn-panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--rim);
}
.dqn-panel-title {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--amber);
  flex: 1;
}
.dqn-metric-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.dqn-metric-label {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--dim);
}
.dqn-metric-val {
  font-family: var(--font-mono);
  font-size: 16px;
  font-weight: 500;
  line-height: 1;
}
.eps-track {
  width: 100%;
  height: 5px;
  background: var(--raised);
  border-radius: 3px;
  overflow: hidden;
  margin-top: 6px;
}
.eps-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease, background 0.5s ease;
}

/* ── Connecting state ── */
.sim-connecting {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 20px;
}
.connecting-ring {
  width: 52px; height: 52px;
  border-radius: 50%;
  border: 2px solid rgba(68,147,248,0.1);
  border-top-color: var(--blue);
}
.connecting-label {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--blue);
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

/* ── Error banner ── */
.sim-error-banner {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 300;
  background: rgba(248,81,73,0.09);
  border: 1px solid rgba(248,81,73,0.35);
  color: var(--rose);
  padding: 12px 22px;
  border-radius: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  max-width: 560px;
  box-shadow: 0 8px 40px rgba(248,81,73,0.2);
  display: flex;
  align-items: center;
  gap: 14px;
  backdrop-filter: blur(8px);
  white-space: nowrap;
}

/* ── Right panel ── */
.sim-right-panel {
  width: 290px;
  flex-shrink: 0;
  border-left: 1px solid var(--rim);
  background: var(--surface);
  overflow-y: auto;
  position: relative;
}
.sim-right-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 120px;
  background: linear-gradient(180deg, rgba(68,147,248,0.03) 0%, transparent 100%);
  pointer-events: none;
}

/* ── Live pulse dot ── */
@keyframes livePulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.4; transform: scale(0.85); }
}
.live-dot { animation: livePulse 1.4s ease-in-out infinite; }

/* ── Corner badge ── */
.corner-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 8px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
`;

// ── SVG icons (no emoji, no encoding issues) ──────────────────────────────────
const Icon = {
    check: (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
        </svg>
    ),
    pause: (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="4" width="4" height="16" rx="1" />
            <rect x="14" y="4" width="4" height="16" rx="1" />
        </svg>
    ),
    play: (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
    ),
    plus: (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
    ),
    arrowRight: (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
        </svg>
    ),
    warning: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
    ),
    clock: (
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
        </svg>
    ),
    bolt: (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
        </svg>
    ),
};

// ── Phase progress bar ─────────────────────────────────────────────────────────
const PHASE_SEGS = [
    { label: 'Observe', color: '#4493f8', phaseVal: 1 },
    { label: 'Train', color: '#a78bfa', phaseVal: 'training' },
    { label: 'Schedule', color: '#e3b341', phaseVal: 3 },
];

function PhaseProgressBar({ phase }: { phase: Phase }) {
    const p1Done = phase === 'training' || phase === 3 || (phase as any) === 'complete';
    const trainDone = phase === 3 || (phase as any) === 'complete';

    function getState(phaseVal: any): 'done' | 'active' | 'idle' {
        if (phaseVal === 1 && p1Done) return 'done';
        if (phaseVal === 'training' && trainDone) return 'done';
        if (phase === phaseVal) return 'active';
        return 'idle';
    }

    return (
        <div className="phase-bar-root">
            {PHASE_SEGS.map((seg, i) => {
                const state = getState(seg.phaseVal);
                const fillW = state === 'done' ? '100%' : state === 'active' ? '55%' : '0%';
                return (
                    <div
                        key={seg.label}
                        className={`phase-seg ${state}`}
                        style={{ color: state !== 'idle' ? seg.color : undefined }}
                    >
                        {/* Bottom fill bar */}
                        <div
                            className="phase-seg-fill"
                            style={{
                                width: fillW,
                                background: seg.color,
                                boxShadow: state === 'active' ? `0 0 8px ${seg.color}` : 'none',
                                transition: 'width 0.7s cubic-bezier(0.22,1,0.36,1)',
                            }}
                        />

                        {/* Step number / check */}
                        <div
                            className={`phase-seg-num ${state}`}
                            style={{ color: state !== 'idle' ? seg.color : undefined }}
                        >
                            {state === 'done' ? Icon.check : i + 1}
                        </div>

                        <span
                            className="phase-seg-label"
                            style={{
                                color: state === 'active' ? seg.color
                                    : state === 'done' ? 'var(--emerald)'
                                        : 'var(--muted)',
                            }}
                        >
                            {seg.label}
                        </span>

                        {state === 'active' && (
                            <div
                                className="live-dot corner-badge"
                                style={{
                                    marginLeft: 'auto',
                                    background: `${seg.color}18`,
                                    color: seg.color,
                                    border: `1px solid ${seg.color}40`,
                                }}
                            >
                                <div style={{
                                    width: 5, height: 5,
                                    borderRadius: '50%',
                                    background: seg.color,
                                }} />
                                Live
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

// ── DQN metrics panel ──────────────────────────────────────────────────────────
function DQNMetricsPanel() {
    const { epsilon, lastQMean } = useSimulationStore(
        useShallow(s => ({
            epsilon: (s as any).epsilon ?? 1.0,
            lastQMean: (s as any).lastQMean ?? 0,
        }))
    );

    const qColor =
        lastQMean > 4 ? 'var(--emerald)' :
            lastQMean > 2 ? 'var(--amber)' : 'var(--rose)';

    const epsilonPct = Math.round(epsilon * 100);
    const epsilonColor = epsilon > 0.1 ? 'var(--amber)' : 'var(--blue)';
    const epsilonLabel = epsilon > 0.3 ? 'Exploring' : epsilon > 0.1 ? 'Converging' : 'Exploiting';

    return (
        <motion.div
            className="dqn-panel"
            initial={{ x: 320, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 320, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 280, damping: 28 }}
            style={{ boxShadow: `0 0 0 1px rgba(227,179,65,0.15), 0 16px 48px rgba(0,0,0,0.5), var(--amber-glow)` }}
        >
            {/* Header */}
            <div className="dqn-panel-header">
                <motion.div
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.4, repeat: Infinity }}
                    style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--amber)', flexShrink: 0 }}
                />
                <span className="dqn-panel-title">DQN Agent — Online</span>
                <div style={{
                    padding: '2px 6px', borderRadius: 4,
                    background: 'rgba(227,179,65,0.1)',
                    border: '1px solid rgba(227,179,65,0.2)',
                    fontFamily: 'var(--font-mono)', fontSize: 8,
                    color: 'var(--amber)', letterSpacing: '0.08em',
                }}>
                    PHASE 2
                </div>
            </div>

            {/* Epsilon */}
            <div style={{ marginBottom: 14 }}>
                <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: 6,
                }}>
                    <span className="dqn-metric-label">
                        Epsilon ({epsilonLabel})
                    </span>
                    <span style={{
                        fontFamily: 'var(--font-mono)', fontSize: 13,
                        fontWeight: 500, color: epsilonColor,
                    }}>
                        {epsilon.toFixed(3)}
                    </span>
                </div>
                <div className="eps-track">
                    <motion.div
                        className="eps-fill"
                        animate={{ width: `${epsilonPct}%`, background: epsilonColor }}
                        transition={{ duration: 0.6, ease: 'easeOut' }}
                        style={{ boxShadow: `0 0 8px ${epsilonColor}` }}
                    />
                </div>
                {/* Threshold markers */}
                <div style={{ position: 'relative', height: 10, marginTop: 2 }}>
                    {[{ x: '30%', label: '0.3' }, { x: '10%', label: '0.1' }].map(m => (
                        <div key={m.label} style={{
                            position: 'absolute', left: m.x,
                            width: 1, height: 6,
                            background: 'var(--rim-hi)',
                            top: 0,
                        }}>
                            <span style={{
                                position: 'absolute', top: 6, left: '-8px',
                                fontFamily: 'var(--font-mono)', fontSize: 8,
                                color: 'var(--muted)', whiteSpace: 'nowrap',
                            }}>{m.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Q-mean */}
            <div className="dqn-metric-row" style={{ marginTop: 8 }}>
                <div>
                    <div className="dqn-metric-label" style={{ marginBottom: 2 }}>Q-Mean</div>
                    <div style={{
                        fontFamily: 'var(--font-mono)', fontSize: 8,
                        color: 'var(--muted)',
                    }}>
                        {lastQMean > 4 ? 'Healthy' : lastQMean > 2 ? 'Converging' : 'Low — learning'}
                    </div>
                </div>
                <motion.div
                    key={Math.round(lastQMean * 10)}
                    initial={{ scale: 1.2, opacity: 0.6 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="dqn-metric-val"
                    style={{ color: qColor, fontSize: 28 }}
                >
                    {lastQMean.toFixed(2)}
                </motion.div>
            </div>

            {/* Q health bar */}
            <div className="eps-track" style={{ marginTop: 6 }}>
                <motion.div
                    animate={{ width: `${Math.min(lastQMean / 8 * 100, 100)}%`, background: qColor }}
                    transition={{ duration: 0.6 }}
                    style={{ height: '100%', borderRadius: 3, boxShadow: `0 0 6px ${qColor}` }}
                />
            </div>
            <div style={{
                display: 'flex', justifyContent: 'space-between',
                marginTop: 4,
                fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--muted)',
            }}>
                <span>0</span>
                <span>Target: 4+</span>
                <span>8</span>
            </div>
        </motion.div>
    );
}

// ── Error banner ───────────────────────────────────────────────────────────────
function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
    return (
        <div className="sim-error-banner">
            <span style={{ color: 'var(--rose)', flexShrink: 0 }}>{Icon.warning}</span>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                Simulation error: {message}
            </span>
            {onDismiss && (
                <button onClick={onDismiss} className="sim-btn" style={{
                    flexShrink: 0, fontSize: 11, padding: '4px 10px',
                    color: 'var(--rose)', borderColor: 'rgba(248,81,73,0.3)',
                }}>
                    Return
                </button>
            )}
        </div>
    );
}

// ── Baseline tabs ──────────────────────────────────────────────────────────────
function BaselineTabs({ ganttBlocks, selectedBaseline, setSelectedBaseline, activePolicy }: {
    ganttBlocks: Record<string, any[]>;
    selectedBaseline: string;
    setSelectedBaseline: (n: string) => void;
    activePolicy: string;
}) {
    return (
        <div className="baseline-tabs">
            {BASELINE_NAMES.map(name => {
                const hasData = (ganttBlocks[name]?.length ?? 0) > 0;
                const isActive = activePolicy.includes(name);
                const isSel = selectedBaseline === name;
                const color = POLICY_COLORS[name] ?? 'var(--dim)';

                return (
                    <button
                        key={name}
                        className="baseline-tab"
                        onClick={() => setSelectedBaseline(name)}
                        style={{
                            borderBottomColor: isSel ? color : 'transparent',
                            color: isSel ? color : hasData ? 'var(--dim)' : 'var(--muted)',
                        }}
                    >
                        {name}
                        {isActive && (
                            <motion.div
                                animate={{ opacity: [1, 0.25, 1] }}
                                transition={{ duration: 1.3, repeat: Infinity }}
                                style={{
                                    width: 5, height: 5,
                                    borderRadius: '50%',
                                    background: color,
                                    boxShadow: `0 0 6px ${color}`,
                                }}
                            />
                        )}
                    </button>
                );
            })}
        </div>
    );
}

// ── Subheader stat block ───────────────────────────────────────────────────────
function StatBlock({ label, value, color }: {
    label: string; value: React.ReactNode; color?: string;
}) {
    return (
        <div className="sim-stat">
            <span className="sim-stat-label">{label}</span>
            <span className="sim-stat-val" style={{ color: color ?? 'var(--sol)' }}>
                {value}
            </span>
        </div>
    );
}

// ── Main SimulationPage ────────────────────────────────────────────────────────
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

    // Auto-navigate to analytics when done (IDENTICAL to original)
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

    const totalDays = simConfig.days_phase1 + simConfig.days_phase2;

    return (
        <>
            <style>{SIM_STYLES}</style>

            <PageTransition>
                <div
                    className="sim-root"
                    style={{
                        height: 'calc(100vh - 56px)',
                        display: 'flex',
                        flexDirection: 'column',
                        background: 'var(--base)',
                        overflow: 'hidden',
                    }}
                >
                    {/* Phase transition overlay */}
                    <PhaseTransitionOverlay
                        visible={phase === 'training'}
                        percent={trainingPercent}
                    />

                    {/* Error banner */}
                    {simulationError && (
                        <ErrorBanner
                            message={simulationError}
                            onDismiss={() => {
                                useSimulationStore.getState().reset();
                                navigate('/');
                            }}
                        />
                    )}

                    {/* DQN floating metrics panel */}
                    <AnimatePresence>
                        {phase === 3 && <DQNMetricsPanel key="dqn-panel" />}
                    </AnimatePresence>

                    {/* Phase progress bar */}
                    <PhaseProgressBar phase={phase} />

                    {/* Subheader */}
                    <div className="sim-subheader">
                        <StatBlock label="Day" value={currentDay} color="var(--sol)" />
                        <StatBlock
                            label="Tick"
                            value={currentTick.toLocaleString()}
                            color="var(--blue)"
                        />
                        <StatBlock
                            label="Policy"
                            value={
                                <span style={{ color: POLICY_COLORS[activePolicy] ?? 'var(--sol)' }}>
                                    {activePolicy || '—'}
                                </span>
                            }
                        />

                        {/* Elapsed */}
                        <div className="sim-stat" style={{ borderRight: 'none' }}>
                            <span className="sim-stat-label">Elapsed</span>
                            <span className="sim-stat-val" style={{
                                color: 'var(--dim)', display: 'flex',
                                alignItems: 'center', gap: 5, fontSize: 13,
                            }}>
                                <span style={{ color: 'var(--muted)' }}>{Icon.clock}</span>
                                {formatElapsed(elapsedSeconds)}
                            </span>
                        </div>

                        <div style={{ flex: 1 }} />

                        {/* Action buttons */}
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <motion.button
                                className={`sim-btn ${isPaused ? 'active' : ''}`}
                                onClick={() => isPaused ? resumeSimulation() : pauseSimulation()}
                                whileTap={{ scale: 0.95 }}
                                style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                            >
                                {isPaused ? Icon.play : Icon.pause}
                                {isPaused ? 'Resume' : 'Pause'}
                            </motion.button>

                            <motion.button
                                className="sim-btn"
                                onClick={() => setShowInjectPanel(p => !p)}
                                whileTap={{ scale: 0.95 }}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: 6,
                                    borderColor: showInjectPanel ? 'var(--violet)' : undefined,
                                    color: showInjectPanel ? 'var(--violet)' : undefined,
                                    background: showInjectPanel ? 'rgba(167,139,250,0.08)' : undefined,
                                }}
                            >
                                {Icon.plus}
                                Inject Task
                            </motion.button>

                            {finalMetrics && (
                                <motion.button
                                    className="sim-btn cta"
                                    onClick={() => navigate('/analytics')}
                                    whileTap={{ scale: 0.97 }}
                                    style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                                >
                                    Analytics {Icon.arrowRight}
                                </motion.button>
                            )}
                        </div>
                    </div>

                    {/* Inject panel — slides down */}
                    <AnimatePresence>
                        {showInjectPanel && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.25, ease: 'easeOut' }}
                                style={{
                                    overflow: 'hidden', flexShrink: 0,
                                    background: 'var(--surface)',
                                    borderBottom: '1px solid var(--rim)',
                                }}
                            >
                                <div style={{ padding: '14px 18px' }}>
                                    <PriorityInjectionPanel
                                        pendingTasks={injectedTasks}
                                        onAdd={t => setInjectedTasks(p => [...p, t])}
                                        onRemove={id => setInjectedTasks(p => p.filter(t => t.task_id !== id))}
                                        onEmit={handleInjectTask}
                                        currentTick={currentTick}
                                    />
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Body */}
                    <div style={{
                        flex: 1, display: 'flex',
                        overflow: 'hidden', minHeight: 0,
                    }}>
                        {/* Left: Workers */}
                        <WorkerSidebar
                            workers={workerStates}
                            simConfig={simConfig}
                            currentTick={currentTick}
                        />

                        {/* Center: Gantt + Queue */}
                        <div style={{
                            flex: 1, display: 'flex',
                            flexDirection: 'column', overflow: 'hidden',
                        }}>
                            {/* Baseline tabs */}
                            {phase === 1 && (
                                <BaselineTabs
                                    ganttBlocks={ganttBlocks}
                                    selectedBaseline={selectedBaseline}
                                    setSelectedBaseline={setSelectedBaseline}
                                    activePolicy={activePolicy}
                                />
                            )}

                            {/* DQN active header */}
                            {phase === 3 && (
                                <div className="dqn-header">
                                    <motion.div
                                        animate={{ opacity: [1, 0.25, 1] }}
                                        transition={{ duration: 1.3, repeat: Infinity }}
                                        style={{
                                            width: 6, height: 6,
                                            borderRadius: '50%',
                                            background: 'var(--amber)',
                                            boxShadow: '0 0 8px var(--amber)',
                                        }}
                                    />
                                    <span style={{
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: 10,
                                        color: 'var(--amber)',
                                        letterSpacing: '0.1em',
                                    }}>
                                        DQN AGENT — ONLINE SCHEDULING ACTIVE
                                    </span>
                                    <div style={{ flex: 1 }} />
                                    <div style={{
                                        fontFamily: 'var(--font-mono)', fontSize: 9,
                                        color: 'var(--dim)',
                                        letterSpacing: '0.06em',
                                    }}>
                                        Day {currentDay} / {totalDays}
                                    </div>
                                </div>
                            )}

                            {/* Connecting spinner */}
                            {phase === 0 && (
                                <div className="sim-connecting">
                                    <motion.div
                                        className="connecting-ring"
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
                                    />
                                    <span className="connecting-label">Connecting</span>
                                </div>
                            )}

                            {/* Gantt chart */}
                            {(phase === 1 || phase === 3) && (
                                <div style={{
                                    flex: 1, overflow: 'hidden',
                                    position: 'relative', minHeight: 0,
                                }}>
                                    <GanttChart
                                        blocks={currentGanttBlocks}
                                        workers={workerStates}
                                        currentTick={currentTick}
                                        phase={phase === 3 ? 2 : 1}
                                        slotsPerDay={16}
                                        numWorkers={simConfig.num_workers}
                                        totalDays={totalDays}
                                    />
                                </div>
                            )}

                            {/* Animated task queue */}
                            {(phase === 1 || phase === 3) && (
                                <div style={{
                                    flexShrink: 0, height: 180,
                                    borderTop: '1px solid var(--rim)',
                                }}>
                                    <AnimatedTaskQueue
                                        queue={queueState}
                                        currentTick={currentTick}
                                    />
                                </div>
                            )}
                        </div>

                        {/* Right: ComparisonStrip */}
                        <div className="sim-right-panel">
                            <ComparisonStrip
                                phase={phase === 3 ? 2 : phase === 1 ? 1 : 0}
                                hhMetrics={hhMetrics}
                                workerCount={simConfig.num_workers}
                            />
                        </div>
                    </div>
                </div>
            </PageTransition>
        </>
    );
}