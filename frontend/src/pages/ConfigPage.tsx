/**
 * ConfigPage.tsx — Visual Overhaul
 * All logic, handlers, state, and API calls are IDENTICAL to the original.
 * Only styles, layout, and visual presentation changed.
 */
import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    RadarChart, PolarGrid, PolarAngleAxis, Radar,
    ResponsiveContainer, Tooltip as RechartsTooltip,
    LineChart, Line, XAxis, YAxis
} from 'recharts';
import {
    DEFAULT_SIM_CONFIG, validateSimConfig, computeTotalTicks, estimateRuntimeSeconds,
    type SimConfig, type ManualWorkerConfig, type InjectedTask
} from '../types/config';
import { useSimulationStore } from '../store/simulationStore';
import PriorityInjectionPanel from '../components/config/PriorityInjectionPanel';
import { PageTransition } from '../components/PageTransition';
import { RevealSection } from '../components/RevealSection';

// ─────────────────────────────────────────────────────────────────────────────
// CSS injected once at module level
// ─────────────────────────────────────────────────────────────────────────────
const STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --void:       #06080f;
  --base:       #090c15;
  --surface:    #0d1120;
  --raised:     #121629;
  --glass:      rgba(13,17,32,0.7);
  --rim:        rgba(255,255,255,0.06);
  --rim-hi:     rgba(255,255,255,0.12);
  --sol:        #e8ecf4;
  --dim:        #6b7694;
  --muted:      #3a4060;
  --blue:       #4f8ef7;
  --blue-glow:  rgba(79,142,247,0.22);
  --cyan:       #2dd4bf;
  --amber:      #f5a623;
  --amber-glow: rgba(245,166,35,0.18);
  --violet:     #a78bfa;
  --rose:       #fb7185;
  --emerald:    #34d399;
  --font-head:  'Syne', sans-serif;
  --font-body:  'DM Sans', sans-serif;
  --font-mono:  'DM Mono', monospace;
}

/* ── Grid noise texture overlay ── */
.cfg-root::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    radial-gradient(ellipse 80% 50% at 10% 20%, rgba(79,142,247,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 70% at 90% 80%, rgba(167,139,250,0.05) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 50% 50%, rgba(45,212,191,0.03) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* Fine grid */
.cfg-root::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

.cfg-root { position: relative; z-index: 1; }

/* ── Stepper ── */
.stepper {
  display: flex;
  align-items: center;
  margin-bottom: 36px;
  padding: 20px 28px;
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 14px;
  position: relative;
  overflow: hidden;
}
.stepper::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(79,142,247,0.04) 0%, transparent 60%);
  pointer-events: none;
}
.stepper-step {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
.stepper-circle {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 500;
  border: 1.5px solid var(--muted);
  color: var(--dim);
  background: var(--raised);
  transition: all 0.3s ease;
}
.stepper-circle.active {
  border-color: var(--blue);
  color: #fff;
  background: var(--blue);
  box-shadow: 0 0 16px var(--blue-glow);
}
.stepper-circle.complete {
  border-color: var(--emerald);
  color: var(--emerald);
  background: rgba(52,211,153,0.1);
}
.stepper-label {
  font-family: var(--font-body);
  font-size: 13px;
  white-space: nowrap;
}
.stepper-line {
  flex: 1;
  height: 1px;
  margin: 0 16px;
  transition: background 0.4s ease;
}

/* ── Section card ── */
.section-card {
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 16px;
  margin-bottom: 20px;
  overflow: hidden;
  transition: border-color 0.2s;
}
.section-card:hover { border-color: var(--rim-hi); }
.section-card-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 22px 28px 18px;
  border-bottom: 1px solid var(--rim);
  background: linear-gradient(135deg, rgba(79,142,247,0.03) 0%, transparent 60%);
}
.section-step-badge {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--blue), #6366f1);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 4px 12px rgba(79,142,247,0.35);
}
.section-title {
  font-family: var(--font-head);
  font-size: 15px;
  font-weight: 700;
  color: var(--sol);
  letter-spacing: -0.01em;
}
.section-card-body {
  padding: 24px 28px;
}

/* ── Form elements ── */
label {
  display: block;
  font-family: var(--font-body);
  font-size: 12px;
  font-weight: 500;
  color: var(--dim);
  letter-spacing: 0.03em;
  margin-bottom: 7px;
  text-transform: uppercase;
}
.cfg-input {
  width: 100%;
  background: var(--raised);
  border: 1px solid var(--rim);
  border-radius: 9px;
  padding: 10px 14px;
  color: var(--sol);
  font-family: var(--font-mono);
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  box-sizing: border-box;
}
.cfg-input:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(79,142,247,0.12), 0 0 16px rgba(79,142,247,0.08);
}
.cfg-select {
  width: 100%;
  background: var(--raised);
  border: 1px solid var(--rim);
  border-radius: 9px;
  padding: 10px 14px;
  color: var(--sol);
  font-family: var(--font-mono);
  font-size: 13px;
  outline: none;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236b7694' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 14px center;
  transition: border-color 0.2s;
}
.cfg-select:focus { border-color: var(--blue); outline: none; }

/* Range track */
input[type=range] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: var(--raised);
  border: 1px solid var(--rim);
  outline: none;
  cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--blue);
  box-shadow: 0 0 10px var(--blue-glow);
  cursor: pointer;
  transition: box-shadow 0.2s;
}
input[type=range]::-webkit-slider-thumb:hover {
  box-shadow: 0 0 16px rgba(79,142,247,0.5);
}
input[type=range].amber-track::-webkit-slider-thumb {
  background: var(--amber);
  box-shadow: 0 0 10px var(--amber-glow);
}

/* ── Segmented control ── */
.seg-control {
  display: flex;
  background: var(--raised);
  border: 1px solid var(--rim);
  border-radius: 9px;
  padding: 3px;
  gap: 2px;
}
.seg-btn {
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 7px;
  font-family: var(--font-body);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  background: transparent;
  color: var(--dim);
}
.seg-btn.active {
  background: var(--blue);
  color: #fff;
  box-shadow: 0 2px 10px rgba(79,142,247,0.35);
}
.seg-btn:not(.active):hover { background: var(--rim); color: var(--sol); }

/* ── Preview chips ── */
.preview-chips {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--rim);
  border-radius: 10px;
  overflow: hidden;
  margin-top: 20px;
}
.preview-chip {
  background: var(--raised);
  padding: 14px 10px;
  text-align: center;
}
.preview-chip-val {
  font-family: var(--font-mono);
  font-size: 20px;
  font-weight: 500;
  color: var(--blue);
  line-height: 1;
  margin-bottom: 5px;
}
.preview-chip-label {
  font-family: var(--font-body);
  font-size: 10px;
  color: var(--dim);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* ── Phase split bar ── */
.phase-bar-wrap {
  margin-top: 12px;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  display: flex;
  background: var(--raised);
}

/* ── Worker row ── */
.worker-row {
  background: var(--raised);
  border: 1px solid var(--rim);
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 10px;
  transition: border-color 0.2s;
}
.worker-row:hover { border-color: var(--rim-hi); }

/* ── Ghost button ── */
.btn-ghost {
  background: transparent;
  border: 1px solid var(--rim-hi);
  color: var(--dim);
  padding: 8px 16px;
  border-radius: 8px;
  font-family: var(--font-body);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.btn-ghost:hover {
  border-color: var(--blue);
  color: var(--blue);
  background: rgba(79,142,247,0.06);
}

/* ── Warning banner ── */
.warn-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 16px;
  background: rgba(245,166,35,0.06);
  border: 1px solid rgba(245,166,35,0.18);
  border-radius: 9px;
  margin-bottom: 20px;
}

/* ── Error banner ── */
.error-banner {
  padding: 14px 18px;
  background: rgba(251,113,133,0.07);
  border: 1px solid rgba(251,113,133,0.22);
  border-radius: 10px;
  color: var(--rose);
  font-family: var(--font-mono);
  font-size: 12px;
  margin-bottom: 20px;
  line-height: 1.6;
}

/* ── Submit button ── */
.submit-btn {
  position: relative;
  width: 100%;
  height: 54px;
  background: linear-gradient(135deg, #3b7cf7, #6366f1);
  color: #fff;
  font-family: var(--font-head);
  font-weight: 700;
  font-size: 15px;
  letter-spacing: 0.02em;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(79,142,247,0.4);
  transition: box-shadow 0.2s, opacity 0.2s;
}
.submit-btn::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 200%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  animation: shimmer 2.5s ease-in-out infinite;
}
.submit-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.submit-btn:disabled::before { animation: none; }

@keyframes shimmer {
  0%   { left: -100%; }
  100% { left: 100%;  }
}

/* ── Right panel cards ── */
.panel-card {
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 14px;
  padding: 22px;
  margin-bottom: 16px;
}
.panel-label {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 16px;
}

/* ── Phase timeline ── */
.phase-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 10px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.2s;
  margin-bottom: 4px;
}
.phase-item:hover, .phase-item.open {
  background: var(--raised);
  border-color: rgba(79,142,247,0.2);
  box-shadow: 0 0 16px rgba(79,142,247,0.06);
}
.phase-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  flex-shrink: 0;
}
.phase-connector {
  width: 1px;
  height: 10px;
  background: var(--rim);
  margin: 2px 0 2px 22px;
}

/* ── Policy legend ── */
.policy-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 7px 10px;
  border-radius: 8px;
  transition: background 0.15s;
}
.policy-row:hover { background: var(--raised); }
.policy-row.dqn {
  background: rgba(245,166,35,0.04);
  border-left: 2px solid var(--amber);
  padding-left: 10px;
  border-radius: 0 8px 8px 0;
}

/* ── Tooltip ── */
.recharts-tooltip-wrapper { outline: none !important; }

/* ── Skill preview rows ── */
.skill-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 0;
  border-bottom: 1px solid var(--rim);
}
.skill-row:last-child { border-bottom: none; }

/* ── Responsive ── */
@media (max-width: 900px) {
  .cfg-two-col { grid-template-columns: 1fr !important; }
  .cfg-right-sticky { position: static !important; }
}
@media (max-width: 600px) {
  .preview-chips { grid-template-columns: repeat(2, 1fr); }
  .stepper-label { display: none; }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (IDENTICAL to original)
// ─────────────────────────────────────────────────────────────────────────────
function seededRandom(seed: number, n: number): number {
    let s = seed;
    for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) & 0xffffffff; }
    return (s >>> 0) / 0xffffffff;
}

function generateWorkerSkillPreview(seed: number, numWorkers: number) {
    return Array.from({ length: numWorkers }, (_, i) => ({
        subject: `W${i + 1}`,
        skill: 0.5 + seededRandom(seed, i * 3 + 1) * 1.0,
        productivity: 0.7 + seededRandom(seed, i * 3 + 2) * 0.6,
        fatigue_resist: 0.7 + seededRandom(seed, i * 3 + 3) * 0.6,
    }));
}

function generateArrivalSparkline(cfg: SimConfig): { day: number; tasks: number }[] {
    const totalDays = cfg.days_phase1 + cfg.days_phase2;
    return Array.from({ length: totalDays }, (_, day) => {
        let tasks = 0;
        if (cfg.arrival_distribution === 'poisson') {
            tasks = (cfg.arrival_params?.rate ?? 3.5) * (0.7 + seededRandom(cfg.seed + day, 7) * 0.6);
        } else if (cfg.arrival_distribution === 'uniform') {
            const min = cfg.arrival_params?.min_per_day ?? 2;
            const max = cfg.arrival_params?.max_per_day ?? 6;
            tasks = min + seededRandom(cfg.seed + day, 9) * (max - min);
        } else if (cfg.arrival_distribution === 'burst') {
            const mult = cfg.arrival_params?.burst_multiplier ?? 3;
            const isBurst = day % 5 === 2;
            tasks = isBurst ? 3.5 * mult : 3.5;
        } else {
            tasks = cfg.arrival_params?.daily_overrides?.[day] ?? 3.5;
        }
        return { day: day + 1, tasks: Math.max(0, Math.round(tasks * 10) / 10) };
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress Stepper (IDENTICAL logic, new visuals)
// ─────────────────────────────────────────────────────────────────────────────
const STEPS = ['Simulation Parameters', 'Worker Setup', 'Task Configuration'];

function ProgressStepper({ active }: { active: number }) {
    return (
        <div className="stepper">
            {STEPS.map((label, i) => {
                const state = i < active ? 'complete' : i === active ? 'active' : 'inactive';
                return (
                    <React.Fragment key={i}>
                        <div className="stepper-step">
                            <div className={`stepper-circle ${state}`}>
                                {state === 'complete' ? '✓' : i + 1}
                            </div>
                            <span className="stepper-label" style={{
                                fontFamily: 'var(--font-body)',
                                color: state === 'active' ? 'var(--sol)'
                                    : state === 'complete' ? 'var(--emerald)'
                                        : 'var(--muted)',
                                fontWeight: state === 'active' ? 600 : 400,
                            }}>{label}</span>
                        </div>
                        {i < STEPS.length - 1 && (
                            <div className="stepper-line" style={{
                                background: i < active ? 'var(--emerald)' : 'var(--rim)',
                            }} />
                        )}
                    </React.Fragment>
                );
            })}
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Section Card
// ─────────────────────────────────────────────────────────────────────────────
function SectionCard({ step, title, children }: {
    step: number; title: string; children: React.ReactNode;
}) {
    return (
        <div className="section-card">
            <div className="section-card-header">
                <div className="section-step-badge">{step}</div>
                <div className="section-title">{title}</div>
            </div>
            <div className="section-card-body">{children}</div>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker Row (IDENTICAL logic, new visuals)
// ─────────────────────────────────────────────────────────────────────────────
function WorkerRow({ worker, index, onChange, onDelete }: {
    worker: ManualWorkerConfig;
    index: number;
    onChange: (i: number, u: ManualWorkerConfig) => void;
    onDelete: (i: number) => void;
}) {
    return (
        <div className="worker-row">
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                <span style={{
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--amber)', fontSize: 11,
                    letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 500,
                }}>Worker {index + 1}</span>
                <button className="btn-ghost" onClick={() => onDelete(index)}
                    style={{
                        marginLeft: 'auto', fontSize: 11, padding: '4px 10px',
                        color: 'var(--rose)', borderColor: 'rgba(251,113,133,0.25)',
                    }}>
                    Remove
                </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div>
                    <label>Name</label>
                    <input className="cfg-input" value={worker.name}
                        onChange={e => onChange(index, { ...worker, name: e.target.value })} />
                </div>
                <div>
                    <label>Productivity Rate&nbsp;
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>
                            {worker.productivity_rate.toFixed(2)}
                        </span>
                    </label>
                    <input type="range" className="amber-track" min={0.5} max={1.5} step={0.01}
                        value={worker.productivity_rate}
                        onChange={e => onChange(index, { ...worker, productivity_rate: Number(e.target.value) })} />
                </div>
                <div>
                    <label>Skill Level&nbsp;
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>
                            {worker.skill_level.toFixed(2)}
                        </span>
                    </label>
                    <input type="range" min={0} max={1} step={0.01}
                        value={worker.skill_level}
                        onChange={e => onChange(index, { ...worker, skill_level: Number(e.target.value) })} />
                </div>
                <div>
                    <label>Fatigue Sensitivity&nbsp;
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--rose)' }}>
                            {worker.fatigue_sensitivity.toFixed(2)}
                        </span>
                    </label>
                    <input type="range" className="amber-track" min={0.05} max={0.3} step={0.01}
                        value={worker.fatigue_sensitivity}
                        onChange={e => onChange(index, { ...worker, fatigue_sensitivity: Number(e.target.value) })} />
                </div>
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase Timeline (IDENTICAL logic, new visuals)
// ─────────────────────────────────────────────────────────────────────────────
// SVG icons — no emoji, no encoding issues, always renders correctly
const PhaseIcons = {
    observe: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
        </svg>
    ),
    train: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
        </svg>
    ),
    schedule: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
        </svg>
    ),
};

const PHASE_STEPS = [
    {
        icon: PhaseIcons.observe,
        label: 'Observe',
        color: '#4f8ef7',
        bg: 'rgba(79,142,247,0.08)',
        description: 'Baseline policies (Greedy, Skill, FIFO, Hybrid, Random) run sequentially. DQN passively records all transitions into its replay buffer.',
    },
    {
        icon: PhaseIcons.train,
        label: 'Train',
        color: '#a78bfa',
        bg: 'rgba(167,139,250,0.08)',
        description: 'DQN trains offline on the collected replay buffer — thousands of gradient steps to bootstrap Q-values before live scheduling.',
    },
    {
        icon: PhaseIcons.schedule,
        label: 'Schedule',
        color: '#f5a623',
        bg: 'rgba(245,166,35,0.08)',
        description: 'DQN takes control on a fresh environment, scheduling tasks in real-time with online learning and epsilon-greedy exploration.',
    },
];

const POLICY_LEGEND = [
    { name: 'Greedy', color: '#4f8ef7', desc: 'Always assigns the most urgent task to the best-match worker.' },
    { name: 'Skill', color: '#2dd4bf', desc: 'Maximizes skill-to-task alignment, ignoring urgency.' },
    { name: 'FIFO', color: '#34d399', desc: 'First-in-first-out: assigns tasks in arrival order.' },
    { name: 'Hybrid', color: '#a78bfa', desc: 'Balances urgency + skill with a weighted scoring function.' },
    { name: 'Random', color: '#fb7185', desc: 'Random baseline — lower bound for comparison.' },
    { name: 'DQN', color: '#f5a623', desc: 'Reinforcement-learning agent trained on observed transitions.' },
];

function PhaseTimeline() {
    const [expanded, setExpanded] = useState<number | null>(null);
    return (
        <div>
            {PHASE_STEPS.map((step, i) => (
                <div key={step.label}>
                    <div
                        className={`phase-item ${expanded === i ? 'open' : ''}`}
                        onClick={() => setExpanded(expanded === i ? null : i)}
                    >
                        <div className="phase-icon" style={{ background: step.bg, color: step.color }}>
                            {step.icon}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{
                                fontFamily: 'var(--font-head)',
                                fontWeight: 700, fontSize: 13,
                                color: expanded === i ? step.color : 'var(--sol)',
                                transition: 'color 0.2s',
                            }}>{step.label}</div>
                        </div>
                        <span style={{ color: 'var(--muted)', display: 'flex', alignItems: 'center' }}>
                            {expanded === i
                                ? <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="18 15 12 9 6 15" /></svg>
                                : <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="6 9 12 15 18 9" /></svg>
                            }
                        </span>
                    </div>
                    <AnimatePresence>
                        {expanded === i && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.22, ease: 'easeOut' }}
                                style={{ overflow: 'hidden' }}
                            >
                                <div style={{
                                    padding: '10px 14px 14px 58px',
                                    fontFamily: 'var(--font-body)',
                                    fontSize: 12, color: 'var(--dim)',
                                    lineHeight: 1.65,
                                }}>
                                    {step.description}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                    {i < PHASE_STEPS.length - 1 && (
                        <div className="phase-connector" />
                    )}
                </div>
            ))}
        </div>
    );
}

function PolicyLegend() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {POLICY_LEGEND.map(p => (
                <div key={p.name} className={`policy-row ${p.name === 'DQN' ? 'dqn' : ''}`}>
                    <div style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: p.color, flexShrink: 0, marginTop: 3,
                    }} />
                    <div style={{ minWidth: 0 }}>
                        <div style={{
                            fontFamily: 'var(--font-mono)', fontSize: 12,
                            fontWeight: 500, color: p.color, lineHeight: 1.3,
                        }}>{p.name}</div>
                        <div style={{
                            fontFamily: 'var(--font-body)', fontSize: 11,
                            color: 'var(--muted)', lineHeight: 1.4, marginTop: 1,
                        }}>{p.desc}</div>
                    </div>
                </div>
            ))}
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tooltip style
// ─────────────────────────────────────────────────────────────────────────────
const TOOLTIP_STYLE = {
    background: '#0d1120',
    border: '1px solid rgba(255,255,255,0.08)',
    fontSize: 11,
    fontFamily: 'DM Mono, monospace',
    borderRadius: 8,
    color: '#e8ecf4',
};

// ─────────────────────────────────────────────────────────────────────────────
// Main ConfigPage — ALL LOGIC IDENTICAL TO ORIGINAL
// ─────────────────────────────────────────────────────────────────────────────
export default function ConfigPage() {
    const navigate = useNavigate();
    const { startSimulation } = useSimulationStore();
    const [cfg, setCfg] = useState<SimConfig>(DEFAULT_SIM_CONFIG);
    const [errors, setErrors] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(false);
    const [pendingTasks, setPendingTasks] = useState<InjectedTask[]>([]);

    const totalTicks = computeTotalTicks(cfg);
    const estimatedSecs = estimateRuntimeSeconds(cfg);
    const skillPreview = useMemo(() =>
        generateWorkerSkillPreview(cfg.worker_seed, cfg.num_workers),
        [cfg.worker_seed, cfg.num_workers]
    );
    const sparklineData = useMemo(() => generateArrivalSparkline(cfg), [cfg]);

    const update = (patch: Partial<SimConfig>) => setCfg(prev => ({ ...prev, ...patch }));

    const addWorker = () => {
        const w: ManualWorkerConfig = {
            name: `Worker ${cfg.manual_workers.length + 1}`,
            skill_level: 0.7, productivity_rate: 1.0,
            fatigue_sensitivity: 0.18, task_type_efficiencies: {}
        };
        update({ manual_workers: [...cfg.manual_workers, w] });
    };
    const updateWorker = (i: number, w: ManualWorkerConfig) => {
        const workers = [...cfg.manual_workers];
        workers[i] = w;
        update({ manual_workers: workers });
    };
    const deleteWorker = (i: number) => {
        update({ manual_workers: cfg.manual_workers.filter((_, idx) => idx !== i) });
    };

    const handleSubmit = async () => {
        const errs = validateSimConfig(cfg);
        if (Object.keys(errs).length > 0) { setErrors(errs); return; }
        setErrors({});
        setLoading(true);
        try {
            const payload = { ...cfg, injected_tasks: pendingTasks };
            const res = await fetch('/api/initialize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) {
                const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
                throw new Error(body.error ?? `API error ${res.status}`);
            }
            useSimulationStore.getState().reset();
            startSimulation(cfg);
            navigate('/simulation');
        } catch (e) {
            setErrors({ submit: String(e) });
        } finally {
            setLoading(false);
        }
    };

    const G2: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 };

    return (
        <>
            {/* Inject styles once */}
            <style>{STYLES}</style>

            <PageTransition>
                <div className="cfg-root" style={{
                    minHeight: 'calc(100vh - 56px)',
                    background: 'var(--base)',
                    overflowY: 'auto',
                    padding: '40px 24px 80px',
                }}>
                    <div style={{ maxWidth: 1180, margin: '0 auto', position: 'relative', zIndex: 1 }}>

                        {/* ── Page header ── */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                            style={{ marginBottom: 40, textAlign: 'center' }}
                        >
                            <div style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: 8,
                                padding: '5px 14px',
                                background: 'rgba(79,142,247,0.08)',
                                border: '1px solid rgba(79,142,247,0.2)',
                                borderRadius: 100,
                                marginBottom: 16,
                            }}>
                                <div style={{
                                    width: 6, height: 6, borderRadius: '50%',
                                    background: 'var(--blue)',
                                    boxShadow: '0 0 8px var(--blue)',
                                    animation: 'pulse 2s ease-in-out infinite',
                                }} />
                                <span style={{
                                    fontFamily: 'var(--font-mono)',
                                    fontSize: 10, color: 'var(--blue)',
                                    letterSpacing: '0.15em', textTransform: 'uppercase',
                                }}>DQN Workforce Scheduler</span>
                            </div>
                            <h1 style={{
                                fontFamily: 'var(--font-head)',
                                fontSize: 'clamp(1.6rem, 3vw, 2.2rem)',
                                fontWeight: 800,
                                color: 'var(--sol)',
                                letterSpacing: '-0.03em',
                                lineHeight: 1.15,
                                marginBottom: 10,
                            }}>
                                Simulation Configuration
                            </h1>
                            <p style={{
                                fontFamily: 'var(--font-body)',
                                color: 'var(--dim)', fontSize: 14,
                                maxWidth: 480, margin: '0 auto',
                                lineHeight: 1.6,
                            }}>
                                Configure all parameters before initializing the two-phase DQN scheduling run.
                            </p>
                        </motion.div>

                        {/* ── Two-column layout ── */}
                        <div className="cfg-two-col" style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 340px',
                            gap: 28, alignItems: 'start',
                        }}>

                            {/* ════════════════════════════════════
                                LEFT COLUMN
                            ════════════════════════════════════ */}
                            <div>
                                <motion.div
                                    initial={{ opacity: 0, y: 16 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.45, delay: 0.1 }}
                                >
                                    <ProgressStepper active={0} />
                                </motion.div>

                                {/* SECTION 1 — Simulation Parameters */}
                                <RevealSection>
                                    <SectionCard step={1} title="Simulation Parameters">
                                        <div className="warn-banner">
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                                                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                                            </svg>
                                            <span style={{
                                                fontFamily: 'var(--font-mono)', fontSize: 11,
                                                color: 'var(--amber)', letterSpacing: '0.05em',
                                            }}>
                                                HARD CONSTRAINT: Workday fixed at 8h (16 x 30min slots). Cannot be modified.
                                            </span>
                                        </div>

                                        <div style={G2}>
                                            <div>
                                                <label>Simulation Days&nbsp;
                                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--blue)' }}>
                                                        {cfg.sim_days}
                                                    </span>
                                                </label>
                                                <input className="cfg-input" type="number" min={1} max={365}
                                                    value={cfg.sim_days}
                                                    onChange={e => {
                                                        const d = Math.max(1, Math.min(365, Number(e.target.value)));
                                                        const p1 = Math.max(1, Math.round(d * cfg.phase1_fraction));
                                                        const p2 = Math.max(1, d - p1);
                                                        update({ sim_days: d, days_phase1: p1, days_phase2: p2 });
                                                    }} />
                                            </div>
                                            <div>
                                                <label>Random Seed</label>
                                                <input className="cfg-input" type="number" value={cfg.seed}
                                                    onChange={e => update({ seed: Number(e.target.value) })} />
                                            </div>
                                        </div>

                                        {/* Phase split */}
                                        <div style={{ marginTop: 22 }}>
                                            <label>
                                                Phase 1 Observation %&nbsp;
                                                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--blue)' }}>
                                                    {Math.round(cfg.phase1_fraction * 100)}%
                                                </span>
                                                <span style={{
                                                    fontFamily: 'var(--font-mono)', fontSize: 11,
                                                    color: 'var(--muted)', marginLeft: 8,
                                                }}>
                                                    ({cfg.days_phase1}d baseline / {cfg.days_phase2}d DQN)
                                                </span>
                                            </label>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--amber)', minWidth: 28 }}>40%</span>
                                                <input type="range" min={40} max={80} step={5}
                                                    value={Math.round(cfg.phase1_fraction * 100)}
                                                    style={{ flex: 1 }}
                                                    onChange={e => {
                                                        const frac = Number(e.target.value) / 100;
                                                        const p1 = Math.max(1, Math.round(cfg.sim_days * frac));
                                                        const p2 = Math.max(1, cfg.sim_days - p1);
                                                        update({ phase1_fraction: frac, days_phase1: p1, days_phase2: p2 });
                                                    }} />
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--blue)', minWidth: 28 }}>80%</span>
                                            </div>
                                            <div className="phase-bar-wrap">
                                                <motion.div
                                                    animate={{ flex: cfg.days_phase1 }}
                                                    transition={{ duration: 0.3 }}
                                                    style={{ background: 'linear-gradient(90deg, var(--violet), var(--blue))' }}
                                                />
                                                <motion.div
                                                    animate={{ flex: cfg.days_phase2 }}
                                                    transition={{ duration: 0.3 }}
                                                    style={{ background: 'var(--amber)', opacity: 0.75 }}
                                                />
                                            </div>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5 }}>
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--blue)' }}>
                                                    ■ Phase 1 — Baseline
                                                </span>
                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--amber)' }}>
                                                    ■ Phase 2 — DQN
                                                </span>
                                            </div>
                                        </div>

                                        {/* Live preview chips */}
                                        <div className="preview-chips">
                                            {[
                                                { label: 'Phase 1 Days', value: cfg.days_phase1 },
                                                { label: 'Phase 2 Days', value: cfg.days_phase2 },
                                                { label: 'Est. Tasks', value: Math.round(cfg.tasks_per_day * (cfg.days_phase1 + cfg.days_phase2)) },
                                                { label: 'Est. Buffer', value: Math.min(12000, Math.max(2000, Math.round(cfg.days_phase1 * cfg.tasks_per_day * 5 * 1.3))) },
                                            ].map(chip => (
                                                <div className="preview-chip" key={chip.label}>
                                                    <div className="preview-chip-val">{chip.value.toLocaleString()}</div>
                                                    <div className="preview-chip-label">{chip.label}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </SectionCard>
                                </RevealSection>

                                {/* SECTION 2 — Worker Setup */}
                                <RevealSection>
                                    <SectionCard step={2} title="Worker Setup">
                                        <div style={{ marginBottom: 20 }}>
                                            <label style={{ marginBottom: 10 }}>Worker Generation Mode</label>
                                            <div className="seg-control">
                                                <button
                                                    className={`seg-btn ${cfg.worker_mode === 'auto' ? 'active' : ''}`}
                                                    onClick={() => update({ worker_mode: 'auto' })}
                                                >
                                                    Auto-Generate (Seed)
                                                </button>
                                                <button
                                                    className={`seg-btn ${cfg.worker_mode === 'manual' ? 'active' : ''}`}
                                                    onClick={() => update({ worker_mode: 'manual' })}
                                                >
                                                    Manual Configuration
                                                </button>
                                            </div>
                                        </div>

                                        {cfg.worker_mode === 'auto' ? (
                                            <div style={G2}>
                                                <div>
                                                    <div style={G2}>
                                                        <div>
                                                            <label>Worker Seed</label>
                                                            <input className="cfg-input" type="number" value={cfg.worker_seed}
                                                                onChange={e => update({ worker_seed: Number(e.target.value) })} />
                                                        </div>
                                                        <div>
                                                            <label>Num Workers&nbsp;
                                                                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--blue)' }}>
                                                                    {cfg.num_workers}
                                                                </span>
                                                            </label>
                                                            <input className="cfg-input" type="number" min={2} max={20}
                                                                value={cfg.num_workers}
                                                                onChange={e => update({ num_workers: Number(e.target.value) })} />
                                                            {errors.num_workers && (
                                                                <p style={{ color: 'var(--rose)', fontSize: 11, marginTop: 4 }}>
                                                                    {errors.num_workers}
                                                                </p>
                                                            )}
                                                        </div>
                                                    </div>

                                                    <div style={{ marginTop: 16 }}>
                                                        <label>
                                                            Max Worker Load&nbsp;
                                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--amber)' }}>
                                                                {cfg.max_worker_load}
                                                            </span>
                                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--muted)', marginLeft: 6 }}>
                                                                tasks/worker
                                                            </span>
                                                        </label>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
                                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', minWidth: 14 }}>3</span>
                                                            <input type="range" className="amber-track" min={3} max={15} step={1}
                                                                value={cfg.max_worker_load}
                                                                style={{ flex: 1 }}
                                                                onChange={e => update({ max_worker_load: Number(e.target.value) })} />
                                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', minWidth: 14 }}>15</span>
                                                        </div>
                                                    </div>

                                                    {/* Skill preview table */}
                                                    <div style={{
                                                        marginTop: 16,
                                                        padding: '12px 14px',
                                                        background: 'var(--void)',
                                                        border: '1px solid var(--rim)',
                                                        borderRadius: 9,
                                                    }}>
                                                        {skillPreview.map((w, i) => (
                                                            <div className="skill-row" key={i}>
                                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--amber)', minWidth: 28 }}>
                                                                    {w.subject}
                                                                </span>
                                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--blue)' }}>
                                                                    skill={w.skill.toFixed(2)}
                                                                </span>
                                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--dim)' }}>
                                                                    prod={w.productivity.toFixed(2)}
                                                                </span>
                                                                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>
                                                                    fat={w.fatigue_resist.toFixed(2)}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>

                                                <div>
                                                    <div style={{
                                                        fontFamily: 'var(--font-body)', fontSize: 11,
                                                        color: 'var(--dim)', marginBottom: 10,
                                                        textTransform: 'uppercase', letterSpacing: '0.06em',
                                                    }}>
                                                        Skill Distribution
                                                    </div>
                                                    <ResponsiveContainer width="100%" height={200}>
                                                        <RadarChart data={skillPreview}>
                                                            <PolarGrid stroke="rgba(255,255,255,0.05)" />
                                                            <PolarAngleAxis dataKey="subject" tick={{
                                                                fill: 'var(--dim)', fontSize: 11,
                                                                fontFamily: 'DM Mono, monospace',
                                                            }} />
                                                            <Radar name="Skill" dataKey="skill"
                                                                stroke="var(--amber)" fill="var(--amber)"
                                                                fillOpacity={0.15} isAnimationActive={false} />
                                                            <Radar name="Productivity" dataKey="productivity"
                                                                stroke="var(--blue)" fill="var(--blue)"
                                                                fillOpacity={0.1} isAnimationActive={false} />
                                                            <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                                        </RadarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        ) : (
                                            <div>
                                                {errors.manual_workers && (
                                                    <p style={{ color: 'var(--rose)', fontSize: 12, marginBottom: 12 }}>
                                                        {errors.manual_workers}
                                                    </p>
                                                )}
                                                {cfg.manual_workers.map((w, i) => (
                                                    <WorkerRow key={i} worker={w} index={i}
                                                        onChange={updateWorker} onDelete={deleteWorker} />
                                                ))}
                                                <button className="btn-ghost" onClick={addWorker}
                                                    style={{ marginTop: 8 }}>
                                                    + Add Worker
                                                </button>
                                            </div>
                                        )}
                                    </SectionCard>
                                </RevealSection>

                                {/* SECTION 3 — Task Configuration */}
                                <RevealSection>
                                    <SectionCard step={3} title="Task Configuration">
                                        <div style={G2}>
                                            <div>
                                                <div style={G2}>
                                                    <div>
                                                        <label>Distribution Type</label>
                                                        <select className="cfg-select"
                                                            value={cfg.arrival_distribution}
                                                            onChange={e => update({
                                                                arrival_distribution: e.target.value as SimConfig['arrival_distribution'],
                                                                arrival_params: {},
                                                            })}>
                                                            <option value="poisson">Poisson</option>
                                                            <option value="uniform">Uniform</option>
                                                            <option value="burst">Burst</option>
                                                            <option value="custom">Custom</option>
                                                        </select>
                                                    </div>
                                                    <div>
                                                        <label>Task Cap (max total)</label>
                                                        <input className="cfg-input" type="number" min={10} max={2000}
                                                            value={cfg.task_count}
                                                            onChange={e => update({ task_count: Number(e.target.value) })} />
                                                        {errors.task_count && (
                                                            <p style={{ color: 'var(--rose)', fontSize: 11, marginTop: 4 }}>
                                                                {errors.task_count}
                                                            </p>
                                                        )}
                                                    </div>
                                                </div>

                                                <div style={{ marginTop: 16 }}>
                                                    <label>
                                                        Tasks per Day&nbsp;
                                                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--blue)' }}>
                                                            {cfg.tasks_per_day}
                                                        </span>
                                                    </label>
                                                    <input type="range" min={1} max={20} step={1}
                                                        value={cfg.tasks_per_day}
                                                        style={{ width: '100%', marginTop: 8 }}
                                                        onChange={e => update({ tasks_per_day: Number(e.target.value) })} />
                                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--muted)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                                                        <span>1/day</span>
                                                        <span style={{ color: 'var(--blue)' }}>
                                                            ~{cfg.tasks_per_day * (cfg.days_phase1 + cfg.days_phase2)} total
                                                        </span>
                                                        <span>20/day</span>
                                                    </div>
                                                </div>

                                                <div style={{ marginTop: 16 }}>
                                                    {cfg.arrival_distribution === 'poisson' && (
                                                        <div>
                                                            <label>Mean Tasks per Day (λ)</label>
                                                            <input className="cfg-input" type="number" step={0.5} min={0.5} max={20}
                                                                value={cfg.arrival_params?.rate ?? 3.5}
                                                                onChange={e => update({ arrival_params: { rate: Number(e.target.value) } })} />
                                                        </div>
                                                    )}
                                                    {cfg.arrival_distribution === 'uniform' && (
                                                        <div style={G2}>
                                                            <div>
                                                                <label>Min Tasks/Day</label>
                                                                <input className="cfg-input" type="number" min={0}
                                                                    value={cfg.arrival_params?.min_per_day ?? 1}
                                                                    onChange={e => update({ arrival_params: { ...cfg.arrival_params, min_per_day: Number(e.target.value) } })} />
                                                            </div>
                                                            <div>
                                                                <label>Max Tasks/Day</label>
                                                                <input className="cfg-input" type="number"
                                                                    value={cfg.arrival_params?.max_per_day ?? 8}
                                                                    onChange={e => update({ arrival_params: { ...cfg.arrival_params, max_per_day: Number(e.target.value) } })} />
                                                            </div>
                                                        </div>
                                                    )}
                                                    {cfg.arrival_distribution === 'burst' && (
                                                        <div>
                                                            <label>Burst Multiplier (every 5th day)</label>
                                                            <input className="cfg-input" type="number" step={0.5} min={1.5} max={10}
                                                                value={cfg.arrival_params?.burst_multiplier ?? 3}
                                                                onChange={e => update({ arrival_params: { burst_multiplier: Number(e.target.value) } })} />
                                                        </div>
                                                    )}
                                                    {cfg.arrival_distribution === 'custom' && (
                                                        <div>
                                                            <label>Daily Task Counts (comma-separated)</label>
                                                            <input className="cfg-input" placeholder="e.g. 2,5,3,8,1,…"
                                                                onChange={e => {
                                                                    const vals = e.target.value.split(',').map(Number).filter(n => !isNaN(n));
                                                                    update({ arrival_params: { daily_overrides: vals } });
                                                                }} />
                                                        </div>
                                                    )}
                                                </div>
                                            </div>

                                            <div>
                                                <div style={{
                                                    fontFamily: 'var(--font-body)', fontSize: 11,
                                                    color: 'var(--dim)', marginBottom: 10,
                                                    textTransform: 'uppercase', letterSpacing: '0.06em',
                                                }}>
                                                    Arrival Rate Preview
                                                </div>
                                                <ResponsiveContainer width="100%" height={180}>
                                                    <LineChart data={sparklineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                                        <XAxis dataKey="day" tick={{
                                                            fontSize: 10, fill: 'var(--muted)',
                                                            fontFamily: 'DM Mono, monospace',
                                                        }} />
                                                        <YAxis tick={{
                                                            fontSize: 10, fill: 'var(--muted)',
                                                            fontFamily: 'DM Mono, monospace',
                                                        }} />
                                                        <Line type="monotone" dataKey="tasks"
                                                            stroke="var(--blue)" dot={false}
                                                            strokeWidth={2} isAnimationActive={false} />
                                                        <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                    </SectionCard>
                                </RevealSection>

                                {/* SECTION 4 — Task Injection (UNCHANGED) */}
                                <RevealSection>
                                    <PriorityInjectionPanel
                                        pendingTasks={pendingTasks}
                                        onAdd={(task) => setPendingTasks(prev => [...prev, task])}
                                        onRemove={(id) => setPendingTasks(prev => prev.filter(t => t.task_id !== id))}
                                    />
                                </RevealSection>

                                {/* Error */}
                                {errors.submit && (
                                    <div className="error-banner">
                                        <strong>Initialization failed:</strong> {errors.submit}
                                    </div>
                                )}

                                {/* Submit */}
                                <div style={{ paddingBottom: 48 }}>
                                    <motion.button
                                        id="initialize-simulation-btn"
                                        className="submit-btn"
                                        onClick={handleSubmit}
                                        disabled={loading}
                                        whileHover={!loading ? {
                                            scale: 1.01,
                                            boxShadow: '0 8px 40px rgba(79,142,247,0.55)',
                                        } : {}}
                                        whileTap={!loading ? { scale: 0.99 } : {}}
                                        style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}
                                    >
                                        {loading ? (
                                            <>
                                                <motion.div
                                                    animate={{ rotate: 360 }}
                                                    transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                                                    style={{
                                                        width: 18, height: 18, borderRadius: '50%',
                                                        border: '2px solid rgba(255,255,255,0.25)',
                                                        borderTopColor: '#fff',
                                                    }}
                                                />
                                                Initializing Simulation…
                                            </>
                                        ) : (
                                            <>
                                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                                                </svg>
                                                Initialize Simulation
                                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.7 }}>
                                                    <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
                                                </svg>
                                            </>
                                        )}
                                    </motion.button>
                                    <p style={{
                                        textAlign: 'center', marginTop: 10,
                                        color: 'var(--muted)', fontSize: 11,
                                        fontFamily: 'var(--font-mono)',
                                    }}>
                                        Begins {cfg.days_phase1 + cfg.days_phase2}-day simulation · {cfg.num_workers} workers
                                    </p>
                                </div>
                            </div>

                            {/* ════════════════════════════════════
                                RIGHT COLUMN (sticky)
                            ════════════════════════════════════ */}
                            <div className="cfg-right-sticky" style={{ position: 'sticky', top: 24 }}>
                                <motion.div
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ duration: 0.5, delay: 0.2 }}
                                >
                                    <div className="panel-card">
                                        <div className="panel-label">Simulation Phases</div>
                                        <PhaseTimeline />
                                    </div>

                                    <div className="panel-card">
                                        <div className="panel-label">Policy Legend</div>
                                        <PolicyLegend />
                                    </div>
                                </motion.div>
                            </div>

                        </div>
                    </div>
                </div>
            </PageTransition>
        </>
    );
}