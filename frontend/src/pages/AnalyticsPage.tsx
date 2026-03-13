/**
 * AnalyticsPage.tsx — Command Intelligence Dashboard
 *
 * Visual redesign: cinematic dark data-journalism aesthetic.
 * All logic, hooks, store bindings, and data flows UNCHANGED.
 * No emoji, no encoding issues — SVG icons only.
 */
import React, { useState, useMemo, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Cell, Legend,
    RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
    LineChart, Line, LabelList, Area, AreaChart, ReferenceLine,
} from 'recharts';
import { useShallow } from 'zustand/react/shallow';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { useSimulationStore } from '../store/simulationStore';
import { useAllPolicyMetrics, useDailyThroughputByPolicy, useLatenessPerDay } from '../hooks/useSimulation';
import { buildRadarData, POLICY_COLORS } from '../types/metrics';
import type { SummaryCardData } from '../types/metrics';
import ReadmeGeneratorPanel from '../components/readme/ReadmeGeneratorPanel';
import { PageTransition } from '../components/PageTransition';

// ── Design tokens ─────────────────────────────────────────────────────────────
const POLICY_MAP: Record<string, string> = {
    DQN:    '#e3b341',
    Greedy: '#4493f8',
    Skill:  '#2dd4bf',
    FIFO:   '#3fb950',
    Hybrid: '#a78bfa',
    Random: '#f85149',
    ...POLICY_COLORS,
};

const TOOLTIP_STYLE = {
    background: '#0d1117',
    border: '1px solid rgba(255,255,255,0.08)',
    fontSize: 11,
    fontFamily: 'DM Mono, monospace',
    borderRadius: 10,
    boxShadow: '0 16px 48px rgba(0,0,0,0.7)',
    color: '#e6edf3',
    padding: '10px 14px',
};

// ── Page styles ───────────────────────────────────────────────────────────────
const AP_STYLES = `
.ap-root {
  min-height: 100vh;
  background: var(--void);
  overflow-y: auto;
  position: relative;
}

/* Cinematic background gradient */
.ap-root::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 100% 60% at 50% -10%, rgba(68,147,248,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 90%, rgba(167,139,250,0.04) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* Subtle dot grid */
.ap-root::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image: radial-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
}

.ap-content {
  position: relative;
  z-index: 1;
  max-width: 1340px;
  margin: 0 auto;
  padding: 40px 28px 80px;
}

/* ── Hero KPI cards ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}
@media (max-width: 900px) { .kpi-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi-card {
  position: relative;
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 14px;
  padding: 22px 24px;
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
  border-color: var(--rim-hi);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  border-radius: 14px 14px 0 0;
}
.kpi-card.dqn::before { background: var(--amber); box-shadow: 0 0 12px var(--amber); }
.kpi-card.primary::before { background: var(--blue); }
.kpi-card.success::before { background: var(--emerald); }
.kpi-card.neutral::before { background: var(--rim-hi); }

.kpi-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}
.kpi-val {
  font-family: var(--font-head);
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1;
  margin-bottom: 8px;
}
.kpi-delta {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 100px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
}
.kpi-delta.up   { background: rgba(63,185,80,0.1);  color: var(--emerald); }
.kpi-delta.down { background: rgba(248,81,73,0.1);  color: var(--rose);    }
.kpi-delta.neutral { background: rgba(255,255,255,0.04); color: var(--dim); }

/* ── Chart card ── */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 16px;
  overflow: hidden;
  margin-bottom: 20px;
  transition: border-color 0.2s;
}
.chart-card:hover { border-color: var(--rim-hi); }
.chart-card-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--rim);
}
.chart-card-title {
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--sol);
  letter-spacing: -0.01em;
}
.chart-card-subtitle {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  margin-top: 1px;
}
.chart-card-body {
  padding: 20px 24px 24px;
}

/* ── Policy pills ── */
.policy-pill {
  padding: 4px 12px;
  border-radius: 100px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  border: 1px solid;
  letter-spacing: 0.04em;
}
.policy-pill.hidden {
  opacity: 0.3;
  filter: grayscale(1);
}

/* ── Tab bar ── */
.ap-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--rim);
  margin-bottom: 20px;
}
.ap-tab {
  position: relative;
  padding: 11px 20px;
  background: none;
  border: none;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
  cursor: pointer;
  transition: color 0.15s;
  white-space: nowrap;
}
.ap-tab.active { color: var(--sol); }
.ap-tab:hover:not(.active) { color: var(--sol); }
.ap-tab-indicator {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  border-radius: 2px 2px 0 0;
}

/* ── Policy comparison table ── */
.policy-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: 12px;
}
.policy-table th {
  text-align: left;
  padding: 9px 14px;
  font-size: 9px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--rim);
  font-weight: 500;
}
.policy-table td {
  padding: 11px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  color: var(--dim);
  transition: background 0.12s;
}
.policy-table tr:hover td { background: rgba(255,255,255,0.02); }
.policy-table tr.dqn-row td {
  color: var(--amber);
  background: rgba(227,179,65,0.03);
}
.policy-table tr.dqn-row:hover td { background: rgba(227,179,65,0.06); }

/* ── Bar rank chart ── */
.rank-row {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 10px;
}
.rank-label {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  width: 62px;
  flex-shrink: 0;
  text-align: right;
}
.rank-track {
  flex: 1;
  height: 28px;
  background: var(--raised);
  border-radius: 6px;
  overflow: hidden;
  position: relative;
}
.rank-fill {
  height: 100%;
  border-radius: 6px;
  display: flex;
  align-items: center;
  padding-left: 10px;
  transition: width 1s cubic-bezier(0.22,1,0.36,1);
  position: relative;
  min-width: 2px;
}
.rank-fill::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 60%, rgba(255,255,255,0.08));
  border-radius: 6px;
}
.rank-val {
  font-family: var(--font-mono);
  font-size: 10px;
  font-weight: 600;
  width: 48px;
  flex-shrink: 0;
  color: var(--dim);
  text-align: right;
}

/* ── Heatmap ── */
.heatmap-cell { cursor: pointer; transition: opacity 0.1s; }
.heatmap-cell:hover { opacity: 0.7; }

/* ── Empty state ── */
.ap-empty {
  min-height: calc(100vh - 56px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 48px;
}
.ap-empty-card {
  background: var(--surface);
  border: 1px solid var(--rim);
  border-radius: 20px;
  padding: 52px 48px;
  text-align: center;
  max-width: 420px;
}

/* ── Shine on DQN rows ── */
@keyframes rowShine {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.dqn-shine {
  background: linear-gradient(
    90deg,
    transparent 0%, rgba(227,179,65,0.15) 50%, transparent 100%
  );
  background-size: 200% auto;
  animation: rowShine 3s linear infinite;
}

/* ── Section divider ── */
.ap-divider {
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 32px 0 24px;
}
.ap-divider-line {
  flex: 1;
  height: 1px;
  background: var(--rim);
}
.ap-divider-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
}

/* ── Two column grid ── */
.ap-grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 20px;
}
@media (max-width: 900px) { .ap-grid-2 { grid-template-columns: 1fr; } }

/* ── Winning badge ── */
.winner-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: rgba(227,179,65,0.12);
  color: var(--amber);
  border: 1px solid rgba(227,179,65,0.25);
  margin-left: 8px;
}
`;

// ── SVG Icons ─────────────────────────────────────────────────────────────────
const Icons = {
    check: (color = 'currentColor') => (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12"/>
        </svg>
    ),
    up: (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="18 15 12 9 6 15"/>
        </svg>
    ),
    down: (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9"/>
        </svg>
    ),
    chart: (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
    ),
    arrowLeft: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
        </svg>
    ),
    download: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
    ),
    doc: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
        </svg>
    ),
    star: (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
        </svg>
    ),
};

// ── Animated counter hook ─────────────────────────────────────────────────────
function useCountUp(target: number, decimals = 0, duration = 1200) {
    const [value, setValue] = useState(0);
    const ref = useRef<number>(0);
    useEffect(() => {
        const start = ref.current;
        const startTime = performance.now();
        const tick = (now: number) => {
            const t = Math.min((now - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3);
            const current = start + (target - start) * ease;
            setValue(parseFloat(current.toFixed(decimals)));
            ref.current = current;
            if (t < 1) requestAnimationFrame(tick);
            else { setValue(target); ref.current = target; }
        };
        requestAnimationFrame(tick);
    }, [target]);
    return value;
}

// ── KPI card ──────────────────────────────────────────────────────────────────
function KpiCard({ label, value, unit = '', rawValue, decimals = 1, accent = 'neutral', delta, deltaPositive, subtitle, delay = 0 }: {
    label: string; value: string; unit?: string; rawValue?: number;
    decimals?: number; accent?: 'dqn' | 'primary' | 'success' | 'neutral';
    delta?: string; deltaPositive?: boolean; subtitle?: string; delay?: number;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const inView = useInView(ref, { once: true });
    const counted = useCountUp(rawValue ?? 0, decimals, 1000);
    const displayVal = rawValue !== undefined && inView
        ? decimals === 0 ? Math.round(counted).toLocaleString() : counted.toFixed(decimals)
        : value;

    const accentColors: Record<string, string> = {
        dqn: 'var(--amber)', primary: 'var(--blue)',
        success: 'var(--emerald)', neutral: 'var(--rim-hi)',
    };

    return (
        <motion.div
            ref={ref}
            className={`kpi-card ${accent}`}
            initial={{ opacity: 0, y: 24 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
        >
            {/* Ambient glow behind value */}
            <div style={{
                position: 'absolute', top: 0, right: 0,
                width: 120, height: 120,
                background: `radial-gradient(circle, ${accentColors[accent]}10 0%, transparent 70%)`,
                borderRadius: '50%',
                pointerEvents: 'none',
            }} />

            <div className="kpi-label">{label}</div>
            <div className="kpi-val" style={{
                color: accent === 'dqn' ? 'var(--amber)'
                    : accent === 'success' ? 'var(--emerald)'
                    : 'var(--sol)',
            }}>
                {displayVal}{unit && <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--dim)', marginLeft: 4 }}>{unit}</span>}
            </div>

            {subtitle && (
                <div style={{ fontFamily: 'var(--font-body)', fontSize: 11, color: 'var(--dim)', marginBottom: 8 }}>
                    {subtitle}
                </div>
            )}

            {delta && (
                <div className={`kpi-delta ${deltaPositive ? 'up' : deltaPositive === false ? 'down' : 'neutral'}`}>
                    {deltaPositive === true ? Icons.up : deltaPositive === false ? Icons.down : null}
                    {delta}
                </div>
            )}
        </motion.div>
    );
}

// ── Animated horizontal rank bar ─────────────────────────────────────────────
function RankBar({ name, value, max, color, isDQN, rank }: {
    name: string; value: number; max: number;
    color: string; isDQN?: boolean; rank: number;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const inView = useInView(ref, { once: true });
    const pct = max > 0 ? (value / max) * 100 : 0;

    return (
        <div className="rank-row" ref={ref}>
            <div className="rank-label" style={{ color: isDQN ? 'var(--amber)' : 'var(--dim)' }}>
                {name}
                {isDQN && (
                    <span className="winner-badge" style={{ display: 'inline-flex', marginLeft: 4 }}>
                        {Icons.star}
                    </span>
                )}
            </div>
            <div className="rank-track">
                <motion.div
                    className="rank-fill"
                    initial={{ width: 0 }}
                    animate={inView ? { width: `${pct}%` } : { width: 0 }}
                    transition={{ duration: 0.9, delay: rank * 0.06, ease: [0.22, 1, 0.36, 1] }}
                    style={{
                        background: isDQN
                            ? `linear-gradient(90deg, ${color}cc, ${color})`
                            : `${color}88`,
                        boxShadow: isDQN ? `0 0 12px ${color}60` : 'none',
                    }}
                />
            </div>
            <div className="rank-val">{value.toFixed(2)}</div>
        </div>
    );
}

// ── Chart card wrapper ────────────────────────────────────────────────────────
function ChartCard({ title, subtitle, badge, children, style }: {
    title: string; subtitle?: string; badge?: string;
    children: React.ReactNode; style?: React.CSSProperties;
}) {
    const ref = useRef<HTMLDivElement>(null);
    const inView = useInView(ref, { once: true, amount: 0.1 });

    return (
        <motion.div
            ref={ref}
            className="chart-card"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            style={style}
        >
            <div className="chart-card-header">
                <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <div className="chart-card-title">{title}</div>
                        {badge && (
                            <div style={{
                                padding: '2px 7px', borderRadius: 4,
                                background: 'rgba(68,147,248,0.1)',
                                border: '1px solid rgba(68,147,248,0.2)',
                                fontFamily: 'var(--font-mono)', fontSize: 8,
                                color: 'var(--blue)', letterSpacing: '0.1em',
                                textTransform: 'uppercase',
                            }}>
                                {badge}
                            </div>
                        )}
                    </div>
                    {subtitle && <div className="chart-card-subtitle">{subtitle}</div>}
                </div>
            </div>
            <div className="chart-card-body">{children}</div>
        </motion.div>
    );
}

// ── Policy filter pills ───────────────────────────────────────────────────────
function PolicyPills({ names, hidden, onToggle }: {
    names: string[]; hidden: Set<string>; onToggle: (n: string) => void;
}) {
    return (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 20 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--muted)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
                Filter
            </span>
            {names.map(name => {
                const color = POLICY_MAP[name] ?? '#555';
                const isHidden = hidden.has(name);
                return (
                    <button
                        key={name}
                        className={`policy-pill ${isHidden ? 'hidden' : ''}`}
                        onClick={() => onToggle(name)}
                        style={{
                            borderColor: color,
                            color: isHidden ? 'var(--muted)' : color,
                            background: isHidden ? 'transparent' : `${color}15`,
                        }}
                    >
                        {name}
                    </button>
                );
            })}
        </div>
    );
}

// ── Heatmap ───────────────────────────────────────────────────────────────────
function UtilizationHeatmap({ data }: {
    data: { worker: string; day: number; utilization: number }[];
}) {
    const [hoveredCell, setHoveredCell] = useState<{ worker: string; day: number; val: number } | null>(null);
    const workers = [...new Set(data.map(d => d.worker))].sort();
    const days    = [...new Set(data.map(d => d.day))].sort((a, b) => a - b);
    const CELL_W = 20, CELL_H = 26, LABEL_W = 36, HEADER_H = 26;

    function utilColor(u: number): string {
        if (u >= 0.9) return '#f85149';
        if (u >= 0.75) return '#e3b341';
        if (u >= 0.5) return '#3fb950';
        if (u >= 0.25) return '#2dd4bf';
        return 'rgba(255,255,255,0.03)';
    }

    return (
        <div style={{ overflowX: 'auto', position: 'relative' }}>
            <svg
                width={LABEL_W + days.length * CELL_W}
                height={HEADER_H + workers.length * CELL_H}
                style={{ display: 'block' }}
            >
                {days.map((day, di) => di % 5 === 0 && (
                    <text key={day}
                        x={LABEL_W + di * CELL_W + CELL_W / 2}
                        y={18}
                        textAnchor="middle"
                        fill="#3d444d"
                        fontSize={8}
                        fontFamily="DM Mono, monospace"
                    >
                        {day}
                    </text>
                ))}
                {workers.map((worker, wi) => (
                    <g key={worker}>
                        <text
                            x={LABEL_W - 5}
                            y={HEADER_H + wi * CELL_H + CELL_H / 2 + 4}
                            textAnchor="end"
                            fill="#7d8590"
                            fontSize={9}
                            fontFamily="DM Mono, monospace"
                        >
                            {worker}
                        </text>
                        {days.map((day, di) => {
                            const entry = data.find(d => d.worker === worker && d.day === day);
                            const u = entry?.utilization ?? 0;
                            const isHovered = hoveredCell?.worker === worker && hoveredCell?.day === day;
                            return (
                                <rect key={day}
                                    className="heatmap-cell"
                                    x={LABEL_W + di * CELL_W + 1}
                                    y={HEADER_H + wi * CELL_H + 1}
                                    width={CELL_W - 2}
                                    height={CELL_H - 2}
                                    fill={utilColor(u)}
                                    rx={3}
                                    opacity={isHovered ? 0.6 : 1}
                                    onMouseEnter={() => setHoveredCell({ worker, day, val: u })}
                                    onMouseLeave={() => setHoveredCell(null)}
                                />
                            );
                        })}
                    </g>
                ))}
            </svg>

            {hoveredCell && (
                <div style={{
                    position: 'absolute', top: 0, left: LABEL_W + 8,
                    background: '#0d1117',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 8, padding: '6px 12px',
                    fontFamily: 'DM Mono, monospace', fontSize: 11, color: '#e6edf3',
                    pointerEvents: 'none', whiteSpace: 'nowrap',
                    boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                    zIndex: 10,
                }}>
                    {hoveredCell.worker} &bull; Day {hoveredCell.day} &bull; {(hoveredCell.val * 100).toFixed(0)}% utilization
                </div>
            )}

            <div style={{ display: 'flex', gap: 14, marginTop: 14, flexWrap: 'wrap', alignItems: 'center' }}>
                {[
                    { label: 'Low', color: 'rgba(255,255,255,0.05)' },
                    { label: '25%+', color: '#2dd4bf' },
                    { label: '50%+', color: '#3fb950' },
                    { label: '75%+', color: '#e3b341' },
                    { label: '90%+', color: '#f85149' },
                ].map(({ label, color }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                            width: 12, height: 12,
                            background: color, borderRadius: 3,
                            border: '1px solid rgba(255,255,255,0.06)',
                        }} />
                        <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, color: 'var(--dim)' }}>
                            {label}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ── Policy comparison table ───────────────────────────────────────────────────
function PolicyTable({ metrics, hidden }: {
    metrics: Record<string, any>; hidden: Set<string>;
}) {
    const rows = Object.entries(metrics)
        .filter(([name]) => !hidden.has(name))
        .sort((a, b) => (b[1].throughput ?? 0) - (a[1].throughput ?? 0));

    const best = rows[0]?.[0];

    return (
        <table className="policy-table">
            <thead>
                <tr>
                    <th>Policy</th>
                    <th>Throughput/Day</th>
                    <th>Completion %</th>
                    <th>Quality Score</th>
                    <th>Lateness %</th>
                    <th>Overload</th>
                </tr>
            </thead>
            <tbody>
                {rows.map(([name, m], i) => {
                    const isDQN = name === 'DQN';
                    const color = POLICY_MAP[name] ?? 'var(--dim)';
                    return (
                        <tr key={name} className={isDQN ? 'dqn-row' : ''}>
                            <td>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <div style={{
                                        width: 8, height: 8, borderRadius: '50%',
                                        background: color, flexShrink: 0,
                                        boxShadow: isDQN ? `0 0 8px ${color}` : 'none',
                                    }} />
                                    <span style={{ color, fontWeight: isDQN ? 600 : 400 }}>{name}</span>
                                    {name === best && !isDQN && (
                                        <span className="winner-badge">Best</span>
                                    )}
                                    {isDQN && (
                                        <span className="dqn-shine" style={{
                                            padding: '1px 6px', borderRadius: 4,
                                            fontFamily: 'var(--font-mono)', fontSize: 8,
                                            color: 'var(--amber)', fontWeight: 700,
                                            letterSpacing: '0.08em',
                                        }}>
                                            Agent
                                        </span>
                                    )}
                                </div>
                            </td>
                            <td style={{ color: isDQN ? 'var(--amber)' : 'var(--sol)', fontWeight: 600 }}>
                                {(m.throughput ?? 0).toFixed(3)}
                            </td>
                            <td>{((m.completion_rate ?? 0) * 100).toFixed(1)}%</td>
                            <td style={{ color: (m.quality_score ?? 0) > 0.4 ? 'var(--emerald)' : 'var(--dim)' }}>
                                {(m.quality_score ?? 0).toFixed(4)}
                            </td>
                            <td style={{ color: (m.lateness_rate ?? 0) > 0.05 ? 'var(--rose)' : 'var(--dim)' }}>
                                {((m.lateness_rate ?? 0) * 100).toFixed(2)}%
                            </td>
                            <td style={{ color: (m.overload_events ?? 0) > 0 ? 'var(--rose)' : 'var(--dim)' }}>
                                {m.overload_events ?? 0}
                            </td>
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

// ── Section divider ───────────────────────────────────────────────────────────
function Divider({ label }: { label: string }) {
    return (
        <div className="ap-divider">
            <div className="ap-divider-line" />
            <span className="ap-divider-label">{label}</span>
            <div className="ap-divider-line" />
        </div>
    );
}

// ── Tab bar ───────────────────────────────────────────────────────────────────
const TABS = [
    { key: 'ranking',    label: 'Rankings'    },
    { key: 'radar',      label: 'Radar'       },
    { key: 'throughput', label: 'Throughput'  },
    { key: 'lateness',   label: 'Lateness'    },
] as const;

type TabKey = typeof TABS[number]['key'];

function TabBar({ active, onChange }: { active: TabKey; onChange: (k: TabKey) => void }) {
    return (
        <div className="ap-tabs">
            {TABS.map(tab => {
                const isActive = tab.key === active;
                return (
                    <button
                        key={tab.key}
                        className={`ap-tab ${isActive ? 'active' : ''}`}
                        onClick={() => onChange(tab.key)}
                    >
                        {tab.label}
                        {isActive && (
                            <motion.div
                                className="ap-tab-indicator"
                                layoutId="tab-indicator"
                                style={{ background: 'var(--amber)' }}
                                transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                            />
                        )}
                    </button>
                );
            })}
        </div>
    );
}

// ── Main AnalyticsPage ────────────────────────────────────────────────────────
export default function AnalyticsPage() {
    const navigate = useNavigate();
    const { finalMetrics, baselineResults, simConfig, dailyMetricsHistory } = useSimulationStore(
        useShallow(s => ({
            finalMetrics:        s.finalMetrics,
            baselineResults:     s.baselineResults,
            simConfig:           s.simConfig,
            dailyMetricsHistory: s.dailyMetricsHistory,
        }))
    );

    const [hiddenPolicies, setHiddenPolicies] = useState<Set<string>>(new Set());
    const [activeTab, setActiveTab]           = useState<TabKey>('ranking');
    const [showReadme, setShowReadme]         = useState(false);

    // Hooks — IDENTICAL to original
    const allMetrics      = useAllPolicyMetrics();
    const dailyThroughput = useDailyThroughputByPolicy();
    const latenessPerDay  = useLatenessPerDay();

    const heatmapData = useMemo(() => {
        const result: { worker: string; day: number; utilization: number }[] = [];
        const numWorkers = simConfig?.num_workers ?? 5;
        const byDay: Record<number, { loadBalance: number }[]> = {};
        dailyMetricsHistory.forEach(row => {
            if (!byDay[row.day]) byDay[row.day] = [];
            byDay[row.day].push({ loadBalance: row.load_balance });
        });
        Object.entries(byDay).forEach(([dayStr, rows]) => {
            const day  = Number(dayStr);
            const avgLB = rows.reduce((s, r) => s + r.loadBalance, 0) / rows.length;
            for (let w = 0; w < numWorkers; w++) {
                const jitter = (w - numWorkers / 2) / numWorkers * avgLB * 0.2;
                result.push({ worker: `W${w + 1}`, day, utilization: Math.min(1, Math.max(0, 0.5 + jitter + (1 - avgLB) * 0.4)) });
            }
        });
        return result;
    }, [dailyMetricsHistory, simConfig]);

    // Empty state
    if (!finalMetrics) {
        return (
            <>
                <style>{AP_STYLES}</style>
                <PageTransition>
                    <div className="ap-empty">
                        <div className="ap-empty-card">
                            <div style={{ marginBottom: 20 }}>
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto', display: 'block' }}>
                                    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
                                    <line x1="6" y1="20" x2="6" y2="14"/>
                                </svg>
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--blue)', letterSpacing: '0.16em', textTransform: 'uppercase', marginBottom: 12 }}>
                                Awaiting Simulation
                            </div>
                            <p style={{ fontFamily: 'var(--font-body)', color: 'var(--dim)', marginBottom: 28, fontSize: 14, lineHeight: 1.6 }}>
                                Complete a simulation run to generate analytics.
                            </p>
                            <motion.button
                                onClick={() => navigate('/')}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                style={{
                                    width: '100%', padding: '11px 0',
                                    background: 'var(--blue)', color: '#fff',
                                    border: 'none', borderRadius: 9,
                                    fontFamily: 'var(--font-head)', fontWeight: 700,
                                    fontSize: 14, cursor: 'pointer',
                                    boxShadow: 'var(--blue-glow)',
                                }}
                            >
                                Configure Simulation
                            </motion.button>
                        </div>
                    </div>
                </PageTransition>
            </>
        );
    }

    // Derived data — IDENTICAL to original
    const policyNames = useMemo(() => Object.keys(allMetrics), [allMetrics]);

    const radarData = useMemo(() => buildRadarData(allMetrics), [allMetrics]);
    const radarPolicies = useMemo(() => [
        ...policyNames.filter(n => n !== 'DQN' && !hiddenPolicies.has(n)),
        ...(policyNames.includes('DQN') && !hiddenPolicies.has('DQN') ? ['DQN'] : []),
    ], [policyNames, hiddenPolicies]);

    const allDays = useMemo(() => [
        ...new Set(Object.values(dailyThroughput).flatMap(arr => arr.map(d => d.day)))
    ].sort((a, b) => a - b), [dailyThroughput]);

    const lineChartData = useMemo(() =>
        allDays.map(day => {
            const row: { day: number; [k: string]: number } = { day };
            Object.entries(dailyThroughput).forEach(([policy, arr]) => {
                row[policy] = arr.find(d => d.day === day)?.value ?? 0;
            });
            return row;
        })
    , [allDays, dailyThroughput]);

    const togglePolicy = (name: string) =>
        setHiddenPolicies(prev => {
            const next = new Set(prev);
            next.has(name) ? next.delete(name) : next.add(name);
            return next;
        });

    // Rank data for horizontal bars
    const throughputRank = useMemo(() =>
        Object.entries(allMetrics)
            .filter(([n]) => !hiddenPolicies.has(n))
            .map(([n, m]) => ({ name: n, value: m.throughput ?? 0 }))
            .sort((a, b) => b.value - a.value)
    , [allMetrics, hiddenPolicies]);

    const qualityRank = useMemo(() =>
        Object.entries(allMetrics)
            .filter(([n]) => !hiddenPolicies.has(n))
            .map(([n, m]) => ({ name: n, value: m.quality_score ?? 0 }))
            .sort((a, b) => b.value - a.value)
    , [allMetrics, hiddenPolicies]);

    const maxThroughput = Math.max(...throughputRank.map(r => r.value), 1);
    const maxQuality    = Math.max(...qualityRank.map(r => r.value), 1);

    const dqnMeta = allMetrics['DQN'];
    const bestBaseline = finalMetrics.best_policy;
    const bestM  = allMetrics[bestBaseline] ?? {};

    return (
        <>
            <style>{AP_STYLES}</style>

            <PageTransition>
                <div className="ap-root">
                    <div className="ap-content">

                        {/* ── Page header ── */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                            style={{ marginBottom: 36, display: 'flex', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}
                        >
                            <div style={{ flex: 1, minWidth: 260 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                                    <div style={{
                                        width: 8, height: 8, borderRadius: '50%',
                                        background: 'var(--emerald)',
                                        boxShadow: '0 0 10px var(--emerald)',
                                    }} />
                                    <span style={{
                                        fontFamily: 'var(--font-mono)', fontSize: 9,
                                        color: 'var(--emerald)', letterSpacing: '0.16em',
                                        textTransform: 'uppercase',
                                    }}>
                                        Simulation Complete
                                    </span>
                                </div>
                                <h1 style={{
                                    fontFamily: 'var(--font-head)',
                                    fontSize: 'clamp(1.6rem, 3vw, 2.4rem)',
                                    fontWeight: 800, color: 'var(--sol)',
                                    letterSpacing: '-0.04em', lineHeight: 1.1,
                                    marginBottom: 8,
                                }}>
                                    Intelligence Report
                                </h1>
                                <p style={{ fontFamily: 'var(--font-body)', color: 'var(--dim)', fontSize: 13, lineHeight: 1.5 }}>
                                    {policyNames.length} policies benchmarked across {simConfig?.days_phase1 + simConfig?.days_phase2} days.
                                    Best baseline: <span style={{ color: POLICY_MAP[bestBaseline] ?? 'var(--sol)', fontWeight: 600 }}>{bestBaseline}</span>.
                                </p>
                            </div>

                            {/* Header actions */}
                            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                                <motion.button
                                    onClick={() => navigate('/simulation')}
                                    whileTap={{ scale: 0.97 }}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: 6,
                                        padding: '8px 14px', background: 'transparent',
                                        border: '1px solid var(--rim-hi)', borderRadius: 8,
                                        fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--dim)',
                                        cursor: 'pointer', transition: 'all 0.15s',
                                    }}
                                >
                                    {Icons.arrowLeft} Simulation
                                </motion.button>
                                <motion.button
                                    onClick={() => navigate('/')}
                                    whileTap={{ scale: 0.97 }}
                                    style={{
                                        padding: '8px 14px', background: 'transparent',
                                        border: '1px solid var(--rim-hi)', borderRadius: 8,
                                        fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--dim)',
                                        cursor: 'pointer',
                                    }}
                                >
                                    New Run
                                </motion.button>
                                <a href="/api/export" download="simulation_results.csv"
                                    style={{
                                        display: 'inline-flex', alignItems: 'center', gap: 6,
                                        padding: '8px 14px',
                                        background: 'rgba(63,185,80,0.08)',
                                        color: 'var(--emerald)',
                                        border: '1px solid rgba(63,185,80,0.2)',
                                        borderRadius: 8,
                                        fontFamily: 'var(--font-body)', fontSize: 12,
                                        fontWeight: 500, textDecoration: 'none',
                                    }}
                                >
                                    {Icons.download} Export CSV
                                </a>
                                <motion.button
                                    onClick={() => setShowReadme(p => !p)}
                                    whileTap={{ scale: 0.97 }}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: 6,
                                        padding: '8px 14px',
                                        background: 'var(--blue)',
                                        color: '#fff',
                                        border: 'none', borderRadius: 8,
                                        fontFamily: 'var(--font-body)', fontSize: 12,
                                        fontWeight: 600, cursor: 'pointer',
                                        boxShadow: 'var(--blue-glow)',
                                    }}
                                >
                                    {Icons.doc} Documentation
                                </motion.button>
                            </div>
                        </motion.div>

                        {showReadme && <ReadmeGeneratorPanel onClose={() => setShowReadme(false)} />}

                        {/* ── KPI hero row ── */}
                        <div className="kpi-grid">
                            <KpiCard
                                label="DQN Quality Score"
                                value={(dqnMeta?.quality_score ?? 0).toFixed(4)}
                                rawValue={dqnMeta?.quality_score ?? 0}
                                decimals={4}
                                accent="dqn"
                                delta={`vs Hybrid ${((dqnMeta?.quality_score ?? 0) - (allMetrics['Hybrid']?.quality_score ?? 0) > 0 ? '+' : '')}${((dqnMeta?.quality_score ?? 0) - (allMetrics['Hybrid']?.quality_score ?? 0)).toFixed(3)}`}
                                deltaPositive={(dqnMeta?.quality_score ?? 0) > (allMetrics['Hybrid']?.quality_score ?? 0)}
                                delay={0}
                            />
                            <KpiCard
                                label="DQN Throughput / Day"
                                value={(dqnMeta?.throughput ?? 0).toFixed(3)}
                                rawValue={dqnMeta?.throughput ?? 0}
                                decimals={3}
                                accent="dqn"
                                delta={`vs Random ${((dqnMeta?.throughput ?? 0) - (allMetrics['Random']?.throughput ?? 0) >= 0 ? '+' : '')}${((dqnMeta?.throughput ?? 0) - (allMetrics['Random']?.throughput ?? 0)).toFixed(3)}`}
                                deltaPositive={(dqnMeta?.throughput ?? 0) > (allMetrics['Random']?.throughput ?? 0)}
                                delay={0.07}
                            />
                            <KpiCard
                                label="On-Time Rate"
                                value={`${(100 - (finalMetrics.overall_lateness_rate ?? 0) * 100).toFixed(1)}`}
                                rawValue={100 - (finalMetrics.overall_lateness_rate ?? 0) * 100}
                                decimals={1}
                                unit="%"
                                accent="success"
                                subtitle="DQN scheduling phase"
                                delay={0.14}
                            />
                            <KpiCard
                                label="Tasks Completed"
                                value={finalMetrics.total_tasks_completed.toLocaleString()}
                                rawValue={finalMetrics.total_tasks_completed}
                                decimals={0}
                                accent="primary"
                                delay={0.21}
                            />
                            <KpiCard
                                label="Peak Overload Events"
                                value={String(finalMetrics.peak_overload_events)}
                                rawValue={finalMetrics.peak_overload_events}
                                decimals={0}
                                accent={finalMetrics.peak_overload_events === 0 ? 'success' : 'neutral'}
                                delta={finalMetrics.peak_overload_events === 0 ? 'Zero overloads' : undefined}
                                deltaPositive={finalMetrics.peak_overload_events === 0 ? true : undefined}
                                delay={0.28}
                            />
                            <KpiCard
                                label="Best Baseline"
                                value={finalMetrics.best_policy}
                                accent="neutral"
                                subtitle={`${(bestM?.throughput ?? 0).toFixed(3)} tasks/day`}
                                delay={0.35}
                            />
                        </div>

                        {/* ── Policy filter ── */}
                        <PolicyPills names={policyNames} hidden={hiddenPolicies} onToggle={togglePolicy} />

                        {/* ── Tab charts ── */}
                        <TabBar active={activeTab} onChange={setActiveTab} />

                        <AnimatePresence mode="wait">
                            {activeTab === 'ranking' && (
                                <motion.div
                                    key="ranking"
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <div className="ap-grid-2">
                                        <ChartCard
                                            title="Throughput Ranking"
                                            subtitle="Average tasks completed per day"
                                            badge="Primary Metric"
                                        >
                                            {throughputRank.map((r, i) => (
                                                <RankBar
                                                    key={r.name} name={r.name}
                                                    value={r.value} max={maxThroughput}
                                                    color={POLICY_MAP[r.name] ?? '#555'}
                                                    isDQN={r.name === 'DQN'}
                                                    rank={i}
                                                />
                                            ))}
                                        </ChartCard>

                                        <ChartCard
                                            title="Quality Ranking"
                                            subtitle="Skill-match quality score (higher = better)"
                                        >
                                            {qualityRank.map((r, i) => (
                                                <RankBar
                                                    key={r.name} name={r.name}
                                                    value={r.value} max={maxQuality}
                                                    color={POLICY_MAP[r.name] ?? '#555'}
                                                    isDQN={r.name === 'DQN'}
                                                    rank={i}
                                                />
                                            ))}
                                        </ChartCard>
                                    </div>

                                    <ChartCard
                                        title="Full Policy Comparison"
                                        subtitle="All metrics across all policies"
                                    >
                                        <PolicyTable metrics={allMetrics} hidden={hiddenPolicies} />
                                    </ChartCard>
                                </motion.div>
                            )}

                            {activeTab === 'radar' && (
                                <motion.div
                                    key="radar"
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <ChartCard
                                        title="Multi-Dimensional Performance Radar"
                                        subtitle="DQN (amber) rendered on top — larger area = better"
                                    >
                                        <ResponsiveContainer width="100%" height={400}>
                                            <RadarChart data={radarData} margin={{ top: 20, right: 40, bottom: 20, left: 40 }}>
                                                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                                                <PolarAngleAxis dataKey="axis" tick={{
                                                    fill: 'var(--dim)', fontSize: 11,
                                                    fontFamily: 'DM Mono, monospace',
                                                }} />
                                                {radarPolicies.map(name => (
                                                    <Radar key={name} name={name} dataKey={name}
                                                        stroke={POLICY_MAP[name] ?? '#555'}
                                                        fill={POLICY_MAP[name] ?? '#555'}
                                                        fillOpacity={name === 'DQN' ? 0.35 : 0.1}
                                                        strokeWidth={name === 'DQN' ? 2.5 : 1.2}
                                                        isAnimationActive={false}
                                                    />
                                                ))}
                                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'DM Mono, monospace', paddingTop: 16 }} />
                                            </RadarChart>
                                        </ResponsiveContainer>
                                    </ChartCard>
                                </motion.div>
                            )}

                            {activeTab === 'throughput' && (
                                <motion.div
                                    key="throughput"
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <ChartCard
                                        title="Daily Throughput by Policy"
                                        subtitle="Tasks completed per working day across all phases"
                                    >
                                        <ResponsiveContainer width="100%" height={340}>
                                            <AreaChart data={lineChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                                <defs>
                                                    {policyNames.map(name => (
                                                        <linearGradient key={name} id={`grad-${name}`} x1="0" y1="0" x2="0" y2="1">
                                                            <stop offset="5%"   stopColor={POLICY_MAP[name] ?? '#555'} stopOpacity={name === 'DQN' ? 0.25 : 0.1} />
                                                            <stop offset="95%"  stopColor={POLICY_MAP[name] ?? '#555'} stopOpacity={0} />
                                                        </linearGradient>
                                                    ))}
                                                </defs>
                                                <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--muted)', fontFamily: 'DM Mono, monospace' }}
                                                    label={{ value: 'Day', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 10 }} />
                                                <YAxis tick={{ fontSize: 10, fill: 'var(--muted)', fontFamily: 'DM Mono, monospace' }} />
                                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'DM Mono, monospace' }} />
                                                {policyNames.filter(n => !hiddenPolicies.has(n)).map(name => (
                                                    <Area key={name} type="monotone" dataKey={name}
                                                        stroke={POLICY_MAP[name] ?? '#555'}
                                                        fill={`url(#grad-${name})`}
                                                        strokeWidth={name === 'DQN' ? 2.5 : 1.2}
                                                        strokeOpacity={name === 'DQN' ? 1 : 0.6}
                                                        dot={false}
                                                        isAnimationActive={false}
                                                    />
                                                ))}
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </ChartCard>
                                </motion.div>
                            )}

                            {activeTab === 'lateness' && (
                                <motion.div
                                    key="lateness"
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <ChartCard
                                        title="Lateness Rate per Day"
                                        subtitle="Fraction of tasks completed past deadline — lower is better"
                                    >
                                        <ResponsiveContainer width="100%" height={340}>
                                            <BarChart data={latenessPerDay} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                                                <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--muted)', fontFamily: 'DM Mono, monospace' }} />
                                                <YAxis tick={{ fontSize: 10, fill: 'var(--muted)', fontFamily: 'DM Mono, monospace' }}
                                                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.02)' }} />
                                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'DM Mono, monospace' }} />
                                                <ReferenceLine y={0.05} stroke="rgba(248,81,73,0.3)" strokeDasharray="4 4"
                                                    label={{ value: '5% threshold', fill: 'var(--rose)', fontSize: 9, fontFamily: 'DM Mono, monospace' }} />
                                                {policyNames.filter(n => !hiddenPolicies.has(n)).map(name => (
                                                    <Bar key={name} dataKey={name}
                                                        fill={POLICY_MAP[name] ?? '#555'}
                                                        fillOpacity={name === 'DQN' ? 1 : 0.65}
                                                        radius={[2, 2, 0, 0]}
                                                        isAnimationActive={false}
                                                    />
                                                ))}
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </ChartCard>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* ── Heatmap ── */}
                        <Divider label="Worker Utilization" />
                        <ChartCard
                            title="Worker x Day Utilization Heatmap"
                            subtitle="Load distribution across all workers and simulation days"
                        >
                            <UtilizationHeatmap data={heatmapData} />
                        </ChartCard>

                    </div>
                </div>
            </PageTransition>
        </>
    );
}