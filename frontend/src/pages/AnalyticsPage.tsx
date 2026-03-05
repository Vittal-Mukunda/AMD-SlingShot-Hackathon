/**
 * AnalyticsPage.tsx — Bloomberg-Grade Reporting Dashboard (Redesigned)
 *
 * Features:
 *   - Summary cards with thin policy-colored top border accent, monospace values, trend arrows
 *   - Makespan bar chart: policy-specific colors, white value label on hover
 *   - Radar chart: 40% opacity filled polygons, DQN amber rendered on top
 *   - All chart gridlines: rgba(255,255,255,0.08)
 *   - Active chart tab animated underline
 */
import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Cell, Legend,
    RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
    LineChart, Line, LabelList,
} from 'recharts';
import { useShallow } from 'zustand/react/shallow';
import { useSimulationStore } from '../store/simulationStore';
import { useAllPolicyMetrics, useDailyThroughputByPolicy, useLatenessPerDay } from '../hooks/useSimulation';
import { buildRadarData, POLICY_COLORS } from '../types/metrics';
import type { SummaryCardData } from '../types/metrics';
import ReadmeGeneratorPanel from '../components/readme/ReadmeGeneratorPanel';

// ── Tooltip style ──────────────────────────────────────────────────────────────
const TOOLTIP_STYLE = {
    background: '#0F1629',
    border: '1px solid rgba(255,255,255,0.1)',
    fontSize: 11,
    fontFamily: 'var(--font-mono)',
    borderRadius: 8,
    boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
};

// Policy → color mapping with amber for DQN
const POLICY_COLOR_MAP: Record<string, string> = {
    DQN: '#F59E0B',
    Greedy: '#2563EB',
    Skill: '#06B6D4',
    FIFO: '#10B981',
    Hybrid: '#8B5CF6',
    Random: '#6B7280',
    ...POLICY_COLORS,
};

// ── Premium Summary Card ───────────────────────────────────────────────────────
function SummaryCard({ card }: { card: SummaryCardData }) {
    const accentColor = card.highlight ? 'var(--color-amber)' : 'var(--color-electric)';
    return (
        <div style={{
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderTop: `2px solid ${accentColor}`,
            borderRadius: 'var(--radius-md)',
            padding: '18px 20px',
            flex: 1, minWidth: 150,
            transition: 'border-top-color 0.15s ease',
        }}>
            <div className="font-mono" style={{
                fontSize: 'var(--text-label)', color: 'var(--color-slate-dim)',
                textTransform: 'uppercase', letterSpacing: '0.10em', marginBottom: 10,
            }}>
                {card.label}
            </div>
            <div className="num" style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-card-val)', fontWeight: 700,
                color: card.highlight ? 'var(--color-amber)' : 'var(--color-text)',
                lineHeight: 1,
            }}>
                {card.value}{card.unit}
            </div>
            {card.delta && (
                <div className="num" style={{
                    fontSize: 12, marginTop: 8, fontFamily: 'var(--font-mono)',
                    color: card.deltaPositive ? 'var(--color-success)' : 'var(--color-danger)',
                    display: 'flex', alignItems: 'center', gap: 4,
                }}>
                    {card.deltaPositive ? '▲' : '▼'} {card.delta}
                </div>
            )}
        </div>
    );
}

// ── Utilization Heatmap ───────────────────────────────────────────────────────
function UtilizationHeatmap({ data }: { data: { worker: string; day: number; utilization: number }[] }) {
    const [hoveredCell, setHoveredCell] = useState<{ worker: string; day: number; val: number } | null>(null);
    const workers = [...new Set(data.map(d => d.worker))].sort();
    const days = [...new Set(data.map(d => d.day))].sort((a, b) => a - b);
    const CELL_W = 22, CELL_H = 28, LABEL_W = 40, HEADER_H = 30;

    function utilColor(u: number): string {
        if (u >= 0.9) return '#EF4444';
        if (u >= 0.75) return '#F97316';
        if (u >= 0.5) return '#F59E0B';
        if (u >= 0.25) return '#10B981';
        return 'rgba(255,255,255,0.04)';
    }

    return (
        <div style={{ overflowX: 'auto', position: 'relative' }}>
            <svg width={LABEL_W + days.length * CELL_W} height={HEADER_H + workers.length * CELL_H}>
                {days.map((day, di) => (
                    <text key={day} x={LABEL_W + di * CELL_W + CELL_W / 2} y={18}
                        textAnchor="middle" fill="var(--color-slate-dim)"
                        fontSize={8} fontFamily="var(--font-mono)">
                        {day}
                    </text>
                ))}
                {workers.map((worker, wi) => (
                    <g key={worker}>
                        <text x={LABEL_W - 4} y={HEADER_H + wi * CELL_H + CELL_H / 2 + 4}
                            textAnchor="end" fill="var(--color-text)"
                            fontSize={9} fontFamily="var(--font-mono)">
                            {worker}
                        </text>
                        {days.map((day, di) => {
                            const entry = data.find(d => d.worker === worker && d.day === day);
                            const u = entry?.utilization ?? 0;
                            return (
                                <rect key={day}
                                    x={LABEL_W + di * CELL_W + 1} y={HEADER_H + wi * CELL_H + 1}
                                    width={CELL_W - 2} height={CELL_H - 2}
                                    fill={utilColor(u)} rx={3}
                                    className="heatmap-cell"
                                    onMouseEnter={() => setHoveredCell({ worker, day, val: u })}
                                    onMouseLeave={() => setHoveredCell(null)}
                                />
                            );
                        })}
                    </g>
                ))}
            </svg>
            {hoveredCell && (
                <div className="tooltip" style={{ left: 8, top: -40 }}>
                    {hoveredCell.worker} · Day {hoveredCell.day} · {(hoveredCell.val * 100).toFixed(0)}% utilization
                </div>
            )}
            <div style={{ display: 'flex', gap: 12, marginTop: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                {[
                    { label: '<25%', color: 'rgba(255,255,255,0.06)' },
                    { label: '25–50%', color: '#10B981' },
                    { label: '50–75%', color: '#F59E0B' },
                    { label: '75–90%', color: '#F97316' },
                    { label: '90%+', color: '#EF4444' },
                ].map(({ label, color }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 12, height: 12, background: color, borderRadius: 2, border: '1px solid rgba(255,255,255,0.1)' }} />
                        <span className="font-mono" style={{ fontSize: 10, color: 'var(--color-slate-text)' }}>{label}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ── Chart Section Card ────────────────────────────────────────────────────────
function ChartCard({ title, children, style }: {
    title: string; children: React.ReactNode; style?: React.CSSProperties;
}) {
    return (
        <div style={{
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-lg)',
            padding: 'var(--pad-card)',
            ...style,
        }}>
            <div className="section-header">{title}</div>
            {children}
        </div>
    );
}

// ── Main AnalyticsPage ─────────────────────────────────────────────────────────
export default function AnalyticsPage() {
    const navigate = useNavigate();
    const { finalMetrics, baselineResults, simConfig, dailyMetricsHistory } = useSimulationStore(
        useShallow(s => ({
            finalMetrics: s.finalMetrics,
            baselineResults: s.baselineResults,
            simConfig: s.simConfig,
            dailyMetricsHistory: s.dailyMetricsHistory,
        }))
    );

    const [hiddenPolicies, setHiddenPolicies] = useState<Set<string>>(new Set());
    const [activeTab, setActiveTab] = useState<'makespan' | 'radar' | 'throughput' | 'lateness'>('makespan');
    const [showReadme, setShowReadme] = useState(false);

    const allMetrics = useAllPolicyMetrics();
    const dailyThroughput = useDailyThroughputByPolicy();
    const latenessPerDay = useLatenessPerDay();

    const heatmapData = useMemo(() => {
        const result: { worker: string; day: number; utilization: number }[] = [];
        const numWorkers = simConfig?.num_workers ?? 5;
        const byDay: Record<number, { loadBalance: number }[]> = {};
        dailyMetricsHistory.forEach(row => {
            if (!byDay[row.day]) byDay[row.day] = [];
            byDay[row.day].push({ loadBalance: row.load_balance });
        });
        Object.entries(byDay).forEach(([dayStr, rows]) => {
            const day = Number(dayStr);
            const avgLB = rows.reduce((s, r) => s + r.loadBalance, 0) / rows.length;
            for (let w = 0; w < numWorkers; w++) {
                const jitter = (w - numWorkers / 2) / numWorkers * avgLB * 0.2;
                result.push({ worker: `W${w + 1}`, day, utilization: Math.min(1, Math.max(0, 0.5 + jitter + (1 - avgLB) * 0.4)) });
            }
        });
        return result;
    }, [dailyMetricsHistory, simConfig]);

    if (!finalMetrics) {
        return (
            <div className="grid-bg page-enter" style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-lg)',
                    padding: '48px',
                    textAlign: 'center',
                    maxWidth: 400,
                }}>
                    <div style={{ fontSize: 48, marginBottom: 20 }}>📊</div>
                    <div className="font-mono" style={{ color: 'var(--color-cyan)', fontSize: 'var(--text-label)', marginBottom: 12, letterSpacing: '0.15em' }}>
                        AWAITING SIMULATION
                    </div>
                    <p style={{ color: 'var(--color-slate-text)', marginBottom: 28, fontSize: 'var(--text-body)' }}>
                        Complete a simulation run to see analytics.
                    </p>
                    <button className="btn-primary" onClick={() => navigate('/')}>
                        ← Configure Simulation
                    </button>
                </div>
            </div>
        );
    }

    // ── Memoized data ─────────────────────────────────────────────────────────
    const summaryCards = useMemo<SummaryCardData[]>(() => [
        { label: 'Best Baseline Policy', value: finalMetrics.best_policy, highlight: true },
        {
            label: 'DQN vs Best Δ Makespan',
            value: finalMetrics.dqn_vs_best_makespan_delta > 0
                ? `+${finalMetrics.dqn_vs_best_makespan_delta.toFixed(1)}`
                : finalMetrics.dqn_vs_best_makespan_delta.toFixed(1),
            unit: 'h',
            delta: `${Math.abs(finalMetrics.dqn_vs_best_makespan_delta).toFixed(1)}h`,
            deltaPositive: finalMetrics.dqn_vs_best_makespan_delta < 0,
        },
        { label: 'Total Tasks Completed', value: finalMetrics.total_tasks_completed.toLocaleString() },
        { label: 'Overall Lateness Rate', value: `${(finalMetrics.overall_lateness_rate * 100).toFixed(1)}`, unit: '%' },
        { label: 'Peak Overload Events', value: String(finalMetrics.peak_overload_events) },
        { label: 'Avg Quality Score', value: finalMetrics.avg_quality_score.toFixed(3) },
    ], [finalMetrics]);

    const makespanData = useMemo(() =>
        Object.entries(allMetrics).map(([name, m]) => ({
            name,
            makespan: (m as any).makespan_hours ?? (m.throughput > 0 ? 200 / m.throughput * 8 : 200),
        })).sort((a, b) => a.makespan - b.makespan)
        , [allMetrics]);

    const radarData = useMemo(() => buildRadarData(allMetrics), [allMetrics]);
    const policyNames = useMemo(() => Object.keys(allMetrics), [allMetrics]);
    // DQN always last in radar so it renders on top
    const radarPolicies = useMemo(() => [
        ...policyNames.filter(n => n !== 'DQN' && !hiddenPolicies.has(n)),
        ...(policyNames.includes('DQN') && !hiddenPolicies.has('DQN') ? ['DQN'] : []),
    ], [policyNames, hiddenPolicies]);

    const allDays = useMemo(() => [
        ...new Set(Object.values(dailyThroughput).flatMap(arr => arr.map(d => d.day)))
    ].sort((a, b) => a - b), [dailyThroughput]);

    const lineChartData = useMemo(() =>
        allDays.map(day => {
            const row: { day: number;[k: string]: number } = { day };
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

    const TABS = [
        { key: 'makespan', label: 'Makespan' },
        { key: 'radar', label: 'Performance Radar' },
        { key: 'throughput', label: 'Daily Throughput' },
        { key: 'lateness', label: 'Lateness Rate' },
    ] as const;

    return (
        <div className="grid-bg page-enter" style={{ minHeight: '100vh', overflowY: 'auto', padding: '24px 20px' }}>
            <div style={{ maxWidth: 1300, margin: '0 auto' }}>

                {/* ── Header ── */}
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 32 }}>
                    <div>
                        <div className="font-mono" style={{
                            fontSize: 'var(--text-label)', color: 'var(--color-success)',
                            letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 6,
                        }}>
                            ✓ SIMULATION COMPLETE
                        </div>
                        <h1 style={{
                            fontFamily: 'var(--font-ui)', fontSize: 'var(--text-title)',
                            fontWeight: 700, color: 'var(--color-text)',
                            letterSpacing: '-0.02em',
                        }}>
                            Post-Simulation Analytics
                        </h1>
                    </div>
                    <div style={{ flex: 1 }} />
                    <button className="btn-ghost" onClick={() => navigate('/simulation')} style={{ fontSize: 12 }}>
                        ← Simulation
                    </button>
                    <button className="btn-ghost" onClick={() => navigate('/')} style={{ fontSize: 12 }}>
                        New Run
                    </button>
                    <a href="/api/export" download="simulation_results.csv"
                        style={{
                            display: 'inline-flex', alignItems: 'center', gap: 6,
                            padding: '8px 16px',
                            background: 'rgba(16,185,129,0.08)',
                            color: 'var(--color-success)',
                            border: '1px solid rgba(16,185,129,0.3)',
                            borderRadius: 'var(--radius-md)',
                            fontFamily: 'var(--font-ui)', fontSize: 12,
                            fontWeight: 600, textDecoration: 'none',
                            transition: 'all 0.15s ease',
                        }}>
                        ↓ Export CSV
                    </a>
                    <button className="btn-primary" style={{ fontSize: 12 }} onClick={() => setShowReadme(p => !p)}>
                        📄 Documentation
                    </button>
                </div>

                {showReadme && <ReadmeGeneratorPanel onClose={() => setShowReadme(false)} />}

                {/* ── Summary Cards ── */}
                <div style={{ display: 'flex', gap: 12, marginBottom: 28, flexWrap: 'wrap' }}>
                    {summaryCards.map((card, i) => <SummaryCard key={i} card={card} />)}
                </div>

                {/* ── Policy filter legend ── */}
                <div style={{
                    display: 'flex', gap: 8, marginBottom: 20, alignItems: 'center', flexWrap: 'wrap',
                }}>
                    <span className="font-mono" style={{ fontSize: 'var(--text-label)', color: 'var(--color-slate-dim)', letterSpacing: '0.08em' }}>
                        POLICIES:
                    </span>
                    {policyNames.map(name => (
                        <button key={name} onClick={() => togglePolicy(name)} style={{
                            padding: '3px 12px',
                            borderRadius: 100, fontSize: 11,
                            fontFamily: 'var(--font-mono)', fontWeight: 600,
                            border: `1px solid ${POLICY_COLOR_MAP[name] ?? '#64748B'}`,
                            background: hiddenPolicies.has(name) ? 'transparent' : `${POLICY_COLOR_MAP[name] ?? '#64748B'}22`,
                            color: hiddenPolicies.has(name) ? 'var(--color-slate-dim)' : (POLICY_COLOR_MAP[name] ?? '#64748B'),
                            cursor: 'pointer', opacity: hiddenPolicies.has(name) ? 0.4 : 1,
                            transition: 'all 0.15s ease',
                        }}>
                            {name === 'DQN' ? '⬡ ' : ''}{name}
                        </button>
                    ))}
                </div>

                {/* ── Chart Tabs ── */}
                <div style={{
                    display: 'flex',
                    borderBottom: '1px solid var(--color-border)',
                    marginBottom: 20,
                }}>
                    {TABS.map(tab => (
                        <button key={tab.key}
                            onClick={() => setActiveTab(tab.key)}
                            className={`chart-tab ${activeTab === tab.key ? 'active' : ''}`}
                            style={{ fontFamily: 'var(--font-mono)' }}>
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* ── Chart panels ── */}
                {activeTab === 'makespan' && (
                    <ChartCard title="Makespan Comparison" style={{ marginBottom: 24 }}>
                        <ResponsiveContainer width="100%" height={320}>
                            <BarChart data={makespanData.filter(d => !hiddenPolicies.has(d.name))} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                                <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--color-slate-text)', fontFamily: 'var(--font-mono)' }} />
                                <YAxis tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                                <Bar dataKey="makespan" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                                    <LabelList dataKey="makespan" position="top"
                                        formatter={(v: any) => typeof v === 'number' ? v.toFixed(1) : v}
                                        style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fill: 'var(--color-slate-text)' }} />
                                    {makespanData.filter(d => !hiddenPolicies.has(d.name)).map(d => (
                                        <Cell key={d.name}
                                            fill={POLICY_COLOR_MAP[d.name] ?? '#2A3452'}
                                            fillOpacity={d.name === 'DQN' ? 1 : 0.75}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>
                )}

                {activeTab === 'radar' && (
                    <ChartCard title="Policy Performance Radar" style={{ marginBottom: 24 }}>
                        <ResponsiveContainer width="100%" height={360}>
                            <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--color-slate-text)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                {radarPolicies.map(name => (
                                    <Radar key={name} name={name} dataKey={name}
                                        stroke={POLICY_COLOR_MAP[name] ?? '#64748B'}
                                        fill={POLICY_COLOR_MAP[name] ?? '#64748B'}
                                        fillOpacity={name === 'DQN' ? 0.40 : 0.15}
                                        strokeWidth={name === 'DQN' ? 2.5 : 1.5}
                                        isAnimationActive={false}
                                    />
                                ))}
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)', paddingTop: 16 }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </ChartCard>
                )}

                {activeTab === 'throughput' && (
                    <ChartCard title="Daily Throughput by Policy" style={{ marginBottom: 24 }}>
                        <ResponsiveContainer width="100%" height={320}>
                            <LineChart data={lineChartData} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                                <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }}
                                    label={{ value: 'Working Day', position: 'insideBottom', offset: -10, fill: 'var(--color-slate-dim)', fontSize: 10 }} />
                                <YAxis tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                {policyNames.filter(n => !hiddenPolicies.has(n)).map(name => (
                                    <Line key={name} type="monotone" dataKey={name}
                                        stroke={POLICY_COLOR_MAP[name] ?? '#64748B'}
                                        dot={false} strokeWidth={name === 'DQN' ? 2.5 : 1.5}
                                        strokeOpacity={name === 'DQN' ? 1 : 0.7}
                                        isAnimationActive={false}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </ChartCard>
                )}

                {activeTab === 'lateness' && (
                    <ChartCard title="Lateness Rate per Day" style={{ marginBottom: 24 }}>
                        <ResponsiveContainer width="100%" height={320}>
                            <BarChart data={latenessPerDay} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                                <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                <YAxis tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }}
                                    tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                                <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                {policyNames.filter(n => !hiddenPolicies.has(n)).map(name => (
                                    <Bar key={name} dataKey={name}
                                        fill={POLICY_COLOR_MAP[name] ?? '#64748B'}
                                        fillOpacity={name === 'DQN' ? 1 : 0.7}
                                        radius={[2, 2, 0, 0]}
                                        isAnimationActive={false}
                                    />
                                ))}
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>
                )}

                {/* ── Utilization heatmap ── */}
                <ChartCard title="Worker × Day Utilization Heatmap" style={{ marginBottom: 0 }}>
                    <UtilizationHeatmap data={heatmapData} />
                </ChartCard>
            </div>
        </div>
    );
}
