/**
 * ComparisonStrip.tsx — Right panel: live head-to-head DQN vs Baselines.
 * Phase 1: Shows task queue depth sparkline and worker fatigue distribution.
 * Phase 2: Shows full head-to-head comparison strip.
 *
 * NO RL training internals exposed (no loss, Q-values, epsilon, replay buffer, etc.)
 */
import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip,
    LineChart, Line, ResponsiveContainer, Cell
} from 'recharts';
import type { HeadToHeadMetrics } from '../../types/metrics';
import { POLICY_COLORS } from '../../types/metrics';

interface Props {
    phase: number;
    hhMetrics: HeadToHeadMetrics;
    workerCount: number;
}

const TOOLTIP_STYLE = {
    background: '#1E2433', border: '1px solid #2A3452',
    fontSize: 11, fontFamily: 'var(--font-mono)'
};

function SectionTitle({ children }: { children: React.ReactNode }) {
    return (
        <div className="font-mono" style={{
            fontSize: 9, color: 'var(--color-slate-dim)',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            marginBottom: 8, paddingBottom: 4,
            borderBottom: '1px solid var(--color-border)'
        }}>{children}</div>
    );
}

function MetricRow({ label, value, color, unit = '' }: {
    label: string; value: string | number; color?: string; unit?: string;
}) {
    return (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span className="font-mono" style={{ fontSize: 10, color: 'var(--color-slate-text)' }}>{label}</span>
            <span className="num font-mono" style={{ fontSize: 12, fontWeight: 700, color: color ?? 'var(--color-text)' }}>
                {value}{unit}
            </span>
        </div>
    );
}

export default function ComparisonStrip({ phase, hhMetrics, workerCount }: Props) {
    // Throughput bar data
    const throughputData = [
        { name: 'DQN', value: hhMetrics.dqn_throughput },
        ...Object.entries(hhMetrics.baseline_throughputs).map(([name, value]) => ({ name, value }))
    ];

    // Lateness rate data
    const latenessData = [
        { name: 'DQN', value: hhMetrics.dqn_lateness_rate },
        ...Object.entries(hhMetrics.baseline_lateness_rates).map(([name, value]) => ({ name, value }))
    ];

    // Queue depth sparkline
    const sparkData = hhMetrics.queue_depth_history.map((v, i) => ({ i, v }));

    // Worker fatigue mini-bars
    const fatigueData = hhMetrics.dqn_worker_fatigue.map((f, i) => ({
        name: `W${i + 1}`, value: f,
        color: f >= 2.6 ? '#EF4444' : f >= 2 ? '#F97316' : f >= 1 ? '#F59E0B' : '#22C55E'
    }));

    const panelPad: React.CSSProperties = { padding: '12px', borderBottom: '1px solid var(--color-border)' };

    return (
        <div style={{ height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <div style={{
                padding: '8px 12px', background: 'var(--color-panel)',
                borderBottom: '1px solid var(--color-border)', position: 'sticky', top: 0, zIndex: 5
            }}>
                <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
                    {phase === 2 ? 'Live Comparison' : 'Session Metrics'}
                </div>
            </div>

            {/* ── Phase 1: basic metrics ── */}
            {phase <= 1 && (
                <div style={panelPad}>
                    <SectionTitle>Observation Phase</SectionTitle>
                    <MetricRow label="Queue Depth" value={sparkData.length > 0 ? sparkData[sparkData.length - 1].v : 0} color="var(--color-amber)" />
                    <MetricRow label="Workers Online" value={workerCount} color="var(--color-success)" />
                    <div style={{ marginTop: 12 }}>
                        <SectionTitle>Queue Depth (last 60 ticks)</SectionTitle>
                        <ResponsiveContainer width="100%" height={80}>
                            <LineChart data={sparkData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                                <Line type="monotone" dataKey="v" stroke="var(--color-amber)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                                <XAxis dataKey="i" hide />
                                <YAxis hide />
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                    <div style={{ marginTop: 12 }}>
                        <SectionTitle>DQN Passive — Recording Transitions</SectionTitle>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0' }}>
                            <span style={{
                                width: 8, height: 8, borderRadius: '50%', background: '#60A5FA',
                                display: 'inline-block', animation: 'pulse-amber 2s ease-in-out infinite'
                            }} />
                            <span className="font-mono" style={{ fontSize: 10, color: '#60A5FA' }}>
                                Buffer filling...
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Phase 2: full comparison ── */}
            {phase === 2 && (
                <>
                    {/* Live Throughput bar */}
                    <div style={panelPad}>
                        <SectionTitle>Throughput (tasks/day)</SectionTitle>
                        <ResponsiveContainer width="100%" height={120}>
                            <BarChart data={throughputData} layout="vertical" margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
                                <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                <YAxis type="category" dataKey="name" tick={{ fontSize: 9, fill: 'var(--color-text)', fontFamily: 'var(--font-mono)' }} width={42} />
                                <Bar dataKey="value" radius={[0, 2, 2, 0]} isAnimationActive={false}>
                                    {throughputData.map((d) => (
                                        <Cell key={d.name}
                                            fill={d.name === 'DQN' ? 'var(--color-amber)' : POLICY_COLORS[d.name] ?? '#64748B'}
                                            fillOpacity={d.name === 'DQN' ? 1 : 0.65}
                                        />
                                    ))}
                                </Bar>
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Lateness rate comparison */}
                    <div style={panelPad}>
                        <SectionTitle>Lateness Rate (%)</SectionTitle>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {latenessData.map(d => (
                                <div key={d.name} style={{
                                    flex: 1, minWidth: 55,
                                    padding: '4px 6px',
                                    background: 'var(--color-bg)', borderRadius: 2,
                                    border: `1px solid ${d.name === 'DQN' ? 'var(--color-amber)' : 'var(--color-border)'}`
                                }}>
                                    <div className="font-mono" style={{ fontSize: 9, color: d.name === 'DQN' ? 'var(--color-amber)' : 'var(--color-slate-dim)' }}>
                                        {d.name}
                                    </div>
                                    <div className="num font-mono" style={{
                                        fontSize: 13, fontWeight: 700,
                                        color: d.value > 0.3 ? 'var(--color-danger)' : d.value > 0.1 ? 'var(--color-amber)' : 'var(--color-success)'
                                    }}>
                                        {(d.value * 100).toFixed(1)}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Worker fatigue distribution stacked mini-bars */}
                    <div style={panelPad}>
                        <SectionTitle>Worker Fatigue</SectionTitle>
                        {fatigueData.length > 0 ? (
                            <ResponsiveContainer width="100%" height={80}>
                                <BarChart data={fatigueData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                                    <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                    <YAxis domain={[0, 3]} hide />
                                    <Bar dataKey="value" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                                        {fatigueData.map((d, i) => (
                                            <Cell key={i} fill={d.color} />
                                        ))}
                                    </Bar>
                                    <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <span className="font-mono" style={{ fontSize: 10, color: 'var(--color-slate-dim)' }}>Waiting...</span>
                        )}
                    </div>

                    {/* Queue depth sparkline */}
                    <div style={panelPad}>
                        <SectionTitle>Queue Depth (last 60 ticks)</SectionTitle>
                        <ResponsiveContainer width="100%" height={70}>
                            <LineChart data={sparkData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                                <Line type="monotone" dataKey="v" stroke="var(--color-success)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                                <XAxis dataKey="i" hide />
                                <YAxis hide />
                                <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {/* DQN status */}
                    <div style={{ ...panelPad, border: 'none' }}>
                        <SectionTitle>DQN Status</SectionTitle>
                        <MetricRow label="Throughput" value={hhMetrics.dqn_throughput.toFixed(2)} color="var(--color-amber)" unit="/day" />
                        <MetricRow label="Lateness" value={(hhMetrics.dqn_lateness_rate * 100).toFixed(1)} color={hhMetrics.dqn_lateness_rate > 0.2 ? 'var(--color-danger)' : 'var(--color-success)'} unit="%" />
                    </div>
                </>
            )}
        </div>
    );
}
