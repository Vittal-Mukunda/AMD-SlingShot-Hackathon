/**
 * EpsilonChart.tsx — Live epsilon decay line chart for Phase 2.
 * Animated line drawing with gradient fill. Shows 0.3 and 0.1 threshold lines.
 */
import React, { useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip,
    ResponsiveContainer, ReferenceLine, Area, AreaChart, CartesianGrid,
} from 'recharts';

interface Props {
    history: number[];   // epsilon values over time
    current: number;
}

const TOOLTIP_STYLE = {
    background: '#0f172a',
    border: '1px solid rgba(255,255,255,0.1)',
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    borderRadius: 6,
};

export default function EpsilonChart({ history, current }: Props) {
    const data = useMemo(() =>
        history.length === 0
            ? [{ step: 0, epsilon: 1.0 }]
            : history.map((v, i) => ({ step: i, epsilon: v })),
        [history]
    );

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <span className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                    ε Exploration Decay
                </span>
                <span className="num font-mono" style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-amber)' }}>
                    ε = {current.toFixed(3)}
                </span>
            </div>
            <ResponsiveContainer width="100%" height={80}>
                <AreaChart data={data} margin={{ top: 2, right: 4, left: -28, bottom: 0 }}>
                    <defs>
                        <linearGradient id="epsilonGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="step" hide />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 8, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                    <Area
                        type="monotone"
                        dataKey="epsilon"
                        stroke="#f59e0b"
                        fill="url(#epsilonGrad)"
                        strokeWidth={1.5}
                        dot={false}
                        isAnimationActive={false}
                    />
                    <ReferenceLine y={0.3} stroke="#3b82f6" strokeDasharray="4 3" strokeOpacity={0.5}
                        label={{ value: '0.3', position: 'right', fontSize: 8, fill: '#3b82f6', fontFamily: 'var(--font-mono)' }} />
                    <ReferenceLine y={0.1} stroke="#10b981" strokeDasharray="4 3" strokeOpacity={0.5}
                        label={{ value: '0.1', position: 'right', fontSize: 8, fill: '#10b981', fontFamily: 'var(--font-mono)' }} />
                    <RechartsTooltip
                        contentStyle={TOOLTIP_STYLE}
                        formatter={(v: any) => [typeof v === 'number' ? v.toFixed(4) : v, 'ε']}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}
