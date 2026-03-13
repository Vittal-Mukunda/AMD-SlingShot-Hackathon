/**
 * CircularGauge.tsx — SVG circular arc gauge for worker fatigue.
 * Arc fills clockwise 0→100%. Color transitions: green→yellow→red.
 * Burns out at 2.6: card pulses red with glow.
 */
import React from 'react';

interface Props {
    value: number;     // 0–3 (fatigue scale)
    max?: number;      // 3.0 default
    size?: number;     // px
    strokeWidth?: number;
    showValue?: boolean;
    label?: string;
}

function fatigueColor(v: number): string {
    if (v >= 2.6) return '#f43f5e';
    if (v >= 2.0) return '#f97316';
    if (v >= 1.0) return '#f59e0b';
    return '#10b981';
}

export default function CircularGauge({
    value, max = 3.0, size = 52, strokeWidth = 4, showValue = true, label
}: Props) {
    const R = (size - strokeWidth * 2) / 2;
    const circumference = 2 * Math.PI * R;
    const pct = Math.min(1, Math.max(0, value / max));
    const dashOffset = circumference * (1 - pct);
    const cx = size / 2;
    const cy = size / 2;
    const color = fatigueColor(value);
    const isBurnout = value >= 2.6;

    return (
        <div
            style={{
                display: 'inline-flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
            }}
        >
            <svg
                width={size}
                height={size}
                viewBox={`0 0 ${size} ${size}`}
                style={isBurnout ? { animation: 'gauge-pulse 1s ease-in-out infinite' } : undefined}
            >
                {/* Background track */}
                <circle
                    cx={cx} cy={cy} r={R}
                    fill="none"
                    stroke="rgba(255,255,255,0.06)"
                    strokeWidth={strokeWidth}
                />
                {/* Colored arc — starts from top (-90°) and fills clockwise */}
                <circle
                    cx={cx} cy={cy} r={R}
                    fill="none"
                    stroke={color}
                    strokeWidth={strokeWidth}
                    strokeDasharray={circumference}
                    strokeDashoffset={dashOffset}
                    strokeLinecap="round"
                    transform={`rotate(-90 ${cx} ${cy})`}
                    style={{
                        transition: 'stroke-dashoffset 0.6s ease, stroke 0.4s ease',
                        filter: isBurnout ? `drop-shadow(0 0 5px ${color})` : undefined,
                    }}
                />
                {/* Glow circle for burnout */}
                {isBurnout && (
                    <circle
                        cx={cx} cy={cy} r={R - 1}
                        fill="none"
                        stroke={color}
                        strokeWidth={1}
                        strokeOpacity={0.25}
                    />
                )}
                {/* Center value */}
                {showValue && (
                    <text
                        x={cx} y={cy + 1}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill={color}
                        fontSize={size > 40 ? 10 : 8}
                        fontWeight={700}
                        fontFamily="var(--font-mono)"
                    >
                        {value.toFixed(1)}
                    </text>
                )}
            </svg>
            {label && (
                <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 8,
                    color: 'var(--color-slate-dim)', letterSpacing: '0.06em',
                }}>
                    {label}
                </span>
            )}
        </div>
    );
}
