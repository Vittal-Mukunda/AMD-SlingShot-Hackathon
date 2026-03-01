/**
 * WorkerSidebar.tsx — Left sidebar showing worker states with circular fatigue gauges.
 * Fatigue levels: 0-1 (fresh/green) → 1-2 (tired/amber) → 2-2.6 (exhausted/orange) → 2.6+ (burnout/red)
 */
import React from 'react';
import type { WorkerState } from '../../types/simulation';
import type { SimConfig } from '../../types/config';

interface Props {
    workers: WorkerState[];
    simConfig: SimConfig;
    currentTick: number;
}

// Circular SVG fatigue gauge
function FatigueGauge({ fatigue }: { fatigue: number }) {
    const RADIUS = 20;
    const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
    const MAX_FATIGUE = 3.0;
    const progress = Math.min(1, fatigue / MAX_FATIGUE);
    const dash = progress * CIRCUMFERENCE;

    const color = fatigue >= 2.6 ? '#EF4444'
        : fatigue >= 2.0 ? '#F97316'
            : fatigue >= 1.0 ? '#F59E0B'
                : '#22C55E';

    return (
        <svg width={50} height={50} viewBox="0 0 50 50">
            {/* Background track */}
            <circle cx={25} cy={25} r={RADIUS} fill="none"
                stroke="var(--color-border)" strokeWidth={4} />
            {/* Fatigue arc */}
            <circle cx={25} cy={25} r={RADIUS} fill="none"
                stroke={color} strokeWidth={4}
                strokeDasharray={`${dash} ${CIRCUMFERENCE}`}
                strokeLinecap="round"
                transform="rotate(-90 25 25)"
                style={{ transition: 'stroke-dasharray 0.5s ease, stroke 0.5s ease' }}
            />
            {/* Center text */}
            <text x={25} y={29} textAnchor="middle"
                fill={color} fontSize={10} fontWeight={700}
                fontFamily="var(--font-mono)">
                {fatigue.toFixed(1)}
            </text>
        </svg>
    );
}

function WorkerCard({ worker, index }: { worker: WorkerState; index: number }) {
    const isUnavailable = worker.availability === 0;
    const isBusy = worker.assigned_tasks.length > 0;

    const statusColor = isUnavailable ? 'var(--color-danger)'
        : isBusy ? 'var(--color-amber)'
            : 'var(--color-success)';
    const statusLabel = isUnavailable ? 'BURNOUT' : isBusy ? 'WORKING' : 'IDLE';

    return (
        <div style={{
            padding: '10px 12px',
            borderBottom: '1px solid var(--color-border)',
            opacity: isUnavailable ? 0.65 : 1,
            transition: 'opacity 0.3s ease',
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <FatigueGauge fatigue={worker.fatigue} />
                <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Worker name */}
                    <div className="font-mono" style={{
                        fontSize: 11, fontWeight: 700, color: 'var(--color-text)',
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'
                    }}>
                        {worker.name}
                    </div>
                    {/* Status */}
                    <div className="font-mono" style={{ fontSize: 9, color: statusColor, marginTop: 2, letterSpacing: '0.06em' }}>
                        ● {statusLabel}
                    </div>
                    {/* Skill */}
                    <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', marginTop: 2 }}>
                        skill={worker.skill_level.toFixed(2)}
                    </div>
                </div>
            </div>

            {/* Active tasks mini-list */}
            {worker.assigned_tasks.length > 0 && (
                <div style={{ marginTop: 6 }}>
                    {worker.assigned_tasks.slice(0, 2).map((tid, ti) => (
                        <div key={ti} className="font-mono" style={{
                            fontSize: 9, color: 'var(--color-amber)',
                            background: 'rgba(245,158,11,0.08)', borderRadius: 2,
                            padding: '2px 6px', marginTop: 3
                        }}>
                            t{tid}
                        </div>
                    ))}
                    {worker.assigned_tasks.length > 2 && (
                        <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', marginTop: 3 }}>
                            +{worker.assigned_tasks.length - 2} more
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default function WorkerSidebar({ workers, simConfig, currentTick }: Props) {
    // If no worker states yet (waiting for first tick), show placeholders
    const displayCount = Math.max(workers.length, simConfig.num_workers);

    return (
        <div style={{
            width: 168, flexShrink: 0, overflow: 'auto',
            borderRight: '1px solid var(--color-border)',
            background: 'var(--color-panel)',
        }}>
            {/* Header */}
            <div style={{
                padding: '8px 12px', background: 'var(--color-panel)',
                borderBottom: '1px solid var(--color-border)',
                position: 'sticky', top: 0, zIndex: 10
            }}>
                <div className="font-mono" style={{
                    fontSize: 9, color: 'var(--color-slate-dim)',
                    letterSpacing: '0.12em', textTransform: 'uppercase'
                }}>
                    Workers ({displayCount})
                </div>
            </div>

            {workers.length > 0 ? (
                workers.map((w, i) => <WorkerCard key={w.id} worker={w} index={i} />)
            ) : (
                // Placeholder skeletons
                Array.from({ length: simConfig.num_workers }, (_, i) => (
                    <div key={i} style={{
                        padding: '10px 12px', borderBottom: '1px solid var(--color-border)',
                        display: 'flex', alignItems: 'center', gap: 10
                    }}>
                        <div style={{
                            width: 50, height: 50, borderRadius: '50%',
                            background: 'var(--color-bg)', border: '1px solid var(--color-border)'
                        }} />
                        <div>
                            <div className="font-mono" style={{ fontSize: 11, color: 'var(--color-slate-dim)' }}>
                                Worker {i + 1}
                            </div>
                            <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', marginTop: 3 }}>
                                Waiting...
                            </div>
                        </div>
                    </div>
                ))
            )}

            {/* Legend */}
            <div style={{ padding: '12px', borderTop: '1px solid var(--color-border)', marginTop: 'auto' }}>
                <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8 }}>
                    Fatigue
                </div>
                {[
                    { label: 'Fresh', color: '#22C55E', range: '0–1' },
                    { label: 'Tired', color: '#F59E0B', range: '1–2' },
                    { label: 'Exhausted', color: '#F97316', range: '2–2.6' },
                    { label: 'Burnout', color: '#EF4444', range: '2.6+' },
                ].map(({ label, color, range }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                        <span className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-text)' }}>
                            {label} <span style={{ color: 'var(--color-slate-dim)' }}>({range})</span>
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}
