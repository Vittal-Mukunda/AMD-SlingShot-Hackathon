/**
 * TaskQueue.tsx — Scrollable task queue panel with urgency badges and skill requirements.
 */
import React from 'react';
import type { QueueItem } from '../../types/simulation';

interface Props {
    queue: QueueItem[];
    currentTick: number;
}

const URGENCY_COLORS: Record<string, string> = {
    low: '#64748B',
    medium: '#3B82F6',
    high: '#F97316',
    critical: '#EF4444',
};

const URGENCY_BG: Record<string, string> = {
    low: 'rgba(100,116,139,0.1)',
    medium: 'rgba(59,130,246,0.1)',
    high: 'rgba(249,115,22,0.1)',
    critical: 'rgba(239,68,68,0.12)',
};

export default function TaskQueue({ queue, currentTick }: Props) {
    const urgent = queue.filter(t => t.slots_remaining <= 8).length;

    return (
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)' }}>
            {/* Header */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '6px 12px', background: 'var(--color-panel)',
                borderBottom: '1px solid var(--color-border)', flexShrink: 0
            }}>
                <span className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                    Task Queue
                </span>
                <span className="badge badge-slate font-mono" style={{ fontSize: 9 }}>
                    {queue.length} pending
                </span>
                {urgent > 0 && (
                    <span className="badge badge-danger font-mono" style={{ fontSize: 9 }}>
                        {urgent} URGENT
                    </span>
                )}
            </div>

            {/* Scrollable list */}
            <div style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden' }}>
                <div style={{ display: 'flex', gap: 6, padding: '8px 12px', height: '100%', alignItems: 'center' }}>
                    {queue.length === 0 ? (
                        <span className="font-mono" style={{ color: 'var(--color-slate-dim)', fontSize: 11 }}>
                            No tasks pending
                        </span>
                    ) : (
                        queue.slice(0, 50).map(task => (
                            <TaskCard key={task.task_id} task={task} currentTick={currentTick} />
                        ))
                    )}
                    {queue.length > 50 && (
                        <span className="font-mono" style={{ color: 'var(--color-slate-dim)', fontSize: 10, flexShrink: 0 }}>
                            +{queue.length - 50} more
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}

function TaskCard({ task, currentTick }: { task: QueueItem; currentTick: number }) {
    const urgency = task.urgency_label;
    const slotsLeft = task.slots_remaining;
    const isUrgent = slotsLeft <= 8;
    const isCritical = task.priority === 3;

    return (
        <div style={{
            flexShrink: 0,
            width: 110,
            padding: '6px 8px',
            background: isCritical ? 'rgba(239,68,68,0.08)' : URGENCY_BG[urgency],
            border: `1px solid ${URGENCY_COLORS[urgency]}`,
            borderRadius: 2,
            position: 'relative',
        }}>
            {/* Urgency badge */}
            <div className="badge" style={{
                background: 'transparent',
                color: URGENCY_COLORS[urgency],
                border: 'none',
                fontSize: 8, padding: 0, marginBottom: 4,
                letterSpacing: '0.08em'
            }}>
                {urgency.toUpperCase()}
            </div>

            {/* Task ID */}
            <div className="font-mono" style={{
                fontSize: 11, fontWeight: 700, color: 'var(--color-text)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
            }}>
                {task.task_id}
            </div>

            {/* Duration */}
            <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', marginTop: 2 }}>
                {task.duration_slots}s · skill≥{task.required_skill.toFixed(1)}
            </div>

            {/* Slots remaining */}
            <div className="num font-mono" style={{
                fontSize: 9, marginTop: 3,
                color: isUrgent ? 'var(--color-danger)' : 'var(--color-slate-text)'
            }}>
                {slotsLeft} slots left
                {isUrgent && <span style={{ marginLeft: 3 }}>⚠</span>}
            </div>
        </div>
    );
}
