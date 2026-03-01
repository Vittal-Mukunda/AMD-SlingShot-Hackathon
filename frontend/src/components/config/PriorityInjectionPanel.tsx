/**
 * PriorityInjectionPanel.tsx — Shared panel for injecting priority tasks
 * Available in ConfigPage (pre-simulation) and SimulationPage (during Phase 2)
 */
import React, { useState } from 'react';
import type { InjectedTask } from '../../types/config';

interface Props {
    pendingTasks: InjectedTask[];
    onAdd: (task: InjectedTask) => void;
    onRemove: (task_id: string) => void;
    // When used in simulation page, provide emit callback
    onEmit?: (task: InjectedTask) => void;
    currentTick?: number;
}

const URGENCY_LABELS = ['Low', 'Medium', 'High', 'Critical'];
const URGENCY_COLORS = ['#64748B', '#3B82F6', '#F97316', '#EF4444'];

export default function PriorityInjectionPanel({
    pendingTasks, onAdd, onRemove, onEmit, currentTick = 0
}: Props) {
    const [form, setForm] = useState<{
        task_id: string; duration: number; urgency: 0 | 1 | 2 | 3;
        required_skill: number; arrival_tick: number;
    }>({
        task_id: `T${Date.now()}`,
        duration: 4, urgency: 2, required_skill: 0.5, arrival_tick: currentTick,
    });

    const handleAdd = () => {
        if (!form.task_id.trim()) return;
        const task: InjectedTask = { ...form };
        onAdd(task);
        if (onEmit) onEmit(task);
        // Reset
        setForm(prev => ({
            ...prev,
            task_id: `T${Date.now()}`,
            arrival_tick: currentTick,
        }));
    };

    return (
        <div style={{
            background: 'var(--color-panel)', border: '1px solid var(--color-border)',
            borderRadius: 4, padding: '1.5rem', marginBottom: '1.5rem'
        }}>
            <div className="section-header">04 — Priority Task Injection</div>
            <p style={{ color: 'var(--color-slate-text)', fontSize: '0.8125rem', marginBottom: '1rem' }}>
                Inject high-priority tasks before or during the simulation run.
            </p>

            {/* Form */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr auto', gap: '0.75rem', alignItems: 'end', marginBottom: '1rem' }}>
                <div>
                    <label>Task ID</label>
                    <input className="input" value={form.task_id}
                        onChange={e => setForm(p => ({ ...p, task_id: e.target.value }))} />
                </div>
                <div>
                    <label>Duration (slots)</label>
                    <input className="input" type="number" min={1} max={32} value={form.duration}
                        onChange={e => setForm(p => ({ ...p, duration: Number(e.target.value) }))} />
                </div>
                <div>
                    <label>Urgency</label>
                    <select className="select" value={form.urgency}
                        onChange={e => setForm(p => ({ ...p, urgency: Number(e.target.value) as 0 | 1 | 2 | 3 }))}>
                        {URGENCY_LABELS.map((l, i) => <option key={i} value={i}>{l}</option>)}
                    </select>
                </div>
                <div>
                    <label>Req. Skill (0–1)</label>
                    <input className="input" type="number" step={0.1} min={0} max={1} value={form.required_skill}
                        onChange={e => setForm(p => ({ ...p, required_skill: Number(e.target.value) }))} />
                </div>
                <div>
                    <label>Arrival Tick</label>
                    <input className="input" type="number" min={0} value={form.arrival_tick}
                        onChange={e => setForm(p => ({ ...p, arrival_tick: Number(e.target.value) }))} />
                </div>
                <button className="btn-primary" onClick={handleAdd} style={{ whiteSpace: 'nowrap' }}>
                    + Add
                </button>
            </div>

            {/* Pending list */}
            {pendingTasks.length > 0 && (
                <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: '1rem' }}>
                    <div style={{ color: 'var(--color-slate-dim)', fontSize: 10, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                        Queued — {pendingTasks.length} task{pendingTasks.length !== 1 ? 's' : ''}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {pendingTasks.map(task => (
                            <div key={task.task_id} style={{
                                display: 'flex', alignItems: 'center', gap: 8,
                                padding: '4px 10px', background: 'var(--color-bg)',
                                border: `1px solid ${URGENCY_COLORS[task.urgency]}`,
                                borderRadius: 2, fontFamily: 'var(--font-mono)', fontSize: 12,
                            }}>
                                <span style={{ color: URGENCY_COLORS[task.urgency], fontWeight: 700 }}>
                                    [{URGENCY_LABELS[task.urgency].toUpperCase()}]
                                </span>
                                <span style={{ color: 'var(--color-text)' }}>{task.task_id}</span>
                                <span style={{ color: 'var(--color-slate-dim)' }}>dur={task.duration}s skill={task.required_skill}</span>
                                <button onClick={() => onRemove(task.task_id)}
                                    style={{ background: 'none', border: 'none', color: 'var(--color-danger)', cursor: 'pointer', fontSize: 14, padding: 0 }}>×</button>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
