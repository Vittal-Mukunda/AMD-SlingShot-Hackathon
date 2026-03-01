/**
 * ConfigPage.tsx — Full-screen simulation configuration wizard (Module 1)
 */
import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
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

// ── Worker skill preview from seed ───────────────────────────────────────────
function seededRandom(seed: number, n: number): number {
    // LCG — deterministic preview matching Python's numpy seed behavior (approximation)
    let s = seed;
    for (let i = 0; i < n; i++) {
        s = (s * 1664525 + 1013904223) & 0xffffffff;
    }
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

// ── Arrival sparkline data ────────────────────────────────────────────────────
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

// ── Manual worker form ────────────────────────────────────────────────────────
function WorkerRow({
    worker, index, onChange, onDelete
}: {
    worker: ManualWorkerConfig;
    index: number;
    onChange: (i: number, updated: ManualWorkerConfig) => void;
    onDelete: (i: number) => void;
}) {
    return (
        <div style={{
            background: 'var(--color-bg)', border: '1px solid var(--color-border)',
            borderRadius: 4, padding: '1rem', marginBottom: '0.75rem'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span className="font-mono" style={{ color: 'var(--color-amber)', fontSize: 12 }}>
                    WORKER {index + 1}
                </span>
                <button
                    className="btn-ghost"
                    style={{ marginLeft: 'auto', fontSize: 11, padding: '2px 8px', color: 'var(--color-danger)' }}
                    onClick={() => onDelete(index)}
                >Remove</button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                    <label>Name</label>
                    <input className="input" value={worker.name}
                        onChange={e => onChange(index, { ...worker, name: e.target.value })} />
                </div>
                <div>
                    <label>Productivity Rate <span className="num" style={{ color: 'var(--color-amber)' }}>
                        {worker.productivity_rate.toFixed(2)}
                    </span></label>
                    <input type="range" min={0.5} max={1.5} step={0.01}
                        value={worker.productivity_rate}
                        onChange={e => onChange(index, { ...worker, productivity_rate: Number(e.target.value) })}
                        style={{ width: '100%', accentColor: 'var(--color-amber)' }} />
                </div>
                <div>
                    <label>Skill Level <span className="num" style={{ color: 'var(--color-amber)' }}>
                        {worker.skill_level.toFixed(2)}
                    </span></label>
                    <input type="range" min={0} max={1} step={0.01}
                        value={worker.skill_level}
                        onChange={e => onChange(index, { ...worker, skill_level: Number(e.target.value) })}
                        style={{ width: '100%', accentColor: 'var(--color-amber)' }} />
                </div>
                <div>
                    <label>Fatigue Sensitivity <span className="num" style={{ color: 'var(--color-amber)' }}>
                        {worker.fatigue_sensitivity.toFixed(2)}
                    </span></label>
                    <input type="range" min={0.05} max={0.3} step={0.01}
                        value={worker.fatigue_sensitivity}
                        onChange={e => onChange(index, { ...worker, fatigue_sensitivity: Number(e.target.value) })}
                        style={{ width: '100%', accentColor: 'var(--color-amber)' }} />
                </div>
            </div>
        </div>
    );
}

// ── Main ConfigPage ───────────────────────────────────────────────────────────
export default function ConfigPage() {
    const navigate = useNavigate();
    const { startSimulation } = useSimulationStore();
    const [cfg, setCfg] = useState<SimConfig>(DEFAULT_SIM_CONFIG);
    const [errors, setErrors] = useState<Record<string, string>>({});
    const [loading, setLoading] = useState(false);
    const [pendingTasks, setPendingTasks] = useState<InjectedTask[]>([]);

    const totalTicks = computeTotalTicks(cfg);
    const estimatedSecs = estimateRuntimeSeconds(cfg);
    const skillPreview = useMemo(
        () => generateWorkerSkillPreview(cfg.worker_seed, cfg.num_workers),
        [cfg.worker_seed, cfg.num_workers]
    );
    const sparklineData = useMemo(() => generateArrivalSparkline(cfg), [cfg]);

    const update = (patch: Partial<SimConfig>) => setCfg(prev => ({ ...prev, ...patch }));

    // Worker manual config handlers
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

    // Form submission
    const handleSubmit = async () => {
        const errs = validateSimConfig(cfg);
        if (Object.keys(errs).length > 0) { setErrors(errs); return; }
        setErrors({});
        setLoading(true);
        try {
            const payload = {
                ...cfg,
                injected_tasks: pendingTasks,
            };
            const res = await fetch('/api/initialize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error(`API error ${res.status}`);
            startSimulation(cfg);
            navigate('/simulation');
        } catch (e) {
            setErrors({ submit: String(e) });
        } finally {
            setLoading(false);
        }
    };

    const PANEL_STYLE: React.CSSProperties = {
        background: 'var(--color-panel)',
        border: '1px solid var(--color-border)',
        borderRadius: 4,
        padding: '1.5rem',
        marginBottom: '1.5rem',
    };

    const GRID3: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' };
    const GRID2: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' };

    return (
        <div className="grid-bg" style={{ minHeight: '100vh', padding: '2rem', overflowY: 'auto' }}>
            {/* Header */}
            <div style={{ maxWidth: 1100, margin: '0 auto' }}>
                <div style={{ marginBottom: '2rem' }}>
                    <div className="font-mono" style={{ color: 'var(--color-amber)', fontSize: 11, letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 8 }}>
                        DQN WORKFORCE SCHEDULER — CONTROL PANEL
                    </div>
                    <h1 className="font-display" style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--color-text)', marginBottom: 4 }}>
                        Simulation Configuration
                    </h1>
                    <p style={{ color: 'var(--color-slate-text)', fontSize: '0.9rem' }}>
                        Configure all parameters before initializing the two-phase DQN scheduling run.
                    </p>
                </div>

                {/* ── SECTION 1: Simulation Horizon ── */}
                <div style={PANEL_STYLE}>
                    <div className="section-header">01 — Simulation Horizon</div>

                    {/* 8-hour workday notice */}
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)',
                        borderRadius: 4, padding: '0.625rem 1rem', marginBottom: '1.25rem'
                    }}>
                        <span style={{ color: 'var(--color-amber)', fontSize: 13 }}>⚡</span>
                        <span className="font-mono" style={{ fontSize: 12, color: 'var(--color-amber)' }}>
                            HARD CONSTRAINT: Workday is fixed at 8 hours (16×30min slots). This cannot be modified.
                        </span>
                    </div>

                    <div style={GRID3}>
                        <div>
                            <label>Phase 1 Days (Baseline Observation)</label>
                            <input className="input" type="number" min={1} max={60} value={cfg.days_phase1}
                                onChange={e => update({ days_phase1: Number(e.target.value) })} />
                            {errors.days_phase1 && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.days_phase1}</p>}
                        </div>
                        <div>
                            <label>Phase 2 Days (DQN Control)</label>
                            <input className="input" type="number" min={1} max={30} value={cfg.days_phase2}
                                onChange={e => update({ days_phase2: Number(e.target.value) })} />
                            {errors.days_phase2 && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.days_phase2}</p>}
                        </div>
                        <div>
                            <label>Random Seed</label>
                            <input className="input" type="number" value={cfg.seed}
                                onChange={e => update({ seed: Number(e.target.value) })} />
                        </div>
                    </div>

                    {/* Computed stats */}
                    <div style={{
                        display: 'flex', gap: '2rem', marginTop: '1.25rem',
                        padding: '1rem', background: 'var(--color-bg)', borderRadius: 4
                    }}>
                        {[
                            ['Total Working Days', `${cfg.days_phase1 + cfg.days_phase2}`],
                            ['Total Ticks (30min slots)', totalTicks.toLocaleString()],
                            ['Hours per Day', '8.0h (fixed)'],
                            ['Est. Runtime', `~${estimatedSecs < 60 ? estimatedSecs.toFixed(0) + 's' : (estimatedSecs / 60).toFixed(1) + 'min'}`],
                        ].map(([label, val]) => (
                            <div key={label}>
                                <div style={{ color: 'var(--color-slate-dim)', fontSize: 10, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</div>
                                <div className="num" style={{ color: 'var(--color-amber)', fontSize: '1.25rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{val}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* ── SECTION 2: Worker Configuration ── */}
                <div style={PANEL_STYLE}>
                    <div className="section-header">02 — Worker Configuration</div>

                    {/* Mode toggle */}
                    <div style={{ display: 'flex', gap: 8, marginBottom: '1.25rem' }}>
                        {(['auto', 'manual'] as const).map(mode => (
                            <button
                                key={mode}
                                onClick={() => update({ worker_mode: mode })}
                                style={{
                                    padding: '0.375rem 1rem',
                                    background: cfg.worker_mode === mode ? 'var(--color-amber)' : 'var(--color-bg)',
                                    color: cfg.worker_mode === mode ? '#000' : 'var(--color-slate-text)',
                                    border: `1px solid ${cfg.worker_mode === mode ? 'var(--color-amber)' : 'var(--color-border)'}`,
                                    borderRadius: 2, fontFamily: 'var(--font-mono)', fontSize: 12,
                                    fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
                                    cursor: 'pointer', transition: 'all 0.15s ease'
                                }}
                            >
                                {mode === 'auto' ? 'Auto-Generate (Seed)' : 'Manual Configuration'}
                            </button>
                        ))}
                    </div>

                    {cfg.worker_mode === 'auto' ? (
                        <div style={GRID2}>
                            <div>
                                <div style={GRID2}>
                                    <div>
                                        <label>Seed</label>
                                        <input className="input" type="number" value={cfg.worker_seed}
                                            onChange={e => update({ worker_seed: Number(e.target.value) })} />
                                    </div>
                                    <div>
                                        <label>Num Workers</label>
                                        <input className="input" type="number" min={1} max={25} value={cfg.num_workers}
                                            onChange={e => update({ num_workers: Number(e.target.value) })} />
                                        {errors.num_workers && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.num_workers}</p>}
                                    </div>
                                </div>
                                <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'var(--color-bg)', borderRadius: 4, fontSize: 12, color: 'var(--color-slate-text)', fontFamily: 'var(--font-mono)' }}>
                                    {skillPreview.map((w, i) => (
                                        <div key={i} style={{ marginBottom: 4 }}>
                                            <span style={{ color: 'var(--color-amber)' }}>{w.subject}</span>
                                            <span style={{ marginLeft: 8 }}>skill={w.skill.toFixed(2)} prod={w.productivity.toFixed(2)} fatigue_res={w.fatigue_resist.toFixed(2)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            {/* Radar thumbnail */}
                            <div>
                                <div style={{ color: 'var(--color-slate-dim)', fontSize: 10, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                                    Skill Distribution Preview
                                </div>
                                <ResponsiveContainer width="100%" height={200}>
                                    <RadarChart data={skillPreview}>
                                        <PolarGrid stroke="var(--color-border)" />
                                        <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--color-slate-text)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                        <Radar name="Skill" dataKey="skill" stroke="var(--color-amber)" fill="var(--color-amber)" fillOpacity={0.25} />
                                        <Radar name="Productivity" dataKey="productivity" stroke="#22C55E" fill="#22C55E" fillOpacity={0.15} />
                                        <RechartsTooltip contentStyle={{ background: '#1E2433', border: '1px solid #2A3452', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    ) : (
                        <div>
                            {errors.manual_workers && (
                                <p style={{ color: 'var(--color-danger)', fontSize: 11, marginBottom: 12 }}>{errors.manual_workers}</p>
                            )}
                            {cfg.manual_workers.map((w, i) => (
                                <WorkerRow key={i} worker={w} index={i} onChange={updateWorker} onDelete={deleteWorker} />
                            ))}
                            <button className="btn-ghost" onClick={addWorker} style={{ marginTop: 8 }}>
                                + Add Worker
                            </button>
                        </div>
                    )}
                </div>

                {/* ── SECTION 3: Task Arrival Distribution ── */}
                <div style={PANEL_STYLE}>
                    <div className="section-header">03 — Task Arrival Distribution</div>
                    <div style={GRID2}>
                        <div>
                            <div style={GRID2}>
                                <div>
                                    <label>Distribution</label>
                                    <select className="select" value={cfg.arrival_distribution}
                                        onChange={e => update({ arrival_distribution: e.target.value as SimConfig['arrival_distribution'], arrival_params: {} })}>
                                        <option value="poisson">Poisson</option>
                                        <option value="uniform">Uniform</option>
                                        <option value="burst">Burst</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </div>
                                <div>
                                    <label>Total Tasks</label>
                                    <input className="input" type="number" min={10} max={500} value={cfg.task_count}
                                        onChange={e => update({ task_count: Number(e.target.value) })} />
                                    {errors.task_count && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.task_count}</p>}
                                </div>
                            </div>
                            {/* Contextual param fields */}
                            <div style={{ marginTop: '1rem' }}>
                                {cfg.arrival_distribution === 'poisson' && (
                                    <div>
                                        <label>Mean Tasks per Day (λ)</label>
                                        <input className="input" type="number" step={0.5} min={0.5} max={20}
                                            value={cfg.arrival_params?.rate ?? 3.5}
                                            onChange={e => update({ arrival_params: { rate: Number(e.target.value) } })} />
                                    </div>
                                )}
                                {cfg.arrival_distribution === 'uniform' && (
                                    <div style={GRID2}>
                                        <div>
                                            <label>Min Tasks/Day</label>
                                            <input className="input" type="number" min={0}
                                                value={cfg.arrival_params?.min_per_day ?? 1}
                                                onChange={e => update({ arrival_params: { ...cfg.arrival_params, min_per_day: Number(e.target.value) } })} />
                                        </div>
                                        <div>
                                            <label>Max Tasks/Day</label>
                                            <input className="input" type="number"
                                                value={cfg.arrival_params?.max_per_day ?? 8}
                                                onChange={e => update({ arrival_params: { ...cfg.arrival_params, max_per_day: Number(e.target.value) } })} />
                                        </div>
                                    </div>
                                )}
                                {cfg.arrival_distribution === 'burst' && (
                                    <div>
                                        <label>Burst Multiplier (every 5th day)</label>
                                        <input className="input" type="number" step={0.5} min={1.5} max={10}
                                            value={cfg.arrival_params?.burst_multiplier ?? 3}
                                            onChange={e => update({ arrival_params: { burst_multiplier: Number(e.target.value) } })} />
                                    </div>
                                )}
                                {cfg.arrival_distribution === 'custom' && (
                                    <div>
                                        <label>Daily Task Counts (comma-separated)</label>
                                        <input className="input"
                                            placeholder={`e.g. 2,5,3,8,1,...`}
                                            onChange={e => {
                                                const vals = e.target.value.split(',').map(Number).filter(n => !isNaN(n));
                                                update({ arrival_params: { daily_overrides: vals } });
                                            }} />
                                    </div>
                                )}
                            </div>
                        </div>
                        {/* Sparkline preview */}
                        <div>
                            <div style={{ color: 'var(--color-slate-dim)', fontSize: 10, fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                                Arrival Rate Preview
                            </div>
                            <ResponsiveContainer width="100%" height={160}>
                                <LineChart data={sparklineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                    <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                    <YAxis tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                    <Line type="monotone" dataKey="tasks" stroke="var(--color-amber)" dot={false} strokeWidth={2} />
                                    <RechartsTooltip contentStyle={{ background: '#1E2433', border: '1px solid #2A3452', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* ── SECTION 4: Priority Task Injection ── */}
                <PriorityInjectionPanel
                    pendingTasks={pendingTasks}
                    onAdd={(task) => setPendingTasks(prev => [...prev, task])}
                    onRemove={(id) => setPendingTasks(prev => prev.filter(t => t.task_id !== id))}
                />

                {/* ── Submit ── */}
                {errors.submit && (
                    <div style={{
                        padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.1)',
                        border: '1px solid rgba(239,68,68,0.3)', borderRadius: 4,
                        color: 'var(--color-danger)', fontFamily: 'var(--font-mono)', fontSize: 12, marginBottom: '1rem'
                    }}>
                        {errors.submit}
                    </div>
                )}
                <button
                    className="btn-primary"
                    style={{ width: '100%', justifyContent: 'center', fontSize: '1rem', padding: '0.875rem' }}
                    onClick={handleSubmit}
                    disabled={loading}
                    id="initialize-simulation-btn"
                >
                    {loading ? '⟳  Initializing...' : 'Initialize Simulation →'}
                </button>
                <p style={{ textAlign: 'center', color: 'var(--color-slate-dim)', fontSize: 11, fontFamily: 'var(--font-mono)', marginTop: '0.75rem' }}>
                    Sends config to backend and navigates to live simulation view
                </p>
            </div>
        </div>
    );
}
