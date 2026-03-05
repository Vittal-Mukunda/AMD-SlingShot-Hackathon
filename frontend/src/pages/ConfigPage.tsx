/**
 * ConfigPage.tsx — Enterprise Onboarding Wizard (Redesigned)
 * Premium dark theme with progress stepper, electric blue CTA, and cyan focus rings
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

// ── Helpers ───────────────────────────────────────────────────────────────────
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

// ── Progress Stepper ──────────────────────────────────────────────────────────
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
                                color: state === 'active'
                                    ? 'var(--color-text)'
                                    : state === 'complete'
                                        ? 'var(--color-success)'
                                        : 'var(--color-slate-dim)',
                                fontWeight: state === 'active' ? 600 : 400,
                            }}>{label}</span>
                        </div>
                        {i < STEPS.length - 1 && (
                            <div className="stepper-line" style={{
                                background: i < active
                                    ? 'var(--color-success)'
                                    : 'var(--color-border-bright)',
                                transition: 'background 0.4s ease',
                            }} />
                        )}
                    </React.Fragment>
                );
            })}
        </div>
    );
}

// ── Metric Chip ───────────────────────────────────────────────────────────────
function MetricChip({ label, value, accent }: { label: string; value: string; accent?: string }) {
    return (
        <div style={{ textAlign: 'center' }}>
            <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-label)',
                color: 'var(--color-slate-dim)', textTransform: 'uppercase',
                letterSpacing: '0.08em', marginBottom: 4,
            }}>{label}</div>
            <div className="num" style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-card-val)',
                fontWeight: 700, color: accent ?? 'var(--color-cyan)',
            }}>{value}</div>
        </div>
    );
}

// ── Manual Worker Row ─────────────────────────────────────────────────────────
function WorkerRow({ worker, index, onChange, onDelete }: {
    worker: ManualWorkerConfig;
    index: number;
    onChange: (i: number, u: ManualWorkerConfig) => void;
    onDelete: (i: number) => void;
}) {
    return (
        <div style={{
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid var(--color-border-bright)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
            marginBottom: '12px',
        }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
                <span className="font-mono" style={{
                    color: 'var(--color-amber)', fontSize: 'var(--text-label)',
                    letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 700,
                }}>
                    WORKER {index + 1}
                </span>
                <button className="btn-ghost" onClick={() => onDelete(index)}
                    style={{
                        marginLeft: 'auto', fontSize: 11, padding: '4px 10px',
                        color: 'var(--color-danger)', borderColor: 'rgba(239,68,68,0.3)'
                    }}>
                    Remove
                </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
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
                        style={{ width: '100%', accentColor: 'var(--color-cyan)' }} />
                </div>
                <div>
                    <label>Skill Level <span className="num" style={{ color: 'var(--color-cyan)' }}>
                        {worker.skill_level.toFixed(2)}
                    </span></label>
                    <input type="range" min={0} max={1} step={0.01}
                        value={worker.skill_level}
                        onChange={e => onChange(index, { ...worker, skill_level: Number(e.target.value) })}
                        style={{ width: '100%', accentColor: 'var(--color-cyan)' }} />
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

// ── Section Card ──────────────────────────────────────────────────────────────
function SectionCard({ step, title, children }: {
    step: number; title: string; children: React.ReactNode;
}) {
    return (
        <div style={{
            background: 'var(--color-surface)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-lg)',
            padding: 'var(--pad-card)',
            marginBottom: '20px',
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
                <div style={{
                    width: 28, height: 28, borderRadius: '50%',
                    background: 'var(--color-electric)',
                    color: '#fff', fontFamily: 'var(--font-mono)',
                    fontSize: 12, fontWeight: 700,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0, boxShadow: '0 0 12px rgba(37,99,235,0.4)',
                }}>{step}</div>
                <div>
                    <div className="section-title">{title}</div>
                </div>
            </div>
            <div style={{
                borderTop: '1px solid var(--color-border)',
                paddingTop: 20,
            }}>
                {children}
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

    const GRID2: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' };
    const GRID3: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' };

    const TOOLTIP_STYLE = {
        background: '#0F1629', border: '1px solid rgba(255,255,255,0.1)',
        fontSize: 11, fontFamily: 'var(--font-mono)', borderRadius: 8,
    };

    return (
        <div className="grid-bg page-enter" style={{ minHeight: '100vh', overflowY: 'auto', padding: '32px 16px' }}>
            <div style={{ maxWidth: 960, margin: '0 auto' }}>

                {/* ── Header ── */}
                <div style={{ marginBottom: 36, textAlign: 'center' }}>
                    <div className="font-mono" style={{
                        color: 'var(--color-cyan)', fontSize: 'var(--text-label)',
                        letterSpacing: '0.20em', textTransform: 'uppercase', marginBottom: 12,
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                    }}>
                        <span className="live-dot" style={{ background: 'var(--color-electric)', boxShadow: '0 0 8px var(--color-electric)' }} />
                        DQN WORKFORCE SCHEDULER
                    </div>
                    <h1 style={{
                        fontFamily: 'var(--font-ui)', fontSize: 'var(--text-title)',
                        fontWeight: 700, color: 'var(--color-text)', marginBottom: 8,
                        letterSpacing: '-0.02em',
                    }}>
                        Simulation Configuration
                    </h1>
                    <p style={{ color: 'var(--color-slate-text)', fontSize: 'var(--text-body)', maxWidth: 520, margin: '0 auto' }}>
                        Configure all parameters before initializing the two-phase DQN scheduling run.
                    </p>
                </div>

                {/* ── Progress Stepper ── */}
                <ProgressStepper active={0} />

                {/* ── SECTION 1: Simulation Horizon ── */}
                <SectionCard step={1} title="Simulation Parameters">
                    {/* Constraint notice */}
                    <div style={{
                        display: 'flex', alignItems: 'center', gap: 10,
                        background: 'rgba(245,158,11,0.06)',
                        border: '1px solid rgba(245,158,11,0.2)',
                        borderRadius: 'var(--radius-md)', padding: '10px 14px', marginBottom: 20,
                    }}>
                        <span style={{ color: 'var(--color-amber)', fontSize: 16 }}>⚡</span>
                        <span className="font-mono" style={{ fontSize: 'var(--text-label)', color: 'var(--color-amber)', letterSpacing: '0.06em' }}>
                            HARD CONSTRAINT: Workday is fixed at 8 hours (16 × 30min slots). This cannot be modified.
                        </span>
                    </div>

                    <div style={GRID2}>
                        <div>
                            <label>Simulation Days <span className="num" style={{ color: 'var(--color-cyan)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{cfg.sim_days}</span></label>
                            <input className="input" type="number" min={1} max={365} value={cfg.sim_days}
                                onChange={e => {
                                    const d = Math.max(1, Math.min(365, Number(e.target.value)));
                                    const p1 = Math.max(1, Math.round(d * cfg.phase1_fraction));
                                    const p2 = Math.max(1, d - p1);
                                    update({ sim_days: d, days_phase1: p1, days_phase2: p2 });
                                }} />
                        </div>
                        <div>
                            <label>Random Seed</label>
                            <input className="input" type="number" value={cfg.seed}
                                onChange={e => update({ seed: Number(e.target.value) })} />
                        </div>
                    </div>

                    {/* Phase 1 Observation % slider */}
                    <div style={{ marginTop: 20 }}>
                        <label>
                            Phase 1 Observation %
                            <span className="num" style={{ color: 'var(--color-electric)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                {Math.round(cfg.phase1_fraction * 100)}%
                            </span>
                            <span style={{ color: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)', fontSize: 11, marginLeft: 8 }}>
                                ({cfg.days_phase1}d baseline / {cfg.days_phase2}d DQN)
                            </span>
                        </label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
                            <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-amber)', minWidth: 28 }}>40%</span>
                            <input type="range" min={40} max={80} step={5}
                                value={Math.round(cfg.phase1_fraction * 100)}
                                style={{ flex: 1, accentColor: 'var(--color-electric)' }}
                                onChange={e => {
                                    const frac = Number(e.target.value) / 100;
                                    const p1 = Math.max(1, Math.round(cfg.sim_days * frac));
                                    const p2 = Math.max(1, cfg.sim_days - p1);
                                    update({ phase1_fraction: frac, days_phase1: p1, days_phase2: p2 });
                                }} />
                            <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-electric)', minWidth: 28 }}>80%</span>
                        </div>
                        <div style={{
                            display: 'flex', gap: 0, height: 6, borderRadius: 4, overflow: 'hidden',
                            marginTop: 6,
                        }}>
                            <div style={{ flex: cfg.days_phase1, background: 'rgba(37,99,235,0.35)', transition: 'flex 0.3s' }} />
                            <div style={{ flex: cfg.days_phase2, background: 'rgba(245,158,11,0.4)', transition: 'flex 0.3s' }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                            <span className="font-mono" style={{ fontSize: 10, color: '#60A5FA' }}>■ Phase 1 — Baseline</span>
                            <span className="font-mono" style={{ fontSize: 10, color: 'var(--color-amber)' }}>■ Phase 2 — DQN</span>
                        </div>
                    </div>

                    {/* Computed stats */}
                    <div style={{
                        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                        gap: 12, marginTop: 20,
                        padding: '20px', background: 'rgba(255,255,255,0.02)',
                        border: '1px solid var(--color-border)',
                        borderRadius: 'var(--radius-md)',
                    }}>
                        <MetricChip label="Total Working Days" value={`${cfg.days_phase1 + cfg.days_phase2}`} />
                        <MetricChip label="Total Ticks" value={totalTicks.toLocaleString()} accent="var(--color-electric)" />
                        <MetricChip label="Hours per Day" value="8.0h" accent="var(--color-slate-text)" />
                        <MetricChip
                            label="Est. Runtime"
                            value={estimatedSecs < 60 ? `${estimatedSecs.toFixed(0)}s` : `${(estimatedSecs / 60).toFixed(1)}min`}
                            accent="var(--color-amber)"
                        />
                    </div>
                </SectionCard>

                {/* ── SECTION 2: Worker Configuration ── */}
                <SectionCard step={2} title="Worker Setup">
                    {/* Segmented control */}
                    <div style={{ marginBottom: 20 }}>
                        <label style={{ marginBottom: 12 }}>Worker Generation Mode</label>
                        <div className="segmented-control">
                            <button
                                className={cfg.worker_mode === 'auto' ? 'active' : ''}
                                onClick={() => update({ worker_mode: 'auto' })}
                            >
                                Auto-Generate (Seed)
                            </button>
                            <button
                                className={cfg.worker_mode === 'manual' ? 'active' : ''}
                                onClick={() => update({ worker_mode: 'manual' })}
                            >
                                Manual Configuration
                            </button>
                        </div>
                    </div>

                    {cfg.worker_mode === 'auto' ? (
                        <div style={GRID2}>
                            <div>
                                <div style={GRID2}>
                                    <div>
                                        <label>Worker Seed</label>
                                        <input className="input" type="number" value={cfg.worker_seed}
                                            onChange={e => update({ worker_seed: Number(e.target.value) })} />
                                    </div>
                                    <div>
                                        <label>Num Workers <span className="num" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cyan)', fontWeight: 700 }}>{cfg.num_workers}</span></label>
                                        <input className="input" type="number" min={2} max={20} value={cfg.num_workers}
                                            onChange={e => update({ num_workers: Number(e.target.value) })} />
                                        {errors.num_workers && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.num_workers}</p>}
                                    </div>
                                </div>
                                {/* Max Worker Load slider */}
                                <div style={{ marginTop: 16 }}>
                                    <label>
                                        Max Worker Load
                                        <span className="num" style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-amber)', fontWeight: 700 }}>{cfg.max_worker_load}</span>
                                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--color-slate-dim)', marginLeft: 6 }}>tasks/worker</span>
                                    </label>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8 }}>
                                        <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-slate-dim)', minWidth: 16 }}>3</span>
                                        <input type="range" min={3} max={15} step={1}
                                            value={cfg.max_worker_load}
                                            style={{ flex: 1, accentColor: 'var(--color-amber)' }}
                                            onChange={e => update({ max_worker_load: Number(e.target.value) })} />
                                        <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-slate-dim)', minWidth: 16 }}>15</span>
                                    </div>
                                </div>
                                <div style={{
                                    marginTop: 16, padding: '12px 14px',
                                    background: 'rgba(255,255,255,0.02)',
                                    border: '1px solid var(--color-border)',
                                    borderRadius: 'var(--radius-md)',
                                    fontSize: 12, fontFamily: 'var(--font-mono)',
                                    color: 'var(--color-slate-text)',
                                }}>
                                    {skillPreview.map((w, i) => (
                                        <div key={i} style={{ marginBottom: 4, display: 'flex', gap: 8 }}>
                                            <span style={{ color: 'var(--color-amber)', minWidth: 32 }}>{w.subject}</span>
                                            <span style={{ color: 'var(--color-cyan)' }}>skill={w.skill.toFixed(2)}</span>
                                            <span>prod={w.productivity.toFixed(2)}</span>
                                            <span style={{ color: 'var(--color-slate-dim)' }}>fat_res={w.fatigue_resist.toFixed(2)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div>
                                <div className="section-header" style={{ marginBottom: 10 }}>Skill Distribution Preview</div>
                                <ResponsiveContainer width="100%" height={200}>
                                    <RadarChart data={skillPreview}>
                                        <PolarGrid stroke="rgba(255,255,255,0.06)" />
                                        <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--color-slate-text)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                                        <Radar name="Skill" dataKey="skill" stroke="var(--color-amber)" fill="var(--color-amber)" fillOpacity={0.2} isAnimationActive={false} />
                                        <Radar name="Productivity" dataKey="productivity" stroke="var(--color-cyan)" fill="var(--color-cyan)" fillOpacity={0.15} isAnimationActive={false} />
                                        <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    ) : (
                        <div>
                            {errors.manual_workers && (
                                <p style={{ color: 'var(--color-danger)', fontSize: 12, marginBottom: 12 }}>{errors.manual_workers}</p>
                            )}
                            {cfg.manual_workers.map((w, i) => (
                                <WorkerRow key={i} worker={w} index={i} onChange={updateWorker} onDelete={deleteWorker} />
                            ))}
                            <button className="btn-ghost" onClick={addWorker} style={{ marginTop: 8 }}>
                                + Add Worker
                            </button>
                        </div>
                    )}
                </SectionCard>

                {/* ── SECTION 3: Task Configuration ── */}
                <SectionCard step={3} title="Task Configuration">
                    <div style={GRID2}>
                        <div>
                            <div style={GRID2}>
                                <div>
                                    <label>Distribution Type</label>
                                    <select className="select" value={cfg.arrival_distribution}
                                        onChange={e => update({ arrival_distribution: e.target.value as SimConfig['arrival_distribution'], arrival_params: {} })}>
                                        <option value="poisson">Poisson</option>
                                        <option value="uniform">Uniform</option>
                                        <option value="burst">Burst</option>
                                        <option value="custom">Custom</option>
                                    </select>
                                </div>
                                <div>
                                    <label>Task Cap (max total)</label>
                                    <input className="input" type="number" min={10} max={2000} value={cfg.task_count}
                                        onChange={e => update({ task_count: Number(e.target.value) })} />
                                    {errors.task_count && <p style={{ color: 'var(--color-danger)', fontSize: 11, marginTop: 4 }}>{errors.task_count}</p>}
                                </div>
                            </div>

                            {/* Tasks per day slider */}
                            <div style={{ marginTop: 16 }}>
                                <label>
                                    Tasks per Day&nbsp;
                                    <span className="num" style={{ color: 'var(--color-cyan)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                        {cfg.tasks_per_day}
                                    </span>
                                </label>
                                <input type="range" min={1} max={20} step={1}
                                    value={cfg.tasks_per_day}
                                    onChange={e => update({ tasks_per_day: Number(e.target.value) })}
                                    style={{ width: '100%', accentColor: 'var(--color-cyan)', marginTop: 6 }} />
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                                    <span>1/day</span>
                                    <span style={{ color: 'var(--color-cyan)' }}>Est. ~{cfg.tasks_per_day * (cfg.days_phase1 + cfg.days_phase2)} total</span>
                                    <span>20/day</span>
                                </div>
                            </div>

                            {/* Distribution contextual fields */}
                            <div style={{ marginTop: 16 }}>
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
                                        <input className="input" placeholder="e.g. 2,5,3,8,1,…"
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
                            <div className="section-header" style={{ marginBottom: 10 }}>Arrival Rate Preview</div>
                            <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={sparklineData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                                    <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                    <YAxis tick={{ fontSize: 10, fill: 'var(--color-slate-dim)', fontFamily: 'var(--font-mono)' }} />
                                    <Line type="monotone" dataKey="tasks" stroke="var(--color-cyan)" dot={false} strokeWidth={2} isAnimationActive={false} />
                                    <RechartsTooltip contentStyle={TOOLTIP_STYLE} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </SectionCard>

                {/* ── SECTION 4: Priority Task Injection ── */}
                <PriorityInjectionPanel
                    pendingTasks={pendingTasks}
                    onAdd={(task) => setPendingTasks(prev => [...prev, task])}
                    onRemove={(id) => setPendingTasks(prev => prev.filter(t => t.task_id !== id))}
                />

                {/* ── Error Banner ── */}
                {errors.submit && (
                    <div style={{
                        padding: '14px 18px',
                        background: 'rgba(239,68,68,0.08)',
                        border: '1px solid rgba(239,68,68,0.3)',
                        borderRadius: 'var(--radius-md)',
                        color: 'var(--color-danger)',
                        fontFamily: 'var(--font-mono)', fontSize: 12,
                        marginBottom: 20,
                        lineHeight: 1.6,
                    }}>
                        <strong>Initialization failed:</strong> {errors.submit}
                    </div>
                )}

                {/* ── Initialize Button ── */}
                <div style={{ paddingBottom: 48 }}>
                    <button
                        id="initialize-simulation-btn"
                        onClick={handleSubmit}
                        disabled={loading}
                        style={{
                            width: '100%',
                            padding: '16px 32px',
                            background: loading ? 'rgba(37,99,235,0.4)' : 'var(--color-electric)',
                            color: '#fff',
                            fontFamily: 'var(--font-ui)',
                            fontWeight: 700,
                            fontSize: 16,
                            letterSpacing: '0.02em',
                            border: 'none',
                            borderRadius: 'var(--radius-lg)',
                            cursor: loading ? 'not-allowed' : 'pointer',
                            transition: 'all 0.15s ease',
                            boxShadow: loading ? 'none' : '0 0 32px rgba(37,99,235,0.4)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 12,
                        }}
                        onMouseEnter={e => {
                            if (!loading) {
                                (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 0 48px rgba(37,99,235,0.6)';
                                (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-1px)';
                            }
                        }}
                        onMouseLeave={e => {
                            (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 0 32px rgba(37,99,235,0.4)';
                            (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
                        }}
                    >
                        {loading ? (
                            <>
                                <div style={{
                                    width: 18, height: 18, borderRadius: '50%',
                                    border: '2px solid rgba(255,255,255,0.3)',
                                    borderTopColor: '#fff',
                                    animation: 'spin 0.8s linear infinite',
                                }} />
                                Initializing Simulation…
                            </>
                        ) : (
                            <>
                                <span style={{ fontSize: 20 }}>⚡</span>
                                Initialize Simulation
                                <span style={{ marginLeft: 4 }}>→</span>
                            </>
                        )}
                    </button>
                    <p style={{
                        textAlign: 'center', marginTop: 12,
                        color: 'var(--color-slate-dim)',
                        fontSize: 'var(--text-label)',
                        fontFamily: 'var(--font-mono)',
                    }}>
                        Sends configuration to backend — begins {cfg.days_phase1 + cfg.days_phase2}-day simulation
                    </p>
                </div>
            </div>
        </div>
    );
}
