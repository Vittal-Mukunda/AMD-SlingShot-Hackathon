/**
 * GanttChart.tsx — Virtualized, scrollable SVG Gantt chart (v3 — full rewrite)
 *
 * Fixes over v2:
 *   - Container is now scrollable (overflowX: 'auto') so multi-day sims don't clip
 *   - Worker rows pre-populated from numWorkers prop at startup (no row collapse on empty workers)
 *   - Tooltip shows task_id, worker NAME (not raw id), start tick, duration, urgency label, policy
 *   - React.memo prevents needless re-renders when only currentTick changes
 *   - Auto-scrolls to keep the current-tick indicator visible during live simulation
 *   - Stable row heights based on fixed ROW_HEIGHT constant
 */
import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import type { GanttBlock, WorkerState } from '../../types/simulation';
import { URGENCY_COLORS } from '../../types/metrics';

interface Props {
    blocks: GanttBlock[];
    workers: WorkerState[];
    currentTick: number;
    phase: number;
    slotsPerDay: number;
    numWorkers?: number;   // Pre-populate rows before workers arrive
    totalDays?: number;    // Total simulation days (for initial SVG width)
}

const ROW_HEIGHT = 38;
const SLOT_WIDTH = 20;       // px per 30-min slot
const HEADER_HEIGHT = 52;
const WORKER_LABEL_WIDTH = 108;
const TOOLTIP_OFFSET = 14;

const URGENCY_LABELS = ['low', 'medium', 'high', 'critical'] as const;

interface TooltipState {
    block: GanttBlock;
    workerName: string;
    x: number;
    y: number;
}

function formatSlotTime(slot: number, slotsPerDay: number): string {
    const daySlot = slot % slotsPerDay;
    const hour = 9 + daySlot * 0.5;
    const h = Math.floor(hour);
    const m = hour % 1 === 0 ? '00' : '30';
    return `${h}:${m}`;
}

function GanttChartInner({
    blocks, workers, currentTick, phase, slotsPerDay, numWorkers = 5, totalDays = 25,
}: Props) {
    const outerRef = useRef<HTMLDivElement>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    // Number of rows: use whichever is larger — received workers or the configured count
    const numRows = Math.max(workers.length, numWorkers, 1);

    // Build worker display name map
    const workerNameMap = useMemo(() => {
        const map: Record<string, string> = {};
        workers.forEach((w, i) => {
            map[w.id] = w.name ?? `Worker ${i + 1}`;
        });
        // Also pre-populate for workers not yet received
        for (let i = 0; i < numRows; i++) {
            const key = `w${i}`;
            if (!map[key]) map[key] = `Worker ${i + 1}`;
        }
        return map;
    }, [workers, numRows]);

    // Worker id → row index map (stable)
    const workerIndexMap = useMemo(() => {
        const map: Record<string, number> = {};
        workers.forEach((w, i) => { map[w.id] = i; });
        // Pre-populate: w0 → 0, w1 → 1, etc.
        for (let i = 0; i < numRows; i++) {
            const key = `w${i}`;
            if (!(key in map)) map[key] = i;
        }
        return map;
    }, [workers, numRows]);

    // Total X width based on simulation length
    const totalSlots = Math.max(currentTick + slotsPerDay * 2, slotsPerDay * (totalDays + 1));
    const totalWidth = totalSlots * SLOT_WIDTH;
    const svgWidth = WORKER_LABEL_WIDTH + totalWidth;
    const svgHeight = HEADER_HEIGHT + numRows * ROW_HEIGHT;

    // Day separator slots
    const dayLines = useMemo(() => {
        const lines: number[] = [];
        for (let d = 0; d * slotsPerDay <= totalSlots; d++) lines.push(d * slotsPerDay);
        return lines;
    }, [totalSlots, slotsPerDay]);

    // Auto-scroll to keep current-tick visible
    useEffect(() => {
        if (!outerRef.current) return;
        const container = outerRef.current;
        const tickX = WORKER_LABEL_WIDTH + currentTick * SLOT_WIDTH;
        const viewRight = container.scrollLeft + container.clientWidth;
        // Scroll right if tick is within 2 slots of the right edge
        if (tickX > viewRight - SLOT_WIDTH * 2) {
            container.scrollLeft = tickX - container.clientWidth * 0.7;
        }
    }, [currentTick]);

    const handleMouseMove = useCallback((e: React.MouseEvent, block: GanttBlock) => {
        const rect = outerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const workerName = workerNameMap[block.worker_id] ?? block.worker_id;
        setTooltip({
            block, workerName,
            x: e.clientX - rect.left + TOOLTIP_OFFSET,
            y: e.clientY - rect.top + TOOLTIP_OFFSET,
        });
    }, [workerNameMap]);

    const handleMouseLeave = useCallback(() => setTooltip(null), []);

    return (
        <div
            ref={outerRef}
            className="gantt-container"
            style={{
                height: '100%',
                background: 'var(--color-bg)',
                position: 'relative',
                overflowX: 'auto',
                overflowY: 'auto',
                scrollBehavior: 'smooth',
            }}
        >
            <svg
                width={svgWidth}
                height={svgHeight}
                style={{ display: 'block', minWidth: svgWidth }}
            >
                {/* ── Frozen worker labels panel ─────────────────────────────── */}
                <rect x={0} y={0} width={WORKER_LABEL_WIDTH} height={svgHeight} fill="#141721" />
                <rect x={WORKER_LABEL_WIDTH - 1} y={0} width={1} height={svgHeight} fill="var(--color-border)" />

                {/* ── Header row ───────────────────────────────────────────── */}
                <rect x={0} y={0} width={svgWidth} height={HEADER_HEIGHT} fill="#0F1318" />
                <rect x={0} y={HEADER_HEIGHT - 1} width={svgWidth} height={1} fill="var(--color-border)" />

                {/* Phase label in header */}
                <text x={8} y={22} fill="var(--color-amber)" fontSize={9}
                    fontFamily="var(--font-mono)" fontWeight={700} letterSpacing={1}>
                    {phase === 2 ? 'DQN PHASE' : 'BASELINE PHASE'}
                </text>

                {/* ── Day labels and hour marks ─────────────────────────────── */}
                {dayLines.map((slot, di) => {
                    const x = WORKER_LABEL_WIDTH + slot * SLOT_WIDTH;
                    return (
                        <g key={di}>
                            {/* Day separator line */}
                            <line x1={x} y1={0} x2={x} y2={svgHeight}
                                stroke="rgba(255,255,255,0.08)" strokeWidth={di === 0 ? 0 : 1} />
                            {/* Day label */}
                            <text x={x + 5} y={20} fill="var(--color-slate-text)"
                                fontSize={10} fontFamily="var(--font-mono)" fontWeight={600}>
                                Day {di + 1}
                            </text>
                            {/* Hour marks within day */}
                            {Array.from({ length: slotsPerDay / 2 }, (_, h) => {
                                const tickX = x + h * 2 * SLOT_WIDTH;
                                return (
                                    <g key={h}>
                                        <line x1={tickX} y1={36} x2={tickX} y2={HEADER_HEIGHT}
                                            stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
                                        <text x={tickX + 2} y={46}
                                            fill="var(--color-slate-dim)" fontSize={8}
                                            fontFamily="var(--font-mono)">
                                            {9 + h}h
                                        </text>
                                    </g>
                                );
                            })}
                        </g>
                    );
                })}

                {/* ── Worker rows ──────────────────────────────────────────── */}
                {Array.from({ length: numRows }, (_, i) => {
                    const y = HEADER_HEIGHT + i * ROW_HEIGHT;
                    const w = workers[i];
                    const bgColor = i % 2 === 0 ? 'var(--color-bg)' : '#0D1017';
                    const fatigue = w?.fatigue ?? 0;
                    const fatigueColor = fatigue >= 2 ? 'var(--color-danger)'
                        : fatigue >= 1 ? 'var(--color-amber)' : 'var(--color-success)';
                    return (
                        <g key={i}>
                            <rect x={0} y={y} width={svgWidth} height={ROW_HEIGHT} fill={bgColor} />
                            <rect x={0} y={y + ROW_HEIGHT - 1} width={svgWidth} height={1}
                                fill="rgba(255,255,255,0.04)" />
                            {/* Worker name */}
                            <text x={8} y={y + ROW_HEIGHT / 2 + 1}
                                fill={w ? 'var(--color-text)' : 'var(--color-slate-dim)'}
                                fontSize={11} fontFamily="var(--font-mono)" fontWeight={500}
                                dominantBaseline="middle">
                                {w?.name ?? `Worker ${i + 1}`}
                            </text>
                            {/* Fatigue sub-label */}
                            {w && (
                                <text x={8} y={y + ROW_HEIGHT / 2 + 14}
                                    fill={fatigueColor}
                                    fontSize={8} fontFamily="var(--font-mono)"
                                    dominantBaseline="middle">
                                    {w.fatigue_level ?? (fatigue >= 2.6 ? 'burnout' :
                                        fatigue >= 2 ? 'exhausted' : fatigue >= 1 ? 'tired' : 'fresh')}
                                </text>
                            )}
                        </g>
                    );
                })}

                {/* ── Current tick indicator ───────────────────────────────── */}
                <line
                    x1={WORKER_LABEL_WIDTH + currentTick * SLOT_WIDTH}
                    y1={0}
                    x2={WORKER_LABEL_WIDTH + currentTick * SLOT_WIDTH}
                    y2={svgHeight}
                    stroke="rgba(245,158,11,0.6)"
                    strokeWidth={2}
                    strokeDasharray="4 3"
                />

                {/* ── Gantt task blocks ────────────────────────────────────── */}
                {blocks.map((block, bi) => {
                    const rawId = block.worker_id.replace('w', '');
                    const rowIdx = workerIndexMap[block.worker_id]
                        ?? (parseInt(rawId, 10) % numRows);
                    const x = WORKER_LABEL_WIDTH + block.start_tick * SLOT_WIDTH;
                    const y = HEADER_HEIGHT + rowIdx * ROW_HEIGHT + 4;
                    const bw = Math.max(4, (block.end_tick - block.start_tick) * SLOT_WIDTH - 2);
                    const bh = ROW_HEIGHT - 8;
                    const color = URGENCY_COLORS[block.urgency] ?? '#64748B';
                    const isDQN = block.policy === 'DQN';

                    return (
                        <g
                            key={`${block.task_id}-${bi}`}
                            onMouseMove={e => handleMouseMove(e, block)}
                            onMouseLeave={handleMouseLeave}
                            style={{ cursor: 'pointer' }}
                        >
                            {/* Shadow for depth */}
                            <rect
                                x={x + 1} y={y + 1} width={bw} height={bh}
                                fill="rgba(0,0,0,0.35)" rx={3}
                            />
                            {/* Main block */}
                            <rect
                                x={x} y={y} width={bw} height={bh}
                                fill={color}
                                fillOpacity={isDQN ? 0.9 : 0.65}
                                stroke={color}
                                strokeOpacity={isDQN ? 1 : 0.8}
                                strokeWidth={isDQN ? 2 : 1}
                                rx={3}
                            />
                            {/* DQN amber outline */}
                            {isDQN && (
                                <rect x={x} y={y} width={bw} height={bh}
                                    fill="none" stroke="var(--color-amber)"
                                    strokeWidth={1.5} strokeOpacity={0.6} rx={3} />
                            )}
                            {/* Task ID label (only if wide enough) */}
                            {bw > 28 && (
                                <text x={x + 4} y={y + bh / 2 + 1}
                                    fill="#fff" fillOpacity={0.9}
                                    fontSize={9} fontFamily="var(--font-mono)"
                                    dominantBaseline="middle">
                                    {block.task_id}
                                </text>
                            )}
                        </g>
                    );
                })}
            </svg>

            {/* ── Hover tooltip ────────────────────────────────────────────── */}
            {tooltip && (
                <div
                    className="tooltip"
                    style={{
                        position: 'absolute',
                        left: tooltip.x,
                        top: tooltip.y,
                        maxWidth: 280,
                        pointerEvents: 'none',
                        zIndex: 100,
                    }}
                >
                    <div style={{ color: 'var(--color-amber)', marginBottom: 8, fontWeight: 700, fontSize: 13 }}>
                        {tooltip.block.task_id}
                    </div>
                    <table style={{ borderCollapse: 'collapse', fontSize: 11, lineHeight: 1.7, width: '100%' }}>
                        <tbody>
                            {[
                                ['Worker', tooltip.workerName],
                                ['Policy', tooltip.block.policy],
                                ['Start tick', `${tooltip.block.start_tick} (${formatSlotTime(tooltip.block.start_tick, slotsPerDay)})`],
                                ['End tick', `${tooltip.block.end_tick} (${formatSlotTime(tooltip.block.end_tick, slotsPerDay)})`],
                                ['Duration', `${tooltip.block.end_tick - tooltip.block.start_tick} slots`],
                                ['Urgency', URGENCY_LABELS[tooltip.block.urgency] ?? '—'],
                            ].map(([label, value]) => (
                                <tr key={label}>
                                    <td style={{ color: 'var(--color-slate-text)', paddingRight: 10 }}>{label}</td>
                                    <td style={{
                                        color: label === 'Urgency'
                                            ? (URGENCY_COLORS[tooltip.block.urgency] ?? 'var(--color-text)')
                                            : label === 'Worker' ? 'var(--color-amber)'
                                                : label === 'Policy' ? 'var(--color-success)'
                                                    : 'var(--color-text)',
                                        fontFamily: 'var(--font-mono)',
                                    }}>
                                        {value}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* ── Empty state ──────────────────────────────────────────────── */}
            {blocks.length === 0 && (
                <div style={{
                    position: 'absolute',
                    top: HEADER_HEIGHT + (numRows * ROW_HEIGHT) / 2,
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                }}>
                    <div className="font-mono" style={{
                        color: 'var(--color-slate-dim)', fontSize: 13,
                        letterSpacing: '0.08em', textAlign: 'center',
                    }}>
                        {phase === 0 ? 'Waiting for simulation to start…'
                            : phase === 1 ? 'Awaiting task assignments…'
                                : 'DQN scheduling in progress…'}
                    </div>
                </div>
            )}
        </div>
    );
}

export default React.memo(GanttChartInner);
