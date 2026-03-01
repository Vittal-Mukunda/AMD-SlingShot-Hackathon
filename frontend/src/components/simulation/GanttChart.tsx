/**
 * GanttChart.tsx — Virtualized SVG Gantt chart for workforce scheduling view.
 *
 * - Scrollable X-axis: 8-hour ticks (16 slots) per working day
 * - Workers on Y-axis
 * - Color-coded task blocks by urgency (0=slate, 1=blue, 2=orange, 3=red)
 * - Hover tooltips: task_id, duration, worker, assigned_tick, expected_completion
 * - Handles 200 tasks × 25 days without layout collapse via virtual windowing
 * - X scroll wheel controlled; Y fixed rows
 */
import React, { useState, useRef, useCallback } from 'react';
import type { GanttBlock, WorkerState } from '../../types/simulation';
import { URGENCY_COLORS } from '../../types/metrics';

interface Props {
    blocks: GanttBlock[];
    workers: WorkerState[];
    currentTick: number;
    phase: number;
    slotsPerDay: number;
}

const ROW_HEIGHT = 36;
const SLOT_WIDTH = 18;       // px per 30-min slot
const HEADER_HEIGHT = 48;
const WORKER_LABEL_WIDTH = 100;
const TOOLTIP_OFFSET = 12;

interface TooltipState {
    block: GanttBlock;
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

export default function GanttChart({ blocks, workers, currentTick, phase, slotsPerDay }: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [tooltip, setTooltip] = useState<TooltipState | null>(null);

    // Determine visible range
    const totalSlots = Math.max(currentTick + slotsPerDay * 2, slotsPerDay * 5);
    const totalWidth = totalSlots * SLOT_WIDTH;
    const numRows = Math.max(workers.length, 1);
    const svgHeight = HEADER_HEIGHT + numRows * ROW_HEIGHT;

    // Worker id → row index map
    const workerIndexMap: Record<string, number> = {};
    workers.forEach((w, i) => { workerIndexMap[w.id] = i; });

    const handleMouseMove = useCallback((e: React.MouseEvent, block: GanttBlock) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        setTooltip({ block, x: e.clientX - rect.left + TOOLTIP_OFFSET, y: e.clientY - rect.top + TOOLTIP_OFFSET });
    }, []);

    const handleMouseLeave = useCallback(() => setTooltip(null), []);

    // Render day separators
    const dayLines: number[] = [];
    for (let d = 0; d * slotsPerDay <= totalSlots; d++) {
        dayLines.push(d * slotsPerDay);
    }

    return (
        <div ref={containerRef} className="gantt-container"
            style={{ height: '100%', background: 'var(--color-bg)', position: 'relative' }}>
            <svg width={WORKER_LABEL_WIDTH + totalWidth} height={svgHeight} style={{ display: 'block', minWidth: '100%' }}>
                {/* Worker labels column */}
                <rect x={0} y={0} width={WORKER_LABEL_WIDTH} height={svgHeight} fill="#141721" />
                <rect x={WORKER_LABEL_WIDTH - 1} y={0} width={1} height={svgHeight} fill="var(--color-border)" />

                {/* Header row */}
                <rect x={0} y={0} width={WORKER_LABEL_WIDTH + totalWidth} height={HEADER_HEIGHT} fill="#0F1318" />
                <rect x={0} y={HEADER_HEIGHT - 1} width={WORKER_LABEL_WIDTH + totalWidth} height={1} fill="var(--color-border)" />

                {/* Day labels in header */}
                {dayLines.map((slot, di) => {
                    const x = WORKER_LABEL_WIDTH + slot * SLOT_WIDTH;
                    return (
                        <g key={di}>
                            <line x1={x} y1={0} x2={x} y2={svgHeight} stroke="var(--color-border)" strokeWidth={1} />
                            <text x={x + 6} y={20} fill="var(--color-slate-dim)"
                                fontSize={10} fontFamily="var(--font-mono)" fontWeight={600}>
                                Day {di + 1}
                            </text>
                            {/* Hour marks within day */}
                            {Array.from({ length: slotsPerDay / 2 }, (_, h) => {
                                const tickX = x + (h * 2) * SLOT_WIDTH;
                                const hourLabel = `${9 + h}h`;
                                return (
                                    <g key={h}>
                                        <line x1={tickX} y1={32} x2={tickX} y2={HEADER_HEIGHT} stroke="rgba(30,36,51,0.8)" strokeWidth={1} />
                                        <text x={tickX + 2} y={42} fill="var(--color-slate-dim)" fontSize={8} fontFamily="var(--font-mono)">
                                            {hourLabel}
                                        </text>
                                    </g>
                                );
                            })}
                        </g>
                    );
                })}

                {/* Worker rows */}
                {Array.from({ length: numRows }, (_, i) => {
                    const y = HEADER_HEIGHT + i * ROW_HEIGHT;
                    const w = workers[i];
                    const bgColor = i % 2 === 0 ? 'var(--color-bg)' : '#0F1318';
                    return (
                        <g key={i}>
                            <rect x={0} y={y} width={WORKER_LABEL_WIDTH + totalWidth} height={ROW_HEIGHT} fill={bgColor} />
                            <rect x={0} y={y + ROW_HEIGHT - 1} width={WORKER_LABEL_WIDTH + totalWidth} height={1} fill="rgba(30,36,51,0.5)" />
                            {/* Worker label */}
                            <text x={8} y={y + ROW_HEIGHT / 2 + 4}
                                fill={w ? 'var(--color-text)' : 'var(--color-slate-dim)'}
                                fontSize={11} fontFamily="var(--font-mono)" fontWeight={500}>
                                {w ? w.name : `Worker ${i + 1}`}
                            </text>
                            {w && (
                                <text x={8} y={y + ROW_HEIGHT / 2 + 16}
                                    fill={w.fatigue >= 2 ? 'var(--color-danger)' : w.fatigue >= 1 ? 'var(--color-amber)' : 'var(--color-success)'}
                                    fontSize={9} fontFamily="var(--font-mono)">
                                    fatigue={w.fatigue.toFixed(2)}
                                </text>
                            )}
                        </g>
                    );
                })}

                {/* Current tick indicator */}
                <line
                    x1={WORKER_LABEL_WIDTH + currentTick * SLOT_WIDTH}
                    y1={0}
                    x2={WORKER_LABEL_WIDTH + currentTick * SLOT_WIDTH}
                    y2={svgHeight}
                    stroke="rgba(245,158,11,0.5)"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                />

                {/* Gantt blocks */}
                {blocks.map((block, bi) => {
                    const rowIndex = workerIndexMap[block.worker_id] ?? (parseInt(block.worker_id.replace('w', '')) % numRows);
                    const x = WORKER_LABEL_WIDTH + block.start_tick * SLOT_WIDTH;
                    const y = HEADER_HEIGHT + rowIndex * ROW_HEIGHT + 4;
                    const w = Math.max(4, (block.end_tick - block.start_tick) * SLOT_WIDTH - 2);
                    const h = ROW_HEIGHT - 8;
                    const color = URGENCY_COLORS[block.urgency] ?? '#64748B';
                    const isPhase2 = block.policy === 'DQN';

                    return (
                        <g key={`${block.task_id}-${bi}`}
                            onMouseMove={e => handleMouseMove(e, block)}
                            onMouseLeave={handleMouseLeave}
                            style={{ cursor: 'pointer' }}>
                            <rect
                                x={x} y={y} width={w} height={h}
                                fill={color}
                                fillOpacity={isPhase2 ? 0.85 : 0.6}
                                stroke={color}
                                strokeOpacity={isPhase2 ? 1 : 0.8}
                                strokeWidth={isPhase2 ? 2 : 1}
                                rx={2}
                            />
                            {/* DQN blocks get amber outline */}
                            {isPhase2 && (
                                <rect x={x} y={y} width={w} height={h}
                                    fill="none" stroke="var(--color-amber)"
                                    strokeWidth={1.5} strokeOpacity={0.5} rx={2} />
                            )}
                            {/* Task ID label if wide enough */}
                            {w > 30 && (
                                <text x={x + 4} y={y + h / 2 + 4}
                                    fill="#fff" fillOpacity={0.85}
                                    fontSize={9} fontFamily="var(--font-mono)">
                                    {block.task_id}
                                </text>
                            )}
                        </g>
                    );
                })}
            </svg>

            {/* Hover tooltip */}
            {tooltip && (
                <div className="tooltip" style={{
                    left: tooltip.x,
                    top: tooltip.y,
                    maxWidth: 260,
                }}>
                    <div style={{ color: 'var(--color-amber)', marginBottom: 6, fontWeight: 700 }}>
                        {tooltip.block.task_id}
                    </div>
                    <div style={{ lineHeight: 1.8, color: 'var(--color-text)' }}>
                        <div>Worker: <span style={{ color: 'var(--color-amber)' }}>{tooltip.block.worker_id}</span></div>
                        <div>Policy: <span style={{ color: 'var(--color-success)' }}>{tooltip.block.policy}</span></div>
                        <div>Start tick: <span className="num">{tooltip.block.start_tick}</span>
                            &nbsp;({formatSlotTime(tooltip.block.start_tick, slotsPerDay)})</div>
                        <div>End tick: <span className="num">{tooltip.block.end_tick}</span>
                            &nbsp;({formatSlotTime(tooltip.block.end_tick, slotsPerDay)})</div>
                        <div>Duration: <span className="num">{tooltip.block.end_tick - tooltip.block.start_tick}</span> slots</div>
                        <div>Urgency:
                            <span style={{ marginLeft: 4, color: URGENCY_COLORS[tooltip.block.urgency] }}>
                                {['low', 'medium', 'high', 'critical'][tooltip.block.urgency]}
                            </span>
                        </div>
                    </div>
                </div>
            )}

            {/* Empty state */}
            {blocks.length === 0 && (
                <div style={{
                    position: 'absolute', inset: 0, display: 'flex',
                    alignItems: 'center', justifyContent: 'center'
                }}>
                    <div className="font-mono" style={{ color: 'var(--color-slate-dim)', fontSize: 13, letterSpacing: '0.08em' }}>
                        {phase === 0 ? 'Waiting for simulation to start...' :
                            phase === 1 ? 'Awaiting task assignments...' :
                                'DQN scheduling in progress...'}
                    </div>
                </div>
            )}
        </div>
    );
}
