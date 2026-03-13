/**
 * TaskQueueCard.tsx — Live Task Intelligence Feed
 *
 * Completely dynamic rebuild:
 * - Cards breathe and pulse in real time based on urgency
 * - SVG deadline ring drains live with color shifting green > amber > red
 * - Left accent bar pulses with glow when urgent
 * - Expandable detail panel on click with spring animation
 * - Critical cards get a persistent CSS attention ring animation
 * - Queue header shows live stats (total, critical count, due-soon count)
 * - Cards auto-sorted by urgency on every render
 * - Scroll fade edges that appear/disappear based on scroll position
 * - Scroll indicator dots at bottom that update on scroll
 * - Staggered spring entrance choreography
 *
 * All logic IDENTICAL. No emoji. Pure SVG icons.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { QueueItem } from '../types/simulation';

// ── Styles ────────────────────────────────────────────────────────────────────
const QUEUE_STYLES = `
.tq-root {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--void);
  position: relative;
  overflow: hidden;
}

/* ── Queue header bar ── */
.tq-header {
  display: flex;
  align-items: center;
  gap: 0;
  height: 32px;
  flex-shrink: 0;
  border-bottom: 1px solid var(--rim);
  padding: 0 14px;
  background: var(--surface);
  position: relative;
  overflow: hidden;
}
.tq-header::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(68,147,248,0.03) 0%, transparent 50%);
  pointer-events: none;
}
.tq-header-label {
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-right: 12px;
  flex-shrink: 0;
}
.tq-stat {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 0 10px;
  border-right: 1px solid var(--rim);
  height: 100%;
  flex-shrink: 0;
}
.tq-stat:last-of-type { border-right: none; }
.tq-stat-label {
  font-family: var(--font-mono);
  font-size: 8px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.tq-stat-val {
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
}

/* ── Scroll track ── */
.tq-scroll-area {
  flex: 1;
  overflow-x: auto;
  overflow-y: hidden;
  position: relative;
  scrollbar-width: none;
}
.tq-scroll-area::-webkit-scrollbar { display: none; }

.tq-fade-left,
.tq-fade-right {
  position: absolute;
  top: 0; bottom: 0;
  width: 40px;
  pointer-events: none;
  z-index: 4;
  transition: opacity 0.3s;
}
.tq-fade-left  { left: 0;  background: linear-gradient(90deg,  var(--void) 0%, transparent 100%); }
.tq-fade-right { right: 0; background: linear-gradient(270deg, var(--void) 0%, transparent 100%); }

.tq-inner {
  display: flex;
  gap: 8px;
  padding: 10px 20px;
  align-items: center;
  min-width: max-content;
  height: 100%;
}

/* ── Empty state ── */
.tq-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.06em;
}

/* ── Overflow badge ── */
.tq-overflow {
  flex-shrink: 0;
  width: 40px;
  height: 120px;
  border-radius: 10px;
  background: linear-gradient(180deg, var(--raised), var(--surface));
  border: 1px solid var(--rim);
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--dim);
  text-align: center;
  line-height: 1.5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  align-self: center;
}

/* ── Task card ── */
.tq-card {
  flex-shrink: 0;
  width: 130px;
  border-radius: 10px;
  position: relative;
  overflow: visible;
  display: flex;
  flex-direction: column;
  cursor: pointer;
}
.tq-card-inner {
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 10px 11px 10px 14px;
  border: 1px solid var(--rim);
  position: relative;
  transition: border-color 0.2s, box-shadow 0.2s;
}

/* Critical pulse ring */
@keyframes criticalRing {
  0%   { transform: scale(1);    opacity: 0.65; }
  70%  { transform: scale(1.13); opacity: 0;    }
  100% { transform: scale(1.13); opacity: 0;    }
}
.tq-critical-ring {
  position: absolute;
  inset: -4px;
  border-radius: 13px;
  border: 1.5px solid #f85149;
  animation: criticalRing 1.8s ease-out infinite;
  pointer-events: none;
  z-index: 0;
}

/* Shimmer sweep on critical */
@keyframes critShimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
.tq-card.is-critical .tq-card-inner::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(248,81,73,0.07) 50%,
    transparent 100%
  );
  background-size: 200% auto;
  animation: critShimmer 2.8s linear infinite;
  pointer-events: none;
}

/* Sections */
.tq-priority-row {
  display: flex;
  align-items: center;
  gap: 5px;
  margin-bottom: 5px;
}
.tq-priority-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  flex-shrink: 0;
}
.tq-priority-label {
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  line-height: 1;
  flex: 1;
}
.tq-task-id {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  color: var(--sol);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.2;
  margin-bottom: 5px;
}
.tq-meta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--dim);
  margin-top: 2px;
  margin-bottom: 5px;
}
.tq-skill-badge {
  padding: 1px 5px;
  background: var(--raised);
  border: 1px solid var(--rim);
  border-radius: 3px;
  font-size: 8px;
  color: var(--dim);
}
.tq-slots-row {
  display: flex;
  align-items: center;
  gap: 4px;
  font-family: var(--font-mono);
  font-size: 9px;
  line-height: 1;
}

/* Drain bar */
.tq-drain-track {
  width: 100%;
  height: 3px;
  background: var(--raised);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 8px;
}

/* Ring container */
.tq-ring-pos {
  position: absolute;
  top: 8px;
  right: 8px;
}

/* Expand panel */
.tq-expand-panel {
  background: var(--surface);
  border: 1px solid var(--rim);
  border-top: none;
  border-radius: 0 0 10px 10px;
  padding: 9px 12px;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--dim);
  overflow: hidden;
}
.tq-expand-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.tq-expand-row:last-child { border-bottom: none; }
.tq-expand-key {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.tq-expand-val {
  color: var(--sol);
  font-weight: 500;
}

/* Scroll dots */
.tq-dots {
  display: flex;
  justify-content: center;
  gap: 4px;
  padding: 4px 0 5px;
  flex-shrink: 0;
  border-top: 1px solid var(--rim);
  background: var(--void);
}
.tq-dot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--muted);
  transition: all 0.2s;
  cursor: pointer;
}
.tq-dot.active {
  background: var(--blue);
  width: 12px;
  border-radius: 2px;
  box-shadow: 0 0 6px var(--blue);
}
`;

// ── Priority config ───────────────────────────────────────────────────────────
const PRIORITY_CONFIG: Record<string, {
    color: string; bg: string; glow: string;
}> = {
    critical: {
        color: '#f85149',
        bg: 'rgba(248,81,73,0.07)',
        glow: '0 0 16px rgba(248,81,73,0.25), 0 4px 20px rgba(0,0,0,0.4)',
    },
    high: {
        color: '#e3b341',
        bg: 'rgba(227,179,65,0.06)',
        glow: '0 0 12px rgba(227,179,65,0.18), 0 4px 16px rgba(0,0,0,0.3)',
    },
    medium: {
        color: '#4493f8',
        bg: 'rgba(68,147,248,0.05)',
        glow: '0 2px 12px rgba(0,0,0,0.3)',
    },
    low: {
        color: '#3d444d',
        bg: 'rgba(255,255,255,0.015)',
        glow: 'none',
    },
};

// ── SVG icons ─────────────────────────────────────────────────────────────────
const WarnIcon = ({ color = 'currentColor' }: { color?: string }) => (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="none"
        stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
);

const InboxIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
        stroke="var(--muted)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
        <path d="M5.45 5.11L2 12v6a2 2 0 002 2h16a2 2 0 002-2v-6l-3.45-6.89A2 2 0 0016.76 4H7.24a2 2 0 00-1.79 1.11z" />
    </svg>
);

const ChevronDown = () => (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="6 9 12 15 18 9" />
    </svg>
);

const ChevronUp = () => (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="18 15 12 9 6 15" />
    </svg>
);

// ── SVG deadline ring ─────────────────────────────────────────────────────────
function DeadlineRing({ pct, color, size = 26 }: {
    pct: number; color: string; size?: number;
}) {
    const r = (size - 4) / 2;
    const circ = 2 * Math.PI * r;
    const fill = circ * Math.max(0, Math.min(1, pct / 100));

    return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
            style={{ transform: 'rotate(-90deg)', display: 'block' }}>
            <circle cx={size / 2} cy={size / 2} r={r}
                fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="2" />
            <circle cx={size / 2} cy={size / 2} r={r}
                fill="none"
                stroke={color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={`${fill} ${circ - fill}`}
                style={{
                    filter: pct < 25 ? `drop-shadow(0 0 3px ${color})` : 'none',
                    transition: 'stroke-dasharray 1s linear, stroke 0.5s ease',
                }}
            />
        </svg>
    );
}

// ── Task card ─────────────────────────────────────────────────────────────────
interface Props {
    task: QueueItem;
    currentTick: number;
    index: number;
}

export function TaskQueueCard({ task, currentTick, index }: Props) {
    const [expanded, setExpanded] = useState(false);

    const urgency = task.urgency_label ?? 'low';
    const cfg = PRIORITY_CONFIG[urgency] ?? PRIORITY_CONFIG.low;
    const slotsLeft = task.slots_remaining ?? 0;
    const isCritical = urgency === 'critical';
    const isUrgent = slotsLeft <= 8 || isCritical;

    // Ring drains from full (32 slots) to 0
    const ringPct = Math.max(0, Math.min(100, (slotsLeft / 32) * 100));

    // Color shifts green -> amber -> red as deadline approaches
    const deadlineColor =
        ringPct > 60 ? '#3fb950' :
            ringPct > 30 ? '#e3b341' : cfg.color;

    // Drain bar fill
    const drainPct = Math.max(0, Math.min(100, ringPct));

    return (
        <div className={`tq-card ${isCritical ? 'is-critical' : ''}`}>
            {/* Attention ring (critical only) */}
            {isCritical && <div className="tq-critical-ring" />}

            <motion.div
                className="tq-card-inner"
                onClick={() => setExpanded(e => !e)}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                animate={{
                    borderColor: isUrgent ? `${cfg.color}55` : 'rgba(255,255,255,0.06)',
                    boxShadow: isUrgent ? cfg.glow : '0 2px 8px rgba(0,0,0,0.2)',
                    background: cfg.bg,
                }}
                transition={{ duration: 0.4 }}
                style={{
                    borderRadius: expanded ? '10px 10px 0 0' : '10px',
                }}
            >
                {/* Left accent bar — pulses when urgent */}
                <motion.div
                    animate={isUrgent ? {
                        opacity: [1, 0.45, 1],
                        boxShadow: [`0 0 6px ${cfg.color}`, `0 0 14px ${cfg.color}`, `0 0 6px ${cfg.color}`],
                    } : { opacity: 1 }}
                    transition={{ duration: 1.5, repeat: isUrgent ? Infinity : 0 }}
                    style={{
                        position: 'absolute',
                        left: 0, top: 0, bottom: 0,
                        width: 3,
                        background: cfg.color,
                        borderRadius: expanded ? '10px 0 0 0' : '10px 0 0 10px',
                    }}
                />

                {/* Deadline ring */}
                <div className="tq-ring-pos">
                    <DeadlineRing pct={ringPct} color={deadlineColor} />
                </div>

                {/* Priority label row */}
                <div className="tq-priority-row">
                    <motion.div
                        className="tq-priority-dot"
                        animate={{ scale: isCritical ? [1, 1.5, 1] : 1 }}
                        transition={{ duration: 0.8, repeat: isCritical ? Infinity : 0 }}
                        style={{
                            background: cfg.color,
                            boxShadow: `0 0 6px ${cfg.color}`,
                        }}
                    />
                    <span className="tq-priority-label" style={{ color: cfg.color }}>
                        {urgency}
                    </span>
                    <span style={{ color: 'var(--muted)', flexShrink: 0, display: 'flex' }}>
                        {expanded ? <ChevronUp /> : <ChevronDown />}
                    </span>
                </div>

                {/* Task ID */}
                <div className="tq-task-id">{task.task_id}</div>

                {/* Duration + skill */}
                <div className="tq-meta-row">
                    <span>{task.duration_slots}s</span>
                    <span className="tq-skill-badge">sk {task.required_skill.toFixed(1)}</span>
                </div>

                {/* Slots remaining — animates on change */}
                <div className="tq-slots-row"
                    style={{ color: isUrgent ? cfg.color : 'var(--dim)' }}>
                    {isUrgent && <WarnIcon color={cfg.color} />}
                    <motion.span
                        key={slotsLeft}
                        initial={{ opacity: 0.4, scale: 0.85 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                    >
                        {slotsLeft}s left
                    </motion.span>
                </div>

                {/* Drain bar */}
                <div className="tq-drain-track">
                    <motion.div
                        animate={{
                            width: `${drainPct}%`,
                            background: deadlineColor,
                            boxShadow: isUrgent ? `0 0 8px ${deadlineColor}` : 'none',
                        }}
                        transition={{ duration: 1.2, ease: 'linear' }}
                        style={{ height: '100%', borderRadius: 2 }}
                    />
                </div>
            </motion.div>

            {/* Expandable detail panel */}
            <AnimatePresence>
                {expanded && (
                    <motion.div
                        className="tq-expand-panel"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.22, ease: 'easeOut' }}
                    >
                        {[
                            { key: 'Priority', val: task.priority ?? urgency, color: cfg.color },
                            { key: 'Deadline', val: task.deadline_tick ?? '—', color: undefined },
                            { key: 'Arrival', val: task.arrival_tick ?? '—', color: undefined },
                            { key: 'Remaining', val: `${ringPct.toFixed(0)}%`, color: deadlineColor },
                        ].map(r => (
                            <div key={r.key} className="tq-expand-row">
                                <span className="tq-expand-key">{r.key}</span>
                                <span className="tq-expand-val" style={{ color: r.color ?? 'var(--sol)' }}>
                                    {String(r.val)}
                                </span>
                            </div>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// ── Live queue header ─────────────────────────────────────────────────────────
function QueueHeader({ queue }: { queue: QueueItem[] }) {
    const critical = queue.filter(t => t.urgency_label === 'critical').length;
    const high = queue.filter(t => t.urgency_label === 'high').length;
    const dueSoon = queue.filter(t => (t.slots_remaining ?? 99) <= 8).length;
    const total = queue.length;

    return (
        <div className="tq-header">
            <span className="tq-header-label">Task Queue</span>

            <div className="tq-stat">
                <span className="tq-stat-label">Total</span>
                <motion.span
                    key={total}
                    initial={{ opacity: 0.3, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="tq-stat-val"
                    style={{ color: 'var(--sol)' }}
                >
                    {total}
                </motion.span>
            </div>

            {critical > 0 && (
                <div className="tq-stat">
                    <motion.span
                        animate={{ opacity: [1, 0.35, 1] }}
                        transition={{ duration: 1.1, repeat: Infinity }}
                        className="tq-stat-val"
                        style={{ color: '#f85149' }}
                    >
                        {critical}
                    </motion.span>
                    <span className="tq-stat-label" style={{ color: '#f85149' }}>Crit</span>
                </div>
            )}

            {high > 0 && (
                <div className="tq-stat">
                    <span className="tq-stat-val" style={{ color: '#e3b341' }}>{high}</span>
                    <span className="tq-stat-label">High</span>
                </div>
            )}

            {dueSoon > 0 && (
                <div className="tq-stat">
                    <motion.span
                        animate={{ opacity: [1, 0.5, 1] }}
                        transition={{ duration: 0.9, repeat: Infinity }}
                        className="tq-stat-val"
                        style={{ color: '#f85149' }}
                    >
                        {dueSoon}
                    </motion.span>
                    <span className="tq-stat-label">Due soon</span>
                </div>
            )}

            <div style={{ flex: 1 }} />

            {/* Live indicator */}
            {total > 0 && (
                <motion.div
                    animate={{ opacity: [1, 0.3, 1], scale: [1, 1.25, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    style={{
                        width: 5, height: 5, borderRadius: '50%',
                        background: '#3fb950',
                        boxShadow: '0 0 8px #3fb950',
                        flexShrink: 0,
                    }}
                />
            )}
        </div>
    );
}

// ── Scroll position dots ──────────────────────────────────────────────────────
function ScrollDots({ count, scrollRef }: {
    count: number; scrollRef: React.RefObject<HTMLDivElement | null>;
}) {
    const [active, setActive] = useState(0);
    const DOT_COUNT = Math.min(count, 10);

    useEffect(() => {
        const el = scrollRef.current;
        if (!el) return;
        const onScroll = () => {
            const max = el.scrollWidth - el.clientWidth;
            if (max <= 0) return;
            setActive(Math.round((el.scrollLeft / max) * (DOT_COUNT - 1)));
        };
        el.addEventListener('scroll', onScroll, { passive: true });
        return () => el.removeEventListener('scroll', onScroll);
    }, [DOT_COUNT, scrollRef]);

    if (DOT_COUNT <= 1) return null;

    return (
        <div className="tq-dots">
            {Array.from({ length: DOT_COUNT }, (_, i) => (
                <div
                    key={i}
                    className={`tq-dot ${i === active ? 'active' : ''}`}
                    onClick={() => {
                        const el = scrollRef.current;
                        if (!el) return;
                        const max = el.scrollWidth - el.clientWidth;
                        el.scrollTo({ left: (i / (DOT_COUNT - 1)) * max, behavior: 'smooth' });
                    }}
                />
            ))}
        </div>
    );
}

// ── Animated queue list ───────────────────────────────────────────────────────
interface QueueListProps {
    queue: QueueItem[];
    currentTick: number;
}

export default function AnimatedTaskQueue({ queue, currentTick }: QueueListProps) {
    const scrollRef = useRef<HTMLDivElement>(null);
    const [atLeft, setAtLeft] = useState(true);
    const [atRight, setAtRight] = useState(false);

    // Sort by urgency so critical cards always appear first
    const sorted = [...queue].sort((a, b) => {
        const order: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
        const ao = order[a.urgency_label ?? 'low'] ?? 3;
        const bo = order[b.urgency_label ?? 'low'] ?? 3;
        return ao !== bo ? ao - bo : (a.slots_remaining ?? 99) - (b.slots_remaining ?? 99);
    });

    const handleScroll = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        setAtLeft(el.scrollLeft < 12);
        setAtRight(el.scrollLeft > el.scrollWidth - el.clientWidth - 12);
    }, []);

    useEffect(() => { handleScroll(); }, [queue.length, handleScroll]);

    return (
        <>
            <style>{QUEUE_STYLES}</style>

            <div className="tq-root">
                <QueueHeader queue={queue} />

                <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                    <div className="tq-fade-left" style={{ opacity: atLeft ? 0 : 1 }} />
                    <div className="tq-fade-right" style={{ opacity: atRight ? 0 : 1 }} />

                    <div
                        className="tq-scroll-area"
                        ref={scrollRef}
                        onScroll={handleScroll}
                    >
                        <div className="tq-inner">
                            <AnimatePresence initial={false}>
                                {sorted.length === 0 ? (
                                    <motion.div
                                        key="empty"
                                        className="tq-empty"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                        style={{ minWidth: 220 }}
                                    >
                                        <InboxIcon />
                                        <span>Queue is clear</span>
                                    </motion.div>
                                ) : (
                                    sorted.slice(0, 50).map((task, index) => (
                                        <motion.div
                                            key={task.task_id}
                                            layout
                                            initial={{ opacity: 0, x: 56, scale: 0.85 }}
                                            animate={{ opacity: 1, x: 0, scale: 1 }}
                                            exit={{
                                                opacity: 0, x: -44, scale: 0.82,
                                                transition: { duration: 0.2 },
                                            }}
                                            transition={{
                                                type: 'spring',
                                                stiffness: 340, damping: 28,
                                                delay: Math.min(index * 0.025, 0.18),
                                            }}
                                            style={{ alignSelf: 'center' }}
                                        >
                                            <TaskQueueCard
                                                task={task}
                                                currentTick={currentTick}
                                                index={index}
                                            />
                                        </motion.div>
                                    ))
                                )}

                                {queue.length > 50 && (
                                    <motion.div
                                        key="overflow"
                                        className="tq-overflow"
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                    >
                                        <span style={{ color: 'var(--blue)', fontWeight: 600, fontSize: 13 }}>
                                            +{queue.length - 50}
                                        </span>
                                        <span style={{
                                            fontSize: 8, color: 'var(--muted)',
                                            letterSpacing: '0.06em', textTransform: 'uppercase',
                                        }}>
                                            more
                                        </span>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>

                <ScrollDots count={Math.min(sorted.length, 50)} scrollRef={scrollRef} />
            </div>
        </>
    );
}