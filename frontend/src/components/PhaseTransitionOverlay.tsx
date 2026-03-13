/**
 * PhaseTransitionOverlay.tsx — Full-screen phase transition overlay.
 * Shows an animated neural network SVG while DQN trains.
 * Dismissed automatically when phase2_ready fires (phase === 3).
 */
import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
    visible: boolean;       // true when phase === 'training'
    percent: number;        // 0–100 training progress
    steps?: number;
}

// Neural network node layout: 3 hidden layers
const LAYERS = [
    [0.12, 0.12, 0.5, 0.88, 0.88],       // input
    [0.1, 0.3, 0.5, 0.7, 0.9],            // hidden 1
    [0.2, 0.4, 0.6, 0.8],                 // hidden 2
    [0.25, 0.5, 0.75],                    // output
];
const LAYER_XS = [0.08, 0.33, 0.60, 0.85];

interface NNNode { lx: number; ly: number; layer: number; idx: number; }
interface NNEdge { from: NNNode; to: NNNode; key: string; }

const buildNetwork = (): { nodes: NNNode[]; edges: NNEdge[] } => {
    const nodes: NNNode[] = [];
    LAYERS.forEach((col, li) => {
        col.forEach((ly, ni) => {
            nodes.push({ lx: LAYER_XS[li], ly, layer: li, idx: ni });
        });
    });
    const edges: NNEdge[] = [];
    for (let li = 0; li < LAYERS.length - 1; li++) {
        const layerA = nodes.filter(n => n.layer === li);
        const layerB = nodes.filter(n => n.layer === li + 1);
        layerA.forEach(a => layerB.forEach(b => {
            edges.push({ from: a, to: b, key: `${a.layer}-${a.idx}-${b.idx}` });
        }));
    }
    return { nodes, edges };
};

const { nodes: NN_NODES, edges: NN_EDGES } = buildNetwork();

function NeuralNetViz({ percent }: { percent: number }) {
    const svgW = 500, svgH = 280;
    const r = 10;

    return (
        <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ opacity: 0.75 }}>
            {/* Edges */}
            {NN_EDGES.map((e, i) => {
                const x1 = e.from.lx * svgW, y1 = e.from.ly * svgH;
                const x2 = e.to.lx * svgW, y2 = e.to.ly * svgH;
                const active = (percent / 100) > (i / NN_EDGES.length);
                return (
                    <line key={e.key}
                        x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke={active ? '#3b82f6' : 'rgba(255,255,255,0.06)'}
                        strokeWidth={active ? 1.2 : 0.5}
                        style={{ transition: 'stroke 0.5s ease' }}
                    />
                );
            })}
            {/* Nodes */}
            {NN_NODES.map((n, i) => {
                const cx = n.lx * svgW, cy = n.ly * svgH;
                const active = n.layer < Math.round((percent / 100) * LAYERS.length);
                const color = n.layer === LAYERS.length - 1 ? '#f59e0b'
                    : active ? '#3b82f6' : '#334155';
                return (
                    <circle key={i}
                        cx={cx} cy={cy} r={r}
                        fill={color}
                        stroke={active ? color : 'rgba(255,255,255,0.1)'}
                        strokeWidth={2}
                        style={{
                            filter: active ? `drop-shadow(0 0 6px ${color})` : 'none',
                            transition: 'fill 0.4s ease, filter 0.4s ease',
                        }}
                    />
                );
            })}
        </svg>
    );
}

export default function PhaseTransitionOverlay({ visible, percent }: Props) {
    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    key="phase-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.4 }}
                    style={{
                        position: 'fixed', inset: 0, zIndex: 200,
                        background: '#020817',
                        display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        gap: 32,
                    }}
                >
                    {/* Ambient rotating glow */}
                    <div className="ambient-glow"
                        style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
                    />

                    {/* Card */}
                    <motion.div
                        initial={{ scale: 0.92, y: 24 }}
                        animate={{ scale: 1, y: 0 }}
                        transition={{ type: 'spring', stiffness: 200, damping: 24 }}
                        style={{
                            position: 'relative', zIndex: 1,
                            background: 'var(--color-surface)',
                            border: '1px solid rgba(245,158,11,0.2)',
                            borderRadius: 20,
                            padding: '40px 56px',
                            textAlign: 'center',
                            maxWidth: 580,
                            boxShadow: '0 0 60px rgba(245,158,11,0.12), 0 32px 80px rgba(0,0,0,0.7)',
                        }}
                    >
                        <div className="font-mono" style={{
                            fontSize: 10, color: 'var(--color-amber)',
                            letterSpacing: '0.20em', textTransform: 'uppercase', marginBottom: 12,
                        }}>
                            PHASE 2 — DQN TRAINING
                        </div>

                        <h2 className="typing-cursor" style={{
                            fontFamily: 'var(--font-ui)', fontSize: 22, fontWeight: 700,
                            color: 'var(--color-text)', lineHeight: 1.4, marginBottom: 24,
                        }}>
                            Training DQN Agent
                        </h2>

                        {/* Neural network visualization */}
                        <NeuralNetViz percent={percent} />

                        {/* Progress bar */}
                        <div style={{
                            width: '100%', height: 4,
                            background: 'rgba(245,158,11,0.10)',
                            borderRadius: 2, overflow: 'hidden', marginTop: 20, marginBottom: 10,
                        }}>
                            <motion.div
                                animate={{ width: `${percent}%` }}
                                transition={{ duration: 0.5 }}
                                style={{
                                    height: '100%',
                                    background: 'linear-gradient(90deg, var(--color-amber), #FBBF24)',
                                    borderRadius: 2,
                                    boxShadow: '0 0 12px rgba(245,158,11,0.5)',
                                }}
                            />
                        </div>

                        <div className="font-mono" style={{
                            fontSize: 10, color: 'var(--color-slate-dim)', letterSpacing: '0.06em',
                        }}>
                            {percent < 100
                                ? `Replaying transitions · Bellman updates · ${percent.toFixed(0)}%`
                                : 'Training complete — switching to live scheduling…'}
                        </div>
                    </motion.div>

                    <div className="font-mono" style={{
                        position: 'relative', zIndex: 1,
                        fontSize: 10, color: 'var(--color-slate-dim)',
                        letterSpacing: '0.1em', textTransform: 'uppercase',
                    }}>
                        Phase 1 complete — All baseline policies benchmarked
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
