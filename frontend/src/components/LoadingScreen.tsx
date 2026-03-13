/**
 * LoadingScreen.tsx — Full-viewport splash screen.
 * Minimum display duration: 1800ms regardless of connection speed.
 * Exit: opacity 0 + scale 1.04 + blur 8px over 400ms (framer-motion).
 */
import React from 'react';
import { motion } from 'framer-motion';

const POLICY_COLORS = ['#388bfd', '#bc8cff', '#39d353', '#d29922', '#f85149'];

export default function LoadingScreen() {
    return (
        <motion.div
            key="loading"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 1.04, filter: 'blur(8px)' }}
            transition={{ duration: 0.4 }}
            style={{
                position: 'fixed', inset: 0, zIndex: 9999,
                background: 'var(--bg-void)',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                overflow: 'hidden',
            }}
        >
            {/* Animated background mesh — 3 rotating radial gradients */}
            <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
                <div className="loading-mesh-1" />
                <div className="loading-mesh-2" />
                <div className="loading-mesh-3" />
            </div>

            {/* Centered content */}
            <div style={{
                position: 'relative', zIndex: 1,
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 0, userSelect: 'none',
            }}>
                {/* AMD label */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.6 }}
                    style={{
                        fontFamily: 'var(--font-ui)',
                        fontSize: '0.75rem',
                        letterSpacing: '0.4em',
                        textTransform: 'uppercase',
                        color: 'var(--text-muted)',
                        marginBottom: 8,
                    }}
                >
                    AMD
                </motion.div>

                {/* SlingShot heading */}
                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
                    style={{
                        fontFamily: 'var(--font-ui)',
                        fontSize: '3.5rem',
                        fontWeight: 800,
                        color: 'var(--text-primary)',
                        lineHeight: 1,
                        margin: 0,
                    }}
                >
                    SlingShot
                </motion.h1>

                {/* Animated underline */}
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 0.7, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
                    style={{
                        height: 2,
                        background: 'var(--accent-blue)',
                        boxShadow: 'var(--glow-blue)',
                        borderRadius: 1,
                        marginTop: 12,
                        marginBottom: 32,
                        alignSelf: 'stretch',
                    }}
                />

                {/* 3 loading dots with stagger */}
                <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                    {[0, 1, 2].map(i => (
                        <motion.div
                            key={i}
                            animate={{ opacity: [0.2, 1, 0.2], y: [0, -6, 0] }}
                            transition={{
                                duration: 1.2,
                                delay: i * 0.2,
                                repeat: Infinity,
                                ease: 'easeInOut',
                            }}
                            style={{
                                width: 8, height: 8,
                                borderRadius: '50%',
                                background: POLICY_COLORS[i],
                            }}
                        />
                    ))}
                </div>
            </div>
        </motion.div>
    );
}
