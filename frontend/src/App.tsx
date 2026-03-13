/**
 * App.tsx — Root layout
 *
 * - LoadingScreen: shown for minimum 1800ms regardless of connection speed
 * - AnimatePresence: wraps router for page transitions
 * - Navbar: glass blur, layoutId active indicator, per-phase status dot
 * - ParticleField: tsParticles background, auto-pauses during simulation
 *
 * Socket.IO handlers, Zustand store shape, API calls — UNCHANGED.
 */
import React, { useState, useEffect } from 'react';
import {
    BrowserRouter, Routes, Route, Navigate,
    NavLink, useLocation,
} from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { useShallow } from 'zustand/react/shallow';
import { useSimulationStore } from './store/simulationStore';
import { useSocket } from './hooks/useSocket';
import ConfigPage from './pages/ConfigPage';
import SimulationPage from './pages/SimulationPage';
import AnalyticsPage from './pages/AnalyticsPage';
import { ErrorBoundary } from './ErrorBoundary';
import LoadingScreen from './components/LoadingScreen';
import ParticleField from './components/ParticleField';
import type { Phase } from './store/simulationStore';

// ── Global styles injected once ───────────────────────────────────────────────
const GLOBAL_STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  /* Backgrounds */
  --void:          #010409;
  --base:          #0d1117;
  --surface:       #161b22;
  --raised:        #21262d;
  --overlay:       rgba(13,17,23,0.92);

  /* Borders */
  --rim:           rgba(255,255,255,0.06);
  --rim-hi:        rgba(255,255,255,0.11);
  --rim-accent:    rgba(56,139,253,0.25);

  /* Text */
  --sol:           #e6edf3;
  --dim:           #7d8590;
  --muted:         #3d444d;

  /* Accents */
  --blue:          #4493f8;
  --blue-lo:       rgba(68,147,248,0.15);
  --blue-glow:     0 0 24px rgba(68,147,248,0.28);
  --cyan:          #2dd4bf;
  --violet:        #a78bfa;
  --amber:         #e3b341;
  --amber-lo:      rgba(227,179,65,0.12);
  --amber-glow:    0 0 20px rgba(227,179,65,0.3);
  --rose:          #f85149;
  --emerald:       #3fb950;
  --emerald-lo:    rgba(63,185,80,0.12);

  /* Typography */
  --font-head:     'Syne', sans-serif;
  --font-body:     'DM Sans', sans-serif;
  --font-mono:     'DM Mono', monospace;
}

html, body {
  background: var(--base);
  color: var(--sol);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  scroll-behavior: smooth;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--dim); }

/* Nav active indicator */
.nav-indicator-pill {
  position: absolute;
  bottom: 0;
  left: 8px;
  right: 8px;
  height: 2px;
  background: var(--amber);
  border-radius: 2px 2px 0 0;
  box-shadow: 0 0 10px rgba(227,179,65,0.5);
}

/* Pulse keyframe for status dot */
@keyframes statusPulse {
  0%   { box-shadow: 0 0 0 0 currentColor; }
  70%  { box-shadow: 0 0 0 6px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}

/* Background grid */
.app-grid-bg {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
  background-size: 52px 52px;
}

/* Background gradient blobs */
.app-blobs {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.app-blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  animation: blobDrift 25s ease-in-out infinite alternate;
}
@keyframes blobDrift {
  0%   { transform: translate(0, 0) scale(1); }
  33%  { transform: translate(30px, -20px) scale(1.05); }
  66%  { transform: translate(-20px, 15px) scale(0.97); }
  100% { transform: translate(10px, -10px) scale(1.02); }
}
`;

// ── Status dot config ────────────────────────────────────────────────────────

function getStatusInfo(phase: Phase, isRunning: boolean) {
    if (!isRunning || phase === 0) return { color: 'var(--muted)', label: 'Ready', pulsing: false };
    if (phase === 1) return { color: 'var(--blue)', label: 'Observing', pulsing: true };
    if (phase === 'training') return { color: 'var(--violet)', label: 'Training', pulsing: true };
    if (phase === 3) return { color: 'var(--amber)', label: 'Scheduling', pulsing: true };
    return { color: 'var(--emerald)', label: 'Complete', pulsing: false };
}

// ── Navbar ───────────────────────────────────────────────────────────────────

const NAV_LINKS = [
    { path: '/', label: 'Config' },
    { path: '/simulation', label: 'Simulation' },
    { path: '/analytics', label: 'Analytics' },
];

function Navbar() {
    const { phase, isRunning } = useSimulationStore(
        useShallow(s => ({ phase: s.phase, isRunning: s.isRunning }))
    );
    const location = useLocation();
    const { color, label, pulsing } = getStatusInfo(phase, isRunning);

    return (
        <nav style={{
            position: 'sticky',
            top: 0,
            zIndex: 100,
            height: 56,
            background: 'rgba(13,17,23,0.82)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            borderBottom: '1px solid var(--rim)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 28px',
        }}>

            {/* Logotype */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <ColorCyclingDot />
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
                    <span style={{
                        fontFamily: 'var(--font-head)',
                        fontSize: 13,
                        color: 'var(--dim)',
                        fontWeight: 600,
                        letterSpacing: '0.06em',
                        textTransform: 'uppercase',
                    }}>
                        AMD
                    </span>
                    <span style={{
                        fontFamily: 'var(--font-head)',
                        fontSize: 15,
                        color: 'var(--sol)',
                        fontWeight: 800,
                        letterSpacing: '-0.01em',
                    }}>
                        SlingShot
                    </span>
                </div>
            </div>

            {/* Nav links */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {NAV_LINKS.map(({ path, label: lbl }) => {
                    const isActive = location.pathname === path;
                    return (
                        <NavLink
                            key={path}
                            to={path}
                            style={{
                                position: 'relative',
                                padding: '16px 18px',
                                fontFamily: 'var(--font-body)',
                                fontSize: 13,
                                fontWeight: isActive ? 600 : 400,
                                color: isActive ? 'var(--sol)' : 'var(--dim)',
                                textDecoration: 'none',
                                transition: 'color 0.2s',
                                whiteSpace: 'nowrap',
                            }}
                        >
                            {lbl}
                            {isActive && (
                                <motion.span
                                    layoutId="nav-indicator"
                                    className="nav-indicator-pill"
                                    transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                                />
                            )}
                        </NavLink>
                    );
                })}
            </div>

            {/* Status indicator */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <motion.div
                    animate={pulsing ? {
                        boxShadow: [`0 0 0 0 ${color}99`, `0 0 0 5px transparent`],
                    } : { boxShadow: '0 0 0 0 transparent' }}
                    transition={{ duration: 1.6, repeat: Infinity, ease: 'easeOut' }}
                    style={{
                        width: 7,
                        height: 7,
                        borderRadius: '50%',
                        background: color,
                        flexShrink: 0,
                        transition: 'background 0.4s',
                    }}
                />
                <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.68rem',
                    color,
                    textTransform: 'uppercase',
                    letterSpacing: '0.12em',
                    transition: 'color 0.4s',
                }}>
                    {label}
                </span>
            </div>
        </nav>
    );
}

// ── Color cycling dot in logotype ─────────────────────────────────────────────
const POLICY_COLORS = ['#4493f8', '#a78bfa', '#2dd4bf', '#e3b341', '#f85149', '#3fb950'];

function ColorCyclingDot() {
    const [idx, setIdx] = useState(0);
    useEffect(() => {
        const id = setInterval(() => setIdx(i => (i + 1) % POLICY_COLORS.length), 5000);
        return () => clearInterval(id);
    }, []);
    return (
        <motion.div
            animate={{ backgroundColor: POLICY_COLORS[idx] }}
            transition={{ duration: 1.2, ease: 'easeInOut' }}
            style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                flexShrink: 0,
                boxShadow: `0 0 8px ${POLICY_COLORS[idx]}`,
            }}
        />
    );
}

// ── Background (grid + blobs, no tsParticles CPU cost) ───────────────────────
function AppBackground() {
    return (
        <>
            <div className="app-blobs">
                <div className="app-blob" style={{
                    width: 600, height: 500,
                    left: '-10%', top: '-15%',
                    background: 'rgba(68,147,248,0.055)',
                    animationDelay: '0s',
                }} />
                <div className="app-blob" style={{
                    width: 500, height: 600,
                    right: '-8%', bottom: '5%',
                    background: 'rgba(167,139,250,0.045)',
                    animationDelay: '-8s',
                }} />
                <div className="app-blob" style={{
                    width: 400, height: 400,
                    left: '40%', top: '40%',
                    background: 'rgba(45,212,191,0.03)',
                    animationDelay: '-15s',
                }} />
            </div>
            <div className="app-grid-bg" />
        </>
    );
}

// ── AppShell ──────────────────────────────────────────────────────────────────
function AppShell() {
    useSocket();

    const tickElapsed = useSimulationStore(s => s.tickElapsed);
    useEffect(() => {
        const id = setInterval(tickElapsed, 1000);
        return () => clearInterval(id);
    }, [tickElapsed]);

    const location = useLocation();

    return (
        <div style={{ minHeight: '100vh', position: 'relative' }}>
            {/* Ambient background layers */}
            <AppBackground />

            {/* ParticleField on top of background blobs */}
            <ParticleField />

            {/* Main content */}
            <div style={{ position: 'relative', zIndex: 2 }}>
                <Navbar />

                <AnimatePresence mode="wait">
                    <Routes location={location} key={location.pathname}>
                        <Route path="/" element={<ConfigPage />} />
                        <Route path="/simulation" element={<SimulationPage />} />
                        <Route path="/analytics" element={<AnalyticsPage />} />
                        <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                </AnimatePresence>
            </div>
        </div>
    );
}

// ── App root ──────────────────────────────────────────────────────────────────
export default function App() {
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => setLoading(false), 1800);
        return () => clearTimeout(timer);
    }, []);

    return (
        <ErrorBoundary>
            {/* Inject global styles */}
            <style>{GLOBAL_STYLES}</style>

            <BrowserRouter>
                <AnimatePresence>
                    {loading && <LoadingScreen key="loading" />}
                </AnimatePresence>
                {!loading && <AppShell />}
            </BrowserRouter>
        </ErrorBoundary>
    );
}