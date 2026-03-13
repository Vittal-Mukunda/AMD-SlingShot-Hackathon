/**
 * ParticleField.tsx — tsParticles v3 animated background.
 * Pauses (opacity → 0) when simulation is running (phase !== 0).
 * Uses @tsparticles/react v3 + tsparticles v3 API.
 */
import React, { useCallback } from 'react';
import Particles from '@tsparticles/react';
import { initParticlesEngine } from '@tsparticles/react';
import { loadSlim } from '@tsparticles/slim';
import { useShallow } from 'zustand/react/shallow';
import { useSimulationStore } from '../store/simulationStore';
import { useEffect, useState } from 'react';

const PARTICLE_OPTIONS = {
    background: { color: { value: 'transparent' } },
    fpsLimit: 30,
    particles: {
        number: { value: 55, density: { enable: true, area: 800 } },
        color: { value: ['#388bfd', '#bc8cff', '#39d353'] },
        opacity: {
            value: 0.06,
            random: { enable: true, minimumValue: 0.03 },
            animation: { enable: true, speed: 0.5, sync: false },
        },
        size: {
            value: 1.5,
            random: { enable: true, minimumValue: 1 },
        },
        move: {
            enable: true, speed: 0.35,
            random: true,
            outModes: { default: 'out' as const },
        },
        links: {
            enable: true,
            distance: 130,
            color: '#388bfd',
            opacity: 0.05,
            width: 1,
        },
    },
    interactivity: {
        events: {
            onHover: { enable: true, mode: 'grab' },
            resize: { enable: true },
        },
        modes: {
            grab: { distance: 140, links: { opacity: 0.2 } },
        },
    },
    detectRetina: true,
};

export default function ParticleField() {
    const phase = useSimulationStore(useShallow(s => s.phase));
    const paused = phase !== 0;
    const [engineReady, setEngineReady] = useState(false);

    useEffect(() => {
        initParticlesEngine(async (engine) => {
            await loadSlim(engine);
        }).then(() => setEngineReady(true));
    }, []);

    if (!engineReady) return null;

    return (
        <div
            style={{
                position: 'fixed', inset: 0, zIndex: 0,
                pointerEvents: 'none',
                opacity: paused ? 0 : 1,
                transition: 'opacity 1s ease',
            }}
        >
            <Particles
                id="tsparticles"
                options={PARTICLE_OPTIONS as any}
                style={{ position: 'absolute', inset: 0 }}
            />
        </div>
    );
}
