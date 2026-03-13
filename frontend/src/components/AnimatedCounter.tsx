/**
 * AnimatedCounter.tsx — Spring-eased numeric counter.
 * Animates from 0 to the target value when the value first appears or changes.
 */
import React, { useEffect, useRef, useState } from 'react';

interface Props {
    value: number;
    duration?: number;   // ms
    decimals?: number;
    prefix?: string;
    suffix?: string;
    style?: React.CSSProperties;
    className?: string;
}

function easeOutExpo(t: number): number {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

export function useCountUp(target: number, duration = 800) {
    const [display, setDisplay] = useState(0);
    const startRef = useRef<number | null>(null);
    const prevTarget = useRef(target);
    const rafRef = useRef<number>(0);

    useEffect(() => {
        if (target === prevTarget.current) return;
        const from = prevTarget.current;
        prevTarget.current = target;
        startRef.current = null;

        const tick = (ts: number) => {
            if (!startRef.current) startRef.current = ts;
            const elapsed = ts - startRef.current;
            const progress = Math.min(1, elapsed / duration);
            const eased = easeOutExpo(progress);
            setDisplay(from + (target - from) * eased);
            if (progress < 1) {
                rafRef.current = requestAnimationFrame(tick);
            } else {
                setDisplay(target);
            }
        };

        cancelAnimationFrame(rafRef.current);
        rafRef.current = requestAnimationFrame(tick);

        return () => cancelAnimationFrame(rafRef.current);
    }, [target, duration]);

    return display;
}

export default function AnimatedCounter({
    value, duration = 800, decimals = 2, prefix = '', suffix = '', style, className
}: Props) {
    const display = useCountUp(value, duration);

    return (
        <span className={`num font-mono ${className ?? ''}`} style={style}>
            {prefix}{display.toFixed(decimals)}{suffix}
        </span>
    );
}
