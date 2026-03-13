/**
 * RevealSection.tsx — Scroll-triggered staggered animation wrapper.
 * Each child fades in + slides up with 80ms stagger.
 */
import React from 'react';
import { motion } from 'framer-motion';
import { useScrollReveal } from '../hooks/useScrollReveal';

interface Props {
    children: React.ReactNode;
    stagger?: number;   // ms delay between children
    className?: string;
    style?: React.CSSProperties;
}

const EASE = [0.22, 1, 0.36, 1] as const;

export function RevealSection({ children, stagger = 80, className, style }: Props) {
    const { ref, isInView } = useScrollReveal();

    const items = React.Children.toArray(children);

    return (
        <div ref={ref} className={className} style={style}>
            {items.map((child, index) => (
                <motion.div
                    key={index}
                    animate={{
                        opacity: isInView ? 1 : 0,
                        y: isInView ? 0 : 40,
                    }}
                    transition={{
                        duration: 0.55,
                        ease: EASE,
                        delay: index * (stagger / 1000),
                    }}
                    style={{ willChange: 'transform, opacity' }}
                >
                    {child}
                </motion.div>
            ))}
        </div>
    );
}

/** Single-element reveal (no stagger, optional clip-path heading reveal) */
export function RevealHeading({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
    const { ref, isInView } = useScrollReveal();
    return (
        <motion.div
            ref={ref}
            animate={{
                clipPath: isInView ? 'inset(0 0% 0 0)' : 'inset(0 100% 0 0)',
            }}
            transition={{ duration: 0.7, ease: EASE, delay }}
            style={{ overflow: 'hidden' }}
        >
            {children}
        </motion.div>
    );
}
