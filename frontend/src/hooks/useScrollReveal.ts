/**
 * useScrollReveal.ts — useInView hook for scroll-triggered animations.
 * Returns { ref, isInView } to animate elements when they enter the viewport.
 */
import { useRef } from 'react';
import { useInView } from 'framer-motion';

export function useScrollReveal(threshold = 0.12) {
    const ref = useRef<HTMLDivElement>(null);
    const isInView = useInView(ref, { once: true, amount: threshold });
    return { ref, isInView };
}
