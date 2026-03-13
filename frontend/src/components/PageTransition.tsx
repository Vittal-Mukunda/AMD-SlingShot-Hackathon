/**
 * PageTransition.tsx — framer-motion page wrapper.
 * opacity 0 + blur 6px + y 20 → 1 + 0px + 0 on enter.
 * opacity 0 + blur 4px + y -12 on exit.
 */
import React from 'react';
import { motion } from 'framer-motion';

const pageVariants = {
    initial: { opacity: 0, y: 20, filter: 'blur(6px)' },
    animate: {
        opacity: 1, y: 0, filter: 'blur(0px)',
        transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] as const },
    },
    exit: {
        opacity: 0, y: -12, filter: 'blur(4px)',
        transition: { duration: 0.3 },
    },
};

interface Props {
    children: React.ReactNode;
}

export function PageTransition({ children }: Props) {
    return (
        <motion.div
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            style={{ width: '100%', minHeight: '100%' }}
        >
            {children}
        </motion.div>
    );
}
