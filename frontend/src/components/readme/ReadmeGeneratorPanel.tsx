/**
 * ReadmeGeneratorPanel.tsx — Module 4: README Generator UI
 *
 * Triggers /api/generate-readme → streams readme_progress WebSocket events →
 * Shows live generation log → completion modal with file tree + root README preview.
 */
import React, { useState, useEffect, useRef } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { useSocket } from '../../hooks/useSocket';

interface Props {
    onClose: () => void;
}

export default function ReadmeGeneratorPanel({ onClose }: Props) {
    const { readmeProgress, readmeComplete } = useSimulationStore();
    const { generateReadme } = useSocket();
    const [started, setStarted] = useState(false);
    const [rootPreview, setRootPreview] = useState('');
    const [showModal, setShowModal] = useState(false);
    const logRef = useRef<HTMLDivElement>(null);

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [readmeProgress]);

    // Show completion modal
    useEffect(() => {
        if (readmeComplete) {
            const rootEntry = readmeProgress.find(p => p.file_path === 'README.md' && p.status === 'done');
            if (rootEntry) setRootPreview(rootEntry.preview_snippet);
            setShowModal(true);
        }
    }, [readmeComplete, readmeProgress]);

    const handleGenerate = async () => {
        await fetch('/api/generate-readme', { method: 'POST' });
        generateReadme();
        setStarted(true);
    };

    const statusIcon = (status: string) => {
        if (status === 'done') return '✓';
        if (status === 'unchanged') return '─';
        if (status === 'generating') return '⟳';
        if (status === 'complete') return '✔';
        return '·';
    };
    const statusColor = (status: string) => {
        if (status === 'done') return 'var(--color-success)';
        if (status === 'unchanged') return 'var(--color-slate-dim)';
        if (status === 'generating') return 'var(--color-amber)';
        return 'var(--color-text)';
    };

    // Deduplicated file list for tree
    const fileTree = [...new Map(
        readmeProgress
            .filter(p => p.file_path !== '__complete__')
            .map(p => [p.file_path, p])
    ).values()];

    return (
        <div style={{
            background: 'var(--color-panel)', border: '1px solid var(--color-border)',
            borderRadius: 4, padding: '1.5rem', marginBottom: '1.5rem',
            position: 'relative'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                <div className="section-header" style={{ margin: 0 }}>README Generator</div>
                <div style={{ flex: 1 }} />
                <button className="btn-ghost" style={{ fontSize: 11 }} onClick={onClose}>✕ Close</button>
                {!started && (
                    <button className="btn-primary" style={{ fontSize: 12 }} onClick={handleGenerate}>
                        ⟳ Generate All READMEs
                    </button>
                )}
            </div>

            {!started && (
                <p style={{ color: 'var(--color-slate-text)', fontSize: '0.8125rem' }}>
                    Regenerates all README.md files across the project repository. The process is idempotent
                    — running it twice produces identical output. Progress streams live below.
                </p>
            )}

            {started && (
                <>
                    <div
                        ref={logRef}
                        style={{
                            background: 'var(--color-bg)', border: '1px solid var(--color-border)',
                            borderRadius: 2, padding: '0.75rem',
                            height: 200, overflowY: 'auto',
                            fontFamily: 'var(--font-mono)', fontSize: 11,
                        }}
                    >
                        {readmeProgress.length === 0 ? (
                            <div style={{ color: 'var(--color-slate-dim)' }}>Initiating README generation...</div>
                        ) : (
                            readmeProgress.map((p, i) => (
                                <div key={i} style={{ marginBottom: 4, color: statusColor(p.status) }}>
                                    <span style={{ marginRight: 8 }}>{statusIcon(p.status)}</span>
                                    <span style={{ color: 'var(--color-text)' }}>{p.file_path}</span>
                                    {p.status === 'generating' && (
                                        <span style={{ color: 'var(--color-slate-dim)', marginLeft: 8, fontSize: 10 }}>
                                            generating...
                                        </span>
                                    )}
                                    {p.status === 'done' && (
                                        <span style={{ color: 'var(--color-success)', marginLeft: 8, fontSize: 10 }}>
                                            updated
                                        </span>
                                    )}
                                    {p.status === 'unchanged' && (
                                        <span style={{ color: 'var(--color-slate-dim)', marginLeft: 8, fontSize: 10 }}>
                                            unchanged (hash match)
                                        </span>
                                    )}
                                    {p.status === 'complete' && (
                                        <span style={{ color: 'var(--color-success)', fontWeight: 700, marginLeft: 8 }}>
                                            ALL DONE
                                        </span>
                                    )}
                                </div>
                            ))
                        )}
                    </div>

                    {/* Completion modal */}
                    {showModal && (
                        <div style={{
                            marginTop: '1rem', background: 'var(--color-bg)',
                            border: '1px solid var(--color-amber-dim)', borderRadius: 4, padding: '1rem'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '1rem' }}>
                                <span style={{ color: 'var(--color-success)', fontSize: 16 }}>✔</span>
                                <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-success)', fontWeight: 700, letterSpacing: '0.08em' }}>
                                    README GENERATION COMPLETE
                                </span>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem' }}>
                                {/* File tree */}
                                <div>
                                    <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
                                        Updated Files
                                    </div>
                                    {fileTree.map((p, i) => (
                                        <div key={i} style={{ marginBottom: 4 }}>
                                            <span style={{ color: statusColor(p.status), marginRight: 6, fontSize: 11 }}>
                                                {statusIcon(p.status)}
                                            </span>
                                            <span className="font-mono" style={{ fontSize: 11, color: 'var(--color-text)' }}>
                                                {p.file_path}
                                            </span>
                                        </div>
                                    ))}
                                </div>

                                {/* Root README preview */}
                                <div>
                                    <div className="font-mono" style={{ fontSize: 9, color: 'var(--color-slate-dim)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
                                        Root README Preview
                                    </div>
                                    <div style={{
                                        background: '#0A0C0F', border: '1px solid var(--color-border)',
                                        borderRadius: 2, padding: '0.75rem',
                                        fontFamily: 'var(--font-mono)', fontSize: 10,
                                        color: 'var(--color-text)', lineHeight: 1.7,
                                        height: 200, overflowY: 'auto', whiteSpace: 'pre-wrap'
                                    }}>
                                        {rootPreview || 'Preview unavailable'}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
