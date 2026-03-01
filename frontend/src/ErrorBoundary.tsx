import React, { type ErrorInfo, type ReactNode } from 'react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
    info: ErrorInfo | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false, error: null, info: null };
    }

    static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error, info: null };
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        console.error("ErrorBoundary caught an error", error, info);
        this.setState({ info });
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{ padding: '2rem', color: '#ff4444', background: '#220000', minHeight: '100vh', fontFamily: 'monospace' }}>
                    <h2>App Crashed!</h2>
                    <pre style={{ whiteSpace: 'pre-wrap', fontWeight: 'bold' }}>{this.state.error && this.state.error.toString()}</pre>
                    <details style={{ whiteSpace: 'pre-wrap', marginTop: '1rem', color: '#ffaaaa' }}>
                        {this.state.info && this.state.info.componentStack}
                    </details>
                </div>
            );
        }
        return this.props.children;
    }
}
