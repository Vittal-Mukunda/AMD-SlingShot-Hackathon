import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useEffect } from 'react';
import { useSimulationStore } from './store/simulationStore';
import { useSocket } from './hooks/useSocket';
import ConfigPage from './pages/ConfigPage';
import SimulationPage from './pages/SimulationPage';
import AnalyticsPage from './pages/AnalyticsPage';

function AppShell() {
    // Mount socket connection at app level to persist across page navigations
    useSocket();

    // Start elapsed timer
    const tickElapsed = useSimulationStore(s => s.tickElapsed);
    useEffect(() => {
        const id = setInterval(tickElapsed, 1000);
        return () => clearInterval(id);
    }, [tickElapsed]);

    return (
        <Routes>
            <Route path="/" element={<ConfigPage />} />
            <Route path="/simulation" element={<SimulationPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    );
}

export default function App() {
    return (
        <BrowserRouter>
            <AppShell />
        </BrowserRouter>
    );
}
