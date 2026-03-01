/* ── Types: config.ts ─────────────────────────────────────────────────────────
   Configuration wizard form data.
──────────────────────────────────────────────────────────────────────────── */

export type WorkerMode = 'auto' | 'manual';
export type ArrivalDist = 'poisson' | 'uniform' | 'burst' | 'custom';

export interface ManualWorkerConfig {
    name: string;
    skill_level: number;         // 0–1 slider
    productivity_rate: number;   // 0.5–1.5
    fatigue_sensitivity: number; // 0.05–0.3
    task_type_efficiencies: Record<string, number>; // e.g. { "A": 0.8, "B": 1.2 }
}

export interface ArrivalParams {
    // Poisson
    rate?: number;              // mean tasks/day
    // Uniform
    min_per_day?: number;
    max_per_day?: number;
    // Burst
    burst_days?: number[];      // which days(0-indexed) are burst days
    burst_multiplier?: number;
    // Custom
    daily_overrides?: number[]; // explicit tasks per day
}

export interface InjectedTask {
    task_id: string;
    duration: number;            // in slots (30min each)
    urgency: 0 | 1 | 2 | 3;
    required_skill: number;      // 0–1
    arrival_tick: number;
}

export interface SimConfig {
    days_phase1: number;
    days_phase2: number;
    worker_mode: WorkerMode;
    worker_seed: number;
    num_workers: number;
    manual_workers: ManualWorkerConfig[];
    arrival_distribution: ArrivalDist;
    arrival_params: ArrivalParams;
    task_count: number;
    seed: number;
}

// Default config values
export const DEFAULT_SIM_CONFIG: SimConfig = {
    days_phase1: 20,
    days_phase2: 5,
    worker_mode: 'auto',
    worker_seed: 42,
    num_workers: 5,
    manual_workers: [],
    arrival_distribution: 'poisson',
    arrival_params: { rate: 3.5 },
    task_count: 200,
    seed: 42,
};

// Validation
export interface FormErrors {
    [field: string]: string;
}

export function validateSimConfig(cfg: SimConfig): FormErrors {
    const errors: FormErrors = {};

    if (cfg.days_phase1 < 1) errors.days_phase1 = 'Must be at least 1 working day';
    if (cfg.days_phase2 < 1) errors.days_phase2 = 'Must be at least 1 working day';
    if (cfg.days_phase1 > 60) errors.days_phase1 = 'Phase 1 must not exceed 60 days';
    if (cfg.days_phase2 > 30) errors.days_phase2 = 'Phase 2 must not exceed 30 days';
    if (cfg.num_workers < 1 || cfg.num_workers > 25) errors.num_workers = 'Workers must be 1–25';
    if (cfg.task_count < 10 || cfg.task_count > 500) errors.task_count = 'Tasks must be 10–500';

    // 8-hour workday hard constraint — users who see this are blocked from submitting
    // (actual enforcement is in config.py SLOTS_PER_DAY = 16, SLOT_HOURS = 0.5)
    // The form just needs to confirm the user knows workdays are fixed at 8h

    if (cfg.worker_mode === 'manual' && cfg.manual_workers.length === 0) {
        errors.manual_workers = 'Add at least one worker in manual mode';
    }

    return errors;
}

// Compute derived display values
export function computeTotalTicks(cfg: SimConfig): number {
    const SLOTS_PER_DAY = 16; // 8h / 0.5h
    return (cfg.days_phase1 + cfg.days_phase2) * SLOTS_PER_DAY;
}

export function estimateRuntimeSeconds(cfg: SimConfig): number {
    // Very rough: ~0.01s per decision, ~3 decisions/slot avg
    return computeTotalTicks(cfg) * 3 * 0.01;
}
