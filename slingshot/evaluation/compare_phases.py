"""
evaluation/compare_phases.py

Loads Phase 1 and Phase 2 metrics CSVs and produces:
  - Console comparison table
  - Saved results/phase_comparison.csv (for external analysis)
"""

import csv
import os
import sys
import numpy as np
from typing import List, Dict

from slingshot.core.settings import config


def load_csv(path: str) -> List[Dict]:
    """Load a metrics CSV into a list of row dicts."""
    if not os.path.exists(path):
        print(f"[WARNING] Metrics file not found: {path}")
        return []
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def agg(rows: List[Dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    vals = [v for v in vals if isinstance(v, float) and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def compare(p1_rows: List[Dict], p2_rows: List[Dict]) -> Dict:
    """Compute comparison dict with deltas."""
    metrics = [
        ('throughput_per_day',  'Throughput / day',   '{:.2f}'),
        ('completion_rate',     'Completion rate',     '{:.2%}'),
        ('lateness_rate',       'Lateness rate',       '{:.2%}'),
        ('quality_score',       'Quality score',       '{:.3f}'),
        ('overload_events',     'Overload events',     '{:.1f}'),
        ('load_balance',        'Load balance (std)',  '{:.3f}'),
        ('mean_loss',           'DQN mean loss',       '{:.4f}'),
        ('mean_q',              'DQN mean Q',          '{:.3f}'),
    ]

    header = f"  {'Metric':<25} {'Phase 1 (Baseline)':>18} {'Phase 2 (DQN)':>14} {'Δ':>8}"
    sep    = f"  {'─'*25} {'─'*18} {'─'*14} {'─'*8}"

    print("\n" + "═" * 70)
    print("  PHASE 1 vs PHASE 2 — Performance Comparison")
    print("═" * 70)
    print(header)
    print(sep)

    comparison_rows = []
    for key, label, fmt in metrics:
        v1 = agg(p1_rows, key)
        v2 = agg(p2_rows, key)
        delta = v2 - v1
        sign  = '+' if delta >= 0 else ''
        v1_str = fmt.format(v1)
        v2_str = fmt.format(v2)
        d_str  = f"{sign}{fmt.format(delta)}"
        print(f"  {label:<25} {v1_str:>18} {v2_str:>14} {d_str:>8}")
        comparison_rows.append({
            'metric': label, 'phase1': v1, 'phase2': v2, 'delta': delta
        })

    print("═" * 70)
    return comparison_rows


def save_comparison(rows, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved → {path}")


def main():
    results_dir = config.RESULTS_DIR
    p1_path = os.path.join(results_dir, 'phase1_metrics.csv')
    p2_path = os.path.join(results_dir, 'phase2_metrics.csv')

    p1_rows = load_csv(p1_path)
    p2_rows = load_csv(p2_path)

    if not p1_rows and not p2_rows:
        print("No metrics found. Run continual_scheduler.py first.")
        return

    comparison = compare(p1_rows, p2_rows)

    out_path = os.path.join(results_dir, 'phase_comparison.csv')
    save_comparison(comparison, out_path)


if __name__ == '__main__':
    main()
