"""
Test script for ablation studies logic.
Runs 1 episode per condition to verify pipeline integrity.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ablation_studies import run_ablation_studies

if __name__ == "__main__":
    print("Running ablation smoke test (1 episode/condition)...")
    try:
        run_ablation_studies(num_episodes=1, output_file='results/ablation_test_results.csv')
        print("✓ Ablation test passed!")
    except Exception as e:
        print(f"✗ Ablation test failed: {e}")
        sys.exit(1)
