"""
Diagnostic utilities for environment stability analysis
Analyzes state normalization, reward distributions, action sparsity, and numerical stability
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from environment.project_env import ProjectEnv


class EnvironmentDiagnostics:
    """
    Collects and analyzes environment statistics for DQN stability
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize diagnostics tracker
        
        Args:
            enable_logging: Whether to enable diagnostic logging
        """
        self.enable_logging = enable_logging
        
        # Statistics trackers
        self.state_ranges = {'min': [], 'max': [], 'mean': [], 'std': []}
        self.reward_components = {
            'action': [], 'completion': [], 'delay': [],
            'overload': [], 'deadline': [], 'total': []
        }
        self.valid_action_counts = []
        self.state_feature_ranges = {f'feature_{i}': {'min': [], 'max': []} for i in range(88)}
        
    def log_step(self, state: np.ndarray, reward_components: Dict[str, float], 
                 num_valid_actions: int, timestep: int):
        """
        Log statistics for a single environment step
        
        Args:
            state: State vector (88-dim)
            reward_components: Dict of reward component values
            num_valid_actions: Number of valid actions at this step
            timestep: Current timestep
        """
        if not self.enable_logging:
            return
        
        # State statistics
        self.state_ranges['min'].append(np.min(state))
        self.state_ranges['max'].append(np.max(state))
        self.state_ranges['mean'].append(np.mean(state))
        self.state_ranges['std'].append(np.std(state))
        
        # Per-feature ranges
        for i in range(len(state)):
            self.state_feature_ranges[f'feature_{i}']['min'].append(state[i])
            self.state_feature_ranges[f'feature_{i}']['max'].append(state[i])
        
        # Reward components
        for key, value in reward_components.items():
            if key in self.reward_components:
                self.reward_components[key].append(value)
        
        # Valid actions
        self.valid_action_counts.append(num_valid_actions)
    
    def analyze(self, num_episodes: int = 50) -> Dict:
        """
        Run diagnostic analysis over multiple episodes
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary of diagnostic results
        """
        print(f"\n{'='*80}")
        print(f"ENVIRONMENT DIAGNOSTICS ({num_episodes} episodes)")
        print(f"{'='*80}\n")
        
        env = ProjectEnv(seed=42)
        
        # Reset trackers
        self.state_ranges = {'min': [], 'max': [], 'mean': [], 'std': []}
        self.reward_components = {
            'action': [], 'completion': [], 'delay': [],
            'overload': [], 'deadline': [], 'total': []
        }
        self.valid_action_counts = []
        
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            timestep = 0
            
            while not done and timestep < config.EPISODE_HORIZON:
                # Get valid actions
                valid_actions = env.get_valid_actions()
                num_valid = len(valid_actions)
                
                if num_valid == 0:
                    break
                
                # Take random action
                action = np.random.choice(valid_actions)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Extract reward components (reconstruct from env internals)
                reward_comps = {
                    'total': reward,
                    'action': 0,  # Approximation
                    'completion': 0,
                    'delay':0,
                    'overload': 0,
                    'deadline': 0
                }
                
                # Log statistics
                self.log_step(state, reward_comps, num_valid, timestep)
                
                state = next_state
                timestep += 1
        
        # Compute summaries
        results = self._compute_summary()
        self._print_report(results)
        
        return results
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics from collected data"""
        
        results = {}
        
        # === 1. STATE NORMALIZATION ANALYSIS ===
        results['state_normalization'] = {
            'global_min': np.min(self.state_ranges['min']),
            'global_max': np.max(self.state_ranges['max']),
            'mean_range': (np.mean(self.state_ranges['min']), np.mean(self.state_ranges['max'])),
            'std_range': (np.min(self.state_ranges['std']), np.max(self.state_ranges['std']))
        }
        
        # Per-feature unbounded growth check
        unbounded_features = []
        for i in range(88):
            feature_mins = self.state_feature_ranges[f'feature_{i}']['min']
            feature_maxs = self.state_feature_ranges[f'feature_{i}']['max']
            if len(feature_maxs) > 0:
                max_value = np.max(feature_maxs)
                min_value = np.min(feature_mins)
                if max_value > 10 or min_value < -10:  # Flag potential unbounded features
                    unbounded_features.append((i, min_value, max_value))
        
        results['unbounded_features'] = unbounded_features
        
        # === 2. REWARD MAGNITUDE ANALYSIS ===
        results['reward_analysis'] = {
            'total_mean': np.mean(self.reward_components['total']) if self.reward_components['total'] else 0,
            'total_std': np.std(self.reward_components['total']) if self.reward_components['total'] else 0,
            'total_min': np.min(self.reward_components['total']) if self.reward_components['total'] else 0,
            'total_max': np.max(self.reward_components['total']) if self.reward_components['total'] else 0,
        }
        
        # === 3. ACTION SPARSITY ANALYSIS ===
        results['action_sparsity'] = {
            'mean_valid_actions': np.mean(self.valid_action_counts) if self.valid_action_counts else 0,
            'min_valid_actions': np.min(self.valid_action_counts) if self.valid_action_counts else 0,
            'max_valid_actions': np.max(self.valid_action_counts) if self.valid_action_counts else 0,
            'sparsity_ratio': np.mean(self.valid_action_counts) / 140 if self.valid_action_counts else 0
        }
        
        return results
    
    def _print_report(self, results: Dict):
        """Print diagnostic report"""
        
        print("\n" + "="*80)
        print("1. STATE NORMALIZATION")
        print("="*80)
        norm = results['state_normalization']
        print(f"Global range: [{norm['global_min']:.4f}, {norm['global_max']:.4f}]")
        print(f"Mean range: [{norm['mean_range'][0]:.4f}, {norm['mean_range'][1]:.4f}]")
        print(f"Std range: [{norm['std_range'][0]:.4f}, {norm['std_range'][1]:.4f}]")
        
        if results['unbounded_features']:
            print(f"\n⚠️  WARNING: {len(results['unbounded_features'])} features may be unbounded:")
            for feat_idx, min_val, max_val in results['unbounded_features'][:5]:
                print(f"   Feature {feat_idx}: [{min_val:.2f}, {max_val:.2f}]")
        else:
            print("✓ No unbounded features detected")
        
        print("\n" + "="*80)
        print("2. REWARD MAGNITUDE ANALYSIS")
        print("="*80)
        rwd = results['reward_analysis']
        print(f"Mean reward: {rwd['total_mean']:.4f}")
        print(f"Std reward: {rwd['total_std']:.4f}")
        print(f"Range: [{rwd['total_min']:.4f}, {rwd['total_max']:.4f}]")
        
        if abs(rwd['total_max']) > 100 or abs(rwd['total_min']) > 100:
            print("⚠️  WARNING: Reward magnitudes exceed 100 (may cause instability)")
        else:
            print("✓ Reward magnitudes reasonable")
        
        print("\n" + "="*80)
        print("3. ACTION SPARSITY ANALYSIS")
        print("="*80)
        sparse = results['action_sparsity']
        print(f"Mean valid actions: {sparse['mean_valid_actions']:.1f} / 140")
        print(f"Range: [{sparse['min_valid_actions']}, {sparse['max_valid_actions']}]")
        print(f"Sparsity ratio: {sparse['sparsity_ratio']:.2%}")
        
        if sparse['mean_valid_actions'] < 10:
            print("⚠️  WARNING: Very sparse action space (< 10 valid actions on average)")
        else:
            print("✓ Action space density acceptable")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    diagnostics = EnvironmentDiagnostics(enable_logging=True)
    results = diagnostics.analyze(num_episodes=50)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("="*80)
