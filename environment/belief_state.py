"""
Bayesian belief state tracker for hidden worker skills
Maintains Beta distribution posteriors and provides uncertainty quantification
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BeliefState:
    """
    Tracks belief distributions over hidden worker skills using Beta distributions
    
    Attributes:
        skill_beliefs (Dict[int, Tuple[float, float]]): Beta(α, β) parameters per worker
        skill_observations (Dict[int, List[float]]): Observed quality scores per worker
    """
    
    def __init__(self, num_workers: int):
        """
        Initialize belief state with uniform priors
        
        Args:
            num_workers: Number of workers to track
        """
        self.num_workers = num_workers
        
        # Initialize with prior: Beta(α=2, β=2) ≈ uniform over [0,1] with slight central bias
        self.skill_beliefs = {
            i: (config.SKILL_PRIOR_ALPHA, config.SKILL_PRIOR_BETA) 
            for i in range(num_workers)
        }
        
        self.skill_observations = {i: [] for i in range(num_workers)}
    
    def update(self, worker_id: int, quality_score: float):
        """
        Bayesian update based on observed task completion quality
        
        Args:
            worker_id: ID of worker who completed task
            quality_score: Observed quality [0, 1]
        """
        if worker_id not in self.skill_beliefs:
            raise ValueError(f"Worker {worker_id} not in belief state")
        
        # Record observation
        self.skill_observations[worker_id].append(quality_score)
        
        # Beta-Bernoulli conjugate update
        # Treat quality as success rate: α += quality, β += (1 - quality)
        alpha, beta = self.skill_beliefs[worker_id]
        alpha += quality_score
        beta += (1.0 - quality_score)
        
        self.skill_beliefs[worker_id] = (alpha, beta)
    
    def get_skill_mean(self, worker_id: int) -> float:
        """
        Get posterior mean skill estimate for worker
        
        Args:
            worker_id: Worker to query
            
        Returns:
            E[skill] = α / (α + β)
        """
        alpha, beta = self.skill_beliefs[worker_id]
        return alpha / (alpha + beta)
    
    def get_skill_variance(self, worker_id: int) -> float:
        """
        Get posterior variance (uncertainty) for worker skill
        
        Args:
            worker_id: Worker to query
            
        Returns:
            Var[skill] = αβ / ((α+β)^2 (α+β+1))
        """
        alpha, beta = self.skill_beliefs[worker_id]
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        return variance
    
    def get_skill_std(self, worker_id: int) -> float:
        """
        Get standard deviation of skill belief
        
        Args:
            worker_id: Worker to query
            
        Returns:
            Std[skill]
        """
        return np.sqrt(self.get_skill_variance(worker_id))
    
    def sample_skill(self, worker_id: int) -> float:
        """
        Thompson sampling: sample skill from posterior distribution
        
        Args:
            worker_id: Worker to sample for
            
        Returns:
            Sampled skill value from Beta distribution
        """
        alpha, beta = self.skill_beliefs[worker_id]
        return np.random.beta(alpha, beta)
    
    def get_ucb_score(self, worker_id: int, exploration_bonus: float = 2.0) -> float:
        """
        Upper confidence bound score for exploration
        
        Args:
            worker_id: Worker to compute UCB for
            exploration_bonus: Multiplier for uncertainty term
            
        Returns:
            UCB score = mean + bonus * std
        """
        mean = self.get_skill_mean(worker_id)
        std = self.get_skill_std(worker_id)
        return mean + exploration_bonus * std
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get belief state representation for all workers (for DQN input)
        
        Returns:
            2 * num_workers dimensional vector: [means..., variances...]
        """
        means = np.array([self.get_skill_mean(i) for i in range(self.num_workers)])
        variances = np.array([self.get_skill_variance(i) for i in range(self.num_workers)])
        return np.concatenate([means, variances])
    
    def reset(self):
        """
        Reset belief state to prior for new episode
        """
        self.skill_beliefs = {
            i: (config.SKILL_PRIOR_ALPHA, config.SKILL_PRIOR_BETA) 
            for i in range(self.num_workers)
        }
        self.skill_observations = {i: [] for i in range(self.num_workers)}
    
    def __repr__(self):
        info = []
        for i in range(min(3, self.num_workers)):  # Show first 3 workers
            mean = self.get_skill_mean(i)
            std = self.get_skill_std(i)
            info.append(f"W{i}: {mean:.2f}±{std:.2f}")
        return f"BeliefState({', '.join(info)})"


if __name__ == "__main__":
    # Unit test
    print("Testing BeliefState class...")
    
    # Test 1: Initialize
    belief = BeliefState(num_workers=5)
    print(f"✓ Initialized: {belief}")
    
    # Test 2: Prior statistics
    prior_mean = belief.get_skill_mean(0)
    prior_var = belief.get_skill_variance(0)
    assert abs(prior_mean - 0.5) < 0.01  # Beta(2,2) has mean 0.5
    print(f"✓ Prior: mean={prior_mean:.3f}, var={prior_var:.3f}")
    
    # Test 3: Update with observations
    for _ in range(10):
        belief.update(worker_id=0, quality_score=0.8)  # High quality
    
    post_mean = belief.get_skill_mean(0)
    post_var = belief.get_skill_variance(0)
    assert post_mean > prior_mean  # Should increase after high-quality observations
    assert post_var < prior_var  # Uncertainty should decrease
    print(f"✓ Posterior (after 10 x 0.8): mean={post_mean:.3f}, var={post_var:.3f}")
    
    # Test 4: Thompson sampling
    samples = [belief.sample_skill(0) for _ in range(100)]
    sample_mean = np.mean(samples)
    assert abs(sample_mean - post_mean) < 0.1  # Sample mean ≈ posterior mean
    print(f"✓ Thompson sample mean: {sample_mean:.3f} (expected {post_mean:.3f})")
    
    # Test 5: UCB score
    ucb = belief.get_ucb_score(0, exploration_bonus=2.0)
    assert ucb > post_mean  # UCB should be higher than mean
    print(f"✓ UCB score: {ucb:.3f} (mean={post_mean:.3f})")
    
    # Test 6: State vector
    state_vec = belief.get_state_vector()
    assert len(state_vec) == 10  # 5 means + 5 variances
    print(f"✓ State vector shape: {state_vec.shape}")
    
    # Test 7: Multiple workers
    belief.update(worker_id=1, quality_score=0.3)  # Low quality
    belief.update(worker_id=2, quality_score=0.9)  # Very high quality
    mean_1 = belief.get_skill_mean(1)
    mean_2 = belief.get_skill_mean(2)
    assert mean_1 < mean_2  # Worker 2 should have higher estimated skill
    print(f"✓ Different workers: W1={mean_1:.3f}, W2={mean_2:.3f}")
    
    print("\nAll BeliefState tests passed!")
