"""
Reward Engineering -- decomposed reward for the docking game.

Instead of using only the raw Vina delta-E, we decompose the reward into
interpretable components:

  total_reward = base_reward * w_base
               + hbond_bonus * w_hbond
               + clash_penalty * w_clash
               + progress_bonus * w_progress

Each component targets a different aspect of good docking:
  - base_reward: Vina energy improvement (the ground truth signal)
  - hbond_bonus: Reward for forming hydrogen bonds with catalytic residues
  - clash_penalty: Penalty for steric clashes
  - progress_bonus: Bonus for getting closer to the catalytic site

Weights are configurable and should be tuned per target.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Configurable weights for reward components."""

    base: float = 1.0          # Weight for Vina delta-E
    hbond: float = 0.5         # Weight for H-bond bonus
    clash: float = -0.3        # Weight for clash penalty (negative = penalize)
    progress: float = 0.2      # Weight for catalytic proximity bonus
    occupancy: float = 0.1     # Weight for pocket occupancy bonus
    water_displacement: float = 0.05  # Weight for water displacement bonus


@dataclass
class RewardComponents:
    """Decomposed reward components for a single step."""

    base_reward: float = 0.0
    hbond_bonus: float = 0.0
    clash_penalty: float = 0.0
    progress_bonus: float = 0.0
    occupancy_bonus: float = 0.0
    water_displacement_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_reward": self.base_reward,
            "hbond_bonus": self.hbond_bonus,
            "clash_penalty": self.clash_penalty,
            "progress_bonus": self.progress_bonus,
            "occupancy_bonus": self.occupancy_bonus,
            "water_displacement_bonus": self.water_displacement_bonus,
            "total": self.total,
        }


class RewardShaper:
    """
    Compute decomposed rewards for the docking game.

    Takes raw environment info and produces a shaped reward that
    encourages binding-relevant behaviors beyond raw Vina delta-E.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        catalytic_distance_threshold: float = 6.0,
        hbond_distance_threshold: float = 3.5,
        clash_distance_threshold: float = 2.0,
    ):
        """
        Args:
            weights: RewardWeights instance (uses defaults if None).
            catalytic_distance_threshold: Distance (A) within which
                catalytic proximity bonus applies.
            hbond_distance_threshold: Distance (A) for counting H-bonds.
            clash_distance_threshold: Distance (A) below which a clash
                is counted.
        """
        self.weights = weights or RewardWeights()
        self.catalytic_dist_thresh = catalytic_distance_threshold
        self.hbond_dist_thresh = hbond_distance_threshold
        self.clash_dist_thresh = clash_distance_threshold

        # Track previous values for progress computation
        self._prev_dist_catalytic: Optional[float] = None

    def reset(self):
        """Reset state at the start of a new episode."""
        self._prev_dist_catalytic = None

    def compute_reward(
        self,
        vina_old: float,
        vina_new: float,
        info: Dict,
        observation: Optional[np.ndarray] = None,
    ) -> RewardComponents:
        """
        Compute the full decomposed reward for a single step.

        Args:
            vina_old: Previous Vina score (kcal/mol).
            vina_new: New Vina score after action.
            info: Info dict from DockingEnv.step().
            observation: Full observation vector (optional, for extracting
                interaction features).

        Returns:
            RewardComponents with all components and total.
        """
        components = RewardComponents()

        # 1. Base reward: Vina energy improvement
        # More negative Vina = better, so reward = old - new
        components.base_reward = vina_old - vina_new

        # 2. H-bond bonus: reward for forming H-bonds near catalytic site
        dist_asp32 = info.get("dist_asp32", 50.0)
        dist_asp228 = info.get("dist_asp228", 50.0)
        min_catalytic_dist = min(dist_asp32, dist_asp228)

        if min_catalytic_dist < self.hbond_dist_thresh:
            # Strong H-bond bonus when very close to catalytic residues
            components.hbond_bonus = 1.0 - (
                min_catalytic_dist / self.hbond_dist_thresh
            )
        elif min_catalytic_dist < self.catalytic_dist_thresh:
            # Mild bonus when in range
            components.hbond_bonus = 0.2 * (
                1.0 - (min_catalytic_dist - self.hbond_dist_thresh)
                / (self.catalytic_dist_thresh - self.hbond_dist_thresh)
            )
        else:
            components.hbond_bonus = 0.0

        # 3. Clash penalty: penalize steric clashes
        if observation is not None and len(observation) > 67:
            # Steric clash count is at interaction index 7 (offset by
            # pocket_dim + ligand_dim + 7)
            # Default layout: 40 + 16 + 7 = 63
            clash_idx = 63
            if clash_idx < len(observation):
                n_clashes = observation[clash_idx]
                components.clash_penalty = float(n_clashes)
        else:
            components.clash_penalty = 0.0

        # 4. Progress bonus: getting closer to catalytic site
        if self._prev_dist_catalytic is not None:
            dist_improvement = self._prev_dist_catalytic - min_catalytic_dist
            if dist_improvement > 0 and min_catalytic_dist < self.catalytic_dist_thresh:
                components.progress_bonus = min(dist_improvement, 2.0)
            elif dist_improvement < 0:
                # Small penalty for moving away from catalytic site
                components.progress_bonus = max(dist_improvement, -0.5)
        self._prev_dist_catalytic = min_catalytic_dist

        # 5. Occupancy bonus: fraction of pocket filled
        if observation is not None and len(observation) > 68:
            occupancy_idx = 64  # pocket_dim + ligand_dim + 8
            if occupancy_idx < len(observation):
                components.occupancy_bonus = float(observation[occupancy_idx])

        # 6. Water displacement bonus
        if observation is not None and len(observation) > 69:
            water_idx = 65  # pocket_dim + ligand_dim + 9
            if water_idx < len(observation):
                components.water_displacement_bonus = float(
                    observation[water_idx]
                ) * 0.1  # Scale down

        # Total shaped reward
        components.total = (
            self.weights.base * components.base_reward
            + self.weights.hbond * components.hbond_bonus
            + self.weights.clash * components.clash_penalty
            + self.weights.progress * components.progress_bonus
            + self.weights.occupancy * components.occupancy_bonus
            + self.weights.water_displacement * components.water_displacement_bonus
        )

        return components

    def get_weights(self) -> Dict[str, float]:
        """Return current weight configuration."""
        return {
            "base": self.weights.base,
            "hbond": self.weights.hbond,
            "clash": self.weights.clash,
            "progress": self.weights.progress,
            "occupancy": self.weights.occupancy,
            "water_displacement": self.weights.water_displacement,
        }

    def set_weights(self, **kwargs):
        """Update reward weights. Only provided keys are changed."""
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
            else:
                raise ValueError(f"Unknown reward weight: {key}")
