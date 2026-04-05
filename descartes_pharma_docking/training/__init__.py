"""
Module 5: Training Loop
========================

RL-style training: the policy network proposes pose adjustments,
Vina scores them, and the policy updates to maximize binding improvement.
Each episode is one ligand being docked.

Components:
    DockingEnv     - Gym-style environment wrapping the docking game
    DockingTrainer - REINFORCE training with baseline and entropy bonus
    RewardShaper   - Decomposed reward engineering
"""

from descartes_pharma_docking.training.docking_env import DockingEnv
from descartes_pharma_docking.training.trainer import DockingTrainer

__all__ = ["DockingEnv", "DockingTrainer"]
