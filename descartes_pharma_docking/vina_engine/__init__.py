"""
Module 3: Vina World Model (The Chess Engine)
==============================================

A perfect simulator that computes the exact binding score for any
ligand pose. NOT learned. NOT approximated. Wraps AutoDock Vina.

Vina evaluations are FREE -- unlike ARC-AGI-3 where each action
costs ticks, here we can evaluate millions of poses without budget.
"""

from descartes_pharma_docking.vina_engine.vina_scorer import (
    VinaWorldModel,
    VinaScore,
)

__all__ = ["VinaWorldModel", "VinaScore"]
