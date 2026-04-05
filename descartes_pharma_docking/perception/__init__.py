"""
Module 1: PocketKnowledge Perception
=====================================

Parse PDB crystal structures into structured pocket features --
the "game board" representation. Hardcoded, rule-based, no ML.

Analogous to CoreKnowledge Perception in the ARC-AGI agent.
"""

from descartes_pharma_docking.perception.pocket_parser import (
    parse_pocket,
    PocketFeatures,
    ResidueFeature,
)

__all__ = ["parse_pocket", "PocketFeatures", "ResidueFeature"]
