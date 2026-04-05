"""
Interaction Features
=====================

Compute pairwise pocket x ligand interaction features:
distances to catalytic residues, hydrogen bonds, hydrophobic contacts,
steric clashes, and pocket occupancy.
"""

from descartes_pharma_docking.interaction.interaction_features import (
    compute_interaction_features,
)

__all__ = ["compute_interaction_features"]
