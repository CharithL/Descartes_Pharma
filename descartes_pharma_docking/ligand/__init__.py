"""
Module 2: Ligand Representation & Action Space
================================================

Represent a drug molecule as a manipulable 3D object with pharmacophore
features, and define the action space for pose adjustments.
This is the "player piece" with its available "moves."
"""

from descartes_pharma_docking.ligand.ligand_features import (
    create_ligand,
    LigandFeatures,
)

__all__ = ["create_ligand", "LigandFeatures"]
