"""
Additional pocket feature extraction with BACE1-specific sub-pocket definitions.

Defines the known sub-pocket geometry for BACE1 (PDB 4IVT) and provides
functions to assign residues to sub-pockets based on distance to known
sub-pocket centers from the crystallographic literature.

Sub-pockets defined:
    S1   - Primary substrate-binding pocket
    S2   - Secondary pocket, accommodates larger substituents
    S3   - Solvent-exposed region near flap
    flap - Flexible flap region (opens/closes over active site)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from descartes_pharma_docking.perception.pocket_parser import (
    PocketFeatures,
    ResidueFeature,
)


# ============================================================
# BACE1-specific sub-pocket definitions (from PDB 4IVT)
# ============================================================
# Approximate centers derived from the crystal structure of
# BACE1 in complex with an inhibitor (PDB: 4IVT).
# Key residue assignments come from the BACE1 literature.

BACE1_SUB_POCKETS: Dict[str, Dict] = {
    "S1": {
        "description": "Primary substrate-binding pocket",
        "center": [26.5, 14.0, 23.0],
        "radius": 6.0,
        "key_residues": {71, 73, 108, 110},  # Tyr71, Thr73, Leu108, Trp110
    },
    "S2": {
        "description": "Secondary pocket, accommodates larger substituents",
        "center": [30.0, 17.0, 20.0],
        "radius": 5.5,
        "key_residues": {198, 200, 226, 229},
    },
    "S3": {
        "description": "Solvent-exposed region near flap",
        "center": [25.0, 18.0, 25.0],
        "radius": 5.0,
        "key_residues": {68, 69, 70},
    },
    "flap": {
        "description": "Flexible flap region (opens/closes)",
        "center": [27.0, 20.0, 24.0],
        "radius": 7.0,
        "key_residues": {67, 68, 69, 70, 71, 72, 73, 74, 75},
    },
}

# Default pocket center: midpoint of Asp32-OD1 (11.9, 14.9, 20.4) and
# Asp228-OD1 (20.0, 26.8, 6.7) from PDB 4IVT crystal structure
BACE1_POCKET_CENTER = (16.0, 20.9, 13.5)
BACE1_POCKET_RADIUS = 15.0  # 15A ensures both catalytic ASPs are captured


def assign_residues_to_sub_pockets(
    pocket: PocketFeatures,
    sub_pocket_defs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Assign pocket residues to sub-pockets based on distance to known
    sub-pocket centers.

    Each residue is assigned to the closest sub-pocket whose center it
    falls within (by that sub-pocket's radius). A residue may appear in
    multiple sub-pockets if the regions overlap.

    Args:
        pocket: Parsed PocketFeatures from pocket_parser.parse_pocket.
        sub_pocket_defs: Definitions dict; defaults to BACE1_SUB_POCKETS.

    Returns:
        Dict mapping sub-pocket name to a dict with keys:
            "center"   -- np.ndarray (3,)
            "radius"   -- float
            "residues" -- List[ResidueFeature]
            "volume"   -- estimated spherical volume in cubic Angstroms
            "description" -- human-readable description
    """
    if sub_pocket_defs is None:
        sub_pocket_defs = BACE1_SUB_POCKETS

    results: Dict[str, Dict] = {}

    for sp_name, sp_def in sub_pocket_defs.items():
        sp_center = np.array(sp_def["center"], dtype=np.float64)
        sp_radius = sp_def["radius"]

        assigned: List[ResidueFeature] = []
        for res in pocket.residues:
            dist = np.linalg.norm(res.center - sp_center)
            if dist <= sp_radius:
                assigned.append(res)

            # Also include residues explicitly listed as key residues
            elif res.resid in sp_def.get("key_residues", set()):
                assigned.append(res)

        # Approximate volume as a sphere
        volume = (4.0 / 3.0) * np.pi * (sp_radius ** 3)

        results[sp_name] = {
            "center": sp_center.tolist(),
            "radius": sp_radius,
            "residues": assigned,
            "volume": round(volume, 1),
            "description": sp_def.get("description", ""),
        }

    return results


def enrich_pocket_with_sub_pockets(
    pocket: PocketFeatures,
    sub_pocket_defs: Optional[Dict[str, Dict]] = None,
) -> PocketFeatures:
    """
    Enrich a PocketFeatures object with sub-pocket assignments.

    Modifies pocket.sub_pockets in place and returns the same object
    for convenience.

    Args:
        pocket: Parsed PocketFeatures.
        sub_pocket_defs: Optional custom sub-pocket definitions.

    Returns:
        The same PocketFeatures object with sub_pockets populated.
    """
    sp_data = assign_residues_to_sub_pockets(pocket, sub_pocket_defs)

    # Store in pocket.sub_pockets in the format expected by
    # PocketFeatures.to_feature_vector()
    for sp_name, sp_info in sp_data.items():
        pocket.sub_pockets[sp_name] = {
            "center": sp_info["center"],
            "volume": sp_info["volume"],
            "radius": sp_info["radius"],
            "residue_names": [r.name for r in sp_info["residues"]],
            "n_residues": len(sp_info["residues"]),
            "description": sp_info["description"],
        }

    return pocket


def compute_sub_pocket_distances(
    ligand_center: np.ndarray,
    sub_pocket_defs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """
    Compute distance from a ligand center to each sub-pocket center.

    Useful for reward shaping: are we getting closer to the right
    sub-pocket?

    Args:
        ligand_center: (3,) array of ligand center of mass.
        sub_pocket_defs: Optional custom sub-pocket definitions.

    Returns:
        Dict mapping sub-pocket name to Euclidean distance in Angstroms.
    """
    if sub_pocket_defs is None:
        sub_pocket_defs = BACE1_SUB_POCKETS

    distances: Dict[str, float] = {}
    for sp_name, sp_def in sub_pocket_defs.items():
        sp_center = np.array(sp_def["center"], dtype=np.float64)
        distances[sp_name] = float(np.linalg.norm(ligand_center - sp_center))

    return distances


def get_bace1_config() -> Dict:
    """
    Return the full BACE1 4IVT pocket configuration.

    Equivalent to the bace1_4ivt.yaml config from the guide,
    expressed as a Python dict for programmatic access.
    """
    return {
        "pdb_id": "4IVT",
        "pdb_path": "data/structures/4IVT.pdb",
        "center": list(BACE1_POCKET_CENTER),
        "radius": BACE1_POCKET_RADIUS,
        "catalytic_residues": [32, 228],
        "sub_pockets": {
            name: {
                "description": sp["description"],
                "key_residues": sorted(sp["key_residues"]),
                "center": sp["center"],
                "radius": sp["radius"],
            }
            for name, sp in BACE1_SUB_POCKETS.items()
        },
    }
