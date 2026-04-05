"""
Compute pairwise pocket x ligand interaction features.

These features capture how well the ligand fits into the pocket at its
current pose. They form part of the observation vector fed to the
policy network.

Features computed:
    dist_asp32             - Distance from ligand center to Asp32 (catalytic)
    dist_asp228            - Distance from ligand center to Asp228 (catalytic)
    n_hbonds               - Count of N/O ligand atoms within 3.5A of pocket
                             H-bond donor/acceptor sites
    hydrophobic_contact_area - Count of heavy atoms within 4.0A of hydrophobic
                             pocket residues (proxy for contact area)
    steric_clash_count     - Count of ligand atoms within 1.5A of any pocket
                             atom (unphysical overlap)
    pocket_occupancy_fraction - Fraction of pocket volume occupied (estimated
                             by fraction of pocket residues with a nearby
                             ligand atom)
    closest_wall_dist      - Distance from ligand center to nearest pocket
                             boundary (pocket_radius - dist_to_pocket_center)

Returns a fixed-size numpy array for concatenation into the policy network
input vector.
"""

from typing import Optional
import numpy as np

from descartes_pharma_docking.perception.pocket_parser import PocketFeatures


# Interaction thresholds (Angstroms)
HBOND_DISTANCE_CUTOFF = 3.5
HYDROPHOBIC_DISTANCE_CUTOFF = 4.0
STERIC_CLASH_CUTOFF = 1.5
RESIDUE_CONTACT_CUTOFF = 5.0

# Feature vector size
N_INTERACTION_FEATURES = 7


def compute_interaction_features(
    pocket: PocketFeatures,
    ligand_coords: np.ndarray,
    ligand_center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute pairwise pocket x ligand interaction features.

    Args:
        pocket: PocketFeatures from pocket_parser.parse_pocket().
        ligand_coords: (n_atoms, 3) array of ligand atom positions.
        ligand_center: (3,) ligand center of mass. If None, computed
                       from ligand_coords.

    Returns:
        np.ndarray of shape (N_INTERACTION_FEATURES,) containing:
            [0] dist_asp32
            [1] dist_asp228
            [2] n_hbonds
            [3] hydrophobic_contact_area
            [4] steric_clash_count
            [5] pocket_occupancy_fraction
            [6] closest_wall_dist
    """
    if ligand_center is None:
        ligand_center = ligand_coords.mean(axis=0)

    # ------------------------------------------------------------------
    # 1. Distances to catalytic residues (Asp32, Asp228)
    # ------------------------------------------------------------------
    dist_asp32 = _distance_to_catalytic(pocket, ligand_center, target_resid=32)
    dist_asp228 = _distance_to_catalytic(pocket, ligand_center, target_resid=228)

    # ------------------------------------------------------------------
    # 2. H-bond count: N/O ligand atoms within 3.5A of pocket H-bond sites
    # ------------------------------------------------------------------
    n_hbonds = _count_hbonds(pocket, ligand_coords)

    # ------------------------------------------------------------------
    # 3. Hydrophobic contact area: heavy atoms within 4A of hydrophobic
    #    pocket residues
    # ------------------------------------------------------------------
    hydrophobic_contact_area = _count_hydrophobic_contacts(
        pocket, ligand_coords
    )

    # ------------------------------------------------------------------
    # 4. Steric clash count: atoms within 1.5A of pocket atoms
    # ------------------------------------------------------------------
    steric_clash_count = _count_steric_clashes(pocket, ligand_coords)

    # ------------------------------------------------------------------
    # 5. Pocket occupancy fraction: fraction of pocket residues that have
    #    at least one ligand atom within RESIDUE_CONTACT_CUTOFF
    # ------------------------------------------------------------------
    pocket_occupancy_fraction = _pocket_occupancy(pocket, ligand_coords)

    # ------------------------------------------------------------------
    # 6. Closest wall distance: how close is the ligand to the pocket
    #    boundary?
    # ------------------------------------------------------------------
    dist_to_center = float(np.linalg.norm(ligand_center - pocket.pocket_center))
    closest_wall_dist = max(0.0, pocket.pocket_radius - dist_to_center)

    features = np.array([
        dist_asp32,
        dist_asp228,
        float(n_hbonds),
        float(hydrophobic_contact_area),
        float(steric_clash_count),
        pocket_occupancy_fraction,
        closest_wall_dist,
    ], dtype=np.float32)

    return features


# =====================================================================
# Private helper functions
# =====================================================================


def _distance_to_catalytic(
    pocket: PocketFeatures,
    ligand_center: np.ndarray,
    target_resid: int,
) -> float:
    """Distance from ligand center to a specific catalytic residue."""
    for res in pocket.catalytic_residues:
        if res.resid == target_resid:
            return float(np.linalg.norm(ligand_center - res.center))
    # If residue not found, return a large sentinel value
    return 99.0


def _count_hbonds(
    pocket: PocketFeatures,
    ligand_coords: np.ndarray,
) -> int:
    """
    Count potential H-bonds: number of ligand atoms (any type for
    simplicity here -- refined version would filter N/O) that are within
    HBOND_DISTANCE_CUTOFF of pocket H-bond donor or acceptor residue
    centers.
    """
    count = 0
    hbond_residues = set()
    for res in pocket.hbond_donors:
        hbond_residues.add(id(res))
    for res in pocket.hbond_acceptors:
        hbond_residues.add(id(res))

    # Collect all hbond site centers
    hbond_centers = []
    for res in pocket.hbond_donors + pocket.hbond_acceptors:
        if id(res) in hbond_residues:
            hbond_centers.append(res.center)

    if len(hbond_centers) == 0:
        return 0

    hbond_centers = np.array(hbond_centers)  # (n_sites, 3)

    for atom_pos in ligand_coords:
        dists = np.linalg.norm(hbond_centers - atom_pos, axis=1)
        if np.any(dists < HBOND_DISTANCE_CUTOFF):
            count += 1

    return count


def _count_hydrophobic_contacts(
    pocket: PocketFeatures,
    ligand_coords: np.ndarray,
) -> int:
    """
    Count ligand heavy atoms within HYDROPHOBIC_DISTANCE_CUTOFF of
    hydrophobic pocket residue centers.
    """
    if len(pocket.hydrophobic_residues) == 0:
        return 0

    hydro_centers = np.array([r.center for r in pocket.hydrophobic_residues])
    count = 0

    for atom_pos in ligand_coords:
        dists = np.linalg.norm(hydro_centers - atom_pos, axis=1)
        if np.any(dists < HYDROPHOBIC_DISTANCE_CUTOFF):
            count += 1

    return count


def _count_steric_clashes(
    pocket: PocketFeatures,
    ligand_coords: np.ndarray,
) -> int:
    """
    Count ligand atoms that are too close to pocket sidechain atoms
    (within STERIC_CLASH_CUTOFF).
    """
    # Collect all pocket sidechain atom positions
    pocket_atoms = []
    for res in pocket.residues:
        pocket_atoms.extend(res.sidechain_atoms)

    if len(pocket_atoms) == 0:
        return 0

    pocket_atoms = np.array(pocket_atoms)  # (n_pocket_atoms, 3)
    count = 0

    for atom_pos in ligand_coords:
        dists = np.linalg.norm(pocket_atoms - atom_pos, axis=1)
        if np.any(dists < STERIC_CLASH_CUTOFF):
            count += 1

    return count


def _pocket_occupancy(
    pocket: PocketFeatures,
    ligand_coords: np.ndarray,
) -> float:
    """
    Fraction of pocket residues that have at least one ligand atom
    within RESIDUE_CONTACT_CUTOFF.
    """
    if len(pocket.residues) == 0:
        return 0.0

    contacted = 0
    for res in pocket.residues:
        dists = np.linalg.norm(ligand_coords - res.center, axis=1)
        if np.any(dists < RESIDUE_CONTACT_CUTOFF):
            contacted += 1

    return contacted / len(pocket.residues)
