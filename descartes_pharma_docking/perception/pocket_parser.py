"""
Parse PDB structure into structured pocket features.

Uses BioPython for structure parsing.
No ML -- pure rule-based feature extraction.

Analogous to CoreKnowledge perception in ARC-AGI agent:
PDB file is the "raw grid", PocketFeatures is the "StructuredPercept".
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    raise ImportError("pip install biopython --break-system-packages")


@dataclass
class ResidueFeature:
    """One amino acid residue in the binding pocket."""
    name: str               # e.g., "ASP32", "TYR71"
    resname: str            # 3-letter code: ASP, TYR, PHE, etc.
    resid: int              # Residue number
    chain: str              # Chain ID
    center: np.ndarray      # Center of mass (x, y, z) in Angstroms

    # Pharmacological properties (hardcoded from amino acid chemistry)
    is_hbond_donor: bool
    is_hbond_acceptor: bool
    is_hydrophobic: bool
    is_charged: bool
    charge_sign: int         # +1, -1, or 0
    is_aromatic: bool
    is_catalytic: bool       # True for Asp32, Asp228 in BACE1

    sidechain_atoms: List[np.ndarray] = field(default_factory=list)


@dataclass
class PocketFeatures:
    """
    Complete structured representation of a binding pocket.

    This is the "StructuredPercept" for the protein pocket -- the game board.
    All features are hardcoded from known biochemistry, no learning needed.
    """

    # Identity
    pdb_id: str
    pocket_center: np.ndarray        # Geometric center of pocket
    pocket_radius: float             # Radius encompassing all pocket residues

    # Residues (the "objects" on the game board)
    residues: List[ResidueFeature] = field(default_factory=list)

    # Catalytic residues (the "goal positions")
    catalytic_residues: List[ResidueFeature] = field(default_factory=list)

    # H-bond network (potential interaction sites)
    hbond_donors: List[ResidueFeature] = field(default_factory=list)
    hbond_acceptors: List[ResidueFeature] = field(default_factory=list)

    # Hydrophobic regions
    hydrophobic_residues: List[ResidueFeature] = field(default_factory=list)

    # Charged residues
    positive_residues: List[ResidueFeature] = field(default_factory=list)
    negative_residues: List[ResidueFeature] = field(default_factory=list)

    # Sub-pockets (the "rooms" in the game board)
    sub_pockets: Dict[str, Dict] = field(default_factory=dict)
    # e.g., {"S1": {"center": [x,y,z], "residues": [...], "volume": 150.0}}

    # Water positions (potential "obstacles" or displaced targets)
    water_positions: List[np.ndarray] = field(default_factory=list)

    def to_feature_vector(self) -> np.ndarray:
        """
        Flatten pocket into a fixed-size feature vector for the policy network.

        Returns: np.ndarray of shape (n_pocket_features,)
        """
        features = []

        # Pocket-level features
        features.extend(self.pocket_center.tolist())    # 3
        features.append(self.pocket_radius)              # 1
        features.append(len(self.residues))              # 1
        features.append(len(self.hbond_donors))          # 1
        features.append(len(self.hbond_acceptors))       # 1
        features.append(len(self.hydrophobic_residues))  # 1
        features.append(len(self.positive_residues))     # 1
        features.append(len(self.negative_residues))     # 1
        features.append(len(self.water_positions))       # 1
        features.append(len(self.catalytic_residues))    # 1

        # Catalytic residue positions (padded to max 4 catalytic residues)
        for i in range(4):
            if i < len(self.catalytic_residues):
                features.extend(self.catalytic_residues[i].center.tolist())  # 3
            else:
                features.extend([0.0, 0.0, 0.0])

        # Sub-pocket centers (padded to max 6 sub-pockets)
        for key in ["S1", "S2", "S3", "S1_prime", "S2_prime", "flap"]:
            if key in self.sub_pockets:
                features.extend(self.sub_pockets[key]["center"])  # 3
                features.append(self.sub_pockets[key].get("volume", 0.0))  # 1
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


# === AMINO ACID PROPERTY LOOKUP (hardcoded biochemistry) ===

HBOND_DONORS = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "TRP", "ARG", "LYS", "CYS"}
HBOND_ACCEPTORS = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "ASP", "GLU", "MET", "CYS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"}
POSITIVE = {"ARG", "LYS", "HIS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}

# BACE1-specific catalytic residues
BACE1_CATALYTIC = {32, 228}  # Asp32 and Asp228


def parse_pocket(pdb_path: str,
                 pocket_center: Tuple[float, float, float],
                 pocket_radius: float = 12.0,
                 catalytic_residue_ids: set = None,
                 pdb_id: str = "unknown") -> PocketFeatures:
    """
    Parse a PDB file and extract pocket features within radius of center.

    Args:
        pdb_path: Path to PDB file
        pocket_center: (x, y, z) center of binding pocket
        pocket_radius: Radius in Angstroms to include residues
        catalytic_residue_ids: Set of residue IDs that are catalytically active
        pdb_id: PDB identifier string

    Returns:
        PocketFeatures: Structured pocket representation
    """
    if catalytic_residue_ids is None:
        catalytic_residue_ids = BACE1_CATALYTIC

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]  # First model

    center = np.array(pocket_center)
    pocket = PocketFeatures(pdb_id=pdb_id, pocket_center=center,
                            pocket_radius=pocket_radius)

    # Extract all atoms for neighbor search
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)

    # Find residues within pocket radius
    nearby_atoms = ns.search(center, pocket_radius, 'R')  # 'R' = residue level
    seen_residues = set()

    for residue in nearby_atoms:
        if not is_aa(residue):
            # Check if it's a water molecule
            if residue.get_resname() == "HOH":
                water_center = np.mean([a.get_vector().get_array()
                                        for a in residue.get_atoms()], axis=0)
                pocket.water_positions.append(water_center)
            continue

        resid = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        key = (chain, resid)
        if key in seen_residues:
            continue
        seen_residues.add(key)

        resname = residue.get_resname()

        # Compute center of mass
        atoms = list(residue.get_atoms())
        coords = np.array([a.get_vector().get_array() for a in atoms])
        res_center = coords.mean(axis=0)

        # Sidechain atoms (exclude backbone N, CA, C, O)
        backbone_names = {"N", "CA", "C", "O"}
        sidechain = [a.get_vector().get_array() for a in atoms
                     if a.get_name() not in backbone_names]

        feat = ResidueFeature(
            name=f"{resname}{resid}",
            resname=resname,
            resid=resid,
            chain=chain,
            center=res_center,
            is_hbond_donor=resname in HBOND_DONORS,
            is_hbond_acceptor=resname in HBOND_ACCEPTORS,
            is_hydrophobic=resname in HYDROPHOBIC,
            is_charged=resname in POSITIVE or resname in NEGATIVE,
            charge_sign=1 if resname in POSITIVE else (-1 if resname in NEGATIVE else 0),
            is_aromatic=resname in AROMATIC,
            is_catalytic=resid in catalytic_residue_ids,
            sidechain_atoms=[np.array(s) for s in sidechain],
        )

        pocket.residues.append(feat)

        if feat.is_catalytic:
            pocket.catalytic_residues.append(feat)
        if feat.is_hbond_donor:
            pocket.hbond_donors.append(feat)
        if feat.is_hbond_acceptor:
            pocket.hbond_acceptors.append(feat)
        if feat.is_hydrophobic:
            pocket.hydrophobic_residues.append(feat)
        if feat.charge_sign > 0:
            pocket.positive_residues.append(feat)
        elif feat.charge_sign < 0:
            pocket.negative_residues.append(feat)

    return pocket
