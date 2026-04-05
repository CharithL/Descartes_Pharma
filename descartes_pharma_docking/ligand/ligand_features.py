"""
Represent a drug molecule as structured features for the policy network.

Uses RDKit for 3D conformer generation and property computation.
No ML -- pure cheminformatics.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdMolTransforms
except ImportError:
    raise ImportError("pip install rdkit --break-system-packages")


@dataclass
class LigandFeatures:
    """
    Structured representation of a drug molecule -- the "player piece."
    """

    # Identity
    smiles: str
    mol: object  # RDKit mol object

    # Current 3D state
    conformer_coords: np.ndarray     # (n_atoms, 3) current atom positions
    center_of_mass: np.ndarray       # (3,) current center

    # Pharmacophore features (what the molecule CAN do)
    n_hbond_donors: int
    n_hbond_acceptors: int
    n_rotatable_bonds: int
    n_aromatic_rings: int
    logp: float
    molecular_weight: float
    tpsa: float  # Topological polar surface area

    # Pharmacophore point positions (where interactions CAN happen)
    hbond_donor_positions: List[np.ndarray] = field(default_factory=list)
    hbond_acceptor_positions: List[np.ndarray] = field(default_factory=list)
    hydrophobic_center_positions: List[np.ndarray] = field(default_factory=list)
    aromatic_ring_centers: List[np.ndarray] = field(default_factory=list)
    charged_group_positions: List[np.ndarray] = field(default_factory=list)

    # Rotatable bond indices (for torsion actions)
    rotatable_bond_indices: List[Tuple[int, int]] = field(default_factory=list)

    # Current pose (6 degrees of freedom)
    pose_translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # (x, y, z) offset
    pose_rotation: np.ndarray = field(
        default_factory=lambda: np.zeros(3))   # (rx, ry, rz) Euler angles

    # Known experimental affinity (if available from BindingDB)
    known_ic50_nm: float = -1.0  # -1 means unknown
    known_ki_nm: float = -1.0

    def to_feature_vector(self) -> np.ndarray:
        """Flatten ligand into fixed-size feature vector."""
        features = []

        # Global properties
        features.extend(self.center_of_mass.tolist())    # 3
        features.append(self.n_hbond_donors)              # 1
        features.append(self.n_hbond_acceptors)           # 1
        features.append(self.n_rotatable_bonds)           # 1
        features.append(self.n_aromatic_rings)            # 1
        features.append(self.logp)                        # 1
        features.append(self.molecular_weight / 500.0)    # 1 (normalized)
        features.append(self.tpsa / 200.0)                # 1 (normalized)

        # Current pose
        features.extend(self.pose_translation.tolist())   # 3
        features.extend(self.pose_rotation.tolist())      # 3

        return np.array(features, dtype=np.float32)


def create_ligand(smiles: str, ic50_nm: float = -1.0,
                  ki_nm: float = -1.0) -> LigandFeatures:
    """
    Create a LigandFeatures from SMILES string.

    Generates a 3D conformer and extracts all pharmacophore features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result == -1:
        # Fallback to random coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    center = coords.mean(axis=0)

    # Pharmacophore properties
    mol_noH = Chem.RemoveHs(mol)

    # Find H-bond donor/acceptor positions
    donor_positions = []
    acceptor_positions = []
    for atom in mol.GetAtoms():
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        # Donors: N-H, O-H
        if atom.GetSymbol() in ("N", "O") and atom.GetTotalNumHs() > 0:
            donor_positions.append(pos)
        # Acceptors: N, O with lone pairs
        if atom.GetSymbol() in ("N", "O"):
            acceptor_positions.append(pos)

    # Hydrophobic centers (C atoms not bonded to heteroatoms)
    hydrophobic_positions = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and not any(
            n.GetSymbol() in ("N", "O", "S") for n in atom.GetNeighbors()):
            hydrophobic_positions.append(
                np.array(conf.GetAtomPosition(atom.GetIdx())))

    # Rotatable bonds
    rot_bonds = mol_noH.GetSubstructMatches(
        Chem.MolFromSmarts("[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]"))

    return LigandFeatures(
        smiles=smiles,
        mol=mol,
        conformer_coords=coords,
        center_of_mass=center,
        n_hbond_donors=rdMolDescriptors.CalcNumHBD(mol_noH),
        n_hbond_acceptors=rdMolDescriptors.CalcNumHBA(mol_noH),
        n_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol_noH),
        n_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol_noH),
        logp=Descriptors.MolLogP(mol_noH),
        molecular_weight=Descriptors.MolWt(mol_noH),
        tpsa=Descriptors.TPSA(mol_noH),
        hbond_donor_positions=donor_positions,
        hbond_acceptor_positions=acceptor_positions,
        hydrophobic_center_positions=hydrophobic_positions,
        rotatable_bond_indices=list(rot_bonds),
        known_ic50_nm=ic50_nm,
        known_ki_nm=ki_nm,
    )
