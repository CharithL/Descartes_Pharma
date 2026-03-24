"""
DESCARTES-PHARMA: Compute known mechanistic features from SMILES.
These serve as "biological ground truth" for probing.
"""

import numpy as np


def compute_mechanistic_features(smiles_list):
    """
    Compute known mechanistic features from SMILES.

    Returns:
        features: np.ndarray of shape (n_compounds, 12)
        feature_names: list of feature names
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([np.nan] * 12)
            continue

        features.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
            rdMolDescriptors.CalcNumHeavyAtoms(mol),
            Descriptors.NumAliphaticRings(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            Descriptors.PEOE_VSA1(mol),
        ])

    feature_names = [
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds',
        'AromaticRings', 'FractionCSP3', 'HeavyAtoms',
        'AliphaticRings', 'AmideBonds', 'PEOE_VSA1'
    ]

    return np.array(features), feature_names


def compute_scaffold(smiles):
    """Compute Murcko scaffold for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'UNKNOWN'
    scaffold = MurckoScaffold.MakeScaffoldGeneric(
        MurckoScaffold.GetScaffoldForMol(mol))
    return Chem.MolToSmiles(scaffold)


def get_scaffold_groups(smiles_list):
    """Return scaffold assignments for a list of SMILES."""
    scaffolds = [compute_scaffold(smi) for smi in smiles_list]
    return np.array(scaffolds)
