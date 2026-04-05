"""
Scaffold-Stratified Permutation Probe (Hardened).

Standard permutation tests shuffle all target values uniformly.
This breaks correlations between scaffolds and activity, which can
inflate R2 (the "scaffold confound").

The hardened version permutes WITHIN scaffold groups only:
molecules with the same Murcko scaffold swap targets among themselves,
but never across scaffolds. This preserves scaffold-level correlations
in the null distribution, giving a more conservative test.

Protocol:
  1. Assign each sample to a scaffold group
  2. Compute real delta-R2 (trained - untrained embeddings)
  3. For each permutation:
     a. Shuffle target values WITHIN each scaffold group
     b. Compute null delta-R2
  4. p-value = fraction of null delta-R2 >= real delta-R2
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple

from descartes_pharma_docking.probing.probe_runner import cv_ridge_r2


def hardened_probe(
    trained_emb: np.ndarray,
    untrained_emb: np.ndarray,
    target: np.ndarray,
    scaffolds: np.ndarray,
    n_perms: int = 200,
    alpha: float = 1.0,
    random_seed: int = 42,
) -> Tuple[float, float, bool]:
    """
    Scaffold-stratified permutation probe.

    Args:
        trained_emb: (N, hidden_dim) hidden states from trained network.
        untrained_emb: (N, hidden_dim) hidden states from untrained network.
        target: (N,) ground truth target values.
        scaffolds: (N,) scaffold group labels (int or string).
            Samples with the same scaffold label are permuted together.
        n_perms: Number of permutations for the null distribution.
        alpha: Ridge regression regularization strength.
        random_seed: Random seed for reproducibility.

    Returns:
        delta_r2: Real delta-R2 (trained - untrained).
        p_value: Scaffold-stratified permutation p-value.
        is_encoded: True if delta_r2 > 0.05 and p_value < 0.05.
    """
    n = len(trained_emb)
    assert len(untrained_emb) == n
    assert len(target) == n
    assert len(scaffolds) == n

    # Real delta-R2
    r2_trained = cv_ridge_r2(trained_emb, target, alpha=alpha)
    r2_untrained = cv_ridge_r2(untrained_emb, target, alpha=alpha)
    delta_r2 = r2_trained - r2_untrained

    # Build scaffold group indices
    unique_scaffolds = np.unique(scaffolds)
    scaffold_indices = {}
    for s in unique_scaffolds:
        scaffold_indices[s] = np.where(scaffolds == s)[0]

    # Scaffold-stratified permutation null
    rng = np.random.default_rng(random_seed)
    null_delta_r2s = []

    for _ in range(n_perms):
        # Permute within each scaffold group
        target_perm = target.copy()
        for s, indices in scaffold_indices.items():
            if len(indices) > 1:
                perm_order = rng.permutation(len(indices))
                target_perm[indices] = target[indices[perm_order]]

        # Compute null delta-R2
        r2_perm_trained = cv_ridge_r2(trained_emb, target_perm, alpha=alpha)
        r2_perm_untrained = cv_ridge_r2(untrained_emb, target_perm, alpha=alpha)
        null_delta = r2_perm_trained - r2_perm_untrained
        null_delta_r2s.append(null_delta)

    null_delta_r2s = np.array(null_delta_r2s)
    p_value = float(np.mean(null_delta_r2s >= delta_r2))

    is_encoded = (delta_r2 > 0.05) and (p_value < 0.05)

    return delta_r2, p_value, is_encoded


def assign_scaffold_groups(
    smiles_list: List[str],
    fallback_to_hash: bool = True,
) -> np.ndarray:
    """
    Assign scaffold group labels based on Murcko scaffold decomposition.

    Args:
        smiles_list: List of SMILES strings for each sample.
        fallback_to_hash: If RDKit is unavailable, use string hashing
            as a rough scaffold proxy.

    Returns:
        (N,) array of integer scaffold group labels.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import (
            GetScaffoldForMol,
            MakeScaffoldGeneric,
        )

        scaffold_map = {}
        labels = []
        next_label = 0

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffold_smi = "INVALID"
            else:
                try:
                    core = MakeScaffoldGeneric(GetScaffoldForMol(mol))
                    scaffold_smi = Chem.MolToSmiles(core)
                except Exception:
                    scaffold_smi = smi[:10]  # Rough fallback

            if scaffold_smi not in scaffold_map:
                scaffold_map[scaffold_smi] = next_label
                next_label += 1
            labels.append(scaffold_map[scaffold_smi])

        return np.array(labels, dtype=np.int64)

    except ImportError:
        if fallback_to_hash:
            # Hash-based fallback when RDKit is not available
            scaffold_map = {}
            labels = []
            next_label = 0

            for smi in smiles_list:
                # Use first 8 characters as a rough scaffold proxy
                key = smi[:8] if smi else "EMPTY"
                if key not in scaffold_map:
                    scaffold_map[key] = next_label
                    next_label += 1
                labels.append(scaffold_map[key])

            return np.array(labels, dtype=np.int64)
        else:
            raise


def run_hardened_probe_suite(
    trained_emb: np.ndarray,
    untrained_emb: np.ndarray,
    targets: Dict[str, np.ndarray],
    scaffolds: np.ndarray,
    n_perms: int = 200,
) -> Dict[str, Dict]:
    """
    Run scaffold-stratified permutation probes for all targets.

    Args:
        trained_emb: (N, hidden_dim) hidden states from trained network.
        untrained_emb: (N, hidden_dim) from untrained network.
        targets: Dict of target name -> (N,) arrays.
        scaffolds: (N,) scaffold group labels.
        n_perms: Permutations per target.

    Returns:
        Dict mapping target name -> {delta_r2, p_value, is_encoded}.
    """
    results = {}
    for name, target in targets.items():
        n = min(len(trained_emb), len(target))
        delta_r2, p_value, is_encoded = hardened_probe(
            trained_emb[:n],
            untrained_emb[:n],
            target[:n],
            scaffolds[:n],
            n_perms=n_perms,
        )
        results[name] = {
            "delta_r2": delta_r2,
            "p_value": p_value,
            "is_encoded": is_encoded,
        }
        status = "ENCODED" if is_encoded else "not encoded"
        print(
            f"  {name:30s} dR2={delta_r2:.4f} "
            f"p={p_value:.4f} [{status}]"
        )

    return results
