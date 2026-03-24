"""
AlphaFold confidence-based zombie flagging.

Uses AlphaFold's own confidence metrics -- per-residue pLDDT and the
Predicted Aligned Error (PAE) matrix -- to flag binding-site residues
whose structural predictions are unreliable and therefore likely to
produce zombie features in downstream drug-discovery models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ConfidenceZombieFlags:
    """Per-residue zombie flags derived from AlphaFold confidence scores.

    Attributes
    ----------
    residue_flags : dict[int, str]
        Mapping from residue index to flag label.  Possible labels:
        ``"HIGH_CONFIDENCE"``, ``"LOW_PLDDT"``, ``"HIGH_PAE"``,
        ``"ZOMBIE_CANDIDATE"``.
    low_plddt_residues : list[int]
        Residue indices with pLDDT below threshold.
    high_pae_residues : list[int]
        Residue indices whose mean PAE to binding-site partners
        exceeds the PAE threshold.
    zombie_candidates : list[int]
        Residues flagged as both low-pLDDT *and* high-PAE within the
        binding site -- the strongest zombie candidates.
    summary : dict
        Aggregate statistics (counts, fractions, mean scores).
    """

    residue_flags: Dict[int, str]
    low_plddt_residues: List[int]
    high_pae_residues: List[int]
    zombie_candidates: List[int]
    summary: Dict[str, float] = field(default_factory=dict)


def alphafold_confidence_zombie_flags(
    plddt_scores: np.ndarray,
    pae_matrix: np.ndarray,
    binding_site_residues: Sequence[int],
    plddt_threshold: float = 70.0,
    pae_threshold: float = 10.0,
) -> ConfidenceZombieFlags:
    """Flag binding-site residues as zombie candidates using AF2 confidence.

    A residue is flagged as a **zombie candidate** when *both* of the
    following conditions hold:

    1. Its predicted Local Distance Difference Test (pLDDT) score is
       below ``plddt_threshold`` (default 70), indicating that
       AlphaFold is not confident in the local geometry.
    2. Its mean Predicted Aligned Error (PAE) to all other binding-site
       residues exceeds ``pae_threshold`` (default 10 Angstrom),
       indicating that the relative positioning with respect to the
       binding site is unreliable.

    Residues that fail only one criterion receive a partial flag
    (``LOW_PLDDT`` or ``HIGH_PAE``).

    Parameters
    ----------
    plddt_scores : np.ndarray, shape (N_residues,)
        Per-residue pLDDT confidence scores in [0, 100].
    pae_matrix : np.ndarray, shape (N_residues, N_residues)
        Predicted Aligned Error matrix in Angstrom.  Entry ``(i, j)``
        is the expected positional error of residue ``j`` when the
        prediction is aligned on residue ``i``.
    binding_site_residues : sequence of int
        Indices of residues that form the binding site of interest.
    plddt_threshold : float
        pLDDT scores below this value are considered low-confidence.
    pae_threshold : float
        Mean PAE above this value (in Angstrom) is considered high.

    Returns
    -------
    ConfidenceZombieFlags
        Dataclass containing per-residue flags, lists of flagged
        residues, and aggregate summary statistics.

    Raises
    ------
    ValueError
        If ``plddt_scores`` and ``pae_matrix`` have incompatible shapes,
        or if any binding-site residue index is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> plddt = np.array([90.0, 55.0, 80.0, 40.0, 95.0])
    >>> pae = np.random.rand(5, 5) * 20
    >>> flags = alphafold_confidence_zombie_flags(
    ...     plddt, pae, binding_site_residues=[1, 2, 3]
    ... )
    >>> flags.zombie_candidates  # residues that are both low-pLDDT and high-PAE
    [...]
    """
    plddt_scores = np.asarray(plddt_scores, dtype=np.float64)
    pae_matrix = np.asarray(pae_matrix, dtype=np.float64)
    n_res = plddt_scores.shape[0]

    # --- Validation ---------------------------------------------------
    if plddt_scores.ndim != 1:
        raise ValueError(
            f"plddt_scores must be 1-D, got shape {plddt_scores.shape}"
        )
    if pae_matrix.shape != (n_res, n_res):
        raise ValueError(
            f"pae_matrix shape {pae_matrix.shape} incompatible with "
            f"plddt_scores length {n_res}"
        )
    binding_site_set = set(binding_site_residues)
    for idx in binding_site_set:
        if idx < 0 or idx >= n_res:
            raise ValueError(
                f"Binding-site residue index {idx} out of range [0, {n_res})"
            )

    # --- Compute flags ------------------------------------------------
    binding_site_list = sorted(binding_site_set)

    low_plddt: List[int] = []
    high_pae: List[int] = []
    zombie_candidates: List[int] = []
    residue_flags: Dict[int, str] = {}

    for res_idx in binding_site_list:
        is_low_plddt = bool(plddt_scores[res_idx] < plddt_threshold)

        # Mean PAE to other binding-site residues
        other_bs = [r for r in binding_site_list if r != res_idx]
        if other_bs:
            mean_pae = float(np.mean(pae_matrix[res_idx, other_bs]))
        else:
            mean_pae = 0.0
        is_high_pae = mean_pae > pae_threshold

        if is_low_plddt:
            low_plddt.append(res_idx)
        if is_high_pae:
            high_pae.append(res_idx)

        if is_low_plddt and is_high_pae:
            residue_flags[res_idx] = "ZOMBIE_CANDIDATE"
            zombie_candidates.append(res_idx)
        elif is_low_plddt:
            residue_flags[res_idx] = "LOW_PLDDT"
        elif is_high_pae:
            residue_flags[res_idx] = "HIGH_PAE"
        else:
            residue_flags[res_idx] = "HIGH_CONFIDENCE"

    # --- Summary statistics -------------------------------------------
    bs_plddt = plddt_scores[binding_site_list]
    summary: Dict[str, float] = {
        "n_binding_site_residues": float(len(binding_site_list)),
        "n_low_plddt": float(len(low_plddt)),
        "n_high_pae": float(len(high_pae)),
        "n_zombie_candidates": float(len(zombie_candidates)),
        "frac_zombie": (
            len(zombie_candidates) / max(len(binding_site_list), 1)
        ),
        "mean_binding_site_plddt": float(np.mean(bs_plddt)) if len(bs_plddt) > 0 else 0.0,
        "plddt_threshold": plddt_threshold,
        "pae_threshold": pae_threshold,
    }

    return ConfidenceZombieFlags(
        residue_flags=residue_flags,
        low_plddt_residues=low_plddt,
        high_pae_residues=high_pae,
        zombie_candidates=zombie_candidates,
        summary=summary,
    )
