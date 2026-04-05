"""
Feature Encoder -- compose input features for the Search Policy Network.

Takes pocket features, ligand features, interaction features, and score
history and concatenates them into the single input vector the GRU expects.

The GRU input layout:
  [pocket_features (40) | ligand_features (16) | interaction_features (20)
   | score_history (10) | current_vina_score (1)]
  Total: 87 dimensions by default.
"""

import numpy as np
import torch
from typing import Optional


def encode_state(
    pocket_features: np.ndarray,
    ligand_features: np.ndarray,
    interaction_features: np.ndarray,
    score_history: np.ndarray,
    current_score: float,
    pocket_dim: int = 40,
    ligand_dim: int = 16,
    interaction_dim: int = 20,
    score_history_len: int = 10,
    as_tensor: bool = False,
    device: str = "cpu",
) -> "np.ndarray | torch.Tensor":
    """
    Compose input features from pocket + ligand + interaction + score history
    into the single input vector the GRU expects.

    Args:
        pocket_features: Pocket feature vector, shape (pocket_dim,) or smaller
            (will be zero-padded to pocket_dim).
        ligand_features: Ligand feature vector, shape (ligand_dim,) or smaller
            (will be zero-padded to ligand_dim).
        interaction_features: Pairwise interaction features, shape
            (interaction_dim,) or smaller (will be zero-padded).
        score_history: Array of recent Vina scores. Will be zero-padded
            on the left to score_history_len.
        current_score: Current Vina binding energy (kcal/mol).
        pocket_dim: Expected pocket feature dimensionality.
        ligand_dim: Expected ligand feature dimensionality.
        interaction_dim: Expected interaction feature dimensionality.
        score_history_len: Fixed length for score history window.
        as_tensor: If True, return a torch.Tensor instead of np.ndarray.
        device: Torch device (only used if as_tensor=True).

    Returns:
        Concatenated feature vector of shape
        (pocket_dim + ligand_dim + interaction_dim + score_history_len + 1,).
    """
    # Pad or truncate pocket features
    pocket_vec = _pad_or_truncate(pocket_features, pocket_dim)

    # Pad or truncate ligand features
    ligand_vec = _pad_or_truncate(ligand_features, ligand_dim)

    # Pad or truncate interaction features
    interaction_vec = _pad_or_truncate(interaction_features, interaction_dim)

    # Score history: right-align (most recent scores at the end),
    # zero-pad on the left
    history_vec = np.zeros(score_history_len, dtype=np.float32)
    if score_history is not None and len(score_history) > 0:
        recent = np.asarray(score_history, dtype=np.float32)
        if len(recent) > score_history_len:
            recent = recent[-score_history_len:]
        history_vec[-len(recent):] = recent

    # Current score as a single scalar
    score_scalar = np.array([float(current_score)], dtype=np.float32)

    # Concatenate all components
    observation = np.concatenate([
        pocket_vec,
        ligand_vec,
        interaction_vec,
        history_vec,
        score_scalar,
    ])

    if as_tensor:
        return torch.tensor(observation, dtype=torch.float32, device=device)

    return observation


def encode_state_batch(
    pocket_features_batch: "list[np.ndarray]",
    ligand_features_batch: "list[np.ndarray]",
    interaction_features_batch: "list[np.ndarray]",
    score_history_batch: "list[np.ndarray]",
    current_scores: "list[float]",
    pocket_dim: int = 40,
    ligand_dim: int = 16,
    interaction_dim: int = 20,
    score_history_len: int = 10,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode a batch of states into a tensor suitable for batched GRU forward.

    Args:
        *_batch: Lists of per-sample feature arrays.
        current_scores: List of current Vina scores.
        Other args: Same as encode_state.

    Returns:
        torch.Tensor of shape (batch_size, total_input_dim).
    """
    batch = []
    for pocket, ligand, interaction, history, score in zip(
        pocket_features_batch,
        ligand_features_batch,
        interaction_features_batch,
        score_history_batch,
        current_scores,
    ):
        obs = encode_state(
            pocket_features=pocket,
            ligand_features=ligand,
            interaction_features=interaction,
            score_history=history,
            current_score=score,
            pocket_dim=pocket_dim,
            ligand_dim=ligand_dim,
            interaction_dim=interaction_dim,
            score_history_len=score_history_len,
            as_tensor=False,
        )
        batch.append(obs)

    return torch.tensor(np.stack(batch), dtype=torch.float32, device=device)


def _pad_or_truncate(arr: Optional[np.ndarray], target_len: int) -> np.ndarray:
    """Zero-pad or truncate an array to a fixed length."""
    result = np.zeros(target_len, dtype=np.float32)
    if arr is not None and len(arr) > 0:
        arr = np.asarray(arr, dtype=np.float32).ravel()
        n = min(len(arr), target_len)
        result[:n] = arr[:n]
    return result
