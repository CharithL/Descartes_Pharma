"""
Define the action space for pose adjustments.

The agent's "moves" in the docking game.
Discretized for RL -- each action is a small pose change.
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class DockingAction(IntEnum):
    """
    Discrete action space for pose adjustments.

    18 actions: 6 translations + 6 rotations + N torsions
    (N depends on the ligand's rotatable bonds)
    """
    # Translations (+/-0.5 Angstrom steps)
    TRANSLATE_X_POS = 0
    TRANSLATE_X_NEG = 1
    TRANSLATE_Y_POS = 2
    TRANSLATE_Y_NEG = 3
    TRANSLATE_Z_POS = 4
    TRANSLATE_Z_NEG = 5

    # Rotations (+/-15 degree steps)
    ROTATE_X_POS = 6
    ROTATE_X_NEG = 7
    ROTATE_Y_POS = 8
    ROTATE_Y_NEG = 9
    ROTATE_Z_POS = 10
    ROTATE_Z_NEG = 11

    # Torsion angles (+/-30 degree steps)
    # These are dynamically added based on rotatable bonds
    # TORSION_0_POS = 12, TORSION_0_NEG = 13, ...


# Step sizes
TRANSLATION_STEP = 0.5    # Angstroms
ROTATION_STEP = 15.0      # Degrees
TORSION_STEP = 30.0       # Degrees

# Base actions list (the 12 rigid-body actions)
ACTION_SPACE = [
    DockingAction.TRANSLATE_X_POS,
    DockingAction.TRANSLATE_X_NEG,
    DockingAction.TRANSLATE_Y_POS,
    DockingAction.TRANSLATE_Y_NEG,
    DockingAction.TRANSLATE_Z_POS,
    DockingAction.TRANSLATE_Z_NEG,
    DockingAction.ROTATE_X_POS,
    DockingAction.ROTATE_X_NEG,
    DockingAction.ROTATE_Y_POS,
    DockingAction.ROTATE_Y_NEG,
    DockingAction.ROTATE_Z_POS,
    DockingAction.ROTATE_Z_NEG,
]


def get_action_count(n_rotatable_bonds: int) -> int:
    """Total number of actions for a ligand with N rotatable bonds."""
    return 12 + 2 * min(n_rotatable_bonds, 5)  # Cap at 5 torsions = 22 max


def apply_action(coords: np.ndarray, center: np.ndarray,
                 action: int, rotatable_bonds: list) -> np.ndarray:
    """
    Apply a discrete action to ligand coordinates.

    Returns new coordinates (does NOT modify in place).

    Args:
        coords: (n_atoms, 3) current atom positions.
        center: (3,) center of mass for rotation pivot.
        action: Integer action index from DockingAction or torsion extension.
        rotatable_bonds: List of (atom_i, atom_j) tuples for torsion bonds.

    Returns:
        new_coords: (n_atoms, 3) updated atom positions.
    """
    new_coords = coords.copy()

    if action < 6:
        # Translation
        direction = action // 2  # 0=X, 1=Y, 2=Z
        sign = 1.0 if action % 2 == 0 else -1.0
        delta = np.zeros(3)
        delta[direction] = sign * TRANSLATION_STEP
        new_coords += delta

    elif action < 12:
        # Rotation around center of mass
        rot_action = action - 6
        axis = rot_action // 2  # 0=X, 1=Y, 2=Z
        sign = 1.0 if rot_action % 2 == 0 else -1.0
        angle_rad = np.radians(sign * ROTATION_STEP)

        # Rotation matrix around the specified axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        if axis == 0:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Rotate around center of mass
        centered = new_coords - center
        rotated = centered @ R.T
        new_coords = rotated + center

    else:
        # Torsion angle rotation
        torsion_idx = (action - 12) // 2
        sign = 1.0 if (action - 12) % 2 == 0 else -1.0

        if torsion_idx < len(rotatable_bonds):
            # Rotate atoms on one side of the bond
            bond = rotatable_bonds[torsion_idx]
            atom_i, atom_j = bond[0], bond[1]
            angle_rad = np.radians(sign * TORSION_STEP)

            # Bond axis vector
            bond_vec = new_coords[atom_j] - new_coords[atom_i]
            bond_vec = bond_vec / (np.linalg.norm(bond_vec) + 1e-8)

            # Rodrigues rotation formula for atoms beyond atom_j
            # Determine which atoms to rotate (simplified: all atoms
            # with index > atom_j in the bond pair)
            pivot = new_coords[atom_i]
            c, s = np.cos(angle_rad), np.sin(angle_rad)

            for k in range(len(new_coords)):
                if k == atom_i:
                    continue
                # Simple heuristic: rotate atoms closer to atom_j side
                d_i = np.linalg.norm(new_coords[k] - new_coords[atom_i])
                d_j = np.linalg.norm(new_coords[k] - new_coords[atom_j])
                if d_j < d_i:
                    v = new_coords[k] - pivot
                    # Rodrigues formula
                    v_rot = (v * c
                             + np.cross(bond_vec, v) * s
                             + bond_vec * np.dot(bond_vec, v) * (1 - c))
                    new_coords[k] = v_rot + pivot

    return new_coords
