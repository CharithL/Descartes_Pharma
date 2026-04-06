"""
Gym-style environment wrapping the docking game.

Each episode:
  1. Load a ligand from BindingDB
  2. Place it in a random initial pose near the pocket
  3. Agent takes T steps of pose adjustments
  4. Each step: Vina scores the new pose -> reward = delta-E
  5. Episode ends after T steps or convergence
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DockingEnv:
    """
    Docking game environment.

    Observation: [pocket_features | ligand_features | interaction_features
                  | score_history | current_score]
    Action: discrete pose adjustment (translate/rotate/torsion)
    Reward: change in Vina binding energy (negative delta-E = improvement)
    """

    # Action space: 12 base movements + up to 10 torsion adjustments
    # Base movements:
    #   0-5: translate +/- x, +/- y, +/- z (by step_size)
    #   6-11: rotate +/- around x, y, z axes (by rotation_step)
    # Torsion adjustments:
    #   12-21: rotate around each rotatable bond (by torsion_step)
    ACTION_NAMES = {
        0: "translate_+x", 1: "translate_-x",
        2: "translate_+y", 3: "translate_-y",
        4: "translate_+z", 5: "translate_-z",
        6: "rotate_+x", 7: "rotate_-x",
        8: "rotate_+y", 9: "rotate_-y",
        10: "rotate_+z", 11: "rotate_-z",
    }
    for i in range(10):
        ACTION_NAMES[12 + i] = f"torsion_{i}"

    def __init__(
        self,
        vina_world_model,
        pocket_features,
        max_steps: int = 200,
        score_history_len: int = 10,
        translation_step: float = 0.5,
        rotation_step: float = 15.0,
        torsion_step: float = 30.0,
        convergence_threshold: float = -9.0,
    ):
        """
        Args:
            vina_world_model: VinaWorldModel instance for scoring poses.
            pocket_features: PocketFeatures instance for the target pocket.
            max_steps: Maximum steps per episode.
            score_history_len: Number of recent scores to include in obs.
            translation_step: Angstroms per translation action.
            rotation_step: Degrees per rotation action.
            torsion_step: Degrees per torsion rotation action.
            convergence_threshold: Vina score below which episode ends early.
        """
        self.wm = vina_world_model
        self.pocket = pocket_features
        self.pocket_vec = pocket_features.to_feature_vector()
        self.max_steps = max_steps
        self.score_history_len = score_history_len
        self.translation_step = translation_step
        self.rotation_step = rotation_step
        self.torsion_step = torsion_step
        self.convergence_threshold = convergence_threshold

        # State (set during reset)
        self.current_ligand = None
        self.current_coords = None
        self.current_score = None
        self.score_history: list = []
        self.step_count = 0
        self.best_score = None

        # Trajectory storage for DESCARTES probing
        self.trajectory: list = []

    def reset(self, ligand_features, initial_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Start a new docking episode with a new ligand.

        Args:
            ligand_features: LigandFeatures object.
            initial_coords: (n_atoms, 3) initial pose, or None for random.

        Returns:
            observation: np.ndarray
        """
        self.current_ligand = ligand_features
        self.step_count = 0
        self.score_history = []
        self.trajectory = []

        if initial_coords is not None:
            self.current_coords = initial_coords.copy()
        else:
            # Use Vina docking for realistic starting pose (fallback to offset)
            self.current_coords = self._get_initial_pose(ligand_features)

        # Score initial pose
        pdbqt = self._coords_to_pdbqt(self.current_coords)
        score_result = self.wm.score_pose(pdbqt)
        self.current_score = score_result.total_energy
        self.best_score = self.current_score

        # Compute interaction features
        interaction_vec = self._compute_interactions()

        obs = self._make_observation(interaction_vec)
        self.trajectory.append({
            "step": 0,
            "action": None,
            "score": self.current_score,
            "coords": self.current_coords.copy(),
            "observation": obs.copy(),
        })

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one action (pose adjustment) and get reward.

        Returns:
            observation: np.ndarray
            reward: float (negative delta-E = good, positive delta-E = bad)
            done: bool
            info: dict with details
        """
        # Apply action to get new coordinates
        new_coords = self._apply_action(action)

        # Score new pose with Vina (FREE -- no budget constraint!)
        pdbqt = self._coords_to_pdbqt(new_coords)
        try:
            score_result = self.wm.score_pose(pdbqt)
            new_score = score_result.total_energy
        except Exception:
            # Invalid pose (e.g., outside box) -- penalize
            new_score = self.current_score + 10.0  # Penalty

        # Reward = improvement in binding energy
        # Vina scores are negative (more negative = better)
        # So reward = old_score - new_score (positive if new is more negative)
        reward = self.current_score - new_score

        # Update state
        self.current_coords = new_coords
        self.current_score = new_score
        self.score_history.append(new_score)
        self.step_count += 1

        if new_score < self.best_score:
            self.best_score = new_score

        # Check done conditions
        done = (
            self.step_count >= self.max_steps
            or new_score <= self.convergence_threshold
        )

        # Compute interaction features for new pose
        interaction_vec = self._compute_interactions()

        info = {
            "vina_score": new_score,
            "best_score": self.best_score,
            "step": self.step_count,
            "reward": reward,
            "action": action,
            "action_name": self.ACTION_NAMES.get(action, f"unknown_{action}"),
            "dist_asp32": self._dist_to_residue(32),
            "dist_asp228": self._dist_to_residue(228),
            "n_evaluations": self.wm.get_evaluation_count()
            if hasattr(self.wm, "get_evaluation_count")
            else self.step_count,
        }

        obs = self._make_observation(interaction_vec)

        # Record trajectory for probing
        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "score": new_score,
            "reward": reward,
            "coords": new_coords.copy(),
            "observation": obs.copy(),
            "info": info,
        })

        return obs, reward, done, info

    def _make_observation(self, interaction_vec: np.ndarray) -> np.ndarray:
        """Compose full observation vector."""
        ligand_vec = self.current_ligand.to_feature_vector()

        # Score history (padded to fixed length)
        history = np.zeros(self.score_history_len, dtype=np.float32)
        if self.score_history:
            recent = self.score_history[-self.score_history_len:]
            history[-len(recent):] = recent

        return np.concatenate([
            self.pocket_vec,
            ligand_vec,
            interaction_vec,
            history,
            np.array([self.current_score], dtype=np.float32),
        ])

    def _compute_interactions(self) -> np.ndarray:
        """
        Compute pairwise interaction features between current ligand pose
        and pocket.

        These are the "CoreKnowledge translation layer" features --
        structured relational representations, not flat descriptors.
        """
        features = []

        ligand_center = self.current_coords.mean(axis=0)

        # Distance to each catalytic residue
        if hasattr(self.pocket, "catalytic_residues"):
            for cat_res in self.pocket.catalytic_residues:
                center = cat_res.center if hasattr(cat_res, "center") else cat_res
                dist = np.linalg.norm(ligand_center - np.asarray(center))
                features.append(dist)
        # Pad to 4 catalytic residues
        while len(features) < 4:
            features.append(50.0)  # Far away default

        # Distance to pocket center
        pocket_center = (
            self.pocket.pocket_center
            if hasattr(self.pocket, "pocket_center")
            else np.zeros(3)
        )
        features.append(np.linalg.norm(ligand_center - pocket_center))

        # Number of close contacts (< 4 A) with H-bond donors/acceptors
        n_close_hbond = 0
        if hasattr(self.pocket, "hbond_donors"):
            for donor in self.pocket.hbond_donors:
                center = donor.center if hasattr(donor, "center") else donor
                if np.linalg.norm(ligand_center - np.asarray(center)) < 4.0:
                    n_close_hbond += 1
        features.append(float(n_close_hbond))

        # Number of close contacts with hydrophobic residues
        n_close_hydrophobic = 0
        if hasattr(self.pocket, "hydrophobic_residues"):
            for hyd in self.pocket.hydrophobic_residues:
                center = hyd.center if hasattr(hyd, "center") else hyd
                if np.linalg.norm(ligand_center - np.asarray(center)) < 5.0:
                    n_close_hydrophobic += 1
        features.append(float(n_close_hydrophobic))

        # Steric clashes (atoms < 2 A from any pocket atom)
        n_clashes = 0
        if hasattr(self.pocket, "residues"):
            for res in self.pocket.residues:
                sidechain = (
                    res.sidechain_atoms
                    if hasattr(res, "sidechain_atoms")
                    else []
                )
                for sc_atom in sidechain:
                    for lig_atom in self.current_coords:
                        if np.linalg.norm(lig_atom - np.asarray(sc_atom)) < 2.0:
                            n_clashes += 1
        features.append(float(min(n_clashes, 20)))  # Cap

        # Fraction of pocket volume occupied
        # (simplified: fraction of pocket residues within 5A of ligand)
        if hasattr(self.pocket, "residues") and len(self.pocket.residues) > 0:
            n_contacted = sum(
                1 for r in self.pocket.residues
                if np.linalg.norm(
                    ligand_center - np.asarray(
                        r.center if hasattr(r, "center") else [0, 0, 0]
                    )
                ) < 5.0
            )
            features.append(n_contacted / max(len(self.pocket.residues), 1))
        else:
            features.append(0.0)

        # Number of waters potentially displaced
        if hasattr(self.pocket, "water_positions"):
            n_waters_displaced = sum(
                1 for w in self.pocket.water_positions
                if np.linalg.norm(ligand_center - np.asarray(w)) < 3.0
            )
            features.append(float(n_waters_displaced))
        else:
            features.append(0.0)

        # Pad to fixed size (20 dimensions)
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20], dtype=np.float32)

    def _dist_to_residue(self, resid: int) -> float:
        """Distance from ligand center to a specific residue."""
        ligand_center = self.current_coords.mean(axis=0)
        if hasattr(self.pocket, "residues"):
            for res in self.pocket.residues:
                if hasattr(res, "resid") and res.resid == resid:
                    center = res.center if hasattr(res, "center") else [0, 0, 0]
                    return float(np.linalg.norm(ligand_center - np.asarray(center)))
        return 50.0  # Not found

    def _get_initial_pose(self, ligand) -> np.ndarray:
        """
        Get a physically realistic starting pose using Vina docking.

        Strategy: let Vina do a quick search (exhaustiveness=4, ~5-10 sec)
        to find a plausible pose, then the RL agent refines it.
        """
        print(f"    [INIT_POSE] Attempting Vina docking for initial pose...")

        try:
            coords = (ligand.conformer_coords
                      if hasattr(ligand, 'conformer_coords') and ligand.conformer_coords is not None
                      else np.random.randn(20, 3))
            self.current_ligand = ligand  # Set so _coords_to_pdbqt can access mol
            pdbqt_str = self._coords_to_pdbqt(coords)
            print(f"    [INIT_POSE] PDBQT generated, {len(pdbqt_str)} chars")

            dock_results = self.wm.dock_ligand(pdbqt_str, n_poses=1, exhaustiveness=4)
            print(f"    [INIT_POSE] dock_ligand returned {len(dock_results)} results")

            if dock_results:
                best = dock_results[0]
                print(f"    [INIT_POSE] Best energy: {best.total_energy:.2f} kcal/mol")
                has_coords = best.docked_coords is not None
                print(f"    [INIT_POSE] Has docked_coords: {has_coords}")

                if has_coords:
                    dc = best.docked_coords
                    print(f"    [INIT_POSE] Docked coords shape: {dc.shape}")
                    print(f"    [INIT_POSE] Docked center: {dc.mean(axis=0).round(2)}")

                    # Check atom count match
                    n_orig = len(coords)
                    n_dock = len(dc)
                    print(f"    [INIT_POSE] Original atoms: {n_orig}, Docked atoms: {n_dock}")

                    if n_dock == n_orig and best.total_energy < 50.0:
                        print(f"    [INIT_POSE] SUCCESS — using Vina-docked pose")
                        return dc
                    elif n_dock != n_orig:
                        print(f"    [INIT_POSE] ATOM COUNT MISMATCH — "
                              f"cannot use docked coords directly")
                        # Use docked center + original shape as fallback
                        centered = coords - coords.mean(axis=0)
                        docked_center = dc.mean(axis=0)
                        print(f"    [INIT_POSE] Using docked CENTER with original shape")
                        return centered + docked_center
                    else:
                        print(f"    [INIT_POSE] Energy too high ({best.total_energy:.1f}), "
                              f"using docked center anyway")
                        centered = coords - coords.mean(axis=0)
                        return centered + dc.mean(axis=0)
                else:
                    print(f"    [INIT_POSE] No docked_coords — coords extraction failed")
            else:
                print(f"    [INIT_POSE] dock_ligand returned empty list!")

        except Exception as e:
            print(f"    [INIT_POSE] EXCEPTION: {type(e).__name__}: {e}")

        print(f"    [INIT_POSE] FALLBACK: using offset placement")
        return self._safe_offset_pose(ligand)

    def _safe_offset_pose(self, ligand) -> np.ndarray:
        """Fallback: place ligand near pocket with safe offset to avoid clashes."""
        pocket_center = (
            self.pocket.pocket_center
            if hasattr(self.pocket, "pocket_center")
            else np.zeros(3)
        )

        if hasattr(ligand, "conformer_coords") and ligand.conformer_coords is not None:
            coords = ligand.conformer_coords
        else:
            n_atoms = (
                ligand.mol.GetNumAtoms()
                if hasattr(ligand, "mol") and ligand.mol is not None
                else 20
            )
            coords = np.random.randn(n_atoms, 3) * 1.5

        center_of_mass = (
            ligand.center_of_mass
            if hasattr(ligand, "center_of_mass")
            else coords.mean(axis=0)
        )
        centered = coords - center_of_mass

        # Compute ligand radius
        ligand_radius = float(np.max(np.linalg.norm(centered, axis=1)))
        box_half = 15.0
        if hasattr(self, 'wm') and hasattr(self.wm, 'box_size'):
            box_half = float(self.wm.box_size[0]) / 2.0
        max_offset = max(0.5, box_half - ligand_radius - 3.0)

        # Offset away from protein core (bias +Z to avoid burying inside)
        rng = np.random.default_rng()
        offset = rng.uniform(-2.0, 2.0, size=3)
        offset = np.clip(offset, -max_offset, max_offset)

        return centered + pocket_center + offset

    def _apply_action(self, action: int) -> np.ndarray:
        """
        Apply a discrete action to the current ligand coordinates.

        Actions 0-5: translations along +/- x, y, z
        Actions 6-11: rotations around x, y, z axes
        Actions 12-21: torsion rotations around rotatable bonds
        """
        coords = self.current_coords.copy()
        center = coords.mean(axis=0)

        if action < 6:
            # Translation
            direction = np.zeros(3)
            axis = action // 2
            sign = 1.0 if action % 2 == 0 else -1.0
            direction[axis] = sign * self.translation_step
            coords = coords + direction

        elif action < 12:
            # Rotation around center of mass
            rot_action = action - 6
            axis = rot_action // 2
            sign = 1.0 if rot_action % 2 == 0 else -1.0
            angle_deg = sign * self.rotation_step
            angle_rad = np.radians(angle_deg)

            # Rotation matrix around the given axis
            rot_mat = _rotation_matrix(axis, angle_rad)
            centered = coords - center
            coords = (rot_mat @ centered.T).T + center

        else:
            # Torsion rotation (around a rotatable bond)
            bond_idx = action - 12
            rotatable_bonds = (
                self.current_ligand.rotatable_bond_indices
                if hasattr(self.current_ligand, "rotatable_bond_indices")
                else []
            )
            if bond_idx < len(rotatable_bonds):
                bond = rotatable_bonds[bond_idx]
                coords = self._rotate_around_bond(
                    coords, bond, np.radians(self.torsion_step)
                )
            # If bond_idx >= number of rotatable bonds, no-op

        return coords

    def _rotate_around_bond(
        self, coords: np.ndarray, bond: tuple, angle: float
    ) -> np.ndarray:
        """Rotate atoms on one side of a bond by the given angle."""
        i, j = bond[0], bond[1]
        if i >= len(coords) or j >= len(coords):
            return coords

        origin = coords[i]
        axis = coords[j] - coords[i]
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return coords
        axis = axis / norm

        # Rodrigues rotation for atoms after j
        new_coords = coords.copy()
        for k in range(j, len(coords)):
            if k == i:
                continue
            v = new_coords[k] - origin
            v_rot = (
                v * np.cos(angle)
                + np.cross(axis, v) * np.sin(angle)
                + axis * np.dot(axis, v) * (1 - np.cos(angle))
            )
            new_coords[k] = v_rot + origin

        return new_coords

    def _coords_to_pdbqt(self, coords: np.ndarray) -> str:
        """Convert numpy coordinates to PDBQT string for Vina.

        Uses meeko if the ligand has an RDKit mol, otherwise manual
        HETATM + ROOT/ENDROOT format. Never writes MODEL/ENDMDL tags.
        """
        # Try meeko with actual mol object for correct atom types
        if hasattr(self, 'current_ligand') and hasattr(self.current_ligand, 'mol'):
            mol = self.current_ligand.mol
            if mol is not None:
                try:
                    return self._mol_coords_to_pdbqt(mol, coords)
                except Exception:
                    pass

        # Manual fallback: HETATM + ROOT/ENDROOT (no MODEL tags)
        lines = ["ROOT"]
        for i, (x, y, z) in enumerate(coords):
            lines.append(
                f"HETATM{i+1:5d}  C   LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000  C"
            )
        lines.append("ENDROOT")
        lines.append("END")
        return "\n".join(lines)

    def _mol_coords_to_pdbqt(self, mol, coords: np.ndarray) -> str:
        """Convert RDKit mol with updated coords to single-model PDBQT."""
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        mol = Chem.RWMol(mol)
        if mol.GetNumConformers() == 0:
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        conf = mol.GetConformer()
        n_atoms = min(len(coords), mol.GetNumAtoms())
        for i in range(n_atoms):
            conf.SetAtomPosition(i, Point3D(
                float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

        # Try meeko
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if is_ok:
                # Strip MODEL/ENDMDL tags
                lines = [l for l in pdbqt_string.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                return '\n'.join(lines)
        except Exception:
            pass

        # Try obabel
        try:
            import tempfile, subprocess, os
            sdf_path = tempfile.mktemp(suffix='.sdf')
            pdbqt_path = tempfile.mktemp(suffix='.pdbqt')
            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()
            result = subprocess.run(
                ['obabel', sdf_path, '-O', pdbqt_path, '-p', '7.4'],
                capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(pdbqt_path):
                with open(pdbqt_path) as f:
                    content = f.read()
                lines = [l for l in content.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                return '\n'.join(lines)
        except Exception:
            pass
        finally:
            for p in [sdf_path, pdbqt_path]:
                try:
                    os.unlink(p)
                except Exception:
                    pass

        # Last resort: manual with proper atom types
        lines = ["ROOT"]
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            sym = atom.GetSymbol()
            x, y, z = coords[i]
            ad = 'A' if atom.GetIsAromatic() and sym == 'C' else (
                 'OA' if sym == 'O' else ('NA' if sym == 'N' and atom.GetTotalNumHs() == 0 else (
                 'N' if sym == 'N' else ('SA' if sym == 'S' else 'C'))))
            lines.append(
                f"HETATM{i+1:5d} {sym:<4s} LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 {ad:>2s}")
        lines.append("ENDROOT")
        lines.append("END")
        return '\n'.join(lines)

    def get_trajectory(self) -> list:
        """Get the full trajectory for this episode (for probing)."""
        return self.trajectory


def _rotation_matrix(axis_idx: int, angle: float) -> np.ndarray:
    """
    3x3 rotation matrix around x (0), y (1), or z (2) axis.

    Args:
        axis_idx: 0=x, 1=y, 2=z
        angle: rotation angle in radians
    """
    c, s = np.cos(angle), np.sin(angle)
    if axis_idx == 0:  # x
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ])
    elif axis_idx == 1:  # y
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])
    else:  # z
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
