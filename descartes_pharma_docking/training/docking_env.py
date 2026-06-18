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
        initial_pose_mode: str = "perturbed_docked",
        pose_cache=None,
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
        # D2: "docked" | "perturbed_docked" | "random"
        self.initial_pose_mode = initial_pose_mode
        self.pose_cache = pose_cache  # D1: optional {smiles: coords} cache

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

    def _docked_coords_from_meeko(self, docked_pdbqt, mol):
        """C1: reconstruct full-atom docked coords in ORIGINAL atom order using
        meeko (which embeds the atom mapping in the PDBQT). Returns (n_atoms, 3)
        or None on any failure."""
        try:
            from rdkit import Chem
            from meeko import PDBQTMolecule, RDKitMolCreate
            try:
                pmol = PDBQTMolecule(docked_pdbqt, is_dlg=False, skip_typing=True)
            except TypeError:
                pmol = PDBQTMolecule(docked_pdbqt)
            rdmols = RDKitMolCreate.from_pdbqt_mol(pmol)
            if not rdmols or rdmols[0] is None:
                return None
            rdmol = rdmols[0]
            if rdmol.GetNumConformers() == 0:
                return None
            if rdmol.GetNumAtoms() < mol.GetNumAtoms():
                rdmol = Chem.AddHs(rdmol, addCoords=True)
            return rdmol.GetConformer().GetPositions()
        except Exception as e:
            print(f"    [INIT_POSE] meeko reconstruction failed: "
                  f"{type(e).__name__}: {e}")
            return None

    def _apply_docked_pose_to_mol(self, mol, docked_heavy_coords):
        """C1: place real docked HEAVY-atom coords onto the ligand's heavy atoms
        (assumes meeko preserved heavy-atom order) and re-place hydrogens from
        that geometry. Returns full-atom coords, or None if counts mismatch."""
        try:
            from rdkit import Chem
            from rdkit.Geometry import Point3D
            heavy = np.asarray(docked_heavy_coords)
            mol_noH = Chem.RemoveHs(Chem.Mol(mol))
            heavy_idx = [a.GetIdx() for a in mol_noH.GetAtoms()
                         if a.GetSymbol() != 'H']
            if len(heavy_idx) != len(heavy):
                return None
            if mol_noH.GetNumConformers() == 0:
                conf = Chem.Conformer(mol_noH.GetNumAtoms())
                mol_noH.AddConformer(conf, assignId=True)
            conf = mol_noH.GetConformer()
            for k, idx in enumerate(heavy_idx):
                conf.SetAtomPosition(idx, Point3D(
                    float(heavy[k][0]), float(heavy[k][1]), float(heavy[k][2])))
            molH = Chem.AddHs(mol_noH, addCoords=True)
            return molH.GetConformer().GetPositions()
        except Exception as e:
            print(f"    [INIT_POSE] heavy-atom apply failed: "
                  f"{type(e).__name__}: {e}")
            return None

    def _dock_and_reconstruct(self, ligand, exhaustiveness: int = 4):
        """Dock the ligand and reconstruct the REAL full-atom docked pose (C1).
        Returns (n_atoms, 3) coords, or None if docking/reconstruction fails
        (caller decides the fallback)."""
        try:
            mol = getattr(ligand, "mol", None)
            coords = (ligand.conformer_coords
                      if getattr(ligand, 'conformer_coords', None) is not None
                      else np.random.randn(20, 3))
            self.current_ligand = ligand  # so _coords_to_pdbqt can access mol
            pdbqt_str = self._coords_to_pdbqt(coords)

            dock_results = self.wm.dock_ligand(
                pdbqt_str, n_poses=1, exhaustiveness=exhaustiveness)
            if not dock_results:
                return None
            best = dock_results[0]
            if best.total_energy >= 50.0:
                print(f"    [INIT_POSE] best energy {best.total_energy:.1f} "
                      f"too high")
                return None

            # Stage 1: meeko reconstruction (correct, original atom order)
            docked_pdbqt = getattr(best, "docked_pdbqt", None)
            if mol is not None and docked_pdbqt:
                full = self._docked_coords_from_meeko(docked_pdbqt, mol)
                if full is not None and len(full) == mol.GetNumAtoms():
                    print(f"    [INIT_POSE] docked pose via meeko "
                          f"({len(full)} atoms)")
                    return np.asarray(full)

            # Stage 2: heavy-atom mapping (meeko preserves heavy-atom order)
            heavy = getattr(best, "docked_heavy_coords", None)
            if mol is not None and heavy is not None and len(heavy) > 0:
                full = self._apply_docked_pose_to_mol(mol, heavy)
                if full is not None:
                    print(f"    [INIT_POSE] docked pose via heavy-atom map "
                          f"({len(full)} atoms)")
                    return np.asarray(full)

            # Stage 3: centroid fallback (orientation/conformation lost)
            dc = best.docked_coords
            if dc is not None:
                print(f"    [INIT_POSE] centroid-only pose (orientation lost)")
                centered = coords - coords.mean(axis=0)
                return centered + np.asarray(dc).mean(axis=0)
        except Exception as e:
            print(f"    [INIT_POSE] EXCEPTION: {type(e).__name__}: {e}")
        return None

    def _perturb_pose(self, coords, rng=None) -> np.ndarray:
        """D2 'perturbed_docked': displace the docked pose by 3-5 A and apply a
        random rotation so the agent must re-find the pocket (a regime Vina has
        not already trivially solved)."""
        rng = rng or np.random.default_rng()
        coords = np.asarray(coords, dtype=float)
        center = coords.mean(axis=0)
        axis = rng.normal(size=3)
        axis /= (np.linalg.norm(axis) + 1e-9)
        R = _axis_angle_matrix(axis, rng.uniform(0, 2 * np.pi))
        out = (R @ (coords - center).T).T + center
        disp = rng.normal(size=3)
        disp /= (np.linalg.norm(disp) + 1e-9)
        return out + disp * rng.uniform(3.0, 5.0)

    def _get_initial_pose(self, ligand) -> np.ndarray:
        """Starting pose, governed by initial_pose_mode (D2) and the optional
        pose cache (D1):
          'random'           -> safe offset placement (no docking)
          'docked'           -> the real Vina-docked pose (cached if available)
          'perturbed_docked' -> docked pose + random 3-5 A shift & rotation
        """
        mode = self.initial_pose_mode
        if mode == "random":
            return self._safe_offset_pose(ligand)

        # Base docked pose: cache first (D1), else dock once and cache it.
        smi = getattr(ligand, "smiles", None)
        base = None
        if (self.pose_cache is not None and smi is not None
                and smi in self.pose_cache):
            base = self.pose_cache.get(smi)
        if base is None:
            print(f"    [INIT_POSE] docking (mode={mode})...")
            base = self._dock_and_reconstruct(ligand, exhaustiveness=4)
            if (base is not None and self.pose_cache is not None
                    and smi is not None):
                self.pose_cache.put(smi, base)
        if base is None:
            print(f"    [INIT_POSE] FALLBACK: offset placement")
            return self._safe_offset_pose(ligand)

        if mode == "perturbed_docked":
            return self._perturb_pose(np.asarray(base))
        return np.asarray(base)  # "docked"

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


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation matrix about an arbitrary unit axis (Rodrigues formula)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
