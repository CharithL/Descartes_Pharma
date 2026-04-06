"""
AutoDock Vina wrapper -- the "chess rules" of the docking game.

Evaluates any ligand pose instantly. No learning needed.
This is what makes the docking game fundamentally easier than ARC-AGI-3:
we KNOW the rules and can SIMULATE freely.

Install: pip install vina --break-system-packages

If the vina pip package is not available, falls back to a simple
RDKit distance-based scoring model (FallbackVinaModel).
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Try to import vina; if unavailable, we use a fallback scorer
_VINA_AVAILABLE = False
try:
    from vina import Vina
    _VINA_AVAILABLE = True
except ImportError:
    logger.warning(
        "AutoDock Vina not installed. Using FallbackVinaModel "
        "(RDKit distance-based scoring). For production use: "
        "pip install vina --break-system-packages"
    )


@dataclass
class VinaScore:
    """Result of scoring one pose."""
    total_energy: float         # Total binding energy (kcal/mol) -- more negative = better
    inter_energy: float         # Intermolecular energy
    intra_energy: float         # Intramolecular (ligand internal) energy

    # Decomposed energy terms (where available)
    # These correspond DIRECTLY to the pharmacological priors:
    #   H_BOND -> gauss1/gauss2 attractive terms
    #   HYDROPHOBIC -> hydrophobic term
    #   STERIC_CLASH -> repulsion term

    # Additional computed features (NOT from Vina, computed separately)
    n_hbonds: int = 0
    hydrophobic_contact_area: float = 0.0
    steric_clash_count: int = 0
    dist_asp32: float = 0.0
    dist_asp228: float = 0.0
    is_penalty: bool = False  # True if score is a penalty (ligand out of bounds)
    docked_coords: object = None  # np.ndarray of docked pose coordinates (if available)


class VinaWorldModel:
    """
    The perfect world model -- knows the rules of molecular binding.

    Usage:
        wm = VinaWorldModel("data/structures/prepared/4IVT_receptor.pdbqt")
        score = wm.score_pose(ligand_pdbqt_string)

    This is the chess engine's rule book.
    The policy network decides WHERE to move.
    Vina tells you what the SCORE is after the move.

    Falls back to FallbackVinaModel if the vina package is not installed.
    """

    def __init__(self, receptor_pdbqt_path: str,
                 center: tuple = (16.0, 20.9, 13.5),
                 box_size: tuple = (25.0, 25.0, 25.0),
                 exhaustiveness: int = 8,
                 allow_fallback: bool = True):
        """
        Initialize Vina with receptor structure.

        Args:
            receptor_pdbqt_path: Prepared receptor file
            center: (x, y, z) center of search box
            box_size: (sx, sy, sz) dimensions of search box in Angstroms
            exhaustiveness: Search thoroughness (higher = slower + better)
            allow_fallback: If False, raise error when Vina fails (no silent fallback)
        """
        self.center = np.array(center)
        self.box_size = np.array(box_size)
        self.exhaustiveness = exhaustiveness
        self.n_evaluations = 0
        self._receptor_path = receptor_pdbqt_path

        if _VINA_AVAILABLE:
            try:
                self.v = Vina(sf_name='vina')
                self.v.set_receptor(receptor_pdbqt_path)
                self.v.compute_vina_maps(center=list(center),
                                         box_size=list(box_size))
                self._fallback = False
                logger.info("Vina loaded successfully with real physics scoring")
            except Exception as e:
                if not allow_fallback:
                    raise RuntimeError(
                        f"Vina failed to load receptor and allow_fallback=False. "
                        f"Fix the PDBQT file or install Open Babel. Error: {e}"
                    )
                logger.warning(f"Vina receptor setup failed: {e}. Using fallback scorer.")
                self.v = None
                self._fallback = True
                self._fallback_model = FallbackVinaModel(
                    receptor_pdbqt_path, center, box_size
                )
        else:
            if not allow_fallback:
                raise RuntimeError(
                    "Vina not installed and allow_fallback=False. "
                    "Install: pip install vina"
                )
            self.v = None
            self._fallback = True
            self._fallback_model = FallbackVinaModel(
                receptor_pdbqt_path, center, box_size
            )

    def score_pose(self, ligand_pdbqt: str) -> VinaScore:
        """
        Score a single ligand pose. This is ONE "chess move evaluation."

        Args:
            ligand_pdbqt: PDBQT string of the ligand in a specific pose

        Returns:
            VinaScore with binding energy (more negative = better binding)
        """
        if self._fallback:
            self.n_evaluations += 1
            return self._fallback_model.score_pose(ligand_pdbqt)

        # Write ligand to temp file (Vina needs file path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt',
                                          delete=False) as f:
            f.write(ligand_pdbqt)
            tmp_path = f.name

        try:
            self.v.set_ligand_from_file(tmp_path)
            try:
                energy = self.v.score()
            except RuntimeError as e:
                if "outside the grid box" in str(e):
                    logger.debug("Ligand outside grid box — returning penalty score")
                    self.n_evaluations += 1
                    return VinaScore(
                        total_energy=100.0, inter_energy=100.0,
                        intra_energy=0.0, is_penalty=True,
                    )
                raise
            self.n_evaluations += 1

            return VinaScore(
                total_energy=energy[0],
                inter_energy=energy[1] if len(energy) > 1 else energy[0],
                intra_energy=energy[2] if len(energy) > 2 else 0.0,
            )
        except (TypeError, RuntimeError) as e:
            if "PDBQT parsing error" in str(e) or "outside the grid box" in str(e):
                logger.debug(f"Vina scoring error: {e} — returning penalty")
                self.n_evaluations += 1
                return VinaScore(
                    total_energy=100.0, inter_energy=100.0,
                    intra_energy=0.0, is_penalty=True,
                )
            raise
        finally:
            os.unlink(tmp_path)

    def dock_ligand(self, ligand_pdbqt: str,
                     n_poses: int = 3,
                     exhaustiveness: int = 4) -> List[VinaScore]:
        """
        Run Vina docking search to find best poses automatically.

        Uses Vina's genetic algorithm + local optimization. More expensive
        than single-pose scoring (~5-10 sec) but gives physically realistic
        starting points for RL refinement.
        """
        if self._fallback:
            return [self._fallback_model.score_pose(ligand_pdbqt)]

        tmp_ligand = tempfile.mktemp(suffix='.pdbqt')
        tmp_output = tempfile.mktemp(suffix='_docked.pdbqt')

        try:
            with open(tmp_ligand, 'w') as f:
                f.write(ligand_pdbqt)

            self.v.set_ligand_from_file(tmp_ligand)
            self.v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

            energies = self.v.energies(n_poses=n_poses)

            # Write docked poses so we can extract coordinates
            self.v.write_poses(tmp_output, n_poses=n_poses)
            coords_list = self._parse_docked_pdbqt(tmp_output)

            results = []
            for i, e in enumerate(energies):
                score = VinaScore(
                    total_energy=e[0],
                    inter_energy=e[1] if len(e) > 1 else e[0],
                    intra_energy=e[2] if len(e) > 2 else 0.0,
                )
                if i < len(coords_list):
                    score.docked_coords = coords_list[i]
                results.append(score)

            self.n_evaluations += 1
            return results

        except (TypeError, RuntimeError) as e:
            logger.debug(f"Docking failed: {e}")
            return []
        finally:
            for p in [tmp_ligand, tmp_output]:
                if os.path.exists(p):
                    os.unlink(p)

    def _parse_docked_pdbqt(self, pdbqt_path: str) -> list:
        """Parse multiple models from a docked PDBQT file."""
        coords_per_model = []
        current_coords = []

        if not os.path.exists(pdbqt_path):
            return coords_per_model

        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith('MODEL'):
                    current_coords = []
                elif line.startswith('ENDMDL'):
                    if current_coords:
                        coords_per_model.append(np.array(current_coords))
                    current_coords = []
                elif line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_coords.append([x, y, z])
                    except (ValueError, IndexError):
                        pass

        if current_coords and not coords_per_model:
            coords_per_model.append(np.array(current_coords))

        return coords_per_model

    def get_evaluation_count(self) -> int:
        """How many poses have been evaluated (for logging)."""
        return self.n_evaluations


# =====================================================================
# Fallback scoring model when AutoDock Vina is not installed
# =====================================================================


class FallbackVinaModel:
    """
    Simple distance-based scoring when AutoDock Vina is unavailable.

    Parses PDBQT atom coordinates and computes a rough binding score
    based on distance to the box center and simple pairwise terms.
    This is NOT a replacement for Vina -- it is a development/testing
    fallback that produces qualitatively reasonable relative rankings.
    """

    def __init__(self, receptor_pdbqt_path: str,
                 center: tuple = (28.0, 15.0, 22.0),
                 box_size: tuple = (25.0, 25.0, 25.0)):
        self.center = np.array(center)
        self.box_size = np.array(box_size)
        self.receptor_coords = self._parse_pdbqt_coords(receptor_pdbqt_path)

    @staticmethod
    def _parse_pdbqt_coords(path: str) -> np.ndarray:
        """Extract atom coordinates from a PDBQT file."""
        coords = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except (ValueError, IndexError):
                            continue
        except FileNotFoundError:
            logger.warning(f"Receptor file not found: {path}")
            return np.zeros((0, 3))

        if len(coords) == 0:
            return np.zeros((0, 3))
        return np.array(coords)

    @staticmethod
    def _parse_pdbqt_string_coords(pdbqt_str: str) -> np.ndarray:
        """Extract atom coordinates from a PDBQT string."""
        coords = []
        for line in pdbqt_str.strip().split('\n'):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue

        if len(coords) == 0:
            return np.zeros((1, 3))
        return np.array(coords)

    def score_pose(self, ligand_pdbqt: str) -> VinaScore:
        """
        Approximate scoring using distance-based heuristics.

        Energy components:
        - Distance penalty: how far the ligand is from box center
        - Steric term: clash penalty for atoms too close to receptor
        - Contact term: reward for atoms near receptor (van der Waals)
        """
        lig_coords = self._parse_pdbqt_string_coords(ligand_pdbqt)
        lig_center = lig_coords.mean(axis=0)

        # Distance to box center (penalty)
        dist_to_center = np.linalg.norm(lig_center - self.center)
        distance_penalty = 0.1 * dist_to_center

        # Contact and clash scoring against receptor
        contact_score = 0.0
        clash_penalty = 0.0

        if len(self.receptor_coords) > 0:
            for atom_pos in lig_coords:
                dists = np.linalg.norm(self.receptor_coords - atom_pos, axis=1)
                min_dist = dists.min()

                # Clash: too close
                if min_dist < 1.5:
                    clash_penalty += 5.0 * (1.5 - min_dist)
                # Good contact: 2-4 Angstroms
                elif min_dist < 4.0:
                    contact_score += 0.3 * (4.0 - min_dist)

        # Approximate total energy (negative = good)
        total = -contact_score + clash_penalty + distance_penalty

        return VinaScore(
            total_energy=total,
            inter_energy=total,
            intra_energy=0.0,
        )
