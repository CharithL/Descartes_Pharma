"""
Receptor preparation for AutoDock Vina docking.

Steps:
    1. Download PDB 4IVT from RCSB if not present locally
    2. Parse with BioPython
    3. Remove waters and non-protein ligands
    4. Write a clean protein-only PDB
    5. Convert to PDBQT format (openbabel subprocess or simple fallback)

This is a one-time preparation step. The resulting PDBQT file is used
by VinaWorldModel for all subsequent scoring.
"""

import os
import logging
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PDB_DIR = os.path.join("data", "structures")
DEFAULT_PREPARED_DIR = os.path.join("data", "structures", "prepared")
PDB_ID = "4IVT"
RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str = PDB_ID,
                 output_dir: str = DEFAULT_PDB_DIR) -> str:
    """
    Download a PDB file from RCSB if not already present.

    Args:
        pdb_id: PDB identifier (e.g., "4IVT").
        output_dir: Directory to save the PDB file.

    Returns:
        Path to the downloaded PDB file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(pdb_path):
        logger.info(f"PDB file already exists: {pdb_path}")
        return pdb_path

    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id)
    logger.info(f"Downloading {pdb_id} from {url}")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, pdb_path)
        logger.info(f"Downloaded {pdb_id} to {pdb_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download PDB {pdb_id} from RCSB: {e}"
        )

    return pdb_path


def clean_pdb(pdb_path: str,
              output_path: Optional[str] = None,
              remove_waters: bool = True,
              remove_ligands: bool = True) -> str:
    """
    Parse a PDB file with BioPython and write a clean protein-only PDB.

    Removes:
        - Water molecules (HOH) if remove_waters is True
        - Non-protein ligands (HETATM records) if remove_ligands is True
        - Alternate conformations (keeps only 'A' or first)

    Args:
        pdb_path: Path to input PDB file.
        output_path: Path for cleaned PDB. If None, derived from input.
        remove_waters: Whether to remove water molecules.
        remove_ligands: Whether to remove non-protein ligands.

    Returns:
        Path to cleaned PDB file.
    """
    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        from Bio.PDB.Polypeptide import is_aa
    except ImportError:
        raise ImportError("pip install biopython --break-system-packages")

    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = f"{base}_clean.pdb"

    class ProteinSelect(Select):
        """Select only protein atoms, removing waters and ligands."""

        def accept_residue(self, residue):
            resname = residue.get_resname().strip()

            # Remove waters
            if remove_waters and resname == "HOH":
                return False

            # Keep standard amino acids
            if is_aa(residue, standard=True):
                return True

            # Remove non-protein ligands
            if remove_ligands:
                hetflag = residue.get_id()[0]
                if hetflag.startswith("H_") or hetflag == "W":
                    return False

            return True

        def accept_atom(self, atom):
            # Skip alternate conformations (keep 'A' or ' ')
            altloc = atom.get_altloc()
            if altloc and altloc not in (' ', 'A'):
                return False
            return True

    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    structure = parser.get_structure(pdb_id, pdb_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ProteinSelect())

    logger.info(f"Cleaned PDB written to {output_path}")
    return output_path


def pdb_to_pdbqt(pdb_path: str,
                 output_path: Optional[str] = None,
                 use_openbabel: bool = True) -> str:
    """
    Convert a PDB file to PDBQT format for AutoDock Vina.

    Attempts to use Open Babel (obabel) via subprocess. If not available,
    falls back to a simple converter that adds Gasteiger charges and
    atom type annotations.

    Args:
        pdb_path: Path to input PDB file.
        output_path: Path for PDBQT output. If None, derived from input.
        use_openbabel: Try Open Babel first if True.

    Returns:
        Path to PDBQT file.
    """
    if output_path is None:
        base = os.path.splitext(pdb_path)[0]
        output_path = f"{base}.pdbqt"

    if use_openbabel:
        success = _convert_with_openbabel(pdb_path, output_path)
        if success:
            return output_path
        logger.warning("Open Babel not available, using fallback converter")

    _convert_fallback(pdb_path, output_path)
    return output_path


def _convert_with_openbabel(pdb_path: str, output_path: str) -> bool:
    """
    Convert PDB to PDBQT using Open Babel (obabel).

    Returns True if conversion succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "obabel", pdb_path,
                "-O", output_path,
                "-xr",           # Receptor mode (rigid)
                "--partialcharge", "gasteiger",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Open Babel conversion successful: {output_path}")
            return True
        else:
            logger.warning(f"Open Babel failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.debug("obabel not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Open Babel conversion timed out")
        return False


# AutoDock atom type mapping for common protein atoms
_AD_ATOM_TYPES = {
    "C": "C",
    "N": "NA",  # Nitrogen acceptor
    "O": "OA",  # Oxygen acceptor
    "S": "SA",  # Sulfur acceptor
    "H": "HD",  # Hydrogen donor (bonded to N/O)
    "F": "F",
    "P": "P",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "FE": "Fe",
    "ZN": "Zn",
    "MG": "Mg",
    "CA": "Ca",
    "MN": "Mn",
}


def _convert_fallback(pdb_path: str, output_path: str) -> None:
    """
    Simple PDB to PDBQT converter that adds dummy partial charges
    and AutoDock atom types.

    This is a minimal fallback for development. For production, use
    Open Babel, MGLTools prepare_receptor4.py, or ADFR Suite.
    """
    logger.warning(
        "Using fallback PDB->PDBQT converter. For production, install "
        "Open Babel: conda install -c conda-forge openbabel"
    )

    lines_out = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Parse element from columns 77-78 (right-justified)
                element = line[76:78].strip().upper() if len(line) > 76 else ""
                if not element:
                    # Fallback: derive from atom name (columns 13-16)
                    atom_name = line[12:16].strip()
                    element = atom_name[0] if atom_name else "C"

                # Map to AutoDock atom type
                ad_type = _AD_ATOM_TYPES.get(element, "C")

                # Assign a dummy partial charge (0.000)
                charge = 0.000

                # Build PDBQT line: PDB line (up to col 54) + occupancy +
                # B-factor + charge + atom type
                # PDBQT format: columns 1-54 same as PDB, then
                # 55-60 occupancy, 61-66 B-factor, 67-76 charge, 77-78 type
                pdb_line = line.rstrip()

                # Ensure line is at least 54 chars
                pdb_line = pdb_line.ljust(54)

                # Take first 54 chars, add occupancy/B-factor from original
                # or defaults
                try:
                    occupancy = float(line[54:60])
                except (ValueError, IndexError):
                    occupancy = 1.00
                try:
                    bfactor = float(line[60:66])
                except (ValueError, IndexError):
                    bfactor = 0.00

                pdbqt_line = (
                    f"{pdb_line[:54]}"
                    f"{occupancy:6.2f}"
                    f"{bfactor:6.2f}"
                    f"    {charge:+8.3f} "
                    f"{ad_type:<2s}"
                )
                lines_out.append(pdbqt_line)

            elif line.startswith("END") or line.startswith("TER"):
                lines_out.append(line.rstrip())

    with open(output_path, 'w') as f:
        for line in lines_out:
            f.write(line + '\n')

    logger.info(f"Fallback PDBQT written to {output_path}")


def prepare_receptor(pdb_id: str = PDB_ID,
                     pdb_dir: str = DEFAULT_PDB_DIR,
                     prepared_dir: str = DEFAULT_PREPARED_DIR,
                     force_download: bool = False) -> str:
    """
    Full receptor preparation pipeline:
        1. Download PDB (if needed)
        2. Clean (remove waters, ligands)
        3. Convert to PDBQT

    Args:
        pdb_id: PDB identifier.
        pdb_dir: Directory for raw PDB files.
        prepared_dir: Directory for prepared files.
        force_download: Re-download even if file exists.

    Returns:
        Path to the prepared PDBQT file.
    """
    os.makedirs(prepared_dir, exist_ok=True)

    pdbqt_path = os.path.join(prepared_dir, f"{pdb_id}_receptor.pdbqt")
    if os.path.exists(pdbqt_path) and not force_download:
        logger.info(f"Prepared receptor already exists: {pdbqt_path}")
        return pdbqt_path

    # Step 1: Download
    pdb_path = download_pdb(pdb_id, pdb_dir)

    # Step 2: Clean
    clean_path = os.path.join(prepared_dir, f"{pdb_id}_clean.pdb")
    clean_pdb(pdb_path, clean_path)

    # Step 3: Convert to PDBQT
    pdb_to_pdbqt(clean_path, pdbqt_path)

    logger.info(f"Receptor preparation complete: {pdbqt_path}")
    return pdbqt_path
