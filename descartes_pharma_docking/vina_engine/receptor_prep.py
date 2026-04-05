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


def _get_ad4_atom_type(element: str, atom_name: str, resname: str) -> str:
    """
    Assign AutoDock4 atom type based on element, atom name, and residue.

    AD4 types used by Vina:
      C  - aliphatic carbon
      A  - aromatic carbon
      N  - nitrogen (non-acceptor: backbone N, Lys NZ, Arg NH1/NH2)
      NA - nitrogen acceptor (His ND1/NE2, Asn/Gln sidechain N)
      OA - oxygen acceptor (all O atoms)
      SA - sulfur acceptor
      HD - polar hydrogen (on N-H or O-H)
      H  - non-polar hydrogen
    """
    element = element.upper().strip()

    if element == "C":
        # Aromatic carbons in aromatic residues
        aromatic_res = {"PHE", "TYR", "TRP", "HIS"}
        aromatic_atoms = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "CH2",
                          "CE3", "CZ2", "CZ3"}
        if resname in aromatic_res and atom_name in aromatic_atoms:
            return "A"
        return "C"
    elif element == "N":
        # Backbone N and donor N: type N
        # Acceptor N (His ND1/NE2): type NA
        acceptor_n = {"ND1", "NE2"}
        if atom_name in acceptor_n:
            return "NA"
        return "N"
    elif element == "O":
        return "OA"
    elif element == "S":
        return "SA"
    elif element == "H":
        return "HD"  # Simplified: all H as HD
    elif element == "P":
        return "P"
    elif element in ("F", "CL", "BR", "I"):
        return {"F": "F", "CL": "Cl", "BR": "Br", "I": "I"}[element]
    elif element in ("FE", "ZN", "MG", "CA", "MN"):
        return {"FE": "Fe", "ZN": "Zn", "MG": "Mg", "CA": "Ca", "MN": "Mn"}[element]
    else:
        return "C"


def _convert_fallback(pdb_path: str, output_path: str) -> None:
    """
    PDB to PDBQT converter with correct AutoDock4 atom types.

    PDBQT format (columns, 1-indexed):
      1-6:   Record type (ATOM/HETATM)
      7-11:  Atom serial
      13-16: Atom name
      17:    Alt location
      18-20: Residue name
      22:    Chain ID
      23-26: Residue sequence number
      31-38: X coordinate
      39-46: Y coordinate
      47-54: Z coordinate
      55-60: Occupancy
      61-66: B-factor
      67-76: Blank
      77-78: Charge (as string +0.000 format, right-justified in cols 69-76)
      79-80: AD4 atom type (right-justified)

    Vina is strict about column alignment. The charge field is cols 71-76
    and the atom type is cols 77-78.
    """
    logger.warning(
        "Using fallback PDB->PDBQT converter. For better results, install "
        "Open Babel: apt-get install -y openbabel"
    )

    lines_out = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                if line.startswith("END") or line.startswith("TER"):
                    lines_out.append(line.rstrip())
                continue

            # Parse atom name and residue name
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()

            # Parse element from columns 77-78 (standard PDB)
            element = line[76:78].strip().upper() if len(line) > 76 else ""
            if not element:
                # Derive from atom name (first non-digit character)
                for ch in atom_name:
                    if ch.isalpha():
                        element = ch
                        break
                else:
                    element = "C"

            # Get correct AD4 type
            ad_type = _get_ad4_atom_type(element, atom_name, resname)

            # Parse occupancy and B-factor
            try:
                occupancy = float(line[54:60])
            except (ValueError, IndexError):
                occupancy = 1.00
            try:
                bfactor = float(line[60:66])
            except (ValueError, IndexError):
                bfactor = 0.00

            # Build PDBQT line with correct column alignment
            # Cols 1-54: same as PDB (record, serial, name, alt, resname, chain, resid, coords)
            # Cols 55-60: occupancy
            # Cols 61-66: B-factor
            # Cols 67-76: partial charge (right-justified, format +0.000)
            # Cols 77-78: AD4 atom type (right-justified)
            base = line[:54].rstrip().ljust(54)
            charge = 0.000
            pdbqt_line = f"{base}{occupancy:6.2f}{bfactor:6.2f}    {charge:+6.3f} {ad_type:>2s}"
            lines_out.append(pdbqt_line)

    with open(output_path, 'w') as f:
        for line in lines_out:
            f.write(line + '\n')

    logger.info(f"Fallback PDBQT written to {output_path} ({len(lines_out)} atom lines)")


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
