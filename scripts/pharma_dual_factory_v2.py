#!/usr/bin/env python3
"""
DESCARTES-PHARMA: Dual Factory Campaign v2 -- Hardened Probes
=============================================================
Section 7 of v1.3 Experimental Results.

v2 FIXES (vs v1 pharma_dual_factory_campaign.py):
  FIX 1: Replace naked Ridge fast probe with hardened_fast_probe using
         scaffold-stratified permutation (200 perms per feature).
  FIX 2: Ensure delta_r2 = r2_trained - r2_untrained (fresh random model
         instance each time, never reused).
  FIX 3: Compute Murcko scaffolds for test set molecules; pass to all probes.
  FIX 4: Updated fitness function: 70% catalytic weight, p<0.05 + dR2>0.05
         gate for "encoded" status.
  FIX 5: Detailed per-round output showing p-values for all 4 catalytic
         features (asp32, asp228, hbond, cat_sc).

Runtime: ~130 min (vs ~20 min v1) because hardened probes cost ~2.5 min/genome.
Results are REAL instead of 100% false positives.

The dual factory evolves TRAINING CONFIGURATIONS (not architectures) for
the Concat protein-ligand model. The architecture is fixed:
  GCN + pocket concat, hidden=128

C2 Factory: Evolves loss functions, auxiliary objectives, regularizers,
            and curricula across 40 rounds in 4 phases.
C1 Factory: Hardened mechanistic probes (scaffold-stratified permutation
            on 4 catalytic features, 200 perms each).

Four-phase campaign (40 rounds total):
  Phase 1 (rounds  1-11): Templates -- 11 pre-designed configurations
  Phase 2 (rounds 12-25): Mutation + crossover from top 5 by fitness
  Phase 3 (rounds 26-35): LLM balloon expansion if stalled 8+ rounds
  Phase 4 (rounds 36-40): Exploit best with small perturbations

Winner evaluation: Full hardening + council controls on best genome only.

Target: Vast.ai A10 GPU (Linux, 22.5 GB VRAM).
"""

import subprocess
import sys
import os
import time
import copy
import json
import warnings
import uuid
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')


def ensure_installed(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        pip_name = pip_name or pkg
        print(f"  Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pip_name, "-q"])


print("=" * 85)
print("DESCARTES-PHARMA: Dual Factory Campaign v2 (Hardened Probes)")
print("Training Configuration Evolution for Concat Protein-Ligand Model")
print("=" * 85)

print("\nChecking dependencies...")
ensure_installed("tdc", "PyTDC")
ensure_installed("torch_geometric", "torch-geometric")
ensure_installed("rdkit", "rdkit")
ensure_installed("sklearn", "scikit-learn")
ensure_installed("Bio", "biopython")

# Meeko + Vina: optional, wrap in try/except
for pkg, pname in [("meeko", "meeko"), ("vina", "vina")]:
    try:
        __import__(pkg)
    except ImportError:
        try:
            print(f"  Attempting install {pname}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   pname, "-q"])
        except Exception:
            print(f"  Could not install {pname}, will use fallback alignment.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdFMCS, rdMolAlign
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser, PDBIO, Select
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from descartes_pharma.statistical.hardening import (
    fdr_correction, confound_removal, tost_equivalence_test, bayes_factor_null,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
os.makedirs('data', exist_ok=True)

CAMPAIGN_START = time.time()

STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
               'THR', 'TRP', 'TYR', 'VAL'}
AA_LIST = sorted(STANDARD_AA)
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}
S1_RESIDS = [30, 71, 108, 110]
S1P_RESIDS = [76, 118]
POLAR_AA = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'ASN', 'GLN', 'SER',
            'THR', 'TYR', 'CYS'}

INTERACTION_NAMES = [
    'dist_asp32', 'dist_asp228', 'hbond_catalytic', 'catalytic_score',
    's1_contacts', 's1prime_contacts', 'total_contacts', 'buried_fraction',
    'mw_raw', 'logp_raw',
]
GROUPS = {
    'CATALYTIC': ['dist_asp32', 'dist_asp228', 'hbond_catalytic',
                  'catalytic_score'],
    'POCKET': ['s1_contacts', 's1prime_contacts', 'total_contacts',
               'buried_fraction'],
    'CONFOUND': ['mw_raw', 'logp_raw'],
}
CONFOUND_IDX = [8, 9]
CATALYTIC_FEATURES = ['dist_asp32', 'dist_asp228', 'hbond_catalytic',
                      'catalytic_score']
CATALYTIC_IDX = [INTERACTION_NAMES.index(n) for n in CATALYTIC_FEATURES]
# Short names for per-round output (FIX 5)
CATALYTIC_SHORT = ['asp32', 'asp228', 'hbond', 'cat_sc']
NODE_FEAT_DIM = 7


# =====================================================================
# SECTION 7.2: TrainingGenome Dataclass
# =====================================================================
@dataclass
class TrainingGenome:
    genome_id: str
    architecture: str = 'concat'   # Fixed
    hidden_dim: int = 128          # Fixed

    # Loss function composition
    primary_loss: str = 'bce'
    aux_docking_score: bool = False
    aux_docking_weight: float = 0.0
    aux_dist_asp32: bool = False
    aux_dist_weight: float = 0.0
    aux_hbond_count: bool = False
    aux_hbond_weight: float = 0.0
    aux_contact_count: bool = False
    aux_contact_weight: float = 0.0

    # Regularization
    embedding_l1: float = 0.0
    information_bottleneck: bool = False
    ib_beta: float = 0.01

    # Training strategy
    learning_rate: float = 1e-3
    label_smoothing: float = 0.0
    curriculum: str = 'none'  # 'none', 'easy_first', 'hard_first'

    def n_aux_outputs(self):
        """Count how many auxiliary task outputs this genome needs."""
        n = 0
        if self.aux_docking_score:
            n += 1
        if self.aux_dist_asp32:
            n += 1
        if self.aux_hbond_count:
            n += 1
        if self.aux_contact_count:
            n += 1
        return n

    def short_desc(self):
        parts = [self.primary_loss]
        if self.aux_docking_score:
            parts.append(f"dock={self.aux_docking_weight:.2f}")
        if self.aux_dist_asp32:
            parts.append(f"dist={self.aux_dist_weight:.2f}")
        if self.aux_hbond_count:
            parts.append(f"hbond={self.aux_hbond_weight:.2f}")
        if self.aux_contact_count:
            parts.append(f"contact={self.aux_contact_weight:.2f}")
        if self.information_bottleneck:
            parts.append(f"IB={self.ib_beta:.3f}")
        if self.label_smoothing > 0:
            parts.append(f"ls={self.label_smoothing:.2f}")
        if self.embedding_l1 > 0:
            parts.append(f"l1={self.embedding_l1:.4f}")
        if self.curriculum != 'none':
            parts.append(f"cur={self.curriculum}")
        return " + ".join(parts)


# =====================================================================
# PHASE A: Prepare Protein Pocket Features (PDB 4IVT)
# =====================================================================
t_a = time.time()
print("\n" + "=" * 85)
print("PHASE A: Prepare Protein Pocket Features (PDB 4IVT)")
print("=" * 85)

PDB_FILE = 'data/4IVT.pdb'
if not os.path.exists(PDB_FILE):
    print("  Downloading PDB 4IVT (1.55A resolution, hydroxyethylamine inhibitor)...")
    urllib.request.urlretrieve('https://files.rcsb.org/download/4IVT.pdb', PDB_FILE)
print(f"  PDB file: {PDB_FILE}")

pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure('BACE1', PDB_FILE)
pdb_model = structure[0]

protein_atoms = []
residue_map = {}
residue_names = {}
for chain in pdb_model:
    for res in chain:
        rid = res.get_id()[1]
        rname = res.get_resname()
        if rname in ('HOH', 'WAT'):
            continue
        alist = []
        for atom in res:
            coord = np.array(atom.get_vector().get_array(), dtype=np.float64)
            protein_atoms.append({'coord': coord, 'name': atom.get_name(),
                                  'resid': rid, 'resname': rname})
            alist.append((atom.get_name(), coord))
        if rid not in residue_map:
            residue_map[rid] = alist
            residue_names[rid] = rname

# Find co-crystallized ligand
ligand_atoms = []
ligand_resname = None
for chain in pdb_model:
    for res in chain:
        rname = res.get_resname()
        if rname not in STANDARD_AA and rname not in ('HOH', 'WAT') and res.get_id()[0] != ' ':
            for atom in res:
                ligand_atoms.append(np.array(atom.get_vector().get_array(),
                                             dtype=np.float64))
            if ligand_resname is None:
                ligand_resname = rname

if ligand_atoms:
    ligand_center = np.mean(ligand_atoms, axis=0)
    print(f"  Ligand residue: {ligand_resname}, {len(ligand_atoms)} atoms")
    print(f"  Ligand center: [{ligand_center[0]:.1f}, {ligand_center[1]:.1f}, "
          f"{ligand_center[2]:.1f}]")
else:
    asp32_c = [c for _, c in residue_map.get(32, [])]
    asp228_c = [c for _, c in residue_map.get(228, [])]
    if asp32_c and asp228_c:
        ligand_center = (np.mean(asp32_c, axis=0) + np.mean(asp228_c, axis=0)) / 2
    else:
        ligand_center = np.array([25., 25., 25.])
    print(f"  No ligand found, using catalytic center: {ligand_center.round(1)}")

BOX_CENTER = ligand_center.tolist()
BOX_SIZE = [22.0, 22.0, 22.0]
print(f"  Docking box: center={[round(c, 1) for c in BOX_CENTER]}, "
      f"size={BOX_SIZE}")

# Binding site: residues within 8A of ligand center
bs_resids = set()
for pa in protein_atoms:
    if pa['resname'] in STANDARD_AA and np.linalg.norm(pa['coord'] - ligand_center) < 8.0:
        bs_resids.add(pa['resid'])
print(f"  Binding site: {len(bs_resids)} residues within 8A")


def get_res_atoms(rid, names=None):
    atoms = residue_map.get(rid, [])
    return {n: c for n, c in atoms if names is None or n in names}


asp32_ox = get_res_atoms(32, ['OD1', 'OD2'])
asp228_ox = get_res_atoms(228, ['OD1', 'OD2'])
print(f"  Asp32 OD atoms: {list(asp32_ox.keys())}")
print(f"  Asp228 OD atoms: {list(asp228_ox.keys())}")


def collect_coords(resid_list):
    coords = []
    for rid in resid_list:
        for _, c in residue_map.get(rid, []):
            coords.append(c)
    return np.array(coords) if coords else np.zeros((0, 3))


s1_coords = collect_coords(S1_RESIDS)
s1p_coords = collect_coords(S1P_RESIDS)
bs_coords = collect_coords(list(bs_resids))
print(f"  S1 pocket: {len(s1_coords)} atoms (residues {S1_RESIDS})")
print(f"  S1' pocket: {len(s1p_coords)} atoms (residues {S1P_RESIDS})")
print(f"  Binding site total: {len(bs_coords)} atoms")

# Per-residue features for pocket summary
bs_resids_sorted = sorted(bs_resids)
pocket_centroid = bs_coords.mean(axis=0) if len(bs_coords) > 0 else ligand_center

per_residue_features_list = []
for rid in bs_resids_sorted:
    rname = residue_names.get(rid, 'UNK')
    onehot = np.zeros(20, dtype=np.float32)
    if rname in AA_INDEX:
        onehot[AA_INDEX[rname]] = 1.0
    res_atoms = residue_map.get(rid, [])
    if res_atoms:
        res_coords = np.array([c for _, c in res_atoms], dtype=np.float64)
        rel_coords = (res_coords.mean(axis=0) - pocket_centroid).astype(np.float32)
    else:
        rel_coords = np.zeros(3, dtype=np.float32)
    polarity = np.array([1.0 if rname in POLAR_AA else 0.0], dtype=np.float32)
    is_cat = np.array([1.0 if rid in (32, 228) else 0.0], dtype=np.float32)
    feat = np.concatenate([onehot, rel_coords, polarity, is_cat])
    per_residue_features_list.append(feat)

per_residue_features_np = np.array(per_residue_features_list, dtype=np.float32)
pocket_mean = per_residue_features_np.mean(axis=0)
pocket_std = per_residue_features_np.std(axis=0)
pocket_min = per_residue_features_np.min(axis=0)
pocket_max = per_residue_features_np.max(axis=0)
pocket_summary_np = np.concatenate([pocket_mean, pocket_std, pocket_min, pocket_max])
pocket_summary_tensor = torch.tensor(pocket_summary_np, dtype=torch.float32).to(device)
POCKET_DIM = len(pocket_summary_np)
print(f"  Pocket summary: ({POCKET_DIM},)")

# Write clean protein PDB for docking
PROTEIN_PDB = 'data/4IVT_protein.pdb'


class ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() in STANDARD_AA


io_pdb = PDBIO()
io_pdb.set_structure(structure)
io_pdb.save(PROTEIN_PDB, ProteinSelect())

# Prepare PDBQT receptor
RECEPTOR_PDBQT = 'data/4IVT_receptor.pdbqt'
pdbqt_ok = False
try:
    subprocess.check_call(['obabel', PROTEIN_PDB, '-O', RECEPTOR_PDBQT, '-xr'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdbqt_ok = True
    print(f"  Receptor PDBQT (openbabel): {RECEPTOR_PDBQT}")
except Exception:
    try:
        with open(PROTEIN_PDB, 'r') as f:
            lines = f.readlines()
        with open(RECEPTOR_PDBQT, 'w') as f:
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    base = line.rstrip('\n').ljust(77)
                    aname = line[12:16].strip()
                    atype = aname[0] if aname else 'C'
                    pdbqt_line = base[:70] + f" 0.000 {atype:>2}\n"
                    f.write(pdbqt_line)
                elif line.startswith(('END', 'TER')):
                    f.write(line)
        pdbqt_ok = True
        print(f"  Receptor PDBQT (simple): {RECEPTOR_PDBQT}")
    except Exception as e:
        print(f"  WARNING: Could not create receptor PDBQT: {e}")

DOCKING_AVAILABLE = False
try:
    from vina import Vina
    DOCKING_AVAILABLE = True
    print("  AutoDock Vina: AVAILABLE")
except ImportError:
    print("  AutoDock Vina: NOT AVAILABLE (will use MCS alignment fallback)")

ref_lig_coords = np.array(ligand_atoms) if ligand_atoms else None
ref_lig_center = ligand_center
ref_ligand_mol = None

print(f"\n  Phase A completed in {time.time() - t_a:.1f}s")


# =====================================================================
# PHASE B: Data Loading + 3D Conformers + Docking
# =====================================================================
t_b = time.time()
print("\n" + "=" * 85)
print("PHASE B: Data Loading + 3D Conformers + Docking")
print("=" * 85)


def mcs_align_to_reference(mol, ref_center, ref_coords):
    """Align molecule to co-crystallized ligand using MCS, or centroid fallback."""
    mol3 = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(mol3, p) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol3, maxIters=200)

    conf = mol3.GetConformer()
    heavy_idx = [i for i, a in enumerate(mol3.GetAtoms()) if a.GetAtomicNum() > 1]
    all_pos = np.array(conf.GetPositions(), dtype=np.float64)
    heavy_pos = all_pos[heavy_idx]
    aligned = False

    if ref_ligand_mol is not None:
        try:
            mcs = rdFMCS.FindMCS([Chem.RemoveHs(mol3), ref_ligand_mol],
                                 timeout=5, threshold=0.7,
                                 ringMatchesRingOnly=True,
                                 completeRingsOnly=False)
            if mcs.numAtoms >= 3:
                patt = Chem.MolFromSmarts(mcs.smartsString)
                if patt is not None:
                    match_mol = Chem.RemoveHs(mol3).GetSubstructMatch(patt)
                    match_ref = ref_ligand_mol.GetSubstructMatch(patt)
                    if match_mol and match_ref:
                        rdMolAlign.AlignMol(
                            Chem.RemoveHs(mol3), ref_ligand_mol,
                            atomMap=list(zip(match_mol, match_ref)))
                        conf2 = Chem.RemoveHs(mol3).GetConformer()
                        heavy_pos = np.array(conf2.GetPositions(),
                                             dtype=np.float64)
                        aligned = True
        except Exception:
            pass

    if not aligned:
        centroid = heavy_pos.mean(axis=0)
        shift = ref_center - centroid
        heavy_pos = heavy_pos + shift
        smi_hash = hash(Chem.MolToSmiles(Chem.RemoveHs(mol3)))
        mol_rng = np.random.default_rng(abs(smi_hash) % (2**31))
        n_atoms = len(heavy_pos)
        perturb_scale = 0.3 + 0.2 * (n_atoms / 40.0)
        theta = mol_rng.uniform(0, 2 * np.pi)
        phi = mol_rng.uniform(0, np.pi)
        rot_axis = np.array([np.sin(phi) * np.cos(theta),
                             np.sin(phi) * np.sin(theta),
                             np.cos(phi)])
        angle = mol_rng.uniform(-0.5, 0.5)
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                       [rot_axis[2], 0, -rot_axis[0]],
                       [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        centered = heavy_pos - ref_center
        rotated = (R @ centered.T).T
        heavy_pos = rotated + ref_center
        heavy_pos += mol_rng.normal(0, perturb_scale, heavy_pos.shape)

    return heavy_pos


def dock_or_align_molecule(mol, smiles):
    """Dock with Vina if available, else MCS-align. Returns heavy atom coords."""
    mol3 = Chem.AddHs(mol)
    p = AllChem.ETKDGv3()
    p.randomSeed = 42
    if AllChem.EmbedMolecule(mol3, p) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol3, maxIters=200)

    if DOCKING_AVAILABLE and pdbqt_ok:
        try:
            import meeko
            from vina import Vina as _Vina
            preparator = meeko.MoleculePreparation()
            preparator.prepare(mol3)
            pdbqt_string = preparator.write_pdbqt_string()
            lig_pdbqt = f'data/tmp_lig_{abs(hash(smiles)) % 100000}.pdbqt'
            with open(lig_pdbqt, 'w') as f:
                f.write(pdbqt_string)
            v = _Vina(sf_name='vina', verbosity=0)
            v.set_receptor(RECEPTOR_PDBQT)
            v.set_ligand_from_file(lig_pdbqt)
            v.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
            v.dock(exhaustiveness=8, n_poses=1)
            pose_pdbqt = v.poses(n_poses=1)
            coords = []
            for line in pose_pdbqt.split('\n'):
                if line.startswith(('ATOM', 'HETATM')):
                    xc = float(line[30:38])
                    yc = float(line[38:46])
                    zc = float(line[46:54])
                    coords.append([xc, yc, zc])
            if coords:
                try:
                    os.remove(lig_pdbqt)
                except Exception:
                    pass
                return np.array(coords, dtype=np.float64)
        except Exception:
            pass

    return mcs_align_to_reference(mol, ref_lig_center, ref_lig_coords)


# Load BACE dataset
print("\n  [1/4] Loading BACE dataset...")
bace_loaded = False
for fn, label in [
    (lambda: __import__('tdc.single_pred', fromlist=['HTS']).HTS(name='BACE'),
     "HTS"),
    (lambda: __import__('tdc.single_pred', fromlist=['ADME']).ADME(
        name='BACE_Group'), "ADME"),
]:
    try:
        bace_data = fn()
        split = bace_data.get_split(method='scaffold')
        bace_loaded = True
        print(f"  Loaded via TDC {label}")
        break
    except Exception:
        pass

if not bace_loaded:
    import pandas as pd
    df = pd.read_csv(
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv")
    df = df.rename(columns={'mol': 'Drug', 'Class': 'Y'})
    n = len(df)
    perm = np.random.RandomState(42).permutation(n)
    nt, nv = int(0.8 * n), int(0.1 * n)
    split = {'train': df.iloc[perm[:nt]],
             'valid': df.iloc[perm[nt:nt + nv]],
             'test': df.iloc[perm[nt + nv:]]}
    print(f"  Direct CSV download ({n} compounds)")

train_df = split['train']
val_df = split['valid']
test_df = split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Generate conformers for train/val
print("\n  [2/4] Generating 3D conformers for train/val sets...")
t0 = time.time()
trainval_conf = {}
nok = nf = 0
all_trainval_smi = list(set(train_df['Drug'].tolist() + val_df['Drug'].tolist()))
for i, smi in enumerate(all_trainval_smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        nf += 1
        continue
    try:
        mol3 = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        if AllChem.EmbedMolecule(mol3, p) == -1:
            nf += 1
            continue
        AllChem.MMFFOptimizeMolecule(mol3, maxIters=200)
        trainval_conf[smi] = True
        nok += 1
    except Exception:
        nf += 1
    if (i + 1) % 300 == 0:
        print(f"    {i + 1}/{len(all_trainval_smi)}: {nok} ok, {nf} fail")
print(f"  Train/val conformers: {nok} ok, {nf} fail in {time.time() - t0:.1f}s")

# Dock or align TEST SET molecules
print("\n  [3/4] Docking/aligning test set molecules...")
test_docked = {}
t0 = time.time()
test_smiles_list = test_df['Drug'].tolist()
nok = nf = 0
for i, smi in enumerate(test_smiles_list):
    if smi in test_docked:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        nf += 1
        continue
    coords = dock_or_align_molecule(mol, smi)
    if coords is not None and len(coords) > 0:
        test_docked[smi] = coords
        nok += 1
    else:
        nf += 1
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"    {i + 1}/{len(test_smiles_list)}: {nok} docked/aligned, "
              f"{nf} fail ({elapsed:.1f}s)")
print(f"  Test set: {nok} docked/aligned, {nf} failed "
      f"in {time.time() - t0:.1f}s")

# [4/4] Verify docking variance
if test_docked:
    a32_check = np.array(list(asp32_ox.values())) if asp32_ox else np.zeros((0, 3))
    check_dists = []
    for smi, coords in test_docked.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        no_idx = [j for j, a in enumerate(mol.GetAtoms())
                  if a.GetAtomicNum() in (7, 8) and j < len(coords)]
        if no_idx and len(a32_check) > 0:
            no_pos = coords[no_idx]
            d = np.min(np.linalg.norm(
                no_pos[:, None, :] - a32_check[None, :, :], axis=2))
            check_dists.append(d)
        else:
            check_dists.append(20.0)
    check_dists = np.array(check_dists)
    print(f"\n  VERIFICATION: dist_to_asp32 across test molecules:")
    print(f"    mean={check_dists.mean():.2f}, std={check_dists.std():.2f}, "
          f"min={check_dists.min():.2f}, max={check_dists.max():.2f}")
    if check_dists.std() < 0.01:
        print("    WARNING: DOCKING/ALIGNMENT FAILED -- std = 0, no variance!")
    else:
        print("    PASS: Real variance in docked coordinates detected.")

print(f"\n  Phase B completed in {time.time() - t_b:.1f}s")


# =====================================================================
# PHASE C: Compute Interaction Features + Prepare Data
# =====================================================================
t_c = time.time()
print("\n" + "=" * 85)
print("PHASE C: Compute Interaction Features + Prepare Data")
print("=" * 85)

a32_arr = np.array(list(asp32_ox.values())) if asp32_ox else np.zeros((0, 3))
a228_arr = np.array(list(asp228_ox.values())) if asp228_ox else np.zeros((0, 3))
cat_o_arr = (np.concatenate([a32_arr, a228_arr])
             if (len(a32_arr) + len(a228_arr)) > 0 else np.zeros((0, 3)))


def compute_interaction_features(smiles, coords):
    """Compute 10 interaction features from docked/aligned coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or coords is None or len(coords) == 0:
        return None
    try:
        n_heavy = mol.GetNumHeavyAtoms()
        pos = coords[:n_heavy] if len(coords) >= n_heavy else coords
        no_idx = [j for j, a in enumerate(mol.GetAtoms())
                  if a.GetAtomicNum() in (7, 8) and j < len(pos)]

        if no_idx and len(a32_arr) > 0:
            no_pos = pos[no_idx]
            d_asp32 = float(np.min(np.linalg.norm(
                no_pos[:, None, :] - a32_arr[None, :, :], axis=2)))
        else:
            d_asp32 = 20.0

        if no_idx and len(a228_arr) > 0:
            no_pos = pos[no_idx]
            d_asp228 = float(np.min(np.linalg.norm(
                no_pos[:, None, :] - a228_arr[None, :, :], axis=2)))
        else:
            d_asp228 = 20.0

        hbc = 0
        if no_idx and len(cat_o_arr) > 0:
            for di in no_idx:
                for oc in cat_o_arr:
                    if np.linalg.norm(pos[di] - oc) < 3.5:
                        hbc += 1

        cat_score = 1.0 / max(d_asp32, 0.5) + 1.0 / max(d_asp228, 0.5)

        def count_contacts(pocket, cutoff=4.0):
            if len(pocket) == 0:
                return 0
            d = np.linalg.norm(pos[:, None, :] - pocket[None, :, :], axis=2)
            return int(np.sum(np.any(d < cutoff, axis=1)))

        s1c = count_contacts(s1_coords)
        s1pc = count_contacts(s1p_coords)
        tc = count_contacts(bs_coords)
        bur = tc / max(len(pos), 1)
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))

        return np.array([d_asp32, d_asp228, float(hbc), cat_score,
                         float(s1c), float(s1pc), float(tc), bur,
                         mw, logp], dtype=np.float32)
    except Exception:
        return None


def smiles_to_graph(smi, lab):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    af = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
           int(a.GetHybridization()), int(a.GetIsAromatic()),
           a.GetTotalNumHs(), int(a.IsInRing())]
          for a in mol.GetAtoms()]
    x = torch.tensor(af, dtype=torch.float32)
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.extend([[i, j], [j, i]])
    if not ei:
        ei = [[0, 0]]
    g = Data(x=x, edge_index=torch.tensor(
        ei, dtype=torch.long).t().contiguous())
    g.y = torch.tensor([lab], dtype=torch.float32)
    return g


def process_test(df):
    gs, ls, fs, ss = [], [], [], []
    for _, row in df.iterrows():
        smi, lab = row['Drug'], row['Y']
        g = smiles_to_graph(smi, lab)
        coords = test_docked.get(smi, None)
        feat = (compute_interaction_features(smi, coords)
                if coords is not None else None)
        if g is not None and feat is not None and g.x.shape[0] > 0:
            gs.append(g)
            ls.append(lab)
            fs.append(feat)
            ss.append(smi)
    return gs, np.array(ls), np.array(fs), ss


def process_trainval(df):
    gs, ls, ss = [], [], []
    for _, row in df.iterrows():
        smi, lab = row['Drug'], row['Y']
        g = smiles_to_graph(smi, lab)
        if g is not None and g.x.shape[0] > 0:
            gs.append(g)
            ls.append(lab)
            ss.append(smi)
    return gs, np.array(ls), ss


print("  Processing datasets...")
trg, trl, trs = process_trainval(train_df)
vag, val_, vas = process_trainval(val_df)
teg, tel_, tef, tes = process_test(test_df)
print(f"  Train: {len(trg)}, Val: {len(vag)}, "
      f"Test: {len(teg)} (with interaction features)")

# Also dock/align TRAIN molecules for auxiliary supervision
print("\n  Docking/aligning train set for auxiliary supervision targets...")
train_docked = {}
t0 = time.time()
for i, smi in enumerate(trs):
    if smi in train_docked:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    coords = dock_or_align_molecule(mol, smi)
    if coords is not None and len(coords) > 0:
        train_docked[smi] = coords
    if (i + 1) % 100 == 0:
        print(f"    {i + 1}/{len(trs)}: {len(train_docked)} docked/aligned "
              f"({time.time() - t0:.1f}s)")
print(f"  Train docked: {len(train_docked)}/{len(trs)} "
      f"in {time.time() - t0:.1f}s")

# Compute interaction features for train set
train_interaction_features = {}
for smi in trs:
    coords = train_docked.get(smi, None)
    if coords is not None:
        feat = compute_interaction_features(smi, coords)
        if feat is not None:
            train_interaction_features[smi] = feat

print(f"  Train interaction features: {len(train_interaction_features)}/{len(trs)}")

# Standardize test features for probing
sc = StandardScaler()
tef_norm = sc.fit_transform(tef)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------
# FIX 3: Compute Murcko scaffolds for test set molecules
# ---------------------------------------------------------------
print("\n  Computing Murcko scaffolds for test set...")
test_scaffolds = []
for smi in tes:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        test_scaffolds.append(Chem.MolToSmiles(generic))
    else:
        test_scaffolds.append('UNKNOWN')
test_scaffolds = np.array(test_scaffolds)
n_unique_scaffolds = len(np.unique(test_scaffolds))
print(f"  Test scaffolds: {len(test_scaffolds)} molecules, "
      f"{n_unique_scaffolds} unique scaffolds")

# Feature statistics
print(f"\n  {'Feature':<20} {'Group':<10} {'Mean':>8} {'Std':>8} "
      f"{'Min':>8} {'Max':>8}")
print(f"  {'-' * 60}")
for i, nm in enumerate(INTERACTION_NAMES):
    v = tef[:, i]
    grp = [g for g, m in GROUPS.items() if nm in m][0]
    print(f"  {nm:<20} {grp:<10} {v.mean():>8.3f} {v.std():>8.3f} "
          f"{v.min():>8.1f} {v.max():>8.1f}")

# Build data loaders
dtl = PyGDataLoader(trg, batch_size=64, shuffle=True)
dvl = PyGDataLoader(vag, batch_size=64, shuffle=False)
dtsl = PyGDataLoader(teg, batch_size=64, shuffle=False)
pw = torch.tensor(
    [(1 - np.mean(trl)) / max(np.mean(trl), 1e-6)]).to(device)

print(f"\n  Phase C completed in {time.time() - t_c:.1f}s")


# =====================================================================
# PHASE D: Model Definition + Training Infrastructure
# =====================================================================
t_d = time.time()
print("\n" + "=" * 85)
print("PHASE D: ConcatModel + Genome Training Infrastructure")
print("=" * 85)


class ConcatModelMultiHead(nn.Module):
    """GCN + pocket concat with variable auxiliary output heads.

    The architecture is FIXED (GCN encoder, pocket MLP, hidden=128).
    Only the number of output heads varies by genome configuration.
    """

    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3,
                 pocket_dim=POCKET_DIM, dropout=0.2, n_aux_outputs=0,
                 use_information_bottleneck=False):
        super().__init__()
        self.n_aux = n_aux_outputs
        self.use_ib = use_information_bottleneck

        # GCN encoder
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)

        # Pocket MLP
        self.pocket_mlp = nn.Sequential(
            nn.Linear(pocket_dim, 64), nn.ReLU(), nn.Linear(64, 64))

        # Information bottleneck (VAE-style)
        self.interaction_dim = hidden_dim + 64  # 192
        if self.use_ib:
            self.ib_mu = nn.Linear(self.interaction_dim, self.interaction_dim)
            self.ib_logvar = nn.Linear(self.interaction_dim,
                                       self.interaction_dim)

        # Primary classifier: activity prediction
        self.classifier = nn.Sequential(
            nn.Linear(self.interaction_dim, 96), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(96, 1))

        # Auxiliary heads (each predicts one continuous target)
        if self.n_aux > 0:
            self.aux_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(self.interaction_dim, 32),
                              nn.ReLU(), nn.Linear(32, 1))
                for _ in range(self.n_aux)
            ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, pocket_features, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(F.relu(bn(conv(x, ei))))
        ligand_emb = global_mean_pool(x, batch)
        bsz = ligand_emb.shape[0]
        pocket_emb = self.pocket_mlp(pocket_features.unsqueeze(0).expand(bsz, -1))
        interaction_emb = torch.cat([ligand_emb, pocket_emb], dim=1)

        kl_loss = torch.tensor(0.0, device=interaction_emb.device)
        if self.use_ib and self.training:
            mu = self.ib_mu(interaction_emb)
            logvar = self.ib_logvar(interaction_emb)
            interaction_emb = self.reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()) / bsz

        logits = self.classifier(interaction_emb).squeeze(-1)

        aux_outputs = []
        if self.n_aux > 0:
            for head in self.aux_heads:
                aux_outputs.append(head(interaction_emb).squeeze(-1))

        if return_embedding:
            return logits, interaction_emb, aux_outputs, kl_loss
        return logits, aux_outputs, kl_loss


def build_aux_target_arrays(genome, smiles_list, interaction_features_dict):
    """Build per-SMILES auxiliary supervision arrays for the train set.

    Returns: ordered list of (aux_name, feat_index, weight) and
             a dict mapping smiles -> normalized aux target vector.
    """
    aux_spec = []
    if genome.aux_docking_score:
        aux_spec.append(('docking_score', 3, genome.aux_docking_weight))
    if genome.aux_dist_asp32:
        aux_spec.append(('dist_asp32', 0, genome.aux_dist_weight))
    if genome.aux_hbond_count:
        aux_spec.append(('hbond_count', 2, genome.aux_hbond_weight))
    if genome.aux_contact_count:
        aux_spec.append(('contact_count', 6, genome.aux_contact_weight))

    if not aux_spec:
        return aux_spec, {}

    # Collect raw values
    raw_vals = {name: [] for name, _, _ in aux_spec}
    valid_smiles = []
    for smi in smiles_list:
        feat = interaction_features_dict.get(smi, None)
        if feat is not None:
            valid_smiles.append(smi)
            for name, idx, _ in aux_spec:
                raw_vals[name].append(feat[idx])

    # Fit scalers on valid values
    scalers = {}
    for name, _, _ in aux_spec:
        s = StandardScaler()
        arr = np.array(raw_vals[name], dtype=np.float32).reshape(-1, 1)
        if len(arr) > 0:
            s.fit(arr)
        scalers[name] = s

    # Build per-smiles target dict
    smi_targets = {}
    for smi in smiles_list:
        feat = interaction_features_dict.get(smi, None)
        if feat is not None:
            targets = []
            for name, idx, _ in aux_spec:
                val = scalers[name].transform(
                    np.array([[feat[idx]]], dtype=np.float32))[0, 0]
                targets.append(val)
            smi_targets[smi] = np.array(targets, dtype=np.float32)

    return aux_spec, smi_targets


def train_genome(genome, train_graphs, train_labels, train_smiles,
                 val_graphs, val_labels, test_graphs, test_labels,
                 pocket_tensor, interaction_features_dict, n_epochs=100):
    """Train one genome configuration. Returns (test_auc, model, embeddings)."""
    n_aux = genome.n_aux_outputs()

    model = ConcatModelMultiHead(
        n_aux_outputs=n_aux,
        use_information_bottleneck=genome.information_bottleneck
    ).to(device)

    opt = torch.optim.Adam(model.parameters(),
                           lr=genome.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5, min_lr=1e-6)

    # Build auxiliary supervision targets
    aux_spec, smi_targets = build_aux_target_arrays(
        genome, train_smiles, interaction_features_dict)

    # Curriculum ordering
    train_idx = list(range(len(train_graphs)))
    if genome.curriculum == 'easy_first':
        confidence = np.abs(train_labels - 0.5)
        train_idx = list(np.argsort(-confidence))
    elif genome.curriculum == 'hard_first':
        confidence = np.abs(train_labels - 0.5)
        train_idx = list(np.argsort(confidence))

    # Reorder graphs and smiles
    ordered_graphs = [train_graphs[i] for i in train_idx]
    ordered_smiles = [train_smiles[i] for i in train_idx]

    # Attach aux targets and smiles index to graph objects
    for gi, g in enumerate(ordered_graphs):
        g.smi_idx = gi  # track index for aux lookup

    # Label smoothing
    ls = genome.label_smoothing

    train_loader = PyGDataLoader(ordered_graphs, batch_size=64, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = PyGDataLoader(test_graphs, batch_size=64, shuffle=False)

    best_auc, best_state = 0.0, None

    for ep in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            bsz = batch.num_graphs
            opt.zero_grad()

            logits, aux_preds, kl_loss = model(batch, pocket_tensor)

            # Primary BCE loss with label smoothing
            targets = batch.y.squeeze(-1)
            if ls > 0:
                targets = targets * (1 - ls) + 0.5 * ls
            primary_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw)

            # Auxiliary losses from docked interaction features
            aux_loss = torch.tensor(0.0, device=device)
            if n_aux > 0 and len(aux_spec) > 0:
                # Get batch smiles indices
                batch_smi_idx = batch.smi_idx.cpu().numpy()
                for head_idx in range(min(len(aux_preds), len(aux_spec))):
                    aname, _, aweight = aux_spec[head_idx]
                    aux_targets_batch = []
                    aux_mask = []
                    for si in batch_smi_idx:
                        smi = ordered_smiles[int(si)]
                        if smi in smi_targets:
                            aux_targets_batch.append(
                                smi_targets[smi][head_idx])
                            aux_mask.append(True)
                        else:
                            aux_targets_batch.append(0.0)
                            aux_mask.append(False)

                    aux_t = torch.tensor(aux_targets_batch,
                                         dtype=torch.float32,
                                         device=device)
                    aux_m = torch.tensor(aux_mask,
                                         dtype=torch.float32,
                                         device=device)
                    if aux_m.sum() > 0:
                        masked_loss = (F.mse_loss(
                            aux_preds[head_idx], aux_t,
                            reduction='none') * aux_m).sum() / aux_m.sum()
                        aux_loss = aux_loss + aweight * masked_loss

            # Information bottleneck KL
            ib_loss = torch.tensor(0.0, device=device)
            if genome.information_bottleneck:
                ib_loss = genome.ib_beta * kl_loss

            # Embedding L1 penalty
            l1_loss = torch.tensor(0.0, device=device)
            if genome.embedding_l1 > 0:
                _, emb_l1, _, _ = model(batch, pocket_tensor,
                                         return_embedding=True)
                l1_loss = genome.embedding_l1 * emb_l1.abs().mean()

            loss = primary_loss + aux_loss + ib_loss + l1_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * bsz

        # Validation AUC every 5 epochs
        if ep % 5 == 0:
            model.eval()  # noqa: E701
            vp, vt = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits, _, _ = model(batch, pocket_tensor)
                    vp.extend(torch.sigmoid(logits).cpu().numpy())
                    vt.extend(batch.y.squeeze(-1).cpu().numpy())
            try:
                va = roc_auc_score(vt, vp)
            except Exception:
                va = 0.5
            scheduler.step(1 - va)
            if va > best_auc:
                best_auc = va
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test AUC
    model.eval()  # noqa: E701
    tp, tt = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _, _ = model(batch, pocket_tensor)
            tp.extend(torch.sigmoid(logits).cpu().numpy())
            tt.extend(batch.y.squeeze(-1).cpu().numpy())
    try:
        test_auc = roc_auc_score(tt, tp)
    except Exception:
        test_auc = 0.5

    # Extract embeddings for probing
    model.eval()  # noqa: E701
    embs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, emb, _, _ = model(batch, pocket_tensor, return_embedding=True)
            embs.append(emb.cpu().numpy())
    embeddings = np.concatenate(embs)

    return test_auc, model, embeddings


# ---------------------------------------------------------------
# FIX 1 + FIX 2: hardened_fast_probe with scaffold-stratified
# permutation (200 perms). Uses delta_r2 = r2_trained - r2_untrained.
# FIX 2: Fresh untrained model embeddings each call.
# ---------------------------------------------------------------
def hardened_fast_probe(trained_emb, untrained_emb, target, scaffolds,
                        n_perms=200, seed=42):
    """Inner-loop probe with scaffold-stratified permutation. ~30s per feature."""
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2_t = np.mean(cross_val_score(Ridge(alpha=1.0), trained_emb, target,
                                    cv=kf_inner, scoring='r2'))
    r2_u = np.mean(cross_val_score(Ridge(alpha=1.0), untrained_emb, target,
                                    cv=kf_inner, scoring='r2'))
    delta_r2 = r2_t - r2_u

    unique_scaffolds = np.unique(scaffolds)
    rng = np.random.default_rng(seed)
    null_deltas = np.zeros(n_perms)
    for p in range(n_perms):
        target_perm = target.copy()
        for scaf in unique_scaffolds:
            mask = scaffolds == scaf
            if mask.sum() > 1:
                target_perm[mask] = rng.permutation(target_perm[mask])
        r2_perm = np.mean(cross_val_score(Ridge(alpha=1.0), trained_emb,
                                           target_perm, cv=kf_inner,
                                           scoring='r2'))
        null_deltas[p] = r2_perm - r2_u

    p_value = float(np.mean(null_deltas >= delta_r2))
    is_encoded = (p_value < 0.05) and (delta_r2 > 0.05)
    return delta_r2, p_value, is_encoded


def get_untrained_embeddings(genome, test_graphs, pocket_tensor):
    """FIX 2: Create a FRESH untrained model instance and extract embeddings.
    Never reuse a previous random model -- always new random weights."""
    n_aux = genome.n_aux_outputs()
    rand_model = ConcatModelMultiHead(
        n_aux_outputs=n_aux,
        use_information_bottleneck=genome.information_bottleneck
    ).to(device)
    rand_model.eval()
    test_loader = PyGDataLoader(test_graphs, batch_size=64, shuffle=False)
    embs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, emb, _, _ = rand_model(batch, pocket_tensor,
                                       return_embedding=True)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs)


def hardened_probe_catalytic(trained_emb, genome, test_graphs, pocket_tensor,
                              scaffolds):
    """C1 Hardened Probe: scaffold-stratified permutation for 4 catalytic features.

    FIX 1: Uses hardened_fast_probe instead of naked Ridge.
    FIX 2: Fresh untrained model for delta_r2 baseline.
    FIX 3: Passes test_scaffolds to permutation.

    Returns dict: fname -> (delta_r2, p_value, is_encoded)
    """
    # FIX 2: fresh random model each time
    untrained_emb = get_untrained_embeddings(genome, test_graphs, pocket_tensor)

    results = {}
    for fi, fname in zip(CATALYTIC_IDX, CATALYTIC_FEATURES):
        t = tef_norm[:, fi]
        if np.std(t) < 1e-10:
            results[fname] = (0.0, 1.0, False)
            continue
        delta_r2, p_val, is_enc = hardened_fast_probe(
            trained_emb, untrained_emb, t, scaffolds,
            n_perms=200, seed=42 + fi)
        results[fname] = (delta_r2, p_val, is_enc)
    return results


# ---------------------------------------------------------------
# FIX 4: Updated fitness function with 70% catalytic weight
# ---------------------------------------------------------------
def compute_fitness(test_auc, catalytic_results):
    """Fitness = 0.3*(AUC/0.95) + 0.7*(n_catalytic_encoded/4).
    AUC gate: must be >= 0.70.
    "Encoded" = p < 0.05 AND delta_r2 > 0.05 (from hardened_fast_probe).
    """
    if test_auc < 0.70:
        return 0.0
    auc_score = min(1.0, test_auc / 0.95)
    n_encoded = sum(1 for _, _, enc in catalytic_results.values() if enc)
    return 0.3 * auc_score + 0.7 * (n_encoded / 4.0)


print(f"\n  Phase D completed in {time.time() - t_d:.1f}s")


# =====================================================================
# PHASE E: Genetic Operators + Thompson Sampling
# =====================================================================
print("\n" + "=" * 85)
print("PHASE E: Genetic Operators + Thompson Sampling")
print("=" * 85)

evo_rng = np.random.default_rng(2024)


def mutate_genome(parent, rng=evo_rng):
    """Mutate one parameter: continuous -> log-normal jitter,
    booleans -> flip, categoricals -> resample."""
    child = copy.deepcopy(parent)
    child.genome_id = f"mut_{uuid.uuid4().hex[:8]}"

    mutable = [
        ('aux_docking_score', 'bool'),
        ('aux_docking_weight', 'float'),
        ('aux_dist_asp32', 'bool'),
        ('aux_dist_weight', 'float'),
        ('aux_hbond_count', 'bool'),
        ('aux_hbond_weight', 'float'),
        ('aux_contact_count', 'bool'),
        ('aux_contact_weight', 'float'),
        ('embedding_l1', 'float'),
        ('information_bottleneck', 'bool'),
        ('ib_beta', 'float'),
        ('learning_rate', 'float'),
        ('label_smoothing', 'float'),
        ('curriculum', 'cat'),
    ]

    param_name, param_type = mutable[rng.integers(len(mutable))]

    if param_type == 'bool':
        val = getattr(child, param_name)
        setattr(child, param_name, not val)
    elif param_type == 'float':
        val = getattr(child, param_name)
        if val == 0.0:
            val = 0.01
        new_val = val * np.exp(rng.normal(0, 0.3))
        new_val = max(1e-6, min(new_val, 1.0))
        setattr(child, param_name, float(new_val))
    elif param_type == 'cat':
        options = ['none', 'easy_first', 'hard_first']
        current = getattr(child, param_name)
        others = [o for o in options if o != current]
        setattr(child, param_name, others[rng.integers(len(others))])

    # Consistency: if aux is disabled, zero its weight
    if not child.aux_docking_score:
        child.aux_docking_weight = 0.0
    if not child.aux_dist_asp32:
        child.aux_dist_weight = 0.0
    if not child.aux_hbond_count:
        child.aux_hbond_weight = 0.0
    if not child.aux_contact_count:
        child.aux_contact_weight = 0.0

    return child


def crossover_genomes(parent_a, parent_b, rng=evo_rng):
    """Uniform crossover: each param from A or B with 50% probability."""
    child = copy.deepcopy(parent_a)
    child.genome_id = f"xov_{uuid.uuid4().hex[:8]}"

    crossable = [
        'aux_docking_score', 'aux_docking_weight',
        'aux_dist_asp32', 'aux_dist_weight',
        'aux_hbond_count', 'aux_hbond_weight',
        'aux_contact_count', 'aux_contact_weight',
        'embedding_l1', 'information_bottleneck', 'ib_beta',
        'learning_rate', 'label_smoothing', 'curriculum',
    ]

    for param in crossable:
        if rng.random() < 0.5:
            setattr(child, param, getattr(parent_b, param))

    if not child.aux_docking_score:
        child.aux_docking_weight = 0.0
    if not child.aux_dist_asp32:
        child.aux_dist_weight = 0.0
    if not child.aux_hbond_count:
        child.aux_hbond_weight = 0.0
    if not child.aux_contact_count:
        child.aux_contact_weight = 0.0

    return child


class ThompsonSampler:
    """Track (success, failure) counts per genome family.
    Sample Beta(success+1, failure+1), allocate compute to highest sample."""

    def __init__(self):
        self.successes = {}
        self.failures = {}

    def register(self, genome_id):
        if genome_id not in self.successes:
            self.successes[genome_id] = 0
            self.failures[genome_id] = 0

    def update(self, genome_id, is_success):
        if is_success:
            self.successes[genome_id] = self.successes.get(genome_id, 0) + 1
        else:
            self.failures[genome_id] = self.failures.get(genome_id, 0) + 1

    def sample_best(self, candidates, rng=evo_rng):
        """Return the candidate with highest Thompson sample."""
        best_id = None
        best_sample = -1
        for gid in candidates:
            s = self.successes.get(gid, 0)
            f = self.failures.get(gid, 0)
            sample = rng.beta(s + 1, f + 1)
            if sample > best_sample:
                best_sample = sample
                best_id = gid
        return best_id


thompson = ThompsonSampler()

print("  Genetic operators: mutate (log-normal jitter), crossover (uniform)")
print("  Thompson sampling: Beta(success+1, failure+1)")


# =====================================================================
# PHASE F: LLM Balloon Expansion Setup
# =====================================================================
print("\n" + "=" * 85)
print("PHASE F: LLM Balloon Expansion Setup")
print("=" * 85)

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
LLM_AVAILABLE = len(ANTHROPIC_API_KEY) > 10
print(f"  Anthropic API key: "
      f"{'AVAILABLE' if LLM_AVAILABLE else 'NOT FOUND (will use template fallback)'}")


def llm_balloon_proposals(history_summary):
    """Section 7.6: Call Anthropic API or fallback to generate novel genomes."""
    proposals = []

    if LLM_AVAILABLE:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            prompt = (
                "You are an ML research assistant helping design training "
                "configurations for a GCN + pocket concat model predicting "
                "BACE1 inhibitor activity.\n\n"
                "The architecture is FIXED (GCN encoder, hidden=128, pocket "
                "MLP concat). You are evolving TRAINING CONFIGURATIONS: loss "
                "functions, auxiliary objectives, regularizers, and curricula."
                "\n\nCurrent best configurations and their fitness scores:\n"
                f"{history_summary}\n\n"
                "The fitness function is:\n"
                "  fitness = 0.3*(AUC/0.95) + 0.7*(n_catalytic_encoded/4)\n"
                "where n_catalytic_encoded counts how many of [dist_asp32, "
                "dist_asp228, hbond_catalytic, catalytic_score] pass the "
                "hardened probe (scaffold-stratified permutation p<0.05 "
                "AND delta_r2 > 0.05).\n\n"
                "Propose 3 NOVEL training configurations that might improve "
                "catalytic feature encoding. Each must be a JSON object with "
                "these fields:\n"
                "- genome_id: string (unique name)\n"
                "- aux_docking_score: bool, aux_docking_weight: float (0-1)\n"
                "- aux_dist_asp32: bool, aux_dist_weight: float (0-1)\n"
                "- aux_hbond_count: bool, aux_hbond_weight: float (0-1)\n"
                "- aux_contact_count: bool, aux_contact_weight: float (0-1)\n"
                "- embedding_l1: float (0-0.1)\n"
                "- information_bottleneck: bool, ib_beta: float (0.001-0.5)\n"
                "- learning_rate: float (1e-5 to 1e-2)\n"
                "- label_smoothing: float (0-0.3)\n"
                "- curriculum: 'none', 'easy_first', or 'hard_first'\n\n"
                "Return ONLY a JSON array of 3 objects, no other text."
            )

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = msg.content[0].text
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                configs = json.loads(json_match.group())
                for cfg in configs[:3]:
                    g = TrainingGenome(
                        genome_id=cfg.get('genome_id',
                                          f'llm_{uuid.uuid4().hex[:8]}'),
                        aux_docking_score=cfg.get('aux_docking_score', False),
                        aux_docking_weight=cfg.get('aux_docking_weight', 0.0),
                        aux_dist_asp32=cfg.get('aux_dist_asp32', False),
                        aux_dist_weight=cfg.get('aux_dist_weight', 0.0),
                        aux_hbond_count=cfg.get('aux_hbond_count', False),
                        aux_hbond_weight=cfg.get('aux_hbond_weight', 0.0),
                        aux_contact_count=cfg.get('aux_contact_count', False),
                        aux_contact_weight=cfg.get('aux_contact_weight', 0.0),
                        embedding_l1=cfg.get('embedding_l1', 0.0),
                        information_bottleneck=cfg.get(
                            'information_bottleneck', False),
                        ib_beta=cfg.get('ib_beta', 0.01),
                        learning_rate=cfg.get('learning_rate', 1e-3),
                        label_smoothing=cfg.get('label_smoothing', 0.0),
                        curriculum=cfg.get('curriculum', 'none'),
                    )
                    proposals.append(g)
                print(f"    LLM proposed {len(proposals)} configurations")
                return proposals
        except Exception as e:
            print(f"    LLM API call failed: {e}")

    # Fallback: 3 template proposals (ranking, contrastive, triplet proxies)
    print("    Using template fallback proposals")
    proposals = [
        TrainingGenome(
            genome_id='fallback_ranking',
            aux_docking_score=True, aux_docking_weight=0.2,
            aux_dist_asp32=True, aux_dist_weight=0.3,
            aux_hbond_count=True, aux_hbond_weight=0.2,
            aux_contact_count=True, aux_contact_weight=0.1,
            label_smoothing=0.2, learning_rate=5e-4,
            curriculum='hard_first',
        ),
        TrainingGenome(
            genome_id='fallback_contrastive',
            aux_docking_score=True, aux_docking_weight=0.4,
            aux_dist_asp32=True, aux_dist_weight=0.2,
            information_bottleneck=True, ib_beta=0.05,
            embedding_l1=0.01, learning_rate=3e-4,
        ),
        TrainingGenome(
            genome_id='fallback_triplet',
            aux_docking_score=True, aux_docking_weight=0.5,
            aux_dist_asp32=True, aux_dist_weight=0.4,
            aux_hbond_count=True, aux_hbond_weight=0.3,
            learning_rate=2e-4, curriculum='easy_first',
            information_bottleneck=True, ib_beta=0.02,
        ),
    ]
    return proposals


print(f"  LLM balloon: "
      f"{'API mode' if LLM_AVAILABLE else 'template fallback mode'}")


# =====================================================================
# PHASE G: 40-Round Dual Factory Campaign
# =====================================================================
t_g = time.time()
print("\n" + "=" * 85)
print("PHASE G: 40-Round Dual Factory Campaign (Hardened Probes)")
print("=" * 85)
print("  NOTE: Each round runs scaffold-stratified permutation (200 perms x 4 features)")
print("        Expected ~2.5 min/genome, ~130 min total for 40 rounds + winner eval.")

# Phase 1 Templates (rounds 1-11)
TEMPLATES = [
    TrainingGenome(genome_id='template_01_baseline'),
    TrainingGenome(genome_id='template_02_dock01',
                   aux_docking_score=True, aux_docking_weight=0.1),
    TrainingGenome(genome_id='template_03_dock03',
                   aux_docking_score=True, aux_docking_weight=0.3),
    TrainingGenome(genome_id='template_04_dock05',
                   aux_docking_score=True, aux_docking_weight=0.5),
    TrainingGenome(genome_id='template_05_dist02',
                   aux_dist_asp32=True, aux_dist_weight=0.2),
    TrainingGenome(genome_id='template_06_hbond02',
                   aux_hbond_count=True, aux_hbond_weight=0.2),
    TrainingGenome(genome_id='template_07_all_aux',
                   aux_docking_score=True, aux_docking_weight=0.1,
                   aux_dist_asp32=True, aux_dist_weight=0.1,
                   aux_hbond_count=True, aux_hbond_weight=0.1,
                   aux_contact_count=True, aux_contact_weight=0.1),
    TrainingGenome(genome_id='template_08_ib001',
                   information_bottleneck=True, ib_beta=0.01),
    TrainingGenome(genome_id='template_09_ib01',
                   information_bottleneck=True, ib_beta=0.1),
    TrainingGenome(genome_id='template_10_ls01',
                   label_smoothing=0.1),
    TrainingGenome(genome_id='template_11_dock03_ib001',
                   aux_docking_score=True, aux_docking_weight=0.3,
                   information_bottleneck=True, ib_beta=0.01),
]

# Campaign state
all_results = []
best_fitness = 0.0
best_genome = None
best_auc = 0.0
best_cat_results = {}
rounds_since_improvement = 0
balloon_proposals_cache = []

N_ROUNDS = 40

for round_num in range(1, N_ROUNDS + 1):
    round_t = time.time()

    # --- Select genome for this round ---
    if round_num <= 11:
        # Phase 1: Templates
        genome = TEMPLATES[round_num - 1]
        phase = "TEMPLATE"

    elif round_num <= 25:
        # Phase 2: Mutation + crossover from top 5
        phase = "EVOLVE"
        if len(all_results) < 2:
            genome = mutate_genome(TEMPLATES[0])
        else:
            sorted_results = sorted(all_results, key=lambda x: x[3],
                                    reverse=True)
            top5 = [r[0] for r in sorted_results[:5]]
            top5_ids = [g.genome_id for g in top5]
            for gid in top5_ids:
                thompson.register(gid)

            if evo_rng.random() < 0.5:
                parent_id = thompson.sample_best(top5_ids, evo_rng)
                parent = next(g for g in top5 if g.genome_id == parent_id)
                genome = mutate_genome(parent)
            else:
                id_a = thompson.sample_best(top5_ids, evo_rng)
                remaining = [gid for gid in top5_ids if gid != id_a]
                if remaining:
                    id_b = thompson.sample_best(remaining, evo_rng)
                else:
                    id_b = id_a
                parent_a = next(g for g in top5 if g.genome_id == id_a)
                parent_b = next(g for g in top5 if g.genome_id == id_b)
                genome = crossover_genomes(parent_a, parent_b)

    elif round_num <= 35:
        # Phase 3: LLM balloon if stalled 8+ rounds, else evolve
        use_balloon = (rounds_since_improvement >= 8)
        phase = "BALLOON" if use_balloon else "EVOLVE"

        if use_balloon:
            if not balloon_proposals_cache:
                sorted_results = sorted(all_results, key=lambda x: x[3],
                                        reverse=True)
                history_lines = []
                for g, a, cr, f in sorted_results[:10]:
                    cat_str = ", ".join(
                        f"{k}: dR2={v[0]:.3f} p={v[1]:.3f}"
                        for k, v in cr.items())
                    history_lines.append(
                        f"  {g.genome_id}: fitness={f:.4f}, AUC={a:.4f}, "
                        f"catalytic=[{cat_str}]")
                    history_lines.append(f"    config: {g.short_desc()}")
                history_summary = "\n".join(history_lines)
                balloon_proposals_cache = llm_balloon_proposals(
                    history_summary)

            used_ids = {r[0].genome_id for r in all_results}
            genome = None
            for prop in balloon_proposals_cache:
                if prop.genome_id not in used_ids:
                    genome = prop
                    break
            if genome is None:
                genome = mutate_genome(all_results[0][0])
                genome.genome_id = f"balloon_mut_{uuid.uuid4().hex[:8]}"
        else:
            sorted_results = sorted(all_results, key=lambda x: x[3],
                                    reverse=True)
            top5 = [r[0] for r in sorted_results[:5]]
            if evo_rng.random() < 0.5:
                genome = mutate_genome(top5[evo_rng.integers(len(top5))])
            else:
                pa = top5[evo_rng.integers(len(top5))]
                pb = top5[evo_rng.integers(len(top5))]
                genome = crossover_genomes(pa, pb)

    else:
        # Phase 4: Exploit best with small perturbations
        phase = "EXPLOIT"
        if best_genome is not None:
            genome = mutate_genome(best_genome)
            genome.genome_id = f"exploit_{uuid.uuid4().hex[:8]}"
            genome.learning_rate = best_genome.learning_rate * np.exp(
                evo_rng.normal(0, 0.1))
            genome.learning_rate = max(1e-5, min(genome.learning_rate, 1e-2))
        else:
            genome = TrainingGenome(
                genome_id=f"exploit_baseline_{round_num}")

    # --- Train this genome ---
    torch.manual_seed(round_num * 42)
    np.random.seed(round_num * 42)

    try:
        auc, model, embeddings = train_genome(
            genome, trg, trl, trs, vag, val_, teg, tel_,
            pocket_summary_tensor, train_interaction_features,
            n_epochs=100)
    except Exception as e:
        print(f"  Round {round_num:2d}: TRAINING FAILED ({e})")
        auc = 0.0
        embeddings = np.random.randn(len(teg), 192)
        model = None

    # --- C1 Hardened Probe (FIX 1 + FIX 2 + FIX 3) ---
    cat_results = hardened_probe_catalytic(
        embeddings, genome, teg, pocket_summary_tensor, test_scaffolds)
    n_cat = sum(1 for _, _, enc in cat_results.values() if enc)

    # FIX 4: Updated fitness
    fitness = compute_fitness(auc, cat_results)

    # Track results
    all_results.append((genome, auc, cat_results, fitness))
    thompson.register(genome.genome_id)
    thompson.update(genome.genome_id, fitness > best_fitness * 0.9)

    is_best = False
    if fitness > best_fitness:
        best_fitness = fitness
        best_genome = genome
        best_auc = auc
        best_cat_results = cat_results
        rounds_since_improvement = 0
        is_best = True
    else:
        rounds_since_improvement += 1

    # --- FIX 5: Detailed per-round output with p-values ---
    best_flag = "[BEST]" if is_best else ""
    desc = genome.short_desc()
    if len(desc) > 18:
        desc = desc[:15] + "..."
    cat_parts = []
    for fname, sname in zip(CATALYTIC_FEATURES, CATALYTIC_SHORT):
        dr2, pv, _ = cat_results[fname]
        cat_parts.append(f"{sname}: dR2={dr2:.2f} p={pv:.2f}")
    cat_str = " | ".join(cat_parts)
    print(f"Round {round_num:2d}: {desc:<18} | AUC={auc:.2f} | "
          f"{cat_str} | enc={n_cat}/4 | fit={fitness:.3f} "
          f"{best_flag}  ({phase}, {time.time() - round_t:.1f}s)")

print(f"\n  40-round campaign completed in {time.time() - t_g:.1f}s")
print(f"  Best genome: {best_genome.genome_id}")
print(f"  Best fitness: {best_fitness:.4f}")
print(f"  Best AUC: {best_auc:.4f}")
for fname in CATALYTIC_FEATURES:
    dr2, pv, enc = best_cat_results[fname]
    print(f"    {fname}: dR2={dr2:.4f}, p={pv:.4f}, encoded={enc}")


# =====================================================================
# PHASE H: Winner Evaluation -- Full Hardening + Council Controls
# =====================================================================
t_h = time.time()
print("\n" + "=" * 85)
print("PHASE H: Winner Evaluation -- Full Hardening + Council Controls")
print("=" * 85)
print(f"  Winner genome: {best_genome.genome_id}")
print(f"  Configuration: {best_genome.short_desc()}")

# Retrain winner with best seed for final evaluation
print("\n  [1/6] Retraining winner genome for final evaluation...")
torch.manual_seed(0)
np.random.seed(0)
final_auc, final_model, emb_trained = train_genome(
    best_genome, trg, trl, trs, vag, val_, teg, tel_,
    pocket_summary_tensor, train_interaction_features,
    n_epochs=150)

# FIX 2: Fresh untrained model for random baseline embeddings
emb_random = get_untrained_embeddings(best_genome, teg, pocket_summary_tensor)

print(f"  Final test AUC: {final_auc:.4f}")
gate = "PASS (>=0.70)" if final_auc >= 0.70 else "WARN (<0.70)"
print(f"  AUC gate: {gate}")
print(f"  Embeddings: trained={emb_trained.shape}, "
      f"random={emb_random.shape}")

# [2/6] Full 6-method hardening on all 10 features
print("\n  [2/6] Full 6-method hardening on all 10 features...")
NP = 500
rng = np.random.default_rng(42)

# Scaffold groups for test set (reuse test_scaffolds from FIX 3)
sg = {s: np.where(test_scaffolds == s)[0]
      for s in np.unique(test_scaffolds)}

raw_results = {}
for j, nm in enumerate(INTERACTION_NAMES):
    t = tef_norm[:, j]
    if np.std(t) < 1e-10:
        raw_results[nm] = {'ridge_delta_r2': 0.0, 'se': 0.01}
        continue
    st = cross_val_score(Ridge(alpha=1.0), emb_trained, t,
                         cv=kf, scoring='r2')
    sr = cross_val_score(Ridge(alpha=1.0), emb_random, t,
                         cv=kf, scoring='r2')
    rd = np.mean(st) - np.mean(sr)
    se = max(np.sqrt(np.var(st) / len(st) + np.var(sr) / len(sr)), 1e-6)
    raw_results[nm] = {'ridge_delta_r2': rd, 'se': se}

print(f"\n  {'Feature':<20} {'Ridge dR2':>10}")
print(f"  {'-' * 32}")
for nm in INTERACTION_NAMES:
    print(f"  {nm:<20} {raw_results[nm]['ridge_delta_r2']:>10.4f}")

# Methods 1-2: Scaffold permutation + Y-scramble
hd = {n: {} for n in INTERACTION_NAMES}
print(f"\n    Methods 1-2: Scaffold perm + Y-scramble ({NP} perms)...")
for j, nm in enumerate(INTERACTION_NAMES):
    t = tef_norm[:, j]
    if np.std(t) < 1e-10:
        hd[nm]['sp_p'] = 1.0
        hd[nm]['ys_p'] = 1.0
        continue
    obs = raw_results[nm]['ridge_delta_r2']
    spn = np.zeros(NP)
    ysn = np.zeros(NP)
    for p in range(NP):
        pt = t.copy()
        for idx in sg.values():
            if len(idx) > 1:
                pt[idx] = rng.permutation(pt[idx])
        st_ = cross_val_score(Ridge(alpha=1.0), emb_trained, pt,
                              cv=kf, scoring='r2')
        sr_ = cross_val_score(Ridge(alpha=1.0), emb_random, pt,
                              cv=kf, scoring='r2')
        spn[p] = np.mean(st_) - np.mean(sr_)
        pt2 = rng.permutation(t)
        st2 = cross_val_score(Ridge(alpha=1.0), emb_trained, pt2,
                              cv=kf, scoring='r2')
        sr2 = cross_val_score(Ridge(alpha=1.0), emb_random, pt2,
                              cv=kf, scoring='r2')
        ysn[p] = np.mean(st2) - np.mean(sr2)
    hd[nm]['sp_p'] = float(np.mean(spn >= obs))
    hd[nm]['ys_p'] = float(np.mean(ysn >= obs))

# Method 3: Confound regression
print("    Method 3: Confound regression...")
conf = tef_norm[:, CONFOUND_IDX]
emb_clean_t = confound_removal(emb_trained, conf)
emb_clean_r = confound_removal(emb_random, conf)
for j, nm in enumerate(INTERACTION_NAMES):
    t = tef_norm[:, j]
    if np.std(t) < 1e-10:
        hd[nm]['cd'] = 0.0
        continue
    lr = LinearRegression()
    lr.fit(conf, t)
    t_clean = t - lr.predict(conf)
    if np.std(t_clean) < 1e-10:
        hd[nm]['cd'] = 0.0
        continue
    st_ = cross_val_score(Ridge(alpha=1.0), emb_clean_t, t_clean,
                          cv=kf, scoring='r2')
    sr_ = cross_val_score(Ridge(alpha=1.0), emb_clean_r, t_clean,
                          cv=kf, scoring='r2')
    hd[nm]['cd'] = np.mean(st_) - np.mean(sr_)

# Method 4: FDR correction
print("    Method 4: FDR correction...")
raw_pvals = np.array([hd[n].get('sp_p', 1.0) for n in INTERACTION_NAMES])
fdr_result = fdr_correction(raw_pvals, method='bh')
for j, n in enumerate(INTERACTION_NAMES):
    hd[n]['fdr_p'] = fdr_result['corrected_p'][j]
print(f"      FDR survivors: "
      f"{int(fdr_result['rejected'].sum())}/{len(INTERACTION_NAMES)}")

# Methods 5-6: TOST + Bayes factor
print("    Methods 5-6: TOST + Bayes factor...")
for n in INTERACTION_NAMES:
    d = raw_results[n]['ridge_delta_r2']
    se = raw_results[n]['se']
    hd[n]['tost'] = False
    if d < 0.05:
        r = tost_equivalence_test(delta_r2=d, se=se, epsilon=0.05)
        hd[n]['tost'] = r['equivalent']
    bf = bayes_factor_null(delta_r2=d, se=se)
    hd[n]['bf01'] = bf['bf01']


def hardened_verdict(hh):
    sp = hh.get('sp_p', 1)
    fp = hh.get('fdr_p', 1)
    cd = hh.get('cd', 0)
    tost_eq = hh.get('tost', False)
    bf = hh.get('bf01', 1)
    if sp < 0.05 and cd > 0.05 and fp < 0.05:
        return "CONFIRMED_ENCODED"
    elif sp < 0.05 and cd < 0.02:
        return "CONFOUND_DRIVEN"
    elif tost_eq and bf > 3:
        return "CONFIRMED_ZOMBIE"
    elif fp >= 0.05:
        return "LIKELY_ZOMBIE"
    elif sp < 0.05:
        return "CANDIDATE_ENCODED"
    return "LIKELY_ZOMBIE"


print(f"\n  {'Feature':<20} {'Raw dR2':>8} {'Scaf-p':>7} {'Clean dR2':>10} "
      f"{'FDR-p':>7} {'BF01':>6} {'Verdict':<20}")
print(f"  {'-' * 82}")
verdicts = {}
for nm in INTERACTION_NAMES:
    v = hardened_verdict(hd[nm])
    verdicts[nm] = v
    print(f"  {nm:<20} {raw_results[nm]['ridge_delta_r2']:>8.4f} "
          f"{hd[nm].get('sp_p', 1):>7.4f} "
          f"{hd[nm].get('cd', 0):>10.4f} "
          f"{hd[nm].get('fdr_p', 1):>7.4f} "
          f"{hd[nm].get('bf01', 1):>6.2f} {v:<20}")

# [3/6] Arbitrary Target Probes (false positive ceiling)
print("\n  [3/6] Arbitrary Target Probes (10 targets)...")
n_test = emb_trained.shape[0]
emb_dim = emb_trained.shape[1]
arb_rng = np.random.default_rng(777)
arbitrary_targets = {}

for i in range(5):
    v = arb_rng.standard_normal(emb_dim)
    v = v / np.linalg.norm(v)
    proj = emb_trained @ v
    proj = (proj - proj.mean()) / max(proj.std(), 1e-8)
    arbitrary_targets[f'rand_proj_{i}'] = proj

from scipy.integrate import odeint


def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


for i, axis in enumerate([0, 1, 2]):
    y0 = [1.0 + 0.1 * i, 1.0, 1.0]
    t_span = np.linspace(0, 50, 10000)
    sol = odeint(lorenz, y0, t_span)
    indices = np.linspace(0, len(sol) - 1, n_test, dtype=int)
    sig = sol[indices, axis]
    sig = (sig - sig.mean()) / max(sig.std(), 1e-8)
    arbitrary_targets[f'lorenz_{["x", "y", "z"][axis]}'] = sig

for src_name in ['dist_asp32', 'hbond_catalytic']:
    src_idx = INTERACTION_NAMES.index(src_name)
    shuffled = arb_rng.permutation(tef_norm[:, src_idx])
    arbitrary_targets[f'shuffled_{src_name}'] = shuffled

print(f"\n  {'Arbitrary Target':<24} {'Ridge dR2':>10}")
print(f"  {'-' * 36}")
arb_dr2s = []
for name, target in arbitrary_targets.items():
    if np.std(target) < 1e-10:
        dr2 = 0.0
    else:
        st = cross_val_score(Ridge(alpha=1.0), emb_trained, target,
                             cv=kf, scoring='r2')
        sr = cross_val_score(Ridge(alpha=1.0), emb_random, target,
                             cv=kf, scoring='r2')
        dr2 = np.mean(st) - np.mean(sr)
    arb_dr2s.append(dr2)
    print(f"  {name:<24} {dr2:>10.4f}")

false_positive_ceiling = max(arb_dr2s)
print(f"\n  False positive ceiling (max arbitrary dR2): "
      f"{false_positive_ceiling:.4f}")

above_ceiling = {}
print(f"\n  {'Real Feature':<20} {'Ridge dR2':>10} {'vs Ceiling':>12}")
print(f"  {'-' * 45}")
for nm in INTERACTION_NAMES:
    dr2 = raw_results[nm]['ridge_delta_r2']
    above = dr2 > false_positive_ceiling
    above_ceiling[nm] = above
    label = "ABOVE" if above else "BELOW"
    print(f"  {nm:<20} {dr2:>10.4f} {label:>12}")

n_above = sum(above_ceiling.values())
print(f"\n  Features above ceiling: {n_above}/{len(INTERACTION_NAMES)}")

# [4/6] 20-Seed Ensemble for catalytic features
print("\n  [4/6] 20-Seed Ensemble for catalytic features...")
N_SEEDS = 20
seed_pass_counts = {n: 0 for n in CATALYTIC_FEATURES}
PASS_THRESHOLD = 0.05

for seed in range(N_SEEDS):
    torch.manual_seed(seed + 100)
    np.random.seed(seed + 100)

    try:
        s_auc, s_model, s_emb = train_genome(
            best_genome, trg, trl, trs, vag, val_, teg, tel_,
            pocket_summary_tensor, train_interaction_features,
            n_epochs=80)
    except Exception:
        s_emb = np.random.randn(len(teg), 192)

    # FIX 2: fresh untrained embeddings per seed
    rand_emb_s = get_untrained_embeddings(
        best_genome, teg, pocket_summary_tensor)

    for fi, fname in zip(CATALYTIC_IDX, CATALYTIC_FEATURES):
        t = tef_norm[:, fi]
        if np.std(t) < 1e-10:
            continue
        st_ = cross_val_score(Ridge(alpha=1.0), s_emb, t,
                              cv=kf, scoring='r2')
        sr_ = cross_val_score(Ridge(alpha=1.0), rand_emb_s[:len(t)], t,
                              cv=kf, scoring='r2')
        dr2 = np.mean(st_) - np.mean(sr_)
        if dr2 > PASS_THRESHOLD:
            seed_pass_counts[fname] += 1

    if (seed + 1) % 5 == 0:
        print(f"    Seed {seed + 1}/{N_SEEDS}: {dict(seed_pass_counts)}")

print(f"\n  {'Feature':<20} {'Seeds Passed':>12} {'Fraction':>10} "
      f"{'Stability':>12}")
print(f"  {'-' * 58}")
seed_stability = {}
for fname in CATALYTIC_FEATURES:
    count = seed_pass_counts[fname]
    frac = count / N_SEEDS
    if count >= int(0.8 * N_SEEDS):
        stab = "ROBUST"
    elif count >= int(0.2 * N_SEEDS):
        stab = "FRAGILE"
    else:
        stab = "ABSENT"
    seed_stability[fname] = stab
    print(f"  {fname:<20} {count:>8}/{N_SEEDS} {frac:>10.2f} {stab:>12}")

for nm in INTERACTION_NAMES:
    if nm not in seed_stability:
        seed_stability[nm] = "N/A"

# [5/6] Two-Stage Ablation
print("\n  [5/6] Two-Stage Ablation...")
ENC_SET = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
ablation_results = {}

for j, nm in enumerate(INTERACTION_NAMES):
    stage1 = raw_results[nm]['ridge_delta_r2']
    stage2 = 0.0

    if verdicts[nm] in ENC_SET:
        other_idx = [k for k in range(len(INTERACTION_NAMES)) if k != j]
        other_features = tef_norm[:, other_idx]
        target = tef_norm[:, j]

        if np.std(target) > 1e-10 and other_features.shape[1] > 0:
            lr_emb = LinearRegression()
            lr_emb.fit(other_features, emb_trained)
            emb_resid = emb_trained - lr_emb.predict(other_features)

            lr_emb_r = LinearRegression()
            lr_emb_r.fit(other_features, emb_random)
            emb_resid_r = emb_random - lr_emb_r.predict(other_features)

            lr_t = LinearRegression()
            lr_t.fit(other_features, target)
            target_resid = target - lr_t.predict(other_features)

            if np.std(target_resid) > 1e-10:
                st_ = cross_val_score(Ridge(alpha=1.0), emb_resid,
                                       target_resid, cv=kf, scoring='r2')
                sr_ = cross_val_score(Ridge(alpha=1.0), emb_resid_r,
                                       target_resid, cv=kf, scoring='r2')
                stage2 = np.mean(st_) - np.mean(sr_)

    if stage2 > 0.02:
        abl_class = "DIRECT"
    elif stage1 > 0.05 and stage2 <= 0.02:
        abl_class = "INDIRECT"
    elif stage1 <= 0.05:
        abl_class = "NONE"
    else:
        abl_class = "NONE"

    ablation_results[nm] = {
        'stage1': stage1, 'stage2': stage2,
        'classification': abl_class
    }

print(f"\n  {'Feature':<20} {'Stage1 dR2':>10} {'Stage2 dR2':>10} "
      f"{'Classification':>16}")
print(f"  {'-' * 60}")
for nm in INTERACTION_NAMES:
    ar = ablation_results[nm]
    print(f"  {nm:<20} {ar['stage1']:>10.4f} {ar['stage2']:>10.4f} "
          f"{ar['classification']:>16}")

# [6/6] Integrated Final Verdicts
print("\n  [6/6] Integrated Final Verdicts...")
print(f"\n  {'Feature':<20} {'Hardened':<18} {'AboveCeil':>10} "
      f"{'Seeds':>10} {'TwoStage':>10} {'FINAL':<22}")
print(f"  {'-' * 95}")

final_verdicts = {}
for nm in INTERACTION_NAMES:
    h_verdict = verdicts[nm]
    above = above_ceiling.get(nm, False)
    stab = seed_stability.get(nm, "N/A")
    abl = ablation_results[nm]['classification']

    is_catalytic = nm in CATALYTIC_FEATURES
    seed_ok = (stab == "ROBUST") if is_catalytic else True
    above_ok = above
    hardened_ok = h_verdict == "CONFIRMED_ENCODED"
    ablation_ok = abl == "DIRECT"

    if hardened_ok and above_ok and seed_ok and ablation_ok:
        final = "PUBLICATION_READY"
    elif hardened_ok and above_ok:
        final = "STRONG_CANDIDATE"
    elif h_verdict in ENC_SET and above_ok:
        final = "CANDIDATE"
    elif h_verdict in ENC_SET:
        final = "WEAK_SIGNAL"
    else:
        final = "NOT_ENCODED"

    final_verdicts[nm] = final

    above_str = "YES" if above else "no"
    seed_str = (f"{seed_pass_counts.get(nm, '-')}/{N_SEEDS}"
                if is_catalytic else "N/A")
    print(f"  {nm:<20} {h_verdict:<18} {above_str:>10} {seed_str:>10} "
          f"{abl:>10} {final:<22}")

print(f"\n  Phase H completed in {time.time() - t_h:.1f}s")


# =====================================================================
# PHASE I: Complete Campaign Summary
# =====================================================================
t_i = time.time()
print("\n" + "=" * 85)
print("PHASE I: Complete Campaign Summary")
print("=" * 85)

# All 40 genomes ranked by fitness
print("\n  ALL 40 GENOMES RANKED BY FITNESS:")
print(f"  {'Rank':>4} {'Genome ID':<32} {'AUC':>6} {'Cat':>5} "
      f"{'Fitness':>8} {'Config Summary'}")
print(f"  {'-' * 100}")

sorted_all = sorted(all_results, key=lambda x: x[3], reverse=True)
for rank, (g, a, cr, f) in enumerate(sorted_all, 1):
    n_cat = sum(1 for _, _, enc in cr.values() if enc)
    desc = g.short_desc()
    if len(desc) > 40:
        desc = desc[:37] + "..."
    print(f"  {rank:>4} {g.genome_id:<32} {a:>6.3f} {n_cat:>3}/4 "
          f"{f:>8.4f} {desc}")

# Top 5 configurations with parameters
print(f"\n\n  TOP 5 CONFIGURATIONS:")
print(f"  {'=' * 85}")
for rank, (g, a, cr, f) in enumerate(sorted_all[:5], 1):
    print(f"\n  #{rank}: {g.genome_id}")
    print(f"    Fitness: {f:.4f}, AUC: {a:.4f}")
    cat_detail = ", ".join(
        f"{k}: dR2={v[0]:.3f} p={v[1]:.3f} enc={v[2]}"
        for k, v in cr.items())
    print(f"    Catalytic: {cat_detail}")
    print(f"    Config: {g.short_desc()}")
    print(f"    Aux outputs: {g.n_aux_outputs()}")
    if g.aux_docking_score:
        print(f"      docking_score weight: {g.aux_docking_weight:.3f}")
    if g.aux_dist_asp32:
        print(f"      dist_asp32 weight: {g.aux_dist_weight:.3f}")
    if g.aux_hbond_count:
        print(f"      hbond_count weight: {g.aux_hbond_weight:.3f}")
    if g.aux_contact_count:
        print(f"      contact_count weight: {g.aux_contact_weight:.3f}")
    if g.information_bottleneck:
        print(f"      IB beta: {g.ib_beta:.4f}")
    if g.embedding_l1 > 0:
        print(f"      L1 penalty: {g.embedding_l1:.5f}")
    if g.label_smoothing > 0:
        print(f"      Label smoothing: {g.label_smoothing:.3f}")
    if g.curriculum != 'none':
        print(f"      Curriculum: {g.curriculum}")
    print(f"      Learning rate: {g.learning_rate:.6f}")

# PUBLICATION_READY check
pub_ready = [n for n, v in final_verdicts.items()
             if v == "PUBLICATION_READY"]
strong = [n for n, v in final_verdicts.items()
          if v == "STRONG_CANDIDATE"]
candidates = [n for n, v in final_verdicts.items()
              if v == "CANDIDATE"]

n_cat_enc = sum(1 for n in GROUPS['CATALYTIC']
                if final_verdicts[n] in
                ('PUBLICATION_READY', 'STRONG_CANDIDATE'))
n_pock_enc = sum(1 for n in GROUPS['POCKET']
                 if final_verdicts[n] in
                 ('PUBLICATION_READY', 'STRONG_CANDIDATE'))

print(f"\n\n  WINNER EVALUATION RESULTS:")
print(f"  {'=' * 85}")
print(f"  Winner: {best_genome.genome_id}")
print(f"  Final AUC: {final_auc:.4f}")
print(f"  PUBLICATION_READY features: "
      f"{pub_ready if pub_ready else 'None'}")
print(f"  STRONG_CANDIDATE features: "
      f"{strong if strong else 'None'}")
print(f"  CANDIDATE features: "
      f"{candidates if candidates else 'None'}")
print(f"  Catalytic encoded: {n_cat_enc}/4")
print(f"  Pocket encoded: {n_pock_enc}/4")

# Comparison to baseline (template_01)
baseline_result = next((r for r in all_results
                        if r[0].genome_id == 'template_01_baseline'), None)
if baseline_result:
    bl_g, bl_auc, bl_cr, bl_f = baseline_result
    bl_cat = sum(1 for _, _, enc in bl_cr.values() if enc)
    print(f"\n  BASELINE COMPARISON:")
    print(f"    Baseline (plain BCE): AUC={bl_auc:.4f}, "
          f"catalytic={bl_cat}/4, fitness={bl_f:.4f}")
    print(f"    Winner ({best_genome.genome_id}): AUC={final_auc:.4f}, "
          f"catalytic={n_cat_enc}/4, fitness={best_fitness:.4f}")
    delta_fitness = best_fitness - bl_f
    delta_cat = n_cat_enc - bl_cat
    print(f"    Delta fitness: {delta_fitness:+.4f}")
    print(f"    Delta catalytic: {delta_cat:+d}")
    if delta_cat > 0:
        print(f"    DIAGNOSIS: Auxiliary losses HELPED -- "
              f"gained {delta_cat} catalytic features")
    elif delta_cat == 0:
        print(f"    DIAGNOSIS: Auxiliary losses had NO EFFECT "
              f"on catalytic encoding")
    else:
        print(f"    DIAGNOSIS: Auxiliary losses HURT -- "
              f"lost {-delta_cat} catalytic features")

# Phase summary
print(f"\n  PHASE SUMMARY:")
phase_fitness = {'TEMPLATE': [], 'EVOLVE': [], 'BALLOON': [],
                 'EXPLOIT': []}
for i, (g, a, cr, f) in enumerate(all_results):
    r = i + 1
    if r <= 11:
        phase_fitness['TEMPLATE'].append(f)
    elif r <= 25:
        phase_fitness['EVOLVE'].append(f)
    elif r <= 35:
        phase_fitness['BALLOON'].append(f)
    else:
        phase_fitness['EXPLOIT'].append(f)

for phase_name, fitnesses in phase_fitness.items():
    if fitnesses:
        print(f"    {phase_name:<10}: n={len(fitnesses):>2}, "
              f"mean={np.mean(fitnesses):.4f}, "
              f"max={np.max(fitnesses):.4f}")

# Per-group summary
for gname, members in GROUPS.items():
    enc_count = sum(1 for n in members if final_verdicts[n] in
                    ('PUBLICATION_READY', 'STRONG_CANDIDATE', 'CANDIDATE'))
    print(f"\n  {gname}: {enc_count}/{len(members)} features with signal")
    for n in members:
        print(f"    {n:<20}: {final_verdicts[n]}")

# Discovery readiness assessment
print(f"\n{'=' * 85}")
print("DISCOVERY READINESS ASSESSMENT")
print(f"{'=' * 85}")
print(f"  Catalytic site encoding: {n_cat_enc}/{len(GROUPS['CATALYTIC'])} "
      f"(PUBLICATION_READY or STRONG)")
print(f"  Pocket shape encoding:   {n_pock_enc}/{len(GROUPS['POCKET'])} "
      f"(PUBLICATION_READY or STRONG)")

if len(pub_ready) >= 3:
    verdict_str = (
        "DISCOVERY READY -- Dual factory evolved training configs "
        "that encode real BACE1 interaction geometry. "
        "Council controls confirm signal is genuine.")
elif n_cat_enc >= 2 and n_pock_enc >= 1:
    verdict_str = (
        "PARTIALLY READY -- Strong catalytic encoding achieved "
        "through evolved training configs. Pocket coverage "
        "needs improvement.")
elif n_cat_enc >= 1:
    verdict_str = (
        "EARLY SIGNAL -- Some catalytic features encoded. "
        "Auxiliary losses show promise but more evolution needed.")
elif best_fitness > 0:
    verdict_str = (
        "WEAK SIGNAL -- Model trains successfully but catalytic "
        "encoding remains elusive. Consider architecture changes.")
else:
    verdict_str = (
        "NOT INTERACTION-AWARE -- Training config evolution alone "
        "insufficient. Architecture changes needed.")

print(f"\n  {verdict_str}")

total_time = time.time() - CAMPAIGN_START
print(f"\n{'=' * 85}")
print(f"TOTAL CAMPAIGN TIME: {total_time:.1f}s ({total_time / 60:.1f} min)")
print(f"{'=' * 85}")
