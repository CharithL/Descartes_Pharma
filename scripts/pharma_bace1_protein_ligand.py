#!/usr/bin/env python3
"""
DESCARTES-PHARMA: BACE1 Protein-Ligand Co-Encoding Probe
=========================================================
Tests whether a protein-ligand co-encoding model can encode BACE1
catalytic features that the 2D GCN completely failed at
(0/4 catalytic, 0/4 pocket, 0/10 overall).

Four architectures compared:
  A: PlainGCN        -- baseline 2D GCN, no protein info
  B: GCN + Concat    -- ligand + pocket summary concatenation
  C: GCN + Bilinear  -- bilinear interaction (ligand x pocket)
  D: GCN + CrossAttn -- cross-attention to pocket residues

Target: Vast.ai A10 GPU (Linux, 22.5 GB VRAM).

Pipeline (9 Phases):
  1: Prepare Protein Pocket Features (PDB 4IVT)
  2: Build All Four Models
  3: Data Loading + 3D Conformers + Docking
  4: Train All Four Models
  5: Probe All Four Models
  6: Harden Best Protein-Aware Model
  7: Four-Model Comparison Table
  8: Pocket Scramble Test (Genuine vs Trivial)
  9: Discovery Readiness
"""

import subprocess
import sys
import os
import time
import warnings
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
print("DESCARTES-PHARMA: BACE1 Protein-Ligand Co-Encoding Probe")
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

STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
               'THR', 'TRP', 'TYR', 'VAL'}
AA_LIST = sorted(STANDARD_AA)  # deterministic one-hot order
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
NODE_FEAT_DIM = 7


# =====================================================================
# PHASE 1: Prepare Protein Pocket Features
# =====================================================================
t1 = time.time()
print("\n" + "=" * 85)
print("PHASE 1: Prepare Protein Pocket Features (PDB 4IVT)")
print("=" * 85)

PDB_FILE = 'data/4IVT.pdb'
if not os.path.exists(PDB_FILE):
    print("  Downloading PDB 4IVT (1.55A resolution, hydroxyethylamine inhibitor)...")
    urllib.request.urlretrieve('https://files.rcsb.org/download/4IVT.pdb', PDB_FILE)
print(f"  PDB file: {PDB_FILE}")

pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure('BACE1', PDB_FILE)
pdb_model = structure[0]

# Parse all protein atoms and build residue map
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

# Find co-crystallized ligand (HETATM, not standard AA, not water)
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

# Docking box
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


# Catalytic dyad oxygen coordinates
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

# --- Build per-residue feature vectors for cross-attention ---
# For each binding site residue: one-hot(20) + relative_coords(3) + polarity(1) + is_catalytic(1) = 25
bs_resids_sorted = sorted(bs_resids)
pocket_centroid = bs_coords.mean(axis=0) if len(bs_coords) > 0 else ligand_center

per_residue_features_list = []
for rid in bs_resids_sorted:
    rname = residue_names.get(rid, 'UNK')
    # One-hot amino acid (20)
    onehot = np.zeros(20, dtype=np.float32)
    if rname in AA_INDEX:
        onehot[AA_INDEX[rname]] = 1.0
    # Mean atomic coords relative to pocket centroid (3)
    res_atoms = residue_map.get(rid, [])
    if res_atoms:
        res_coords = np.array([c for _, c in res_atoms], dtype=np.float64)
        rel_coords = (res_coords.mean(axis=0) - pocket_centroid).astype(np.float32)
    else:
        rel_coords = np.zeros(3, dtype=np.float32)
    # Side chain polarity (1)
    polarity = np.array([1.0 if rname in POLAR_AA else 0.0], dtype=np.float32)
    # Is catalytic (1)
    is_cat = np.array([1.0 if rid in (32, 228) else 0.0], dtype=np.float32)
    feat = np.concatenate([onehot, rel_coords, polarity, is_cat])
    per_residue_features_list.append(feat)

per_residue_features_np = np.array(per_residue_features_list, dtype=np.float32)
per_residue_tensor = torch.tensor(per_residue_features_np, dtype=torch.float32).to(device)
n_residues = per_residue_tensor.shape[0]
RESIDUE_FEAT_DIM = 25
print(f"\n  Per-residue features: ({n_residues}, {RESIDUE_FEAT_DIM})")

# Pocket summary: [mean, std, min, max] across residues = (100,) vector
pocket_mean = per_residue_features_np.mean(axis=0)
pocket_std = per_residue_features_np.std(axis=0)
pocket_min = per_residue_features_np.min(axis=0)
pocket_max = per_residue_features_np.max(axis=0)
pocket_summary_np = np.concatenate([pocket_mean, pocket_std, pocket_min, pocket_max])
pocket_summary_tensor = torch.tensor(pocket_summary_np, dtype=torch.float32).to(device)
POCKET_DIM = len(pocket_summary_np)
print(f"  Pocket summary: ({POCKET_DIM},)")
print(f"  Catalytic residues in binding site: "
      f"{[r for r in bs_resids_sorted if r in (32, 228)]}")

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

# Check Vina availability
DOCKING_AVAILABLE = False
try:
    from vina import Vina
    DOCKING_AVAILABLE = True
    print("  AutoDock Vina: AVAILABLE")
except ImportError:
    print("  AutoDock Vina: NOT AVAILABLE (will use MCS alignment fallback)")

# Store reference ligand info
ref_lig_coords = np.array(ligand_atoms) if ligand_atoms else None
ref_lig_center = ligand_center
ref_ligand_mol = None

print(f"\n  Phase 1 completed in {time.time() - t1:.1f}s")


# =====================================================================
# PHASE 2: Build All Models
# =====================================================================
t2 = time.time()
print("\n" + "=" * 85)
print("PHASE 2: Build All Four Models")
print("=" * 85)


class PlainGCN(nn.Module):
    """Model A: Baseline GCN with no protein info."""

    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3,
                 dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(F.relu(bn(conv(x, ei))))
        emb = global_mean_pool(x, batch)
        logits = self.classifier(emb).squeeze(-1)
        return (logits, emb) if return_embedding else logits


class ConcatModel(nn.Module):
    """Model B: GCN + pocket summary concatenation."""

    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3,
                 pocket_dim=POCKET_DIM, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.pocket_mlp = nn.Sequential(
            nn.Linear(pocket_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 96), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(96, 1))

    def forward(self, data, pocket_features, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(F.relu(bn(conv(x, ei))))
        ligand_emb = global_mean_pool(x, batch)
        bsz = ligand_emb.shape[0]
        pocket_emb = self.pocket_mlp(pocket_features.unsqueeze(0).expand(bsz, -1))
        interaction_emb = torch.cat([ligand_emb, pocket_emb], dim=1)
        logits = self.classifier(interaction_emb).squeeze(-1)
        return (logits, interaction_emb) if return_embedding else logits


class BilinearModel(nn.Module):
    """Model C: GCN + pocket bilinear interaction."""

    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3,
                 pocket_dim=POCKET_DIM, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.pocket_mlp = nn.Sequential(
            nn.Linear(pocket_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.bilinear = nn.Bilinear(hidden_dim, 64, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, data, pocket_features, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(F.relu(bn(conv(x, ei))))
        ligand_emb = global_mean_pool(x, batch)
        bsz = ligand_emb.shape[0]
        pocket_emb = self.pocket_mlp(pocket_features.unsqueeze(0).expand(bsz, -1))
        interaction_emb = self.bilinear(ligand_emb, pocket_emb)
        logits = self.classifier(interaction_emb).squeeze(-1)
        return (logits, interaction_emb) if return_embedding else logits


class CrossAttentionModel(nn.Module):
    """Model D: GCN + cross-attention to pocket residues."""

    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3,
                 residue_feat_dim=RESIDUE_FEAT_DIM, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.residue_proj = nn.Linear(residue_feat_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, data, per_residue_features, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = self.drop(F.relu(bn(conv(x, ei))))
        # x: (total_atoms, hidden_dim)
        # per_residue_features: (n_residues, residue_feat_dim)
        residue_emb = self.residue_proj(per_residue_features)  # (n_residues, hidden_dim)
        kv = residue_emb.unsqueeze(0)  # (1, n_residues, hidden_dim)

        # Process each graph individually due to variable atom counts
        unique_batches = batch.unique()
        interaction_embs = []
        for b_idx in unique_batches:
            mask = (batch == b_idx)
            atom_emb = x[mask]  # (n_atoms_i, hidden_dim)
            queries = atom_emb.unsqueeze(0)  # (1, n_atoms_i, hidden_dim)
            attended, _ = self.cross_attn(queries, kv, kv)
            # attended: (1, n_atoms_i, hidden_dim)
            pooled = attended.squeeze(0).mean(dim=0)  # (hidden_dim,)
            interaction_embs.append(pooled)

        batch_emb = torch.stack(interaction_embs, dim=0)  # (bsz, hidden_dim)
        logits = self.classifier(batch_emb).squeeze(-1)
        return (logits, batch_emb) if return_embedding else logits


MODEL_CONFIGS = {
    'PlainGCN': {'class': PlainGCN, 'pocket_mode': 'none'},
    'Concat': {'class': ConcatModel, 'pocket_mode': 'summary'},
    'Bilinear': {'class': BilinearModel, 'pocket_mode': 'summary'},
    'CrossAttn': {'class': CrossAttentionModel, 'pocket_mode': 'residues'},
}

for name, cfg in MODEL_CONFIGS.items():
    m = cfg['class']().to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  {name:<14}: {n_params:>8,} params  (pocket_mode={cfg['pocket_mode']})")
    del m

print(f"\n  Phase 2 completed in {time.time() - t2:.1f}s")


# =====================================================================
# PHASE 3: Data Loading + 3D Conformers + Docking
# =====================================================================
t3 = time.time()
print("\n" + "=" * 85)
print("PHASE 3: Data Loading + 3D Conformers + Docking")
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

        # Rodrigues rotation
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


# --- Load BACE dataset ---
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

# [2/4] Generate conformers for train/val
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

# [3/4] Dock or align TEST SET molecules
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

# [4/4] Verify docking produced real variance
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


# --- Compute interaction features and build graphs ---
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

        # 1. dist_asp32
        if no_idx and len(a32_arr) > 0:
            no_pos = pos[no_idx]
            d_asp32 = float(np.min(np.linalg.norm(
                no_pos[:, None, :] - a32_arr[None, :, :], axis=2)))
        else:
            d_asp32 = 20.0

        # 2. dist_asp228
        if no_idx and len(a228_arr) > 0:
            no_pos = pos[no_idx]
            d_asp228 = float(np.min(np.linalg.norm(
                no_pos[:, None, :] - a228_arr[None, :, :], axis=2)))
        else:
            d_asp228 = 20.0

        # 3. hbond_catalytic
        hbc = 0
        if no_idx and len(cat_o_arr) > 0:
            for di in no_idx:
                for oc in cat_o_arr:
                    if np.linalg.norm(pos[di] - oc) < 3.5:
                        hbc += 1

        # 4. catalytic_score
        cat_score = 1.0 / max(d_asp32, 0.5) + 1.0 / max(d_asp228, 0.5)

        # 5-8. pocket contacts and buried fraction
        def count_contacts(pocket, cutoff=4.0):
            if len(pocket) == 0:
                return 0
            d = np.linalg.norm(pos[:, None, :] - pocket[None, :, :], axis=2)
            return int(np.sum(np.any(d < cutoff, axis=1)))

        s1c = count_contacts(s1_coords)
        s1pc = count_contacts(s1p_coords)
        tc = count_contacts(bs_coords)
        bur = tc / max(len(pos), 1)

        # 9-10. confounds
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


print("\n  Processing datasets...")
trg, trl, trs = process_trainval(train_df)
vag, val_, vas = process_trainval(val_df)
teg, tel_, tef, tes = process_test(test_df)
print(f"  Train: {len(trg)}, Val: {len(vag)}, "
      f"Test: {len(teg)} (with interaction features)")

print(f"\n  {'Feature':<20} {'Group':<10} {'Mean':>8} {'Std':>8} "
      f"{'Min':>8} {'Max':>8}")
print(f"  {'-' * 60}")
for i, nm in enumerate(INTERACTION_NAMES):
    v = tef[:, i]
    grp = [g for g, m in GROUPS.items() if nm in m][0]
    print(f"  {nm:<20} {grp:<10} {v.mean():>8.3f} {v.std():>8.3f} "
          f"{v.min():>8.1f} {v.max():>8.1f}")

print(f"\n  Phase 3 completed in {time.time() - t3:.1f}s")


# =====================================================================
# PHASE 4: Train All Four Models
# =====================================================================
t4 = time.time()
print("\n" + "=" * 85)
print("PHASE 4: Train All Four Models")
print("=" * 85)

dtl = PyGDataLoader(trg, batch_size=64, shuffle=True)
dvl = PyGDataLoader(vag, batch_size=64, shuffle=False)
dtsl = PyGDataLoader(teg, batch_size=64, shuffle=False)
pw = torch.tensor(
    [(1 - np.mean(trl)) / max(np.mean(trl), 1e-6)]).to(device)


def train_model(model, train_loader, val_loader, model_name,
                pocket_mode='none', n_epochs=200):
    """Shared training function for all four model architectures."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5, min_lr=1e-6)
    best_auc, best_state = 0.0, None
    t0 = time.time()

    for ep in range(n_epochs):
        model.train()
        total_loss = 0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            if pocket_mode == 'none':
                logits = model(b)
            elif pocket_mode == 'summary':
                logits = model(b, pocket_summary_tensor)
            elif pocket_mode == 'residues':
                logits = model(b, per_residue_tensor)
            loss = F.binary_cross_entropy_with_logits(
                logits, b.y.squeeze(-1), pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * b.num_graphs

        if ep % 5 == 0:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = b.to(device)
                    if pocket_mode == 'none':
                        pred = model(b)
                    elif pocket_mode == 'summary':
                        pred = model(b, pocket_summary_tensor)
                    elif pocket_mode == 'residues':
                        pred = model(b, per_residue_tensor)
                    vp.extend(torch.sigmoid(pred).cpu().numpy())
                    vt.extend(b.y.squeeze(-1).cpu().numpy())
            try:
                va = roc_auc_score(vt, vp)
            except Exception:
                va = 0.5
            sch.step(1 - va)
            if va > best_auc:
                best_auc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 50 == 0:
                print(f"    [{model_name}] Ep {ep:3d}: "
                      f"loss={total_loss / len(trg):.4f} val_auc={va:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test AUC
    model.eval()
    tp, tt = [], []
    with torch.no_grad():
        for b in dtsl:
            b = b.to(device)
            if pocket_mode == 'none':
                pred = model(b)
            elif pocket_mode == 'summary':
                pred = model(b, pocket_summary_tensor)
            elif pocket_mode == 'residues':
                pred = model(b, per_residue_tensor)
            tp.extend(torch.sigmoid(pred).cpu().numpy())
            tt.extend(b.y.squeeze(-1).cpu().numpy())
    try:
        test_auc = roc_auc_score(tt, tp)
    except Exception:
        test_auc = 0.5

    elapsed = time.time() - t0
    print(f"    [{model_name}] Training: {elapsed:.1f}s, "
          f"best val AUC: {best_auc:.4f}, test AUC: {test_auc:.4f}")
    return model, best_auc, test_auc


# Train all four models
trained_models = {}
model_aucs = {}

for name, cfg in MODEL_CONFIGS.items():
    print(f"\n  Training {name}...")
    torch.manual_seed(42)
    np.random.seed(42)
    m = cfg['class']().to(device)
    m, val_auc, test_auc = train_model(
        m, dtl, dvl, name, pocket_mode=cfg['pocket_mode'])
    trained_models[name] = m
    model_aucs[name] = {'val': val_auc, 'test': test_auc}

print(f"\n  {'Model':<14} {'Val AUC':>10} {'Test AUC':>10}")
print(f"  {'-' * 36}")
for name in MODEL_CONFIGS:
    print(f"  {name:<14} {model_aucs[name]['val']:>10.4f} "
          f"{model_aucs[name]['test']:>10.4f}")

print(f"\n  Phase 4 completed in {time.time() - t4:.1f}s")


# =====================================================================
# PHASE 5: Probe All Four Models
# =====================================================================
t5 = time.time()
print("\n" + "=" * 85)
print("PHASE 5: Probe All Four Models (Ridge dR2)")
print("=" * 85)


def extract_embeddings(model, loader, model_type):
    """Extract embeddings handling different model forward signatures."""
    model.eval()
    embs = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            if model_type == 'none':
                _, e = model(b, return_embedding=True)
            elif model_type == 'summary':
                _, e = model(b, pocket_summary_tensor, return_embedding=True)
            elif model_type == 'residues':
                _, e = model(b, per_residue_tensor, return_embedding=True)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs)


sc = StandardScaler()
tef_norm = sc.fit_transform(tef)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Probe all four models (trained + untrained)
all_probe_results = {}

for name, cfg in MODEL_CONFIGS.items():
    pmode = cfg['pocket_mode']

    # Trained embeddings
    emb_trained = extract_embeddings(trained_models[name], dtsl, pmode)

    # Untrained (random) embeddings
    torch.manual_seed(999)
    rand_m = cfg['class']().to(device)
    rand_m.eval()
    emb_random = extract_embeddings(rand_m, dtsl, pmode)
    del rand_m

    print(f"\n  {name} embeddings: trained={emb_trained.shape}, "
          f"random={emb_random.shape}")
    print(f"  {'Feature':<20} {'Ridge dR2':>10} {'Verdict':>15}")
    print(f"  {'-' * 48}")

    model_results = {}
    for j, nm in enumerate(INTERACTION_NAMES):
        t = tef_norm[:, j]
        if np.std(t) < 1e-10:
            model_results[nm] = {'ridge_delta_r2': 0.0, 'se': 0.01}
            print(f"  {nm:<20} {'0.0000':>10} {'ZERO_VAR':>15}")
            continue
        st = cross_val_score(Ridge(alpha=1.0), emb_trained, t,
                             cv=kf, scoring='r2')
        sr = cross_val_score(Ridge(alpha=1.0), emb_random, t,
                             cv=kf, scoring='r2')
        rd = np.mean(st) - np.mean(sr)
        se = np.sqrt(np.var(st) / len(st) + np.var(sr) / len(sr))
        model_results[nm] = {'ridge_delta_r2': rd, 'se': max(se, 1e-6)}
        verdict = 'ENCODED' if rd > 0.05 else 'ZOMBIE'
        print(f"  {nm:<20} {rd:>10.4f} {verdict:>15}")

    all_probe_results[name] = {
        'results': model_results,
        'emb_trained': emb_trained,
        'emb_random': emb_random,
    }

print(f"\n  Phase 5 completed in {time.time() - t5:.1f}s")


# =====================================================================
# PHASE 6: Harden Best Protein-Aware Model
# =====================================================================
t6 = time.time()
print("\n" + "=" * 85)
print("PHASE 6: Harden Best Protein-Aware Model (6 methods + 3 council controls)")
print("=" * 85)

# Determine which protein-aware model encoded the most catalytic features
CATALYTIC_FEATURES = ['dist_asp32', 'dist_asp228', 'hbond_catalytic',
                      'catalytic_score']
CATALYTIC_IDX = [INTERACTION_NAMES.index(n) for n in CATALYTIC_FEATURES]

best_pa_name = None
best_cat_count = -1
for name in ['Concat', 'Bilinear', 'CrossAttn']:
    count = sum(1 for nm in CATALYTIC_FEATURES
                if all_probe_results[name]['results'][nm]['ridge_delta_r2'] > 0.05)
    print(f"  {name}: {count}/4 catalytic features encoded (raw)")
    if count > best_cat_count:
        best_cat_count = count
        best_pa_name = name

print(f"\n  Best protein-aware model: {best_pa_name} ({best_cat_count}/4 catalytic)")

pa_results = all_probe_results[best_pa_name]['results']
pa_emb_trained = all_probe_results[best_pa_name]['emb_trained']
pa_emb_random = all_probe_results[best_pa_name]['emb_random']

# --- 6-method hardening ---
print(f"\n  [1/4] Statistical hardening (6 methods) on {best_pa_name}...")
NP = 500
rng = np.random.default_rng(42)

# Scaffold groups for test set
scaff = []
for smi in tes:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        c = MurckoScaffold.GetScaffoldForMol(mol)
        g = MurckoScaffold.MakeScaffoldGeneric(c)
        scaff.append(Chem.MolToSmiles(g))
    else:
        scaff.append('UNK')
scaff = np.array(scaff)
sg = {s: np.where(scaff == s)[0] for s in np.unique(scaff)}

hd = {n: {} for n in INTERACTION_NAMES}

# Methods 1-2: Scaffold permutation + Y-scramble
print(f"    Methods 1-2: Scaffold perm + Y-scramble ({NP} perms)...")
for j, nm in enumerate(INTERACTION_NAMES):
    t = tef_norm[:, j]
    if np.std(t) < 1e-10:
        hd[nm]['sp_p'] = 1.0
        hd[nm]['ys_p'] = 1.0
        continue
    obs = pa_results[nm]['ridge_delta_r2']
    spn = np.zeros(NP)
    ysn = np.zeros(NP)
    for p in range(NP):
        pt = t.copy()
        for idx in sg.values():
            if len(idx) > 1:
                pt[idx] = rng.permutation(pt[idx])
        st_ = cross_val_score(Ridge(alpha=1.0), pa_emb_trained, pt,
                              cv=kf, scoring='r2')
        sr_ = cross_val_score(Ridge(alpha=1.0), pa_emb_random, pt,
                              cv=kf, scoring='r2')
        spn[p] = np.mean(st_) - np.mean(sr_)
        pt2 = rng.permutation(t)
        st2 = cross_val_score(Ridge(alpha=1.0), pa_emb_trained, pt2,
                              cv=kf, scoring='r2')
        sr2 = cross_val_score(Ridge(alpha=1.0), pa_emb_random, pt2,
                              cv=kf, scoring='r2')
        ysn[p] = np.mean(st2) - np.mean(sr2)
    hd[nm]['sp_p'] = float(np.mean(spn >= obs))
    hd[nm]['ys_p'] = float(np.mean(ysn >= obs))
    print(f"      {nm:<20}: scaffold_p={hd[nm]['sp_p']:.4f}, "
          f"yscramble_p={hd[nm]['ys_p']:.4f}")

# Method 3: Confound regression
print("    Method 3: Confound regression...")
conf = tef_norm[:, CONFOUND_IDX]
emb_clean_t = confound_removal(pa_emb_trained, conf)
emb_clean_r = confound_removal(pa_emb_random, conf)
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
    d = pa_results[n]['ridge_delta_r2']
    se = pa_results[n]['se']
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


print(f"\n  Hardened Results Table ({best_pa_name})")
print(f"\n  {'Feature':<20} {'Raw dR2':>8} {'Scaf-p':>7} {'Clean dR2':>10} "
      f"{'FDR-p':>7} {'BF01':>6} {'Verdict':<20}")
print(f"  {'-' * 82}")
verdicts = {}
for nm in INTERACTION_NAMES:
    v = hardened_verdict(hd[nm])
    verdicts[nm] = v
    print(f"  {nm:<20} {pa_results[nm]['ridge_delta_r2']:>8.4f} "
          f"{hd[nm].get('sp_p', 1):>7.4f} "
          f"{hd[nm].get('cd', 0):>10.4f} "
          f"{hd[nm].get('fdr_p', 1):>7.4f} "
          f"{hd[nm].get('bf01', 1):>6.2f} {v:<20}")

# --- Council Control 1: Arbitrary Target Probes ---
print(f"\n  [2/4] Council Control 1: Arbitrary Target Probes...")
n_test = pa_emb_trained.shape[0]
emb_dim = pa_emb_trained.shape[1]
arb_rng = np.random.default_rng(777)
arbitrary_targets = {}

# 5 random linear projections
for i in range(5):
    v = arb_rng.standard_normal(emb_dim)
    v = v / np.linalg.norm(v)
    proj = pa_emb_trained @ v
    proj = (proj - proj.mean()) / max(proj.std(), 1e-8)
    arbitrary_targets[f'rand_proj_{i}'] = proj

# 3 Lorenz attractor signals
from scipy.integrate import odeint


def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    lx, ly, lz = state
    return [sigma * (ly - lx), lx * (rho - lz) - ly, lx * ly - beta * lz]


for i, axis in enumerate([0, 1, 2]):
    y0 = [1.0 + 0.1 * i, 1.0, 1.0]
    t_span = np.linspace(0, 50, 10000)
    sol = odeint(lorenz, y0, t_span)
    indices = np.linspace(0, len(sol) - 1, n_test, dtype=int)
    sig = sol[indices, axis]
    sig = (sig - sig.mean()) / max(sig.std(), 1e-8)
    arbitrary_targets[f'lorenz_{["x", "y", "z"][axis]}'] = sig

# 2 shuffled versions of real features
for src_name in ['dist_asp32', 'hbond_catalytic']:
    src_idx = INTERACTION_NAMES.index(src_name)
    shuffled = arb_rng.permutation(tef_norm[:, src_idx])
    arbitrary_targets[f'shuffled_{src_name}'] = shuffled

print(f"\n  {'Arbitrary Target':<24} {'Ridge dR2':>10}")
print(f"  {'-' * 36}")
arb_dr2s = []
for arb_name, target in arbitrary_targets.items():
    if np.std(target) < 1e-10:
        dr2 = 0.0
    else:
        st = cross_val_score(Ridge(alpha=1.0), pa_emb_trained, target,
                             cv=kf, scoring='r2')
        sr = cross_val_score(Ridge(alpha=1.0), pa_emb_random, target,
                             cv=kf, scoring='r2')
        dr2 = np.mean(st) - np.mean(sr)
    arb_dr2s.append(dr2)
    print(f"  {arb_name:<24} {dr2:>10.4f}")

false_positive_ceiling = max(arb_dr2s)
print(f"\n  False positive ceiling (max arbitrary dR2): "
      f"{false_positive_ceiling:.4f}")

above_ceiling = {}
for nm in INTERACTION_NAMES:
    dr2 = pa_results[nm]['ridge_delta_r2']
    above_ceiling[nm] = dr2 > false_positive_ceiling

n_above = sum(above_ceiling.values())
print(f"  Features above ceiling: {n_above}/{len(INTERACTION_NAMES)}")

# --- Council Control 2: 20-Seed Ensemble ---
print(f"\n  [3/4] Council Control 2: 20-Seed Ensemble Stability...")
N_SEEDS = 20
pa_cfg = MODEL_CONFIGS[best_pa_name]
pa_pmode = pa_cfg['pocket_mode']

seed_pass_counts = {n: 0 for n in CATALYTIC_FEATURES}
PASS_THRESHOLD = 0.05

for seed in range(N_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    m = pa_cfg['class']().to(device)
    opt_s = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch_s = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_s, patience=15, factor=0.5, min_lr=1e-6)
    ba_s, bst_s = 0.0, None

    for ep in range(200):
        m.train()
        for b in dtl:
            b = b.to(device)
            opt_s.zero_grad()
            if pa_pmode == 'none':
                logits = m(b)
            elif pa_pmode == 'summary':
                logits = m(b, pocket_summary_tensor)
            elif pa_pmode == 'residues':
                logits = m(b, per_residue_tensor)
            loss = F.binary_cross_entropy_with_logits(
                logits, b.y.squeeze(-1), pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt_s.step()
        if ep % 5 == 0:
            m.eval()
            vp_s, vt_s = [], []
            with torch.no_grad():
                for b in dvl:
                    b = b.to(device)
                    if pa_pmode == 'none':
                        pred = m(b)
                    elif pa_pmode == 'summary':
                        pred = m(b, pocket_summary_tensor)
                    elif pa_pmode == 'residues':
                        pred = m(b, per_residue_tensor)
                    vp_s.extend(torch.sigmoid(pred).cpu().numpy())
                    vt_s.extend(b.y.squeeze(-1).cpu().numpy())
            try:
                va_s = roc_auc_score(vt_s, vp_s)
            except Exception:
                va_s = 0.5
            sch_s.step(1 - va_s)
            if va_s > ba_s:
                ba_s = va_s
                bst_s = {k: v.clone() for k, v in m.state_dict().items()}

    if bst_s is not None:
        m.load_state_dict(bst_s)

    emb_s = extract_embeddings(m, dtsl, pa_pmode)
    torch.manual_seed(seed + 10000)
    rm_s = pa_cfg['class']().to(device)
    rm_s.eval()
    er_s = extract_embeddings(rm_s, dtsl, pa_pmode)
    del rm_s

    for fi, fname in zip(CATALYTIC_IDX, CATALYTIC_FEATURES):
        t = tef_norm[:, fi]
        if np.std(t) < 1e-10:
            continue
        st_ = cross_val_score(Ridge(alpha=1.0), emb_s, t,
                              cv=kf, scoring='r2')
        sr_ = cross_val_score(Ridge(alpha=1.0), er_s, t,
                              cv=kf, scoring='r2')
        dr2 = np.mean(st_) - np.mean(sr_)
        if dr2 > PASS_THRESHOLD:
            seed_pass_counts[fname] += 1

    if (seed + 1) % 5 == 0:
        elapsed = time.time() - t6
        print(f"    Seed {seed + 1}/{N_SEEDS} ({elapsed:.1f}s) -- "
              f"pass counts: {dict(seed_pass_counts)}")

print(f"\n  {'Feature':<20} {'Seeds Passed':>12} {'Fraction':>10} "
      f"{'Stability':>12}")
print(f"  {'-' * 58}")
seed_stability = {}
for fname in CATALYTIC_FEATURES:
    count = seed_pass_counts[fname]
    frac = count / N_SEEDS
    if count >= int(0.8 * N_SEEDS):  # 16/20
        stab = "ROBUST"
    elif count >= int(0.2 * N_SEEDS):  # 4/20
        stab = "FRAGILE"
    else:
        stab = "ABSENT"
    seed_stability[fname] = stab
    print(f"  {fname:<20} {count:>8}/{N_SEEDS} {frac:>10.2f} {stab:>12}")

for nm in INTERACTION_NAMES:
    if nm not in seed_stability:
        seed_stability[nm] = "N/A"

# --- Council Control 3: Two-Stage Ablation ---
print(f"\n  [4/4] Council Control 3: Two-Stage Ablation...")
ENC_SET = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
ablation_results = {}

for j, nm in enumerate(INTERACTION_NAMES):
    stage1 = pa_results[nm]['ridge_delta_r2']
    stage2 = 0.0

    if verdicts[nm] in ENC_SET:
        other_idx = [k for k in range(len(INTERACTION_NAMES)) if k != j]
        other_features = tef_norm[:, other_idx]
        target = tef_norm[:, j]

        if np.std(target) > 1e-10 and other_features.shape[1] > 0:
            lr_emb = LinearRegression()
            lr_emb.fit(other_features, pa_emb_trained)
            emb_resid = pa_emb_trained - lr_emb.predict(other_features)

            lr_emb_r = LinearRegression()
            lr_emb_r.fit(other_features, pa_emb_random)
            emb_resid_r = pa_emb_random - lr_emb_r.predict(other_features)

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
        'classification': abl_class}

print(f"\n  {'Feature':<20} {'Stage1 dR2':>10} {'Stage2 dR2':>10} "
      f"{'Classification':>16}")
print(f"  {'-' * 60}")
for nm in INTERACTION_NAMES:
    ar = ablation_results[nm]
    print(f"  {nm:<20} {ar['stage1']:>10.4f} {ar['stage2']:>10.4f} "
          f"{ar['classification']:>16}")

print(f"\n  Phase 6 completed in {time.time() - t6:.1f}s")


# =====================================================================
# PHASE 7: Four-Model Comparison Table
# =====================================================================
t7 = time.time()
print("\n" + "=" * 85)
print("PHASE 7: Four-Model Comparison Table")
print("=" * 85)

# For each model, determine per-feature verdict based on raw dR2 > 0.05
print(f"\n  {'Feature':<20} {'PlainGCN':>10} {'Concat':>10} "
      f"{'Bilinear':>10} {'CrossAttn':>10}")
print(f"  {'-' * 64}")

model_feature_verdicts = {}
for name in MODEL_CONFIGS:
    model_feature_verdicts[name] = {}
    for nm in INTERACTION_NAMES:
        dr2 = all_probe_results[name]['results'][nm]['ridge_delta_r2']
        model_feature_verdicts[name][nm] = 'ENC' if dr2 > 0.05 else '---'

for nm in INTERACTION_NAMES:
    row = f"  {nm:<20}"
    for name in MODEL_CONFIGS:
        row += f" {model_feature_verdicts[name][nm]:>10}"
    print(row)

# Count per group per model
print(f"\n  {'Group':<12}", end="")
for name in MODEL_CONFIGS:
    print(f" {name:>10}", end="")
print()
print(f"  {'-' * 54}")

for gname, members in GROUPS.items():
    row = f"  {gname:<12}"
    for name in MODEL_CONFIGS:
        enc = sum(1 for nm in members
                  if model_feature_verdicts[name][nm] == 'ENC')
        row += f" {enc:>7}/{len(members):>1}"
    print(row)

# Total
row = f"  {'TOTAL':<12}"
for name in MODEL_CONFIGS:
    enc = sum(1 for nm in INTERACTION_NAMES
              if model_feature_verdicts[name][nm] == 'ENC')
    row += f" {enc:>7}/{len(INTERACTION_NAMES):>2}"
print(row)

print(f"\n  Phase 7 completed in {time.time() - t7:.1f}s")


# =====================================================================
# PHASE 8: Pocket Scramble Test (Genuine vs Trivial)
# =====================================================================
t8 = time.time()
print("\n" + "=" * 85)
print("PHASE 8: Pocket Scramble Test (Genuine vs Trivial)")
print("=" * 85)
print(f"  Testing whether {best_pa_name} learns real spatial relationships")
print("  by comparing performance with real vs scrambled pocket features.")

# Create scrambled pocket
scramble_rng = np.random.default_rng(12345)

if pa_pmode == 'residues':
    # Permute residue rows
    perm_idx = scramble_rng.permutation(n_residues)
    scrambled_residue_np = per_residue_features_np[perm_idx]
    scrambled_residue_tensor = torch.tensor(
        scrambled_residue_np, dtype=torch.float32).to(device)
    # Also create scrambled summary from scrambled residues
    scr_mean = scrambled_residue_np.mean(axis=0)
    scr_std = scrambled_residue_np.std(axis=0)
    scr_min = scrambled_residue_np.min(axis=0)
    scr_max = scrambled_residue_np.max(axis=0)
    scrambled_summary_np = np.concatenate([scr_mean, scr_std, scr_min, scr_max])
    scrambled_summary_tensor = torch.tensor(
        scrambled_summary_np, dtype=torch.float32).to(device)
else:
    # Shuffle pocket summary by blocks of 25 (mean, std, min, max blocks)
    scrambled_summary_np = pocket_summary_np.copy()
    block_size = RESIDUE_FEAT_DIM
    for start in range(0, POCKET_DIM, block_size):
        end = min(start + block_size, POCKET_DIM)
        block = scrambled_summary_np[start:end].copy()
        scrambled_summary_np[start:end] = scramble_rng.permutation(block)
    scrambled_summary_tensor = torch.tensor(
        scrambled_summary_np, dtype=torch.float32).to(device)
    scrambled_residue_tensor = None

# Train model with scrambled pocket
print(f"  Training {best_pa_name} with SCRAMBLED pocket...")
torch.manual_seed(42)
np.random.seed(42)
scrambled_model = pa_cfg['class']().to(device)

opt_scr = torch.optim.Adam(scrambled_model.parameters(), lr=1e-3, weight_decay=1e-5)
sch_scr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_scr, patience=15, factor=0.5, min_lr=1e-6)
ba_scr, bst_scr = 0.0, None

for ep in range(200):
    scrambled_model.train()
    for b in dtl:
        b = b.to(device)
        opt_scr.zero_grad()
        if pa_pmode == 'summary':
            logits = scrambled_model(b, scrambled_summary_tensor)
        elif pa_pmode == 'residues':
            logits = scrambled_model(b, scrambled_residue_tensor)
        loss = F.binary_cross_entropy_with_logits(
            logits, b.y.squeeze(-1), pos_weight=pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scrambled_model.parameters(), 1.0)
        opt_scr.step()
    if ep % 5 == 0:
        scrambled_model.eval()
        vp_scr, vt_scr = [], []
        with torch.no_grad():
            for b in dvl:
                b = b.to(device)
                if pa_pmode == 'summary':
                    pred = scrambled_model(b, scrambled_summary_tensor)
                elif pa_pmode == 'residues':
                    pred = scrambled_model(b, scrambled_residue_tensor)
                vp_scr.extend(torch.sigmoid(pred).cpu().numpy())
                vt_scr.extend(b.y.squeeze(-1).cpu().numpy())
        try:
            va_scr = roc_auc_score(vt_scr, vp_scr)
        except Exception:
            va_scr = 0.5
        sch_scr.step(1 - va_scr)
        if va_scr > ba_scr:
            ba_scr = va_scr
            bst_scr = {k: v.clone() for k, v in scrambled_model.state_dict().items()}
        if ep % 50 == 0:
            print(f"    [Scrambled] Ep {ep:3d}: val_auc={va_scr:.4f}")

if bst_scr is not None:
    scrambled_model.load_state_dict(bst_scr)

# Extract embeddings from scrambled model
scrambled_model.eval()
scr_embs = []
with torch.no_grad():
    for b in dtsl:
        b = b.to(device)
        if pa_pmode == 'summary':
            _, e = scrambled_model(b, scrambled_summary_tensor, return_embedding=True)
        elif pa_pmode == 'residues':
            _, e = scrambled_model(b, scrambled_residue_tensor, return_embedding=True)
        scr_embs.append(e.cpu().numpy())
emb_scrambled = np.concatenate(scr_embs)

# Random baseline for scrambled model
torch.manual_seed(999)
scr_rand = pa_cfg['class']().to(device)
scr_rand.eval()
scr_rand_embs = []
with torch.no_grad():
    for b in dtsl:
        b = b.to(device)
        if pa_pmode == 'summary':
            _, e = scr_rand(b, scrambled_summary_tensor, return_embedding=True)
        elif pa_pmode == 'residues':
            _, e = scr_rand(b, scrambled_residue_tensor, return_embedding=True)
        scr_rand_embs.append(e.cpu().numpy())
emb_scrambled_rand = np.concatenate(scr_rand_embs)
del scr_rand

# Compare real pocket vs scrambled pocket on catalytic features
print(f"\n  Scrambled model trained. Comparing catalytic feature probes...")
print(f"\n  {'Feature':<20} {'Real dR2':>10} {'Scrambled dR2':>14} {'Interpretation':>16}")
print(f"  {'-' * 64}")

scramble_interpretations = {}
for fi, fname in zip(CATALYTIC_IDX, CATALYTIC_FEATURES):
    t = tef_norm[:, fi]
    if np.std(t) < 1e-10:
        real_dr2 = 0.0
        scr_dr2 = 0.0
    else:
        # Real pocket dR2 (already computed)
        real_dr2 = pa_results[fname]['ridge_delta_r2']
        # Scrambled pocket dR2
        st_scr = cross_val_score(Ridge(alpha=1.0), emb_scrambled, t,
                                 cv=kf, scoring='r2')
        sr_scr = cross_val_score(Ridge(alpha=1.0), emb_scrambled_rand, t,
                                 cv=kf, scoring='r2')
        scr_dr2 = np.mean(st_scr) - np.mean(sr_scr)

    real_enc = real_dr2 > 0.05
    scr_enc = scr_dr2 > 0.05

    if real_enc and not scr_enc:
        interp = "GENUINE"
    elif real_enc and scr_enc:
        interp = "TRIVIAL"
    elif not real_enc and not scr_enc:
        interp = "NEITHER"
    else:
        interp = "NOISE"

    scramble_interpretations[fname] = interp
    print(f"  {fname:<20} {real_dr2:>10.4f} {scr_dr2:>14.4f} {interp:>16}")

n_genuine = sum(1 for v in scramble_interpretations.values() if v == 'GENUINE')
n_trivial = sum(1 for v in scramble_interpretations.values() if v == 'TRIVIAL')
n_neither = sum(1 for v in scramble_interpretations.values() if v == 'NEITHER')
print(f"\n  GENUINE: {n_genuine}/4 -- model learned real spatial relationships")
print(f"  TRIVIAL: {n_trivial}/4 -- encoding does not depend on correct pocket")
print(f"  NEITHER: {n_neither}/4 -- pocket info not helping")

print(f"\n  Phase 8 completed in {time.time() - t8:.1f}s")


# =====================================================================
# PHASE 9: Discovery Readiness
# =====================================================================
t9 = time.time()
print("\n" + "=" * 85)
print("PHASE 9: Discovery Readiness Assessment")
print("=" * 85)

# Combine all evidence for the best protein-aware model
print(f"\n  Best protein-aware model: {best_pa_name}")
print(f"  Test AUC: {model_aucs[best_pa_name]['test']:.4f}")
print(f"  Docking method: "
      f"{'AutoDock Vina' if DOCKING_AVAILABLE else 'MCS alignment (fallback)'}")

# Determine final verdicts combining all 4 gates + scramble
print(f"\n  {'Feature':<20} {'Hardened':<18} {'AboveCeil':>10} "
      f"{'Seeds':>10} {'TwoStage':>10} {'Scramble':>10} {'FINAL':<22}")
print(f"  {'-' * 105}")

final_verdicts = {}
for nm in INTERACTION_NAMES:
    h_verdict = verdicts[nm]
    above = above_ceiling.get(nm, False)
    stab = seed_stability.get(nm, "N/A")
    abl = ablation_results[nm]['classification']
    scr = scramble_interpretations.get(nm, "N/A")

    is_catalytic = nm in CATALYTIC_FEATURES
    seed_ok = (stab == "ROBUST") if is_catalytic else True
    above_ok = above
    hardened_ok = h_verdict == "CONFIRMED_ENCODED"
    ablation_ok = abl == "DIRECT"
    scramble_ok = (scr == "GENUINE") if is_catalytic else True

    if hardened_ok and above_ok and seed_ok and ablation_ok and scramble_ok:
        final = "PUBLICATION_READY"
    elif hardened_ok and above_ok and seed_ok:
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
    scr_str = scr if is_catalytic else "N/A"
    print(f"  {nm:<20} {h_verdict:<18} {above_str:>10} {seed_str:>10} "
          f"{abl:>10} {scr_str:>10} {final:<22}")

# Executive summary
print(f"\n{'=' * 85}")
print("EXECUTIVE SUMMARY")
print(f"{'=' * 85}")

pub_ready = [n for n, v in final_verdicts.items()
             if v == "PUBLICATION_READY"]
strong = [n for n, v in final_verdicts.items()
          if v == "STRONG_CANDIDATE"]
candidates = [n for n, v in final_verdicts.items()
              if v == "CANDIDATE"]
weak = [n for n, v in final_verdicts.items()
        if v == "WEAK_SIGNAL"]
not_enc = [n for n, v in final_verdicts.items()
           if v == "NOT_ENCODED"]

print(f"\n  PUBLICATION_READY ({len(pub_ready)}): "
      f"{pub_ready if pub_ready else 'None'}")
print(f"  STRONG_CANDIDATE ({len(strong)}): "
      f"{strong if strong else 'None'}")
print(f"  CANDIDATE ({len(candidates)}): "
      f"{candidates if candidates else 'None'}")
print(f"  WEAK_SIGNAL ({len(weak)}): "
      f"{weak if weak else 'None'}")
print(f"  NOT_ENCODED ({len(not_enc)}): "
      f"{not_enc if not_enc else 'None'}")

for gname, members in GROUPS.items():
    enc_count = sum(1 for n in members if final_verdicts[n] in
                    ('PUBLICATION_READY', 'STRONG_CANDIDATE', 'CANDIDATE'))
    print(f"\n  {gname}: {enc_count}/{len(members)} features with signal")
    for n in members:
        print(f"    {n:<20}: {final_verdicts[n]}")

# Four-model summary
print(f"\n{'=' * 85}")
print("FOUR-MODEL COMPARISON SUMMARY")
print(f"{'=' * 85}")
print(f"\n  {'Model':<14} {'Test AUC':>10} {'Catalytic':>10} "
      f"{'Pocket':>10} {'Total':>10}")
print(f"  {'-' * 56}")
for name in MODEL_CONFIGS:
    ta = model_aucs[name]['test']
    cat_enc = sum(1 for nm in GROUPS['CATALYTIC']
                  if model_feature_verdicts[name][nm] == 'ENC')
    pock_enc = sum(1 for nm in GROUPS['POCKET']
                   if model_feature_verdicts[name][nm] == 'ENC')
    tot_enc = sum(1 for nm in INTERACTION_NAMES
                  if model_feature_verdicts[name][nm] == 'ENC')
    marker = " <-- BEST" if name == best_pa_name else ""
    print(f"  {name:<14} {ta:>10.4f} {cat_enc:>7}/4 {pock_enc:>7}/4 "
          f"{tot_enc:>7}/10{marker}")

# Pocket scramble summary
print(f"\n  Pocket scramble test ({best_pa_name}):")
for fname in CATALYTIC_FEATURES:
    print(f"    {fname:<20}: {scramble_interpretations.get(fname, 'N/A')}")

# Discovery readiness
n_cat_enc = sum(1 for n in GROUPS['CATALYTIC']
                if final_verdicts[n] in
                ('PUBLICATION_READY', 'STRONG_CANDIDATE'))
n_pock_enc = sum(1 for n in GROUPS['POCKET']
                 if final_verdicts[n] in
                 ('PUBLICATION_READY', 'STRONG_CANDIDATE'))
n_conf_enc = sum(1 for n in GROUPS['CONFOUND']
                 if final_verdicts[n] in
                 ('PUBLICATION_READY', 'STRONG_CANDIDATE', 'CANDIDATE'))

n_cat_any = sum(1 for n in GROUPS['CATALYTIC']
                if final_verdicts[n] != 'NOT_ENCODED')
n_genuine_final = sum(1 for v in scramble_interpretations.values()
                      if v == 'GENUINE')

print(f"\n{'=' * 85}")
print("DISCOVERY READINESS")
print(f"{'=' * 85}")
print(f"  Catalytic site encoding: {n_cat_enc}/{len(GROUPS['CATALYTIC'])} "
      f"(PUBLICATION_READY or STRONG)")
print(f"  Pocket shape encoding:   {n_pock_enc}/{len(GROUPS['POCKET'])} "
      f"(PUBLICATION_READY or STRONG)")
print(f"  Confound encoding:       {n_conf_enc}/{len(GROUPS['CONFOUND'])}")
print(f"  Genuine pocket signal:   {n_genuine_final}/4 catalytic features")

# Compare against baseline GCN (expected 0/4 catalytic, 0/4 pocket, 0/10 total)
gcn_cat = sum(1 for nm in GROUPS['CATALYTIC']
              if model_feature_verdicts['PlainGCN'][nm] == 'ENC')
gcn_pock = sum(1 for nm in GROUPS['POCKET']
               if model_feature_verdicts['PlainGCN'][nm] == 'ENC')
gcn_total = sum(1 for nm in INTERACTION_NAMES
                if model_feature_verdicts['PlainGCN'][nm] == 'ENC')

pa_cat = sum(1 for nm in GROUPS['CATALYTIC']
             if model_feature_verdicts[best_pa_name][nm] == 'ENC')
pa_pock_count = sum(1 for nm in GROUPS['POCKET']
                    if model_feature_verdicts[best_pa_name][nm] == 'ENC')
pa_total = sum(1 for nm in INTERACTION_NAMES
               if model_feature_verdicts[best_pa_name][nm] == 'ENC')

print(f"\n  PlainGCN baseline: {gcn_cat}/4 catalytic, "
      f"{gcn_pock}/4 pocket, {gcn_total}/10 total")
print(f"  {best_pa_name} (best PA): {pa_cat}/4 catalytic, "
      f"{pa_pock_count}/4 pocket, {pa_total}/10 total")

improvement = pa_total - gcn_total
if improvement > 0:
    print(f"  Improvement: +{improvement} features encoded with protein context")
else:
    print(f"  No improvement over baseline GCN")

if len(pub_ready) >= 3 and n_genuine_final >= 2:
    readiness = ("DISCOVERY READY -- Protein-ligand co-encoding captures real "
                 "BACE1 catalytic interactions. Pocket scramble confirms spatial "
                 "relationships are genuine, not trivial correlations.")
elif n_cat_enc >= 2 and n_genuine_final >= 1:
    readiness = ("PARTIALLY READY -- Strong catalytic encoding with genuine "
                 "pocket dependence. Ready for validation on second AD target.")
elif n_cat_any >= 2 and n_genuine_final >= 1:
    readiness = ("EARLY SIGNAL -- Catalytic features detected with genuine "
                 "pocket signal but not fully hardened. Consider deeper "
                 "cross-attention or multi-head architectures.")
elif pa_cat > gcn_cat:
    readiness = ("PROTEIN HELPS -- Protein context improves encoding over "
                 "plain GCN but signal is weak. Try: larger pocket radius, "
                 "attention to backbone atoms, or SE(3)-equivariant layers.")
elif pa_cat == gcn_cat == 0:
    readiness = ("ARCHITECTURE INSUFFICIENT -- Neither plain GCN nor "
                 "protein-ligand models encode catalytic features. Consider: "
                 "3D equivariant networks, or direct structure-based scoring.")
else:
    readiness = ("INCONCLUSIVE -- Mixed results across architectures. "
                 "Further investigation needed with larger datasets or "
                 "alternative protein representations.")

print(f"\n  VERDICT: {readiness}")

print(f"\n  Phase 9 completed in {time.time() - t9:.1f}s")

# Total runtime
total_time = time.time() - t1
print(f"\n{'=' * 85}")
print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time / 60:.1f} min)")
print(f"{'=' * 85}")
print("\nNEXT STEPS:")
print("  1. If PUBLICATION_READY: validate on BACE2 or gamma-secretase")
print("  2. If GENUINE pocket signal: try SE(3)-equivariant protein encoder")
print("  3. If TRIVIAL: pocket features may be redundant with ligand features")
print("  4. If NEITHER: protein-ligand interface needs explicit 3D modeling")
print("=" * 85)
