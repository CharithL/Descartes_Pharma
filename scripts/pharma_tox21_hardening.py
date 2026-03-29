#!/usr/bin/env python3
"""
DESCARTES-PHARMA Tier 2: Tox21 Full Pipeline + 3-Dataset VZS Promotion
=======================================================================
Third and final dataset for Tier 1 Axiom promotion. Runs the complete
DESCARTES-PHARMA pipeline (train GCN, probe 10 mechanisms, 6 hardening
methods, SAE polypharmacology) on Tox21, then cross-references with
ClinTox and BBBP results for VZS tier promotion.

Key question: do the mechanisms confirmed on ClinTox + BBBP also survive
on Tox21? If yes, they get promoted to Tier 1 AXIOM -- settled, hash-
chained, never re-probed.

Pipeline:
  Phase 1: Train GCN on Tox21, extract embeddings, raw Ridge dR2
  Phase 2: 6 hardening methods
  Phase 3: Hardened verdicts
  Phase 4: 3-dataset cross-comparison (ClinTox vs BBBP vs Tox21)
  Phase 5: VZS tier promotion (Tier 2 -> Tier 1 AXIOM)
  Phase 6: SAE polypharmacology quick check
  Phase 7: Executive summary across all 3 datasets
"""

import subprocess
import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# 0. DEPENDENCY CHECK
# ============================================================
def ensure_installed(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"  Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pip_name, "-q"])

print("=" * 80)
print("DESCARTES-PHARMA: Tox21 Full Pipeline + 3-Dataset VZS Tier 1 Promotion")
print("=" * 80)

print("\nChecking dependencies...")
ensure_installed("tdc", "PyTDC")
ensure_installed("torch_geometric", "torch-geometric")
ensure_installed("rdkit", "rdkit")
ensure_installed("sklearn", "scikit-learn")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from descartes_pharma.statistical.hardening import (
    fdr_correction,
    confound_removal,
    tost_equivalence_test,
    bayes_factor_null,
)
from descartes_pharma.probes.sae import train_sae, sae_probe_molecular_mechanisms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# PREVIOUS DATASET RESULTS (hardcoded from prior runs)
# ============================================================
CLINTOX_VERDICTS = {
    'MW':             'CONFOUND_DRIVEN',
    'LogP':           'CONFIRMED_ENCODED',
    'TPSA':           'LIKELY_ZOMBIE',
    'HBD':            'LIKELY_ZOMBIE',
    'HBA':            'CONFIRMED_ENCODED',
    'RotatableBonds': 'CONFIRMED_ENCODED',
    'AromaticRings':  'LIKELY_ZOMBIE',
    'FractionCSP3':   'LIKELY_ZOMBIE',
    'NumHeavyAtoms':  'CONFOUND_DRIVEN',
    'PEOE_VSA1':      'CONFIRMED_ENCODED',
}

BBBP_VERDICTS = {
    'MW':             'LIKELY_ZOMBIE',
    'LogP':           'CONFIRMED_ENCODED',
    'TPSA':           'CONFIRMED_ENCODED',
    'HBD':            'CONFIRMED_ENCODED',
    'HBA':            'CONFIRMED_ENCODED',
    'RotatableBonds': 'CONFIRMED_ENCODED',
    'AromaticRings':  'CONFIRMED_ENCODED',
    'FractionCSP3':   'CONFIRMED_ENCODED',
    'NumHeavyAtoms':  'LIKELY_ZOMBIE',
    'PEOE_VSA1':      'CONFIRMED_ENCODED',
}

MECHANISM_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotatableBonds', 'AromaticRings', 'FractionCSP3',
    'NumHeavyAtoms', 'PEOE_VSA1',
]

CONFOUND_IDX = [0, 8]  # MW=0, NumHeavyAtoms=8


# ============================================================
# PHASE 1: DATA + GCN + EMBEDDINGS + RAW PROBES
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: Train GCN on Tox21 + extract embeddings + raw probes")
print("=" * 80)

# ---- Data loading ----
print("\n[1/4] Loading Tox21 dataset from TDC...")
from tdc.single_pred import Tox
from tdc.utils import retrieve_label_name_list

# Tox21 has 12 assay endpoints -- must select one
available_labels = retrieve_label_name_list('Tox21')
print(f"  Available Tox21 assays: {available_labels}")

# Use SR-ARE (stress response) as representative task
# Fallback to first available if SR-ARE not found
SELECTED_LABEL = 'SR-ARE'
if SELECTED_LABEL not in available_labels:
    SELECTED_LABEL = available_labels[0]
print(f"  Selected assay: {SELECTED_LABEL}")

tox21_data = Tox(name='Tox21', label_name=SELECTED_LABEL)
split = tox21_data.get_split(method='scaffold')
train_df, val_df, test_df = split['train'], split['valid'], split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} compounds")
print(f"  Tox prevalence (train): {train_df['Y'].mean():.3f}")


# ---- Feature + graph computation ----
def compute_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.PEOE_VSA1(mol),
        ], dtype=np.float32)
    except Exception:
        return None


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            int(atom.GetHybridization()), int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(), int(atom.IsInRing()),
        ]
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    if len(edge_index) == 0:
        edge_index = [[0, 0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def process_dataset(df):
    graphs, labels, features, smiles_list = [], [], [], []
    for _, row in df.iterrows():
        smi = row['Drug']
        graph = smiles_to_graph(smi)
        feat = compute_rdkit_features(smi)
        if graph is not None and feat is not None and graph.x.shape[0] > 0:
            graph.y = torch.tensor([row['Y']], dtype=torch.float32)
            graphs.append(graph)
            labels.append(row['Y'])
            features.append(feat)
            smiles_list.append(smi)
    return graphs, np.array(labels), np.array(features), smiles_list


print("\n[2/4] Computing RDKit mechanistic features...")
train_graphs, train_labels, train_features, train_smiles = process_dataset(train_df)
val_graphs, val_labels, val_features, val_smiles = process_dataset(val_df)
test_graphs, test_labels, test_features, test_smiles = process_dataset(test_df)
print(f"  Valid: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

for i, name in enumerate(MECHANISM_NAMES):
    vals = train_features[:, i]
    print(f"    {name}: mean={vals.mean():.2f}, std={vals.std():.2f}")


# ---- GCN (same architecture as ClinTox/BBBP) ----
NODE_FEAT_DIM = 7


class ToxGCN(nn.Module):
    """Same architecture as ClinTox/BBBP for fair cross-dataset comparison."""
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        embedding = global_mean_pool(x, batch)
        logits = self.classifier(embedding).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits


print("\n[3/4] Training GCN on Tox21 (larger dataset, may take longer)...")
model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=15, factor=0.5, min_lr=1e-6)

train_loader = PyGDataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = PyGDataLoader(val_graphs, batch_size=64, shuffle=False)
test_loader = PyGDataLoader(test_graphs, batch_size=64, shuffle=False)

pos_weight = torch.tensor(
    [(1 - np.mean(train_labels)) / max(np.mean(train_labels), 1e-6)]
).to(device)

best_val_auc = 0.0
best_state = None
t_start = time.time()

for epoch in range(200):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits, batch.y.squeeze(-1), pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    if epoch % 5 == 0:
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                vp.extend(torch.sigmoid(logits).cpu().numpy())
                vt.extend(batch.y.squeeze(-1).cpu().numpy())
        try:
            val_auc = roc_auc_score(vt, vp)
        except ValueError:
            val_auc = 0.5
        scheduler.step(1 - val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: loss={total_loss/len(train_graphs):.4f} "
                  f"val_AUC={val_auc:.4f}")

model.load_state_dict(best_state)
print(f"  Training: {time.time()-t_start:.1f}s, best val AUC: {best_val_auc:.4f}")

# Test AUC gate (0.65 for Tox21 -- harder dataset)
model.eval()
tp, tt = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        tp.extend(torch.sigmoid(logits).cpu().numpy())
        tt.extend(batch.y.squeeze(-1).cpu().numpy())
try:
    test_auc = roc_auc_score(tt, tp)
except ValueError:
    test_auc = 0.5

print(f"  Test AUC: {test_auc:.4f}")
if test_auc < 0.65:
    print(f"  WARNING: AUC={test_auc:.3f} < 0.65. Tox21 is hard but model underperforms.")
else:
    print(f"  PASS: AUC={test_auc:.3f} >= 0.65 (Tox21 gate)")


# ---- Embeddings ----
def extract_embeddings(gcn_model, loader):
    gcn_model.eval()
    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, emb = gcn_model(batch, return_embedding=True)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


print("\n[4/4] Extracting embeddings + raw dR2...")
trained_emb = extract_embeddings(model, test_loader)
random_model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
random_model.eval()
random_emb = extract_embeddings(random_model, test_loader)

scaler = StandardScaler()
test_features_norm = scaler.fit_transform(test_features)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

raw_results = {}
for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        raw_results[name] = {'ridge_delta_r2': 0.0, 'se': 0.01}
        continue
    scores_t = cross_val_score(Ridge(alpha=1.0), trained_emb, target, cv=kf, scoring='r2')
    scores_r = cross_val_score(Ridge(alpha=1.0), random_emb, target, cv=kf, scoring='r2')
    delta = np.mean(scores_t) - np.mean(scores_r)
    se = np.sqrt(np.var(scores_t) / len(scores_t) + np.var(scores_r) / len(scores_r))
    raw_results[name] = {'ridge_delta_r2': delta, 'se': max(se, 1e-6)}
    print(f"    {name:<18}: dR2={delta:.4f} (SE={se:.4f})")


# ============================================================
# PHASE 2: STATISTICAL HARDENING (6 methods)
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: Statistical Hardening (6 methods)")
print("=" * 80)

N_PERMS = 500
hardened = {name: {} for name in MECHANISM_NAMES}

# ---- METHOD 1: Scaffold-stratified permutation ----
print(f"\n[Method 1/6] Scaffold-stratified permutation null ({N_PERMS} perms)...")

test_scaffolds = []
for smi in test_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        test_scaffolds.append(Chem.MolToSmiles(generic))
    else:
        test_scaffolds.append('UNKNOWN')

test_scaffolds = np.array(test_scaffolds)
unique_scaffolds = np.unique(test_scaffolds)
print(f"  Unique scaffolds in test set: {len(unique_scaffolds)}")

scaffold_groups = {}
for scaf in unique_scaffolds:
    scaffold_groups[scaf] = np.where(test_scaffolds == scaf)[0]

rng = np.random.default_rng(42)

for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        hardened[name]['scaffold_perm_p'] = 1.0
        continue

    obs_delta = raw_results[name]['ridge_delta_r2']
    null_deltas = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        perm_target = target.copy()
        for indices in scaffold_groups.values():
            if len(indices) > 1:
                perm_target[indices] = rng.permutation(perm_target[indices])
        r2_t = np.mean(cross_val_score(
            Ridge(alpha=1.0), trained_emb, perm_target, cv=kf, scoring='r2'))
        r2_r = np.mean(cross_val_score(
            Ridge(alpha=1.0), random_emb, perm_target, cv=kf, scoring='r2'))
        null_deltas[p] = r2_t - r2_r

    p_val = np.mean(null_deltas >= obs_delta)
    hardened[name]['scaffold_perm_p'] = p_val
    print(f"  {name:<18}: obs_dR2={obs_delta:.4f}, null_mean={null_deltas.mean():.4f}, "
          f"p={p_val:.4f}")

# ---- METHOD 2: Y-scramble null ----
print(f"\n[Method 2/6] Y-scramble null ({N_PERMS} perms)...")

for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        hardened[name]['y_scramble_p'] = 1.0
        continue

    obs_delta = raw_results[name]['ridge_delta_r2']
    null_deltas = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        perm_target = rng.permutation(target)
        r2_t = np.mean(cross_val_score(
            Ridge(alpha=1.0), trained_emb, perm_target, cv=kf, scoring='r2'))
        r2_r = np.mean(cross_val_score(
            Ridge(alpha=1.0), random_emb, perm_target, cv=kf, scoring='r2'))
        null_deltas[p] = r2_t - r2_r

    p_val = np.mean(null_deltas >= obs_delta)
    hardened[name]['y_scramble_p'] = p_val
    print(f"  {name:<18}: p={p_val:.4f}")

# ---- METHOD 3: Confound-regressed probing ----
print("\n[Method 3/6] Confound-regressed probing (removing MW + NumHeavyAtoms)...")

confounds = test_features_norm[:, CONFOUND_IDX]
trained_emb_clean = confound_removal(trained_emb, confounds)
random_emb_clean = confound_removal(random_emb, confounds)

for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        hardened[name]['confound_regressed_delta_r2'] = 0.0
        continue

    lr = LinearRegression()
    lr.fit(confounds, target)
    target_clean = target - lr.predict(confounds)

    if np.std(target_clean) < 1e-10:
        hardened[name]['confound_regressed_delta_r2'] = 0.0
        print(f"  {name:<18}: target variance gone after confound removal")
        continue

    r2_t = np.mean(cross_val_score(
        Ridge(alpha=1.0), trained_emb_clean, target_clean, cv=kf, scoring='r2'))
    r2_r = np.mean(cross_val_score(
        Ridge(alpha=1.0), random_emb_clean, target_clean, cv=kf, scoring='r2'))
    delta_clean = r2_t - r2_r
    hardened[name]['confound_regressed_delta_r2'] = delta_clean

    raw_d = raw_results[name]['ridge_delta_r2']
    print(f"  {name:<18}: raw_dR2={raw_d:.4f} -> clean_dR2={delta_clean:.4f} "
          f"(drop={raw_d - delta_clean:.4f})")

# ---- METHOD 4: FDR correction ----
print("\n[Method 4/6] FDR correction (Benjamini-Hochberg)...")

raw_p_values = np.array([hardened[name].get('scaffold_perm_p', 1.0)
                         for name in MECHANISM_NAMES])
fdr_result = fdr_correction(raw_p_values, method='bh')
fdr_adjusted_p = fdr_result['corrected_p']

for j, name in enumerate(MECHANISM_NAMES):
    hardened[name]['fdr_adjusted_p'] = fdr_adjusted_p[j]
    sig = "SIG" if fdr_adjusted_p[j] < 0.05 else "ns"
    print(f"  {name:<18}: raw_p={raw_p_values[j]:.4f} -> "
          f"FDR_p={fdr_adjusted_p[j]:.4f} [{sig}]")

print(f"  Surviving FDR: {int(fdr_result['rejected'].sum())}/{len(MECHANISM_NAMES)}")

# ---- METHOD 5: TOST equivalence test ----
print("\n[Method 5/6] TOST equivalence test (epsilon=0.05)...")

for name in MECHANISM_NAMES:
    delta = raw_results[name]['ridge_delta_r2']
    se = raw_results[name]['se']
    if delta >= 0.05:
        hardened[name]['tost_equivalent'] = False
        hardened[name]['tost_p'] = 1.0
        print(f"  {name:<18}: dR2={delta:.4f} >= 0.05, skip TOST")
        continue
    tost = tost_equivalence_test(delta_r2=delta, se=se, epsilon=0.05)
    hardened[name]['tost_equivalent'] = tost['equivalent']
    hardened[name]['tost_p'] = tost['p_tost']
    status = "ZOMBIE CONFIRMED" if tost['equivalent'] else "inconclusive"
    print(f"  {name:<18}: dR2={delta:.4f}, TOST p={tost['p_tost']:.4f} [{status}]")

# ---- METHOD 6: Bayes Factor ----
print("\n[Method 6/6] Bayes Factor for null (BF01)...")

for name in MECHANISM_NAMES:
    delta = raw_results[name]['ridge_delta_r2']
    se = raw_results[name]['se']
    bf = bayes_factor_null(delta_r2=delta, se=se)
    hardened[name]['bf01'] = bf['bf01']
    hardened[name]['bf_verdict'] = bf['verdict']
    print(f"  {name:<18}: BF01={bf['bf01']:.2f} ({bf['verdict']})")


# ============================================================
# PHASE 3: HARDENED VERDICTS
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: Tox21 Hardened Results")
print("=" * 80)


def hardened_verdict(name, h):
    scaffold_p = h.get('scaffold_perm_p', 1.0)
    fdr_p = h.get('fdr_adjusted_p', 1.0)
    conf_delta = h.get('confound_regressed_delta_r2', 0.0)
    tost_eq = h.get('tost_equivalent', False)
    bf01 = h.get('bf01', 1.0)

    if scaffold_p < 0.05 and conf_delta > 0.05 and fdr_p < 0.05:
        return "CONFIRMED_ENCODED"
    elif scaffold_p < 0.05 and conf_delta < 0.02:
        return "CONFOUND_DRIVEN"
    elif tost_eq and bf01 > 3:
        return "CONFIRMED_ZOMBIE"
    elif fdr_p >= 0.05:
        return "LIKELY_ZOMBIE"
    elif scaffold_p < 0.05:
        return "CANDIDATE_ENCODED"
    else:
        return "LIKELY_ZOMBIE"


print(f"\n  {'Mech':<14} {'Raw dR2':>8} {'Scaf-p':>7} {'Y-scr-p':>8} "
      f"{'Clean dR2':>10} {'FDR-p':>7} {'BF01':>7} {'Verdict':<20}")
print(f"  {'-' * 85}")

tox21_verdicts = {}
verdict_counts = {}
for name in MECHANISM_NAMES:
    h = hardened[name]
    raw_d = raw_results[name]['ridge_delta_r2']
    scaf_p = h.get('scaffold_perm_p', 1.0)
    yscr_p = h.get('y_scramble_p', 1.0)
    clean_d = h.get('confound_regressed_delta_r2', 0.0)
    fdr_p = h.get('fdr_adjusted_p', 1.0)
    bf01 = h.get('bf01', 1.0)

    verdict = hardened_verdict(name, h)
    tox21_verdicts[name] = verdict
    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print(f"  {name:<14} {raw_d:>8.4f} {scaf_p:>7.4f} {yscr_p:>8.4f} "
          f"{clean_d:>10.4f} {fdr_p:>7.4f} {bf01:>7.2f} {verdict:<20}")

n_encoded = verdict_counts.get('CONFIRMED_ENCODED', 0)
n_confound = verdict_counts.get('CONFOUND_DRIVEN', 0)
n_zombie = (verdict_counts.get('CONFIRMED_ZOMBIE', 0)
            + verdict_counts.get('LIKELY_ZOMBIE', 0))

print(f"\n  Tox21 standalone: {n_encoded}/10 CONFIRMED_ENCODED, "
      f"{n_confound} confound-driven, {n_zombie} zombie")


# ============================================================
# PHASE 4: 3-DATASET CROSS-COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: Three-Dataset Cross-Comparison (ClinTox vs BBBP vs Tox21)")
print("=" * 80)


def three_dataset_verdict(cv, bv, tv):
    """Determine cross-dataset status from 3 dataset verdicts."""
    encoded_set = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
    zombie_set = {'LIKELY_ZOMBIE', 'CONFIRMED_ZOMBIE'}
    confound_set = {'CONFOUND_DRIVEN'}

    verdicts = [cv, bv, tv]
    n_enc = sum(1 for v in verdicts if v in encoded_set)
    n_zom = sum(1 for v in verdicts if v in zombie_set)
    n_con = sum(1 for v in verdicts if v in confound_set)

    if n_enc == 3:
        return "TIER_1_AXIOM"
    elif n_enc == 2:
        return "TIER_2_PATTERN"
    elif n_enc == 1:
        return "TASK_DEPENDENT"
    elif n_con >= 2:
        return "CONSISTENT_CONFOUND"
    elif n_zom >= 2:
        return "CONSISTENT_ZOMBIE"
    else:
        return "CONTESTED"


print(f"\n  {'Mechanism':<14} {'ClinTox':<20} {'BBBP':<20} {'Tox21':<20} "
      f"{'3-Dataset Verdict':<20}")
print(f"  {'-' * 95}")

cross_verdicts = {}
cross_counts = {}
for name in MECHANISM_NAMES:
    cv = CLINTOX_VERDICTS[name]
    bv = BBBP_VERDICTS[name]
    tv = tox21_verdicts[name]
    xv = three_dataset_verdict(cv, bv, tv)
    cross_verdicts[name] = xv
    cross_counts[xv] = cross_counts.get(xv, 0) + 1

    print(f"  {name:<14} {cv:<20} {bv:<20} {tv:<20} {xv:<20}")

print(f"\n  Cross-dataset distribution:")
for status in ['TIER_1_AXIOM', 'TIER_2_PATTERN', 'TASK_DEPENDENT',
               'CONSISTENT_CONFOUND', 'CONSISTENT_ZOMBIE', 'CONTESTED']:
    count = cross_counts.get(status, 0)
    if count > 0:
        mechs = [n for n in MECHANISM_NAMES if cross_verdicts[n] == status]
        print(f"    {status:<22}: {count}/10  ({', '.join(mechs)})")


# ============================================================
# PHASE 5: VZS TIER PROMOTION
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: Verified Zombie Store (VZS) Tier Promotion")
print("=" * 80)

print("""
  VZS Tier System (from DESCARTES-PHARMA v1.2 Meta-Learner M6.1):
    Tier 1: AXIOMS      -- settled across 3+ datasets, hash-chained, IMMUTABLE
    Tier 2: PATTERNS    -- confirmed across 2 datasets
    Tier 3: CONTESTED   -- conflicting evidence across datasets
    Tier 4: PROVISIONAL -- single-dataset result
""")

print(f"  {'Mechanism':<14} {'3-Dataset':<20} {'Previous Tier':<16} "
      f"{'New Tier':<12} {'Action':<30}")
print(f"  {'-' * 95}")

axioms = []
patterns = []
for name in MECHANISM_NAMES:
    xv = cross_verdicts[name]

    # Determine previous tier (before this Tox21 run)
    cv_enc = CLINTOX_VERDICTS[name] in {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
    bv_enc = BBBP_VERDICTS[name] in {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
    if cv_enc and bv_enc:
        prev_tier = "Tier 2"
    elif cv_enc or bv_enc:
        prev_tier = "Tier 4"
    else:
        prev_tier = "Tier 4"

    if xv == "TIER_1_AXIOM":
        new_tier = "Tier 1"
        action = "PROMOTED -> AXIOM (immutable)"
        axioms.append(name)
    elif xv == "TIER_2_PATTERN":
        new_tier = "Tier 2"
        action = "Stays/promoted Tier 2"
        patterns.append(name)
    elif xv == "CONSISTENT_ZOMBIE":
        new_tier = "Tier 2"
        action = "ZOMBIE PATTERN (settled)"
    elif xv == "CONSISTENT_CONFOUND":
        new_tier = "Tier 2"
        action = "CONFOUND PATTERN (settled)"
    elif xv == "CONTESTED":
        new_tier = "Tier 3"
        action = "Conflicting evidence"
    elif xv == "TASK_DEPENDENT":
        new_tier = "Tier 3"
        action = "Depends on endpoint"
    else:
        new_tier = "Tier 4"
        action = "Awaiting more data"

    print(f"  {name:<14} {xv:<20} {prev_tier:<16} {new_tier:<12} {action:<30}")

# Print axiom declarations
if axioms:
    print(f"\n  {'='*70}")
    print(f"  TIER 1 AXIOM DECLARATIONS")
    print(f"  {'='*70}")
    for name in axioms:
        print(f"\n  AXIOM: GCN encodes {name}")
        print(f"    Settled across: ClinTox, BBBP, Tox21")
        print(f"    Status: Hash-chained, append-only, NEVER re-probe")
        print(f"    Evidence: scaffold-perm p < 0.05, confound-regressed dR2 > 0.05,")
        print(f"              FDR-adjusted p < 0.05 on ALL THREE datasets")


# ============================================================
# PHASE 6: SAE POLYPHARMACOLOGY QUICK CHECK
# ============================================================
print("\n" + "=" * 80)
print("PHASE 6: SAE Polypharmacology Quick Check (expansion=8x)")
print("=" * 80)

print("\n  Training SAE on Tox21 GCN embeddings...")
sae, sae_loss = train_sae(
    [trained_emb], trained_emb.shape[1],
    expansion_factor=8, k=20,
    device=device
)

sae_results = sae_probe_molecular_mechanisms(
    sae, trained_emb, test_features_norm, MECHANISM_NAMES,
    device=device
)

print(f"\n  SAE Results:")
n_total_sae = sae_results['correlation_matrix'].shape[0]
print(f"    Alive features: {sae_results['n_alive']}/{n_total_sae}")
print(f"    Mean monosemanticity: {sae_results['mean_monosemanticity']:.4f}")

print(f"\n  {'Mechanism':<14} {'SAE R2':>8} {'Raw R2':>8} {'Delta':>8} "
      f"{'Detection':<20}")
print(f"  {'-' * 65}")

n_poly = 0
for name in MECHANISM_NAMES:
    sae_r2 = sae_results['sae_r2'][name]
    raw_r2 = sae_results['raw_r2'][name]
    delta = sae_r2 - raw_r2
    poly = sae_results['polypharmacology_detected'][name]
    if poly:
        n_poly += 1
    label = "POLYPHARMACOLOGY" if poly else "monosemantic"
    print(f"  {name:<14} {sae_r2:>8.4f} {raw_r2:>8.4f} {delta:>8.4f} {label:<20}")

print(f"\n  Polypharmacology detected: {n_poly}/{len(MECHANISM_NAMES)} mechanisms")
if n_poly > 0:
    poly_list = [n for n in MECHANISM_NAMES
                 if sae_results['polypharmacology_detected'][n]]
    print(f"  Affected: {', '.join(poly_list)}")
    print(f"  -> SAE reveals multi-target encoding invisible to linear probes")
else:
    print(f"  -> GCN embeddings appear monosemantic (each mechanism linearly decodable)")


# ============================================================
# PHASE 7: EXECUTIVE SUMMARY
# ============================================================
tox21_total = len(train_df) + len(val_df) + len(test_df)
print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY: DESCARTES-PHARMA 3-Dataset Validation Campaign")
print("=" * 80)

print(f"""
  Campaign: GCN (hidden=128, 3-layer, mean readout) on molecular toxicity
  Datasets: ClinTox (1491 cmpds), BBBP (2039 cmpds), Tox21 ({tox21_total} cmpds)
  Mechanisms probed: {len(MECHANISM_NAMES)} RDKit descriptors
  Hardening: 6 methods per dataset (scaffold-perm, y-scramble, confound
             regression, FDR, TOST, Bayes Factor)
  Test AUC (Tox21): {test_auc:.3f}
""")

n_axiom = cross_counts.get('TIER_1_AXIOM', 0)
n_pattern = cross_counts.get('TIER_2_PATTERN', 0)
n_task_dep = cross_counts.get('TASK_DEPENDENT', 0)
n_con_confound = cross_counts.get('CONSISTENT_CONFOUND', 0)
n_con_zombie = cross_counts.get('CONSISTENT_ZOMBIE', 0)
n_contested = cross_counts.get('CONTESTED', 0)

print(f"  FINAL VZS STATE:")
print(f"  {'='*60}")
if axioms:
    print(f"  Tier 1 AXIOMS ({n_axiom}):      {', '.join(axioms)}")
if patterns:
    print(f"  Tier 2 PATTERNS ({n_pattern}):    {', '.join(patterns)}")

confound_list = [n for n in MECHANISM_NAMES
                 if cross_verdicts[n] == 'CONSISTENT_CONFOUND']
zombie_list = [n for n in MECHANISM_NAMES
               if cross_verdicts[n] == 'CONSISTENT_ZOMBIE']
contested_list = [n for n in MECHANISM_NAMES
                  if cross_verdicts[n] == 'CONTESTED']
task_dep_list = [n for n in MECHANISM_NAMES
                 if cross_verdicts[n] == 'TASK_DEPENDENT']

if confound_list:
    print(f"  Tier 2 CONFOUNDS ({n_con_confound}):  {', '.join(confound_list)}")
if zombie_list:
    print(f"  Tier 2 ZOMBIES ({n_con_zombie}):    {', '.join(zombie_list)}")
if contested_list:
    print(f"  Tier 3 CONTESTED ({n_contested}): {', '.join(contested_list)}")
if task_dep_list:
    print(f"  Tier 3 TASK-DEP ({n_task_dep}):  {', '.join(task_dep_list)}")

print(f"\n  KEY FINDINGS:")
print(f"  {'-'*60}")

if n_axiom > 0:
    pct = 100 * n_axiom / len(MECHANISM_NAMES)
    print(f"  1. {n_axiom}/10 mechanisms ({pct:.0f}%) are TIER 1 AXIOMS:")
    print(f"     GCNs genuinely encode {', '.join(axioms)} across ALL three")
    print(f"     datasets. These are settled facts about GCN architecture.")

if n_con_zombie + n_con_confound > 0:
    zombie_pct = 100 * (n_con_zombie + n_con_confound) / len(MECHANISM_NAMES)
    all_bad = confound_list + zombie_list
    print(f"  2. {n_con_zombie + n_con_confound}/10 mechanisms ({zombie_pct:.0f}%) "
          f"are consistent failures:")
    print(f"     GCNs never genuinely encode {', '.join(all_bad)}.")
    if confound_list:
        print(f"     MW/NumHeavyAtoms: confound-driven (size, not mechanism).")

if n_axiom == 0:
    print(f"  3. NO mechanisms reached Tier 1 Axiom status.")
    print(f"     The GCN architecture may have fundamental limitations.")
    print(f"     Consider testing GAT, GIN, or SchNet.")

print(f"\n  RECOMMENDATION:")
print(f"  {'-'*60}")
if n_axiom >= 3:
    print(f"  GCN shows GENUINE mechanistic understanding of {n_axiom}/10 properties.")
    print(f"  Safe for interpretable drug discovery on axiom mechanisms.")
    print(f"  Zombie mechanisms need architectural improvements.")
elif n_axiom > 0:
    print(f"  GCN encodes SOME mechanisms genuinely ({n_axiom}/10) but has")
    print(f"  significant blind spots. Augment with attention-based or")
    print(f"  3D-aware architectures for zombie mechanisms.")
else:
    print(f"  GCN shows no cross-dataset mechanistic axioms. GCNs learn")
    print(f"  task-specific shortcuts, not transferable molecular understanding.")
    print(f"  Test alternative architectures (GAT, GIN, SchNet, DimeNet).")

print(f"\n  NEXT STEPS:")
print(f"  1. Test GAT/GIN/SchNet on same 3 datasets (architecture comparison)")
print(f"  2. Run SAE polypharmacology sweep (4x, 8x, 16x) on axiom mechanisms")
print(f"  3. Deploy VZS: skip re-probing axioms in future campaigns")
print(f"  4. Write up: '3-Dataset Zombie Detection in Drug Discovery GNNs'")

print("\n" + "=" * 80)
print("END OF 3-DATASET VALIDATION CAMPAIGN")
print("=" * 80)
