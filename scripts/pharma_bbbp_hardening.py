#!/usr/bin/env python3
"""
DESCARTES-PHARMA Tier 2: BBBP Full Pipeline + Cross-Dataset Validation
=======================================================================
Runs the complete DESCARTES-PHARMA pipeline on the Blood-Brain Barrier
Penetration dataset (BBBP): train GCN, probe 10 mechanisms, apply 6
hardening methods, then compare with ClinTox results for cross-dataset
Verified Zombie Store (VZS) tier promotion.

Key question: do the 4 mechanisms that survived hardening on ClinTox
(LogP, HBA, RotatableBonds, PEOE_VSA1) also survive on BBBP?
If yes, they get promoted from Tier 4 (provisional) to Tier 2 (pattern).

Pipeline:
  Phase 1: Train GCN on BBBP, extract embeddings, raw Ridge dR2
  Phase 2: 6 hardening methods (scaffold-perm, y-scramble, confound
           regression, FDR, TOST, Bayes Factor)
  Phase 3: Hardened verdicts
  Phase 4: Cross-dataset comparison (BBBP vs ClinTox)
  Phase 5: VZS tier promotion summary
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

print("=" * 75)
print("DESCARTES-PHARMA: BBBP Full Pipeline + Cross-Dataset Validation")
print("=" * 75)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# ClinTox hardened results (from pharma_clintox_hardening.py)
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

MECHANISM_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotatableBonds', 'AromaticRings', 'FractionCSP3',
    'NumHeavyAtoms', 'PEOE_VSA1',
]

CONFOUND_IDX = [0, 8]  # MW=0, NumHeavyAtoms=8

# ============================================================
# PHASE 1: DATA + GCN + EMBEDDINGS + RAW PROBES
# ============================================================
print("\n" + "=" * 75)
print("PHASE 1: Train GCN on BBBP + extract embeddings + raw probes")
print("=" * 75)

# ---- Data loading ----
print("\n[1/4] Loading BBBP dataset from TDC...")
from tdc.single_pred import ADME

bbbp_data = ADME(name='BBB_Martins')
split = bbbp_data.get_split(method='scaffold')
train_df, val_df, test_df = split['train'], split['valid'], split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"  BBB+ prevalence (train): {train_df['Y'].mean():.3f}")


# ---- Feature computation ----
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

# ---- GCN ----
NODE_FEAT_DIM = 7


class ToxGCN(nn.Module):
    """Same architecture as ClinTox for fair cross-dataset comparison."""
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


print("\n[3/4] Training GCN on BBBP...")
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

# Test AUC gate
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
if test_auc < 0.7:
    print(f"  WARNING: AUC={test_auc:.3f} < 0.7. Model underperforms.")
else:
    print(f"  PASS: AUC={test_auc:.3f} >= 0.7")


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
print("\n" + "=" * 75)
print("PHASE 2: Statistical Hardening (6 methods)")
print("=" * 75)

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
print("\n" + "=" * 75)
print("PHASE 3: BBBP Hardened Results")
print("=" * 75)


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

bbbp_verdicts = {}
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
    bbbp_verdicts[name] = verdict
    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print(f"  {name:<14} {raw_d:>8.4f} {scaf_p:>7.4f} {yscr_p:>8.4f} "
          f"{clean_d:>10.4f} {fdr_p:>7.4f} {bf01:>7.2f} {verdict:<20}")

n_encoded_bbbp = verdict_counts.get('CONFIRMED_ENCODED', 0)
n_confound_bbbp = verdict_counts.get('CONFOUND_DRIVEN', 0)
n_zombie_bbbp = (verdict_counts.get('CONFIRMED_ZOMBIE', 0)
                 + verdict_counts.get('LIKELY_ZOMBIE', 0))

print(f"\n  BBBP standalone: {n_encoded_bbbp}/10 CONFIRMED_ENCODED, "
      f"{n_confound_bbbp} confound-driven, {n_zombie_bbbp} zombie")


# ============================================================
# PHASE 4: CROSS-DATASET COMPARISON (BBBP vs ClinTox)
# ============================================================
print("\n" + "=" * 75)
print("PHASE 4: Cross-Dataset Comparison (BBBP vs ClinTox)")
print("=" * 75)


def cross_dataset_verdict(clintox_v, bbbp_v):
    """Determine cross-dataset status for a mechanism."""
    encoded_set = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
    zombie_set = {'LIKELY_ZOMBIE', 'CONFIRMED_ZOMBIE'}
    confound_set = {'CONFOUND_DRIVEN'}

    if clintox_v in encoded_set and bbbp_v in encoded_set:
        return "AXIOM_CANDIDATE"
    elif clintox_v in zombie_set and bbbp_v in zombie_set:
        return "CONSISTENT_ZOMBIE"
    elif clintox_v in confound_set and bbbp_v in confound_set:
        return "CONSISTENT_CONFOUND"
    elif (clintox_v in encoded_set and bbbp_v not in encoded_set) or \
         (bbbp_v in encoded_set and clintox_v not in encoded_set):
        return "DATASET_SPECIFIC"
    else:
        return "INCONSISTENT"


print(f"\n  {'Mechanism':<16} {'ClinTox':<22} {'BBBP':<22} {'Cross-Dataset':<20}")
print(f"  {'-' * 80}")

cross_verdicts = {}
cross_counts = {}
for name in MECHANISM_NAMES:
    cv = CLINTOX_VERDICTS[name]
    bv = bbbp_verdicts[name]
    xv = cross_dataset_verdict(cv, bv)
    cross_verdicts[name] = xv
    cross_counts[xv] = cross_counts.get(xv, 0) + 1

    print(f"  {name:<16} {cv:<22} {bv:<22} {xv:<20}")

print(f"\n  Cross-dataset distribution:")
for status in ['AXIOM_CANDIDATE', 'DATASET_SPECIFIC', 'CONSISTENT_ZOMBIE',
               'CONSISTENT_CONFOUND', 'INCONSISTENT']:
    count = cross_counts.get(status, 0)
    if count > 0:
        mechs = [n for n in MECHANISM_NAMES if cross_verdicts[n] == status]
        print(f"    {status:<22}: {count}/10  ({', '.join(mechs)})")


# ============================================================
# PHASE 5: VZS TIER PROMOTION SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("PHASE 5: Verified Zombie Store (VZS) Tier Promotion")
print("=" * 75)

print("""
  VZS Tier System (from DESCARTES-PHARMA v1.2 Meta-Learner):
    Tier 1: AXIOMS      -- settled across 3+ datasets, hash-chained
    Tier 2: PATTERNS    -- confirmed across 2 datasets (ClinTox + BBBP)
    Tier 3: CONTESTED   -- conflicting evidence across datasets
    Tier 4: PROVISIONAL -- single-dataset result, awaiting replication
""")

print(f"  {'Mechanism':<16} {'Status':<22} {'Promotion':<35} {'New Tier':<10}")
print(f"  {'-' * 85}")

promoted = []
for name in MECHANISM_NAMES:
    xv = cross_verdicts[name]

    if xv == "AXIOM_CANDIDATE":
        promotion = "Tier 4 -> Tier 2 (PATTERN)"
        new_tier = "Tier 2"
        promoted.append(name)
    elif xv == "CONSISTENT_ZOMBIE":
        promotion = "Tier 4 -> Tier 2 (ZOMBIE PATTERN)"
        new_tier = "Tier 2"
    elif xv == "CONSISTENT_CONFOUND":
        promotion = "Tier 4 -> Tier 2 (CONFOUND PATTERN)"
        new_tier = "Tier 2"
    elif xv == "DATASET_SPECIFIC":
        promotion = "Stays Tier 4 (needs 3rd dataset)"
        new_tier = "Tier 4"
    elif xv == "INCONSISTENT":
        promotion = "-> Tier 3 (CONTESTED)"
        new_tier = "Tier 3"
    else:
        promotion = "Stays Tier 4"
        new_tier = "Tier 4"

    print(f"  {name:<16} {xv:<22} {promotion:<35} {new_tier:<10}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("EXECUTIVE SUMMARY")
print("=" * 75)

print(f"\n  Datasets tested: ClinTox (toxicity), BBBP (BBB penetration)")
print(f"  Architecture: GCN (hidden=128, 3 layers, mean readout)")
print(f"  BBBP test AUC: {test_auc:.3f}")

n_axiom = cross_counts.get('AXIOM_CANDIDATE', 0)
n_ds_specific = cross_counts.get('DATASET_SPECIFIC', 0)
n_con_zombie = cross_counts.get('CONSISTENT_ZOMBIE', 0)
n_con_confound = cross_counts.get('CONSISTENT_CONFOUND', 0)

if n_axiom > 0:
    print(f"\n  AXIOM CANDIDATES ({n_axiom}): {', '.join(promoted)}")
    print(f"  -> These mechanisms are GENUINELY ENCODED by GCNs across datasets")
    print(f"  -> Promoted to VZS Tier 2 (Pattern). Need 1 more dataset for Tier 1.")
    print(f"  -> Next step: test on Tox21 or MoleculeNet HIV for Tier 1 promotion.")

if n_con_zombie > 0:
    zombie_list = [n for n in MECHANISM_NAMES if cross_verdicts[n] == 'CONSISTENT_ZOMBIE']
    print(f"\n  CONSISTENT ZOMBIES ({n_con_zombie}): {', '.join(zombie_list)}")
    print(f"  -> GCNs consistently FAIL to encode these mechanisms.")
    print(f"  -> Reproducible architectural limitation, not noise.")

if n_con_confound > 0:
    conf_list = [n for n in MECHANISM_NAMES if cross_verdicts[n] == 'CONSISTENT_CONFOUND']
    print(f"\n  CONSISTENT CONFOUNDS ({n_con_confound}): {', '.join(conf_list)}")
    print(f"  -> GCNs encode these via trivial size/weight correlation.")

if n_ds_specific > 0:
    ds_list = [n for n in MECHANISM_NAMES if cross_verdicts[n] == 'DATASET_SPECIFIC']
    print(f"\n  DATASET-SPECIFIC ({n_ds_specific}): {', '.join(ds_list)}")
    print(f"  -> Encoded on one dataset but not the other.")
    print(f"  -> May depend on the endpoint, not the architecture.")

print(f"\n  Overall: {n_axiom}/10 mechanisms replicate across both datasets.")
print(f"  GCN zombie fraction (cross-dataset): "
      f"{(n_con_zombie + n_con_confound)}/10 = "
      f"{100*(n_con_zombie + n_con_confound)/10:.0f}%")

print("\n" + "=" * 75)
print("NEXT STEPS")
print("=" * 75)
print(f"  1. Run Tox21 or HIV pipeline for Tier 1 axiom promotion")
print(f"  2. Test GAT/GIN architectures to see if zombie patterns persist")
print(f"  3. Run SAE polypharmacology on BBBP (detect multi-target effects)")
print(f"  4. Compare with 3D-equivariant models (SchNet/DimeNet)")
print("=" * 75)
