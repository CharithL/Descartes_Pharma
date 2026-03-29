#!/usr/bin/env python3
"""
DESCARTES-PHARMA Tier 2: ClinTox Statistical Hardening
=======================================================
Applies the 6 most important methods from the DESCARTES-PHARMA
Statistical Hardening Suite (Section 8) to the ClinTox probing
results. This answers: do the "10/10 ENCODED" verdicts from
pharma_clintox_test.py survive rigorous statistical testing,
or are they scaffold confounds and multiple-testing artifacts?

Methods applied:
  1. Scaffold-stratified permutation null (500 perms)
  2. Y-scramble null (500 perms)
  3. Confound-regressed probing (remove MW + NumHeavyAtoms)
  4. FDR correction (Benjamini-Hochberg across 10 mechanisms)
  5. TOST equivalence test (confirm zombie status for dR2 < 0.05)
  6. Bayes Factor for null (BF01 > 3 = moderate zombie evidence)
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

print("=" * 70)
print("DESCARTES-PHARMA: ClinTox Statistical Hardening Suite")
print("=" * 70)

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

# Import hardening functions from the project module
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
# 1. DATA LOADING + GCN TRAINING (reused from pharma_clintox_test.py)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: Reproduce baseline (GCN + embeddings + raw probes)")
print("=" * 70)

print("\n[1/3] Loading ClinTox dataset...")
from tdc.single_pred import Tox

tox_data = Tox(name='ClinTox')
split = tox_data.get_split(method='scaffold')
train_df, val_df, test_df = split['train'], split['valid'], split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

MECHANISM_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotatableBonds', 'AromaticRings', 'FractionCSP3',
    'NumHeavyAtoms', 'PEOE_VSA1',
]

# Indices of confound features: MW=0, NumHeavyAtoms=8
CONFOUND_IDX = [0, 8]
CONFOUND_NAMES = ['MW', 'NumHeavyAtoms']


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


train_graphs, train_labels, train_features, train_smiles = process_dataset(train_df)
val_graphs, val_labels, val_features, val_smiles = process_dataset(val_df)
test_graphs, test_labels, test_features, test_smiles = process_dataset(test_df)

print(f"  Valid: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

NODE_FEAT_DIM = 7


class ToxGCN(nn.Module):
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


# Train GCN
print("\n[2/3] Training GCN...")
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
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(batch.y.squeeze(-1).cpu().numpy())
        try:
            val_auc = roc_auc_score(val_true, val_preds)
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
test_preds, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        test_preds.extend(torch.sigmoid(logits).cpu().numpy())
        test_true.extend(batch.y.squeeze(-1).cpu().numpy())
try:
    test_auc = roc_auc_score(test_true, test_preds)
except ValueError:
    test_auc = 0.5
print(f"  Test AUC: {test_auc:.4f}")


# Extract embeddings
def extract_embeddings(gcn_model, loader):
    gcn_model.eval()
    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, emb = gcn_model(batch, return_embedding=True)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


print("\n[3/3] Extracting embeddings...")
trained_emb = extract_embeddings(model, test_loader)
random_model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
random_model.eval()
random_emb = extract_embeddings(random_model, test_loader)

scaler = StandardScaler()
test_features_norm = scaler.fit_transform(test_features)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Compute raw Ridge dR2 for each mechanism (baseline)
print("\n  Computing raw Ridge dR2 baseline...")
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
    raw_results[name] = {
        'ridge_delta_r2': delta,
        'se': max(se, 1e-6),
        'fold_scores_t': scores_t,
        'fold_scores_r': scores_r,
    }
    print(f"    {name:<18}: dR2={delta:.4f} (SE={se:.4f})")

# ============================================================
# PHASE 2: STATISTICAL HARDENING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Statistical Hardening (6 methods)")
print("=" * 70)

N_PERMS = 500
hardened = {name: {} for name in MECHANISM_NAMES}

# ----------------------------------------------------------
# METHOD 1: Scaffold-stratified permutation null
# ----------------------------------------------------------
print(f"\n[Method 1/6] Scaffold-stratified permutation null ({N_PERMS} perms)...")

# Compute Murcko scaffolds for test molecules
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

# Build scaffold group index once
scaffold_groups = {}
for scaf in unique_scaffolds:
    scaffold_groups[scaf] = np.where(test_scaffolds == scaf)[0]

rng = np.random.default_rng(42)

for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        hardened[name]['scaffold_perm_p'] = 1.0
        continue

    # Observed dR2
    obs_delta = raw_results[name]['ridge_delta_r2']

    # Generate null distribution by permuting within scaffolds, then re-probing
    null_deltas = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        perm_target = target.copy()
        for indices in scaffold_groups.values():
            if len(indices) > 1:
                perm_target[indices] = rng.permutation(perm_target[indices])
        # Probe with permuted targets
        r2_t_perm = np.mean(cross_val_score(
            Ridge(alpha=1.0), trained_emb, perm_target, cv=kf, scoring='r2'))
        r2_r_perm = np.mean(cross_val_score(
            Ridge(alpha=1.0), random_emb, perm_target, cv=kf, scoring='r2'))
        null_deltas[p] = r2_t_perm - r2_r_perm

    p_val = np.mean(null_deltas >= obs_delta)
    hardened[name]['scaffold_perm_p'] = p_val
    print(f"  {name:<18}: obs_dR2={obs_delta:.4f}, "
          f"null_mean={null_deltas.mean():.4f}, p={p_val:.4f}")

# ----------------------------------------------------------
# METHOD 2: Y-scramble null
# ----------------------------------------------------------
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
        r2_t_perm = np.mean(cross_val_score(
            Ridge(alpha=1.0), trained_emb, perm_target, cv=kf, scoring='r2'))
        r2_r_perm = np.mean(cross_val_score(
            Ridge(alpha=1.0), random_emb, perm_target, cv=kf, scoring='r2'))
        null_deltas[p] = r2_t_perm - r2_r_perm

    p_val = np.mean(null_deltas >= obs_delta)
    hardened[name]['y_scramble_p'] = p_val
    print(f"  {name:<18}: p={p_val:.4f}")

# ----------------------------------------------------------
# METHOD 3: Confound-regressed probing
# ----------------------------------------------------------
print("\n[Method 3/6] Confound-regressed probing (removing MW + NumHeavyAtoms)...")

confounds = test_features_norm[:, CONFOUND_IDX]

# Regress confounds out of BOTH embeddings and targets
trained_emb_clean = confound_removal(trained_emb, confounds)
random_emb_clean = confound_removal(random_emb, confounds)

for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]
    if np.std(target) < 1e-10:
        hardened[name]['confound_regressed_delta_r2'] = 0.0
        continue

    # Regress confounds out of target too
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
    drop = raw_d - delta_clean
    print(f"  {name:<18}: raw_dR2={raw_d:.4f} -> clean_dR2={delta_clean:.4f} "
          f"(drop={drop:.4f})")

# ----------------------------------------------------------
# METHOD 4: FDR correction (Benjamini-Hochberg)
# ----------------------------------------------------------
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

n_rejected = int(fdr_result['rejected'].sum())
print(f"  Mechanisms surviving FDR: {n_rejected}/{len(MECHANISM_NAMES)}")

# ----------------------------------------------------------
# METHOD 5: TOST equivalence test (zombie confirmation)
# ----------------------------------------------------------
print("\n[Method 5/6] TOST equivalence test (epsilon=0.05)...")

for name in MECHANISM_NAMES:
    delta = raw_results[name]['ridge_delta_r2']
    se = raw_results[name]['se']

    if delta >= 0.05:
        hardened[name]['tost_equivalent'] = False
        hardened[name]['tost_p'] = 1.0
        print(f"  {name:<18}: dR2={delta:.4f} >= 0.05, skipping TOST (not zombie candidate)")
        continue

    tost_result = tost_equivalence_test(delta_r2=delta, se=se, epsilon=0.05)
    hardened[name]['tost_equivalent'] = tost_result['equivalent']
    hardened[name]['tost_p'] = tost_result['p_tost']
    status = "ZOMBIE CONFIRMED" if tost_result['equivalent'] else "inconclusive"
    print(f"  {name:<18}: dR2={delta:.4f}, TOST p={tost_result['p_tost']:.4f} [{status}]")

# ----------------------------------------------------------
# METHOD 6: Bayes Factor for null
# ----------------------------------------------------------
print("\n[Method 6/6] Bayes Factor for null (BF01)...")

for name in MECHANISM_NAMES:
    delta = raw_results[name]['ridge_delta_r2']
    se = raw_results[name]['se']
    bf_result = bayes_factor_null(delta_r2=delta, se=se)
    hardened[name]['bf01'] = bf_result['bf01']
    hardened[name]['bf_verdict'] = bf_result['verdict']
    print(f"  {name:<18}: BF01={bf_result['bf01']:.2f} ({bf_result['verdict']})")


# ============================================================
# PHASE 3: HARDENED VERDICTS
# ============================================================
print("\n" + "=" * 70)
print("HARDENED RESULTS TABLE")
print("=" * 70)


# Verdict logic
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


# Header
print(f"\n  {'Mech':<14} {'Raw dR2':>8} {'Scaf-p':>7} {'Y-scr-p':>8} "
      f"{'Clean dR2':>10} {'FDR-p':>7} {'BF01':>7} {'Verdict':<20}")
print(f"  {'-' * 85}")

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
    h['verdict'] = verdict
    verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print(f"  {name:<14} {raw_d:>8.4f} {scaf_p:>7.4f} {yscr_p:>8.4f} "
          f"{clean_d:>10.4f} {fdr_p:>7.4f} {bf01:>7.2f} {verdict:<20}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("HARDENING SUMMARY -- ClinTox Tier 2")
print("=" * 70)

print(f"\n  Test AUC: {test_auc:.3f}")
print(f"  Unique test scaffolds: {len(unique_scaffolds)}")
print(f"  Permutations per method: {N_PERMS}")

print(f"\n  Verdict distribution:")
for verdict in ['CONFIRMED_ENCODED', 'CANDIDATE_ENCODED', 'CONFOUND_DRIVEN',
                 'LIKELY_ZOMBIE', 'CONFIRMED_ZOMBIE']:
    count = verdict_counts.get(verdict, 0)
    if count > 0:
        mechs = [n for n in MECHANISM_NAMES if hardened[n].get('verdict') == verdict]
        print(f"    {verdict:<22}: {count}/10  ({', '.join(mechs)})")

n_naive_encoded = sum(1 for name in MECHANISM_NAMES
                      if raw_results[name]['ridge_delta_r2'] > 0.05)
n_hardened_encoded = verdict_counts.get('CONFIRMED_ENCODED', 0)
n_confound = verdict_counts.get('CONFOUND_DRIVEN', 0)
n_zombie = (verdict_counts.get('CONFIRMED_ZOMBIE', 0)
            + verdict_counts.get('LIKELY_ZOMBIE', 0))

print(f"\n  Naive count (dR2 > 0.05): {n_naive_encoded}/10 mechanisms 'encoded'")
print(f"  After hardening:          {n_hardened_encoded}/10 CONFIRMED_ENCODED")
print(f"  Confound-driven:          {n_confound}/10 (encoding was just MW/size)")
print(f"  Zombie (confirmed+likely): {n_zombie}/10")

if n_hardened_encoded > n_naive_encoded * 0.5:
    print(f"\n  RESULT: Majority of encodings survive hardening!")
    print(f"  -> GCN genuinely encodes toxicological mechanisms.")
elif n_confound > 0:
    print(f"\n  RESULT: {n_confound} mechanisms were CONFOUND-DRIVEN!")
    print(f"  -> GCN partially relies on trivial MW/size features.")
    print(f"  -> True mechanistic encoding is weaker than naive results suggest.")
if n_zombie >= 5:
    print(f"\n  RESULT: {n_zombie}/10 mechanisms are zombies after hardening!")
    print(f"  -> Model may be a partial pharmaceutical zombie.")

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
  CONFIRMED_ENCODED:  Mechanism survives scaffold permutation, confound
                      removal, AND FDR correction. This is real encoding.

  CANDIDATE_ENCODED:  Significant after scaffold permutation but borderline
                      on confound regression. Needs further investigation.

  CONFOUND_DRIVEN:    Significant in raw probing but disappears after
                      removing MW + NumHeavyAtoms. The GCN encoded molecule
                      size, not the mechanism itself.

  LIKELY_ZOMBIE:      Not significant after FDR correction. The naive
                      "ENCODED" verdict was a multiple-testing artifact.

  CONFIRMED_ZOMBIE:   TOST equivalence test confirms AND Bayes Factor > 3
                      provides positive evidence for the null. The mechanism
                      is definitively NOT encoded.
""")
print("=" * 70)
