#!/usr/bin/env python3
"""
DESCARTES-PHARMA: BACE1 Alzheimer's Disease Discovery Pipeline
================================================================
First disease-targeted probe. Instead of generic molecular descriptors,
probes for BACE1-specific mechanistic features: pharmacophore cores,
catalytic site H-bonding, brain penetration, and safety liabilities.

A non-zombie model MUST encode these disease-specific features to be
trusted for Alzheimer's drug screening.

Pipeline:
  Phase 1: Load BACE dataset, compute 10 disease-specific features
  Phase 2: Train GCN on BACE1 activity, extract embeddings
  Phase 3: Raw Ridge/MLP probes for all 10 features
  Phase 4: 6-method statistical hardening
  Phase 5: BACE1-specific discovery readiness assessment
"""

import subprocess
import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def ensure_installed(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"  Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pip_name, "-q"])

print("=" * 80)
print("DESCARTES-PHARMA: BACE1 Alzheimer's Disease Discovery Pipeline")
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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from descartes_pharma.statistical.hardening import (
    fdr_correction, confound_removal, tost_equivalence_test, bayes_factor_null,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# BACE1-SPECIFIC MECHANISM FEATURE DEFINITIONS
# ============================================================
MECHANISM_NAMES = [
    'hydroxyethylamine_core', 'hbond_donors_catalytic_asp',
    'hydrophobic_s1_pocket', 'sp3_character',
    'cns_mpo_score', 'bbb_tpsa',
    'herg_risk_basic_nitrogen', 'mw_drug_range',
    'mw_raw', 'logp_raw',
]

GROUPS = {
    'BINDING_MECHANISM': ['hydroxyethylamine_core', 'hbond_donors_catalytic_asp',
                          'hydrophobic_s1_pocket', 'sp3_character'],
    'BRAIN_PENETRATION': ['cns_mpo_score', 'bbb_tpsa'],
    'SAFETY': ['herg_risk_basic_nitrogen', 'mw_drug_range'],
    'CONFOUNDS': ['mw_raw', 'logp_raw'],
}

HYDROXYETHYLAMINE_SMARTS = [
    "[NX3][CX4][CX4](O)[CX4]",
    "[CX4](O)[CX4][CX3]=[OX1]",
    "[NX3][CX4][CX4](O)",
]
BASIC_NITROGEN_SMARTS = "[NX3;H2,H1;!$(NC=O);!$(NS=O);!$(N=*)]"
CONFOUND_IDX = [8, 9]  # mw_raw, logp_raw


def compute_bace1_features(smiles):
    """Compute 10 BACE1-specific mechanistic features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # A1: Hydroxyethylamine core (transition-state isostere)
        has_core = 0.0
        for smarts_str in HYDROXYETHYLAMINE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                has_core = 1.0
                break
        # A2: H-bond donors (catalytic Asp proxy)
        hbd = float(rdMolDescriptors.CalcNumHBD(mol))
        # A3: Aromatic rings (S1' pocket filling)
        arom = float(Descriptors.NumAromaticRings(mol))
        # A4: FractionCSP3 (3D shape)
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        # B5: CNS MPO score
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        cns = float((1.0 <= logp <= 3.0) + (mw < 450) + (tpsa < 90)
                     + (hbd <= 3) + (2.0 < logp < 5.0) + (rot <= 8))
        # B6: BBB TPSA
        bbb_tpsa = float(tpsa)
        # C7: hERG risk basic nitrogens
        bn_pat = Chem.MolFromSmarts(BASIC_NITROGEN_SMARTS)
        herg = float(len(mol.GetSubstructMatches(bn_pat))) if bn_pat else 0.0
        # C8: MW drug range
        mw_dr = 1.0 if 300 <= mw <= 600 else 0.0
        # D9-10: Confounds
        return np.array([has_core, hbd, arom, fsp3, cns, bbb_tpsa,
                         herg, mw_dr, float(mw), float(logp)], dtype=np.float32)
    except Exception:
        return None


# ============================================================
# PHASE 1: LOAD BACE DATASET
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: Load BACE dataset + compute disease-specific features")
print("=" * 80)

print("\n[1/3] Loading BACE dataset...")
bace_loaded = False

# Try TDC loaders
for attempt_fn, attempt_label in [
    (lambda: __import__('tdc.single_pred', fromlist=['HTS']).HTS(name='BACE'), "HTS('BACE')"),
    (lambda: __import__('tdc.single_pred', fromlist=['ADME']).ADME(name='BACE_Group'), "ADME('BACE_Group')"),
]:
    try:
        bace_data = attempt_fn()
        split = bace_data.get_split(method='scaffold')
        bace_loaded = True
        print(f"  Loaded via {attempt_label}")
        break
    except Exception as e:
        print(f"  {attempt_label} failed: {e}")

if not bace_loaded:
    # Fallback: direct CSV download
    print("  Trying direct MoleculeNet download...")
    try:
        import pandas as pd
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        df = pd.read_csv(url)
        df = df.rename(columns={'mol': 'Drug', 'Class': 'Y'})
        n = len(df)
        perm = np.random.RandomState(42).permutation(n)
        n_tr, n_va = int(0.8 * n), int(0.1 * n)
        split = {
            'train': df.iloc[perm[:n_tr]],
            'valid': df.iloc[perm[n_tr:n_tr+n_va]],
            'test': df.iloc[perm[n_tr+n_va:]],
        }
        bace_loaded = True
        print(f"  Loaded directly ({len(df)} compounds)")
    except Exception as e:
        print(f"  FATAL: {e}")
        sys.exit(1)

train_df, val_df, test_df = split['train'], split['valid'], split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"  Activity prevalence (train): {train_df['Y'].mean():.3f}")

# ---- Graph + feature computation ----
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    af = []
    for atom in mol.GetAtoms():
        af.append([atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
                   int(atom.GetHybridization()), int(atom.GetIsAromatic()),
                   atom.GetTotalNumHs(), int(atom.IsInRing())])
    x = torch.tensor(af, dtype=torch.float32)
    ei = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei.extend([[i, j], [j, i]])
    if not ei:
        ei = [[0, 0]]
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous())


def process_dataset(df):
    graphs, labels, features, smiles_list = [], [], [], []
    for _, row in df.iterrows():
        smi = row['Drug']
        g = smiles_to_graph(smi)
        f = compute_bace1_features(smi)
        if g is not None and f is not None and g.x.shape[0] > 0:
            g.y = torch.tensor([row['Y']], dtype=torch.float32)
            graphs.append(g)
            labels.append(row['Y'])
            features.append(f)
            smiles_list.append(smi)
    return graphs, np.array(labels), np.array(features), smiles_list


print("\n[2/3] Computing BACE1-specific mechanistic features...")
train_graphs, train_labels, train_features, train_smiles = process_dataset(train_df)
val_graphs, val_labels, val_features, val_smiles = process_dataset(val_df)
test_graphs, test_labels, test_features, test_smiles = process_dataset(test_df)
print(f"  Valid: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

for i, name in enumerate(MECHANISM_NAMES):
    vals = train_features[:, i]
    grp = [g for g, m in GROUPS.items() if name in m][0]
    print(f"    [{grp[:4]}] {name:<30}: mean={vals.mean():.3f}, std={vals.std():.3f}")

n_core = int(np.sum(train_features[:, 0] > 0))
print(f"\n  Pharmacophore core in {n_core}/{len(train_features)} "
      f"train compounds ({100*n_core/len(train_features):.1f}%)")


# ============================================================
# PHASE 2: TRAIN GCN
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: Train GCN on BACE1 activity")
print("=" * 80)

NODE_FEAT_DIM = 7

class ToxGCN(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3, dropout=0.2):
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
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index); x = bn(x); x = F.relu(x); x = self.dropout(x)
        embedding = global_mean_pool(x, batch)
        logits = self.classifier(embedding).squeeze(-1)
        return (logits, embedding) if return_embedding else logits


model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

train_loader = PyGDataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = PyGDataLoader(val_graphs, batch_size=64, shuffle=False)
test_loader = PyGDataLoader(test_graphs, batch_size=64, shuffle=False)
pos_weight = torch.tensor([(1-np.mean(train_labels))/max(np.mean(train_labels),1e-6)]).to(device)

best_val_auc, best_state = 0.0, None
t0 = time.time()
for epoch in range(200):
    model.train()
    tl = 0
    for batch in train_loader:
        batch = batch.to(device); optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(batch), batch.y.squeeze(-1), pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        tl += loss.item() * batch.num_graphs
    if epoch % 5 == 0:
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device); vp.extend(torch.sigmoid(model(b)).cpu().numpy()); vt.extend(b.y.squeeze(-1).cpu().numpy())
        try: va = roc_auc_score(vt, vp)
        except ValueError: va = 0.5
        scheduler.step(1-va)
        if va > best_val_auc: best_val_auc = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0: print(f"  Epoch {epoch:3d}: loss={tl/len(train_graphs):.4f} val_AUC={va:.4f}")

model.load_state_dict(best_state)
print(f"  Training: {time.time()-t0:.1f}s, best val AUC: {best_val_auc:.4f}")

model.eval(); tp, tt = [], []
with torch.no_grad():
    for b in test_loader:
        b = b.to(device); tp.extend(torch.sigmoid(model(b)).cpu().numpy()); tt.extend(b.y.squeeze(-1).cpu().numpy())
try: test_auc = roc_auc_score(tt, tp)
except ValueError: test_auc = 0.5
print(f"  Test AUC: {test_auc:.4f}")
if test_auc < 0.70: print(f"  WARNING: AUC < 0.70")
else: print(f"  PASS: AUC >= 0.70")


# ============================================================
# PHASE 3: RAW PROBES
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: Raw probes for BACE1-specific mechanism features")
print("=" * 80)

def extract_emb(m, loader):
    m.eval(); out = []
    with torch.no_grad():
        for b in loader: b = b.to(device); _, e = m(b, return_embedding=True); out.append(e.cpu().numpy())
    return np.concatenate(out, axis=0)

trained_emb = extract_emb(model, test_loader)
rand_m = ToxGCN(hidden_dim=128, n_layers=3).to(device); rand_m.eval()
random_emb = extract_emb(rand_m, test_loader)

scaler = StandardScaler()
tfn = scaler.fit_transform(test_features)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Mechanism':<30} {'Ridge dR2':>10} {'MLP dR2':>9} {'Verdict':>10}")
print(f"  {'-' * 65}")

raw_results = {}
for j, name in enumerate(MECHANISM_NAMES):
    target = tfn[:, j]
    if np.std(target) < 1e-10:
        raw_results[name] = {'ridge_delta_r2': 0.0, 'mlp_delta_r2': 0.0, 'se': 0.01}
        print(f"  {name:<30} {'CONSTANT':>32}"); continue
    st = cross_val_score(Ridge(alpha=1.0), trained_emb, target, cv=kf, scoring='r2')
    sr = cross_val_score(Ridge(alpha=1.0), random_emb, target, cv=kf, scoring='r2')
    rd = np.mean(st) - np.mean(sr)
    se = np.sqrt(np.var(st)/len(st) + np.var(sr)/len(sr))
    mt = np.mean(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, early_stopping=True, random_state=42), trained_emb, target, cv=kf, scoring='r2'))
    mr = np.mean(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, early_stopping=True, random_state=42), random_emb, target, cv=kf, scoring='r2'))
    md = mt - mr
    v = "ENCODED" if (rd > 0.05 or md > 0.05) and md <= rd + 0.1 else ("NONLINEAR" if md > 0.05 else "ZOMBIE")
    raw_results[name] = {'ridge_delta_r2': rd, 'mlp_delta_r2': md, 'se': max(se, 1e-6)}
    print(f"  {name:<30} {rd:>10.4f} {md:>9.4f} {v:>10}")


# ============================================================
# PHASE 4: STATISTICAL HARDENING
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: Statistical Hardening (6 methods)")
print("=" * 80)

N_PERMS = 500
hardened = {n: {} for n in MECHANISM_NAMES}
rng = np.random.default_rng(42)

# Scaffolds
scaff_arr = []
for smi in test_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        c = MurckoScaffold.GetScaffoldForMol(mol)
        g = MurckoScaffold.MakeScaffoldGeneric(c)
        scaff_arr.append(Chem.MolToSmiles(g))
    else:
        scaff_arr.append('UNK')
scaff_arr = np.array(scaff_arr)
scaff_groups = {s: np.where(scaff_arr == s)[0] for s in np.unique(scaff_arr)}
print(f"  Unique scaffolds: {len(scaff_groups)}")

# M1: Scaffold perm
print(f"\n[M1] Scaffold-stratified permutation ({N_PERMS} perms)...")
for j, name in enumerate(MECHANISM_NAMES):
    t = tfn[:, j]
    if np.std(t) < 1e-10: hardened[name]['sp_p'] = 1.0; continue
    obs = raw_results[name]['ridge_delta_r2']
    nulls = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        pt = t.copy()
        for idx in scaff_groups.values():
            if len(idx) > 1: pt[idx] = rng.permutation(pt[idx])
        nulls[p] = np.mean(cross_val_score(Ridge(alpha=1.0), trained_emb, pt, cv=kf, scoring='r2')) - np.mean(cross_val_score(Ridge(alpha=1.0), random_emb, pt, cv=kf, scoring='r2'))
    hardened[name]['sp_p'] = float(np.mean(nulls >= obs))
    print(f"  {name:<30}: p={hardened[name]['sp_p']:.4f}")

# M2: Y-scramble
print(f"\n[M2] Y-scramble ({N_PERMS} perms)...")
for j, name in enumerate(MECHANISM_NAMES):
    t = tfn[:, j]
    if np.std(t) < 1e-10: hardened[name]['ys_p'] = 1.0; continue
    obs = raw_results[name]['ridge_delta_r2']
    nulls = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        pt = rng.permutation(t)
        nulls[p] = np.mean(cross_val_score(Ridge(alpha=1.0), trained_emb, pt, cv=kf, scoring='r2')) - np.mean(cross_val_score(Ridge(alpha=1.0), random_emb, pt, cv=kf, scoring='r2'))
    hardened[name]['ys_p'] = float(np.mean(nulls >= obs))
    print(f"  {name:<30}: p={hardened[name]['ys_p']:.4f}")

# M3: Confound regression
print("\n[M3] Confound regression (remove mw_raw + logp_raw)...")
conf = tfn[:, CONFOUND_IDX]
tc = confound_removal(trained_emb, conf)
rc = confound_removal(random_emb, conf)
for j, name in enumerate(MECHANISM_NAMES):
    t = tfn[:, j]
    if np.std(t) < 1e-10: hardened[name]['cd'] = 0.0; continue
    lr = LinearRegression(); lr.fit(conf, t)
    tc2 = t - lr.predict(conf)
    if np.std(tc2) < 1e-10: hardened[name]['cd'] = 0.0; continue
    hardened[name]['cd'] = np.mean(cross_val_score(Ridge(alpha=1.0), tc, tc2, cv=kf, scoring='r2')) - np.mean(cross_val_score(Ridge(alpha=1.0), rc, tc2, cv=kf, scoring='r2'))
    print(f"  {name:<30}: raw={raw_results[name]['ridge_delta_r2']:.4f} -> clean={hardened[name]['cd']:.4f}")

# M4: FDR
print("\n[M4] FDR correction...")
rp = np.array([hardened[n].get('sp_p', 1.0) for n in MECHANISM_NAMES])
fdr = fdr_correction(rp, method='bh')
for j, n in enumerate(MECHANISM_NAMES): hardened[n]['fdr_p'] = fdr['corrected_p'][j]
print(f"  Surviving FDR: {int(fdr['rejected'].sum())}/10")

# M5: TOST
print("\n[M5] TOST equivalence...")
for n in MECHANISM_NAMES:
    d, se = raw_results[n]['ridge_delta_r2'], raw_results[n]['se']
    if d >= 0.05: hardened[n]['tost'] = False; continue
    r = tost_equivalence_test(delta_r2=d, se=se, epsilon=0.05)
    hardened[n]['tost'] = r['equivalent']
    if r['equivalent']: print(f"  {n:<30}: ZOMBIE CONFIRMED")

# M6: Bayes Factor
print("\n[M6] Bayes Factor...")
for n in MECHANISM_NAMES:
    d, se = raw_results[n]['ridge_delta_r2'], raw_results[n]['se']
    bf = bayes_factor_null(delta_r2=d, se=se)
    hardened[n]['bf01'] = bf['bf01']
    print(f"  {n:<30}: BF01={bf['bf01']:.2f} ({bf['verdict']})")


# ============================================================
# PHASE 5: BACE1 DISCOVERY READINESS ASSESSMENT
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: BACE1 Discovery Readiness Assessment")
print("=" * 80)

def bv(n, h):
    sp = h.get('sp_p', 1.0); fp = h.get('fdr_p', 1.0)
    cd = h.get('cd', 0.0); tost = h.get('tost', False); bf = h.get('bf01', 1.0)
    if sp < 0.05 and cd > 0.05 and fp < 0.05: return "CONFIRMED_ENCODED"
    elif sp < 0.05 and cd < 0.02: return "CONFOUND_DRIVEN"
    elif tost and bf > 3: return "CONFIRMED_ZOMBIE"
    elif fp >= 0.05: return "LIKELY_ZOMBIE"
    elif sp < 0.05: return "CANDIDATE_ENCODED"
    return "LIKELY_ZOMBIE"

print(f"\n  {'Mechanism':<30} {'Raw dR2':>8} {'Scaf-p':>7} {'Clean dR2':>10} {'FDR-p':>7} {'BF01':>7} {'Verdict':<20}")
print(f"  {'-' * 92}")

verdicts = {}
for name in MECHANISM_NAMES:
    h = hardened[name]; v = bv(name, h); verdicts[name] = v
    print(f"  {name:<30} {raw_results[name]['ridge_delta_r2']:>8.4f} "
          f"{h.get('sp_p',1.0):>7.4f} {h.get('cd',0.0):>10.4f} "
          f"{h.get('fdr_p',1.0):>7.4f} {h.get('bf01',1.0):>7.2f} {v:<20}")

# Disease-specific interpretation
desc = {
    'hydroxyethylamine_core': 'Does model know the pharmacophore?',
    'hbond_donors_catalytic_asp': 'Does model encode H-bonding to active site?',
    'hydrophobic_s1_pocket': 'Does model encode pocket filling?',
    'sp3_character': 'Does model encode 3D shape?',
    'cns_mpo_score': 'Does model know drug must reach brain?',
    'bbb_tpsa': 'Does model encode BBB crossing?',
    'herg_risk_basic_nitrogen': 'Does model encode cardiac risk?',
    'mw_drug_range': 'Does model encode drug-likeness?',
    'mw_raw': 'Is model just learning size?',
    'logp_raw': 'Is model just learning greasiness?',
}
labels = {'BINDING_MECHANISM': 'BINDING MECHANISM (A)', 'BRAIN_PENETRATION': 'BRAIN PENETRATION (B)',
          'SAFETY': 'SAFETY (C)', 'CONFOUNDS': 'CONFOUND CHECK (D)'}

print(f"\n{'='*80}")
print("BACE1 DISCOVERY READINESS ASSESSMENT")
print(f"{'='*80}")
for gn, members in GROUPS.items():
    print(f"\n  {labels[gn]}:")
    for n in members:
        print(f"    {n:<30}: {verdicts[n]:<20} <- \"{desc[n]}\"")

binding = GROUPS['BINDING_MECHANISM']
brain = GROUPS['BRAIN_PENETRATION']
enc_set = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}
nb = sum(1 for n in binding if verdicts[n] in enc_set)
nbr = sum(1 for n in brain if verdicts[n] in enc_set)

print(f"\n{'='*80}")
print("OVERALL VERDICT")
print(f"{'='*80}")
print(f"  Test AUC: {test_auc:.3f}")
print(f"  Binding features encoded: {nb}/4")
print(f"  Brain features encoded:   {nbr}/2")

if nb >= 3 and nbr >= 1:
    print(f"\n  DISCOVERY READY -- Model genuinely understands BACE1 mechanism.")
    print(f"  Safe to use for virtual screening of Alzheimer's drug candidates.")
elif nb < 3 and (verdicts['mw_raw'] in enc_set or verdicts['logp_raw'] in enc_set):
    zb = [n for n in binding if verdicts[n] not in enc_set]
    print(f"\n  ZOMBIE WARNING -- Model learning size/lipophilicity shortcuts.")
    print(f"  Zombie binding features: {', '.join(zb)}")
    print(f"  Do NOT use for drug screening. Try 3D-equivariant architecture.")
else:
    print(f"\n  NOT DISCOVERY READY -- Model does not encode BACE1 mechanism ({nb}/4).")
    print(f"  Try: larger model, GAT, or SchNet + AlphaFold BACE1 structure (PDB: 6EJ2)")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print(f"  1. If DISCOVERY READY: run virtual screening pipeline")
print(f"  2. If ZOMBIE: try SchNet + AlphaFold BACE1 structure")
print(f"  3. Compare with tau aggregation probe (second AD target)")
print(f"  4. Cross-reference with generic probes from ClinTox/BBBP/Tox21")
print("=" * 80)
