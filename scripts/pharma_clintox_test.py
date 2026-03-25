#!/usr/bin/env python3
"""
DESCARTES-PHARMA Tier 2: ClinTox GCN Zombie Detection
======================================================
Tests whether a GNN trained on ClinTox toxicity prediction
encodes real molecular mechanisms (MW, LogP, TPSA, etc.) or
statistical shortcuts. This is the first REAL pharma test
after the HH simulator unit test validates the probing framework.

Pipeline:
1. Load ClinTox from TDC with scaffold split
2. Compute RDKit mechanistic ground truth features
3. Train GCN on SMILES -> toxicity (binary classification)
4. Validate output: AUC >= 0.7 gate
5. Probe GCN embeddings for 10 mechanistic features
6. SAE polypharmacology decomposition
7. Zombie verdict per mechanism
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

print("Checking dependencies...")
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 1. LOAD CLINTOX DATASET
# ============================================================
print("\n[1/7] Loading ClinTox dataset from TDC...")
from tdc.single_pred import Tox

tox_data = Tox(name='ClinTox')
split = tox_data.get_split(method='scaffold')

train_df = split['train']
val_df = split['valid']
test_df = split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} compounds")
print(f"  Toxicity prevalence (train): {train_df['Y'].mean():.3f}")

# ============================================================
# 2. COMPUTE RDKit MECHANISTIC GROUND TRUTH
# ============================================================
print("\n[2/7] Computing RDKit mechanistic features...")

MECHANISM_NAMES = [
    'MW', 'LogP', 'TPSA', 'HBD', 'HBA',
    'RotatableBonds', 'AromaticRings', 'FractionCSP3',
    'NumHeavyAtoms', 'PEOE_VSA1',
]


def compute_rdkit_features(smiles):
    """Compute 10 mechanistic ground truth features for a SMILES string."""
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
    """Convert SMILES to PyG Data object with atom features and bonds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features: atomic number, degree, formal charge, hybridization,
    # aromatic, num Hs, in ring -- 7 features
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
        ]
        atom_features.append(feat)

    x = torch.tensor(atom_features, dtype=torch.float32)

    # Edge index (undirected)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if len(edge_index) == 0:
        edge_index = [[0, 0]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def process_dataset(df):
    """Process a TDC dataframe into graphs, labels, and mechanistic features."""
    graphs = []
    labels = []
    features = []

    for _, row in df.iterrows():
        smiles = row['Drug']
        graph = smiles_to_graph(smiles)
        feat = compute_rdkit_features(smiles)

        if graph is not None and feat is not None and graph.x.shape[0] > 0:
            graph.y = torch.tensor([row['Y']], dtype=torch.float32)
            graphs.append(graph)
            labels.append(row['Y'])
            features.append(feat)

    return graphs, np.array(labels), np.array(features)


train_graphs, train_labels, train_features = process_dataset(train_df)
val_graphs, val_labels, val_features = process_dataset(val_df)
test_graphs, test_labels, test_features = process_dataset(test_df)

print(f"  Valid molecules -- Train: {len(train_graphs)}, "
      f"Val: {len(val_graphs)}, Test: {len(test_graphs)}")
print(f"  Mechanistic features: {len(MECHANISM_NAMES)}")
for i, name in enumerate(MECHANISM_NAMES):
    vals = train_features[:, i]
    print(f"    {name}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
          f"range=[{vals.min():.2f}, {vals.max():.2f}]")

# ============================================================
# 3. DEFINE GCN MODEL
# ============================================================
print("\n[3/7] Training GCN on toxicity prediction...")

NODE_FEAT_DIM = 7


class ToxGCN(nn.Module):
    """3-layer GCN with mean readout for toxicity classification."""

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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Mean readout -- this is the embedding we probe
        embedding = global_mean_pool(x, batch)

        logits = self.classifier(embedding).squeeze(-1)

        if return_embedding:
            return logits, embedding
        return logits


# ============================================================
# 4. TRAIN GCN
# ============================================================
model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=15, factor=0.5, min_lr=1e-6)

train_loader = PyGDataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = PyGDataLoader(val_graphs, batch_size=64, shuffle=False)
test_loader = PyGDataLoader(test_graphs, batch_size=64, shuffle=False)

# Handle class imbalance
pos_weight = torch.tensor(
    [(1 - np.mean(train_labels)) / max(np.mean(train_labels), 1e-6)]
).to(device)

best_val_auc = 0.0
best_state = None
n_epochs = 200
t_start = time.time()

for epoch in range(n_epochs):
    # Train
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

    avg_loss = total_loss / len(train_graphs)

    # Validate
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
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} "
                  f"val_AUC={val_auc:.4f} lr={lr:.6f}")

model.load_state_dict(best_state)
elapsed = time.time() - t_start
print(f"  Training time: {elapsed:.1f}s")
print(f"  Best validation AUC: {best_val_auc:.4f}")

# ============================================================
# 5. OUTPUT VALIDATION GATE: AUC >= 0.7
# ============================================================
print("\n[4/7] Output validation gate...")
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

if test_auc < 0.7:
    print(f"\n  WARNING: Test AUC = {test_auc:.3f} < 0.7")
    print(f"  Model hasn't learned toxicity prediction well enough.")
    print(f"  Zombie verdicts below reflect model failure, not probe failure.")
    print(f"  Try: more epochs, larger hidden_dim, or different architecture.\n")
else:
    print(f"  PASS: AUC = {test_auc:.3f} >= 0.7 -- model learned toxicity!")

# ============================================================
# 6. EXTRACT EMBEDDINGS
# ============================================================
print("\n[5/7] Extracting GCN embeddings...")


def extract_embeddings(gcn_model, loader):
    """Extract graph-level embeddings from GCN."""
    gcn_model.eval()
    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, emb = gcn_model(batch, return_embedding=True)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


trained_emb = extract_embeddings(model, test_loader)

random_model = ToxGCN(hidden_dim=128, n_layers=3).to(device)
random_model.eval()
random_emb = extract_embeddings(random_model, test_loader)

print(f"  Trained embeddings: {trained_emb.shape}")
print(f"  Random embeddings: {random_emb.shape}")
print(f"  Test mechanistic features: {test_features.shape}")

# ============================================================
# 7. PROBE: Ridge dR2 + MLP dR2
# ============================================================
print("\n[6/7] Probing embeddings for mechanistic features...")

scaler = StandardScaler()
test_features_norm = scaler.fit_transform(test_features)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Mechanism':<18} {'Ridge(T)':>10} {'Ridge(R)':>10} {'Ridge dR2':>10} "
      f"{'MLP(T)':>10} {'MLP(R)':>10} {'MLP dR2':>10} {'Verdict':>12}")
print(f"  {'-' * 95}")

results = {}
for j, name in enumerate(MECHANISM_NAMES):
    target = test_features_norm[:, j]

    if np.std(target) < 1e-10:
        results[name] = {'encoding_type': 'CONSTANT',
                         'ridge_delta_r2': 0, 'mlp_delta_r2': 0}
        print(f"  {name:<18} {'CONSTANT':>92}")
        continue

    # Ridge probe
    r2_t = np.mean(cross_val_score(
        Ridge(alpha=1.0), trained_emb, target, cv=kf, scoring='r2'))
    r2_r = np.mean(cross_val_score(
        Ridge(alpha=1.0), random_emb, target, cv=kf, scoring='r2'))
    ridge_delta = r2_t - r2_r

    # MLP probe
    mlp_t = np.mean(cross_val_score(
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                     early_stopping=True, random_state=42),
        trained_emb, target, cv=kf, scoring='r2'))
    mlp_r = np.mean(cross_val_score(
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                     early_stopping=True, random_state=42),
        random_emb, target, cv=kf, scoring='r2'))
    mlp_delta = mlp_t - mlp_r

    # Classify encoding
    if ridge_delta > 0.05 or mlp_delta > 0.05:
        if mlp_delta > ridge_delta + 0.1:
            verdict = "NONLINEAR"
        else:
            verdict = "ENCODED"
    else:
        verdict = "ZOMBIE"

    results[name] = {
        'ridge_trained_r2': r2_t, 'ridge_random_r2': r2_r,
        'ridge_delta_r2': ridge_delta,
        'mlp_trained_r2': mlp_t, 'mlp_random_r2': mlp_r,
        'mlp_delta_r2': mlp_delta,
        'encoding_type': verdict,
    }

    print(f"  {name:<18} {r2_t:>10.4f} {r2_r:>10.4f} {ridge_delta:>10.4f} "
          f"{mlp_t:>10.4f} {mlp_r:>10.4f} {mlp_delta:>10.4f} {verdict:>12}")

# ============================================================
# 8. SAE POLYPHARMACOLOGY DECOMPOSITION
# ============================================================
print("\n[7/7] SAE polypharmacology decomposition...")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from descartes_pharma.probes.sae import train_sae, sae_probe_molecular_mechanisms

for expansion in [4, 8]:
    print(f"\n  SAE expansion={expansion}x:")
    sae, loss = train_sae(
        [trained_emb], trained_emb.shape[1],
        expansion_factor=expansion, k=20,
        device=device
    )
    sae_results = sae_probe_molecular_mechanisms(
        sae, trained_emb, test_features_norm, MECHANISM_NAMES,
        device=device
    )
    print(f"    Alive features: {sae_results['n_alive']}")
    print(f"    Mean monosemanticity: {sae_results['mean_monosemanticity']:.4f}")
    for name in MECHANISM_NAMES:
        poly = sae_results['polypharmacology_detected'][name]
        print(f"    {name:<18}: SAE R2={sae_results['sae_r2'][name]:.4f} "
              f"Raw R2={sae_results['raw_r2'][name]:.4f} "
              f"{'POLYPHARMACOLOGY' if poly else 'monosemantic'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("VERDICT SUMMARY -- ClinTox Tier 2")
print("=" * 70)

n_encoded = sum(1 for r in results.values()
                if r['encoding_type'] in ('ENCODED', 'NONLINEAR'))
n_zombie = sum(1 for r in results.values()
               if r['encoding_type'] == 'ZOMBIE')
n_total = len(results)

print(f"  Test AUC: {test_auc:.3f}")
print(f"  Encoded:  {n_encoded}/{n_total} mechanisms")
print(f"  Zombie:   {n_zombie}/{n_total} mechanisms")

print(f"\n  {'Mechanism':<18} {'dR2 (Ridge)':>12} {'dR2 (MLP)':>12} {'Verdict':>12}")
print(f"  {'-' * 56}")
for name in MECHANISM_NAMES:
    r = results[name]
    v = r['encoding_type']
    rd = r.get('ridge_delta_r2', 0)
    md = r.get('mlp_delta_r2', 0)
    print(f"  {name:<18} {rd:>12.4f} {md:>12.4f} {v:>12}")

if test_auc >= 0.7 and n_encoded >= n_total * 0.5:
    print(f"\n  RESULT: Model encodes real toxicological mechanisms!")
    print(f"  -> This is a MECHANISTICALLY VALID model, not a zombie.")
elif test_auc >= 0.7 and n_encoded < n_total * 0.5:
    print(f"\n  RESULT: Model predicts toxicity but doesn't encode mechanisms!")
    print(f"  -> PHARMACEUTICAL ZOMBIE detected!")
    print(f"  -> The model found statistical shortcuts, not real biology.")
    zombie_list = [n for n, r in results.items()
                   if r['encoding_type'] == 'ZOMBIE']
    print(f"  -> Zombie mechanisms: {', '.join(zombie_list)}")
elif test_auc < 0.7:
    print(f"\n  RESULT: Model didn't learn toxicity well enough (AUC < 0.7)")
    print(f"  -> Improve training before interpreting probe results.")

print("=" * 70)
print("\nNEXT STEPS:")
print("  If ENCODED: Run statistical hardening (scaffold-stratified null)")
print("  If ZOMBIE:  Try larger model (hidden=256), or different architecture")
print("  If AUC low: Tune GCN hyperparameters or try GAT/GIN architecture")
print("=" * 70)
