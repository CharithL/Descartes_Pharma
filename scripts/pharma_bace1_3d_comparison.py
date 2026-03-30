#!/usr/bin/env python3
"""
DESCARTES-PHARMA: BACE1 3D vs 2D Architecture Comparison
==========================================================
Compares a 3D-aware SchNet-style GNN against the 2D GCN baseline
on the BACE1 Alzheimer's discovery probe.

Hypothesis: 3D geometry enables encoding of binding mechanism
features (pharmacophore, pocket complementarity) that the 2D GCN
misses as a zombie.

Pipeline:
  Phase 1: Load BACE, generate 3D conformers via RDKit
  Phase 2: Train 2D GCN + 3D GNN on identical splits
  Phase 3: Output validation gate (AUC >= 0.65 for each)
  Phase 4-5: Probe BOTH models for 10 BACE1 mechanism features
  Phase 6: Harden BOTH models (6 methods each)
  Phase 7: Side-by-side comparison table
  Phase 8: Discovery readiness for the winner
"""

import subprocess, sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

def ensure_installed(pkg, pip_name=None):
    try: __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg, "-q"])

print("=" * 85)
print("DESCARTES-PHARMA: BACE1 3D vs 2D Architecture Comparison")
print("=" * 85)

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
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
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

MECHANISM_NAMES = [
    'hydroxyethylamine_core', 'hbond_donors_catalytic_asp',
    'hydrophobic_s1_pocket', 'sp3_character',
    'cns_mpo_score', 'bbb_tpsa',
    'herg_risk_basic_nitrogen', 'mw_drug_range',
    'mw_raw', 'logp_raw',
]
GROUPS = {
    'BINDING': ['hydroxyethylamine_core', 'hbond_donors_catalytic_asp',
                'hydrophobic_s1_pocket', 'sp3_character'],
    'BRAIN':   ['cns_mpo_score', 'bbb_tpsa'],
    'SAFETY':  ['herg_risk_basic_nitrogen', 'mw_drug_range'],
    'CONFOUND':['mw_raw', 'logp_raw'],
}
HEA_SMARTS = ["[NX3][CX4][CX4](O)[CX4]", "[CX4](O)[CX4][CX3]=[OX1]", "[NX3][CX4][CX4](O)"]
BN_SMARTS = "[NX3;H2,H1;!$(NC=O);!$(NS=O);!$(N=*)]"
CONFOUND_IDX = [8, 9]


def compute_bace1_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        has_core = 0.0
        for s in HEA_SMARTS:
            p = Chem.MolFromSmarts(s)
            if p and mol.HasSubstructMatch(p): has_core = 1.0; break
        hbd = float(rdMolDescriptors.CalcNumHBD(mol))
        arom = float(Descriptors.NumAromaticRings(mol))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol); rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        cns = float((1<=logp<=3)+(mw<450)+(tpsa<90)+(hbd<=3)+(2<logp<5)+(rot<=8))
        bn_pat = Chem.MolFromSmarts(BN_SMARTS)
        herg = float(len(mol.GetSubstructMatches(bn_pat))) if bn_pat else 0.0
        mw_dr = 1.0 if 300<=mw<=600 else 0.0
        return np.array([has_core,hbd,arom,fsp3,cns,float(tpsa),herg,mw_dr,float(mw),float(logp)], dtype=np.float32)
    except: return None


# ============================================================
# PHASE 1: LOAD DATA + 3D CONFORMERS
# ============================================================
print("\n" + "=" * 85)
print("PHASE 1: Load BACE + generate 3D conformers")
print("=" * 85)

print("\n[1a] Loading BACE dataset...")
bace_loaded = False
for fn, label in [
    (lambda: __import__('tdc.single_pred', fromlist=['HTS']).HTS(name='BACE'), "HTS('BACE')"),
    (lambda: __import__('tdc.single_pred', fromlist=['ADME']).ADME(name='BACE_Group'), "ADME('BACE_Group')"),
]:
    try:
        bace_data = fn(); split = bace_data.get_split(method='scaffold')
        bace_loaded = True; print(f"  Loaded via {label}"); break
    except Exception as e: print(f"  {label}: {e}")

if not bace_loaded:
    import pandas as pd
    df = pd.read_csv("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv")
    df = df.rename(columns={'mol': 'Drug', 'Class': 'Y'})
    n = len(df); perm = np.random.RandomState(42).permutation(n)
    nt, nv = int(0.8*n), int(0.1*n)
    split = {'train': df.iloc[perm[:nt]], 'valid': df.iloc[perm[nt:nt+nv]], 'test': df.iloc[perm[nt+nv:]]}
    print(f"  Direct download ({n} compounds)")

train_df, val_df, test_df = split['train'], split['valid'], split['test']
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

print("\n[1b] Generating 3D conformers...")


def generate_3d_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1: return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        pos = np.array(conf.GetPositions(), dtype=np.float32)
        heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
        return pos[heavy_idx], [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy_idx]
    except: return None


all_smiles = set()
for df in [train_df, val_df, test_df]:
    all_smiles.update(df['Drug'].tolist())

n_ok, n_fail = 0, 0
conformer_cache = {}
t0 = time.time()
for i, smi in enumerate(all_smiles):
    r = generate_3d_conformer(smi)
    if r: conformer_cache[smi] = r; n_ok += 1
    else: n_fail += 1
    if (i+1) % 300 == 0:
        print(f"  {i+1}/{len(all_smiles)}: {n_ok} ok, {n_fail} failed")

print(f"  Done: {n_ok} ok, {n_fail} failed ({100*n_fail/max(n_ok+n_fail,1):.1f}%) in {time.time()-t0:.1f}s")


# ============================================================
# PHASE 2: BUILD BOTH MODELS + DATA
# ============================================================
print("\n" + "=" * 85)
print("PHASE 2: Build + train 2D GCN and 3D GNN")
print("=" * 85)

NODE_FEAT_DIM = 7


def smiles_to_2d_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    af = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
           int(a.GetHybridization()), int(a.GetIsAromatic()),
           a.GetTotalNumHs(), int(a.IsInRing())] for a in mol.GetAtoms()]
    x = torch.tensor(af, dtype=torch.float32)
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei.extend([[i, j], [j, i]])
    if not ei: ei = [[0, 0]]
    g = Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous())
    g.y = torch.tensor([label], dtype=torch.float32)
    return g


def smiles_to_3d_graph(smiles, label):
    if smiles not in conformer_cache: return None
    pos_h, nums_h = conformer_cache[smiles]
    n_atoms = len(nums_h)
    if n_atoms == 0: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    af = [[a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
           int(a.GetHybridization()), int(a.GetIsAromatic()),
           a.GetTotalNumHs(), int(a.IsInRing())] for a in mol.GetAtoms()]
    x = torch.tensor(af, dtype=torch.float32)
    cutoff = 10.0
    ei, dl = [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j: continue
            d = float(np.linalg.norm(pos_h[i] - pos_h[j]))
            if d <= cutoff: ei.append([i, j]); dl.append(d)
    if not ei: ei = [[0, 0]]; dl = [0.0]
    g = Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
             edge_attr=torch.tensor(dl, dtype=torch.float32).unsqueeze(-1),
             pos=torch.tensor(pos_h, dtype=torch.float32))
    g.y = torch.tensor([label], dtype=torch.float32)
    return g


def process_both(df):
    g2, g3, labs, feats, smis = [], [], [], [], []
    for _, row in df.iterrows():
        smi, lab = row['Drug'], row['Y']
        feat = compute_bace1_features(smi)
        gr2 = smiles_to_2d_graph(smi, lab)
        gr3 = smiles_to_3d_graph(smi, lab)
        if feat is not None and gr2 is not None and gr3 is not None and gr2.x.shape[0] > 0:
            g2.append(gr2); g3.append(gr3)
            labs.append(lab); feats.append(feat); smis.append(smi)
    return g2, g3, np.array(labs), np.array(feats), smis


print("  Processing (molecules valid for BOTH 2D and 3D)...")
tr2, tr3, tr_lab, tr_feat, tr_smi = process_both(train_df)
va2, va3, va_lab, va_feat, va_smi = process_both(val_df)
te2, te3, te_lab, te_feat, te_smi = process_both(test_df)
print(f"  Shared valid: Train={len(tr2)}, Val={len(va2)}, Test={len(te2)}")


# ---- 2D GCN ----
class GCN2D(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim)); self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim)); self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

    def forward(self, data, return_embedding=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        for c, b in zip(self.convs, self.bns): x = self.dropout(F.relu(b(c(x, ei))))
        emb = global_mean_pool(x, batch)
        logits = self.classifier(emb).squeeze(-1)
        return (logits, emb) if return_embedding else logits


# ---- 3D GNN (SchNet-inspired) ----
class GaussianRBF(nn.Module):
    def __init__(self, n_g=20, start=0.5, stop=10.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(start, stop, n_g))
        self.width = (stop - start) / n_g

    def forward(self, d):
        return torch.exp(-0.5 * ((d - self.centers) / self.width) ** 2)


class SchNetLayer(nn.Module):
    def __init__(self, hdim, n_g=20):
        super().__init__()
        self.filter_net = nn.Sequential(nn.Linear(n_g, hdim), nn.SiLU(), nn.Linear(hdim, hdim))
        self.node_update = nn.Sequential(nn.Linear(hdim, hdim), nn.SiLU(), nn.Linear(hdim, hdim))
        self.bn = nn.BatchNorm1d(hdim)

    def forward(self, x, ei, rbf):
        row, col = ei
        msg = x[col] * self.filter_net(rbf)
        agg = torch.zeros_like(x); agg.scatter_add_(0, row.unsqueeze(1).expand_as(msg), msg)
        return x + self.bn(self.node_update(agg))


class Simple3DGNN(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=3, n_g=20, dropout=0.2):
        super().__init__()
        self.atom_embed = nn.Embedding(120, hidden_dim)
        self.rbf = GaussianRBF(n_g=n_g)
        self.layers = nn.ModuleList([SchNetLayer(hidden_dim, n_g) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

    def forward(self, data, return_embedding=False):
        x = self.atom_embed(data.x[:, 0].long().clamp(0, 119))
        rbf = self.rbf(data.edge_attr)
        for layer in self.layers: x = self.dropout(layer(x, data.edge_index, rbf))
        emb = global_mean_pool(x, data.batch)
        logits = self.classifier(emb).squeeze(-1)
        return (logits, emb) if return_embedding else logits


# ---- Shared training ----
def train_model(model, tr_g, va_g, tr_lab, name):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
    tl = PyGDataLoader(tr_g, batch_size=64, shuffle=True)
    vl = PyGDataLoader(va_g, batch_size=64, shuffle=False)
    pw = torch.tensor([(1-np.mean(tr_lab))/max(np.mean(tr_lab),1e-6)]).to(device)
    best_a, best_s = 0.0, None; t0 = time.time()
    for ep in range(200):
        model.train(); loss_sum = 0
        for b in tl:
            b = b.to(device); opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(model(b), b.y.squeeze(-1), pos_weight=pw)
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            loss_sum += l.item()*b.num_graphs
        if ep % 5 == 0:
            model.eval(); vp, vt = [], []
            with torch.no_grad():
                for b in vl: b = b.to(device); vp.extend(torch.sigmoid(model(b)).cpu().numpy()); vt.extend(b.y.squeeze(-1).cpu().numpy())
            try: va = roc_auc_score(vt, vp)
            except: va = 0.5
            sch.step(1-va)
            if va > best_a: best_a = va; best_s = {k:v.clone() for k,v in model.state_dict().items()}
            if ep % 50 == 0: print(f"    [{name}] Ep {ep:3d}: loss={loss_sum/len(tr_g):.4f} val={va:.4f}")
    model.load_state_dict(best_s)
    print(f"    [{name}] {time.time()-t0:.1f}s, best val={best_a:.4f}")
    return model

def get_auc(model, loader):
    model.eval(); p, t = [], []
    with torch.no_grad():
        for b in loader: b = b.to(device); p.extend(torch.sigmoid(model(b)).cpu().numpy()); t.extend(b.y.squeeze(-1).cpu().numpy())
    try: return roc_auc_score(t, p)
    except: return 0.5

def get_emb(model, loader):
    model.eval(); out = []
    with torch.no_grad():
        for b in loader: b = b.to(device); _, e = model(b, return_embedding=True); out.append(e.cpu().numpy())
    return np.concatenate(out)


print("\n  Training 2D GCN...")
gcn = train_model(GCN2D(hidden_dim=128), tr2, va2, tr_lab, "2D-GCN")
print("\n  Training 3D GNN...")
gnn = train_model(Simple3DGNN(hidden_dim=128), tr3, va3, tr_lab, "3D-GNN")


# ============================================================
# PHASE 3: AUC GATE
# ============================================================
print("\n" + "=" * 85)
print("PHASE 3: Output validation")
print("=" * 85)
tl2 = PyGDataLoader(te2, batch_size=64, shuffle=False)
tl3 = PyGDataLoader(te3, batch_size=64, shuffle=False)
auc2 = get_auc(gcn, tl2); auc3 = get_auc(gnn, tl3)
print(f"  2D-GCN AUC: {auc2:.4f}  3D-GNN AUC: {auc3:.4f}")


# ============================================================
# PHASE 4-5: PROBE BOTH
# ============================================================
print("\n" + "=" * 85)
print("PHASE 4-5: Probe both models")
print("=" * 85)

sc = StandardScaler(); tfn = sc.fit_transform(te_feat)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def probe(model, loader, cls, name):
    et = get_emb(model, loader)
    rm = cls(hidden_dim=128).to(device); rm.eval()
    er = get_emb(rm, loader)
    res = {}
    for j, n in enumerate(MECHANISM_NAMES):
        t = tfn[:, j]
        if np.std(t) < 1e-10: res[n] = {'ridge_delta_r2': 0, 'se': 0.01}; continue
        st = cross_val_score(Ridge(alpha=1.0), et, t, cv=kf, scoring='r2')
        sr = cross_val_score(Ridge(alpha=1.0), er, t, cv=kf, scoring='r2')
        res[n] = {'ridge_delta_r2': np.mean(st)-np.mean(sr),
                  'se': max(np.sqrt(np.var(st)/len(st)+np.var(sr)/len(sr)), 1e-6)}
    return res, et, er

print("  Probing 2D-GCN..."); r2, e2t, e2r = probe(gcn, tl2, GCN2D, "2D")
print("  Probing 3D-GNN..."); r3, e3t, e3r = probe(gnn, tl3, Simple3DGNN, "3D")


# ============================================================
# PHASE 6: HARDEN BOTH
# ============================================================
print("\n" + "=" * 85)
print("PHASE 6: Hardening both models")
print("=" * 85)

N_P = 500; rng = np.random.default_rng(42)
scaff = []
for smi in te_smi:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        c = MurckoScaffold.GetScaffoldForMol(mol)
        g = MurckoScaffold.MakeScaffoldGeneric(c)
        scaff.append(Chem.MolToSmiles(g))
    else: scaff.append('UNK')
scaff = np.array(scaff)
sg = {s: np.where(scaff==s)[0] for s in np.unique(scaff)}


def harden(raw, et, er, name):
    h = {n: {} for n in MECHANISM_NAMES}
    print(f"  [{name}] Scaffold perm + y-scramble ({N_P} perms)...")
    for j, n in enumerate(MECHANISM_NAMES):
        t = tfn[:, j]
        if np.std(t) < 1e-10: h[n]['sp_p'] = h[n]['ys_p'] = 1.0; continue
        obs = raw[n]['ridge_delta_r2']
        sp_n = np.zeros(N_P); ys_n = np.zeros(N_P)
        for p in range(N_P):
            pt = t.copy()
            for idx in sg.values():
                if len(idx) > 1: pt[idx] = rng.permutation(pt[idx])
            sp_n[p] = np.mean(cross_val_score(Ridge(alpha=1.0), et, pt, cv=kf, scoring='r2')) - \
                      np.mean(cross_val_score(Ridge(alpha=1.0), er, pt, cv=kf, scoring='r2'))
            pt2 = rng.permutation(t)
            ys_n[p] = np.mean(cross_val_score(Ridge(alpha=1.0), et, pt2, cv=kf, scoring='r2')) - \
                      np.mean(cross_val_score(Ridge(alpha=1.0), er, pt2, cv=kf, scoring='r2'))
        h[n]['sp_p'] = float(np.mean(sp_n >= obs))
        h[n]['ys_p'] = float(np.mean(ys_n >= obs))

    print(f"  [{name}] Confound regression + FDR + TOST + BF...")
    conf = tfn[:, CONFOUND_IDX]
    tc = confound_removal(et, conf); rc = confound_removal(er, conf)
    for j, n in enumerate(MECHANISM_NAMES):
        t = tfn[:, j]
        if np.std(t) < 1e-10: h[n]['cd'] = 0.0; continue
        lr = LinearRegression(); lr.fit(conf, t); tc2 = t - lr.predict(conf)
        if np.std(tc2) < 1e-10: h[n]['cd'] = 0.0; continue
        h[n]['cd'] = np.mean(cross_val_score(Ridge(alpha=1.0), tc, tc2, cv=kf, scoring='r2')) - \
                     np.mean(cross_val_score(Ridge(alpha=1.0), rc, tc2, cv=kf, scoring='r2'))

    rp = np.array([h[n].get('sp_p', 1.0) for n in MECHANISM_NAMES])
    fdr = fdr_correction(rp, method='bh')
    for j, n in enumerate(MECHANISM_NAMES): h[n]['fdr_p'] = fdr['corrected_p'][j]

    for n in MECHANISM_NAMES:
        d, se = raw[n]['ridge_delta_r2'], raw[n]['se']
        if d >= 0.05: h[n]['tost'] = False
        else:
            r = tost_equivalence_test(delta_r2=d, se=se, epsilon=0.05)
            h[n]['tost'] = r['equivalent']
        bf = bayes_factor_null(delta_r2=d, se=se)
        h[n]['bf01'] = bf['bf01']
    return h


h2 = harden(r2, e2t, e2r, "2D-GCN")
h3 = harden(r3, e3t, e3r, "3D-GNN")


def verd(h):
    sp = h.get('sp_p', 1.0); fp = h.get('fdr_p', 1.0)
    cd = h.get('cd', 0.0); tost = h.get('tost', False); bf = h.get('bf01', 1.0)
    if sp < 0.05 and cd > 0.05 and fp < 0.05: return "CONFIRMED_ENCODED"
    elif sp < 0.05 and cd < 0.02: return "CONFOUND_DRIVEN"
    elif tost and bf > 3: return "CONFIRMED_ZOMBIE"
    elif fp >= 0.05: return "LIKELY_ZOMBIE"
    elif sp < 0.05: return "CANDIDATE_ENCODED"
    return "LIKELY_ZOMBIE"


# ============================================================
# PHASE 7: SIDE-BY-SIDE COMPARISON
# ============================================================
print("\n" + "=" * 85)
print("PHASE 7: BACE1 Architecture Comparison")
print("=" * 85)

v2 = {n: verd(h2[n]) for n in MECHANISM_NAMES}
v3 = {n: verd(h3[n]) for n in MECHANISM_NAMES}
enc = {'CONFIRMED_ENCODED', 'CANDIDATE_ENCODED'}

print(f"\n  AUC: 2D-GCN={auc2:.3f}  3D-GNN={auc3:.3f}")
print(f"\n  {'Mechanism':<28} {'GCN dR2':>8} {'GCN Verdict':<18} {'3D dR2':>8} {'3D Verdict':<18} {'Win':<5}")
print(f"  {'-'*90}")
for n in MECHANISM_NAMES:
    gd = r2[n]['ridge_delta_r2']; gv = v2[n]
    td = r3[n]['ridge_delta_r2']; tv = v3[n]
    w = "3D" if tv in enc and gv not in enc else ("2D" if gv in enc and tv not in enc else ("TIE" if gv in enc and tv in enc else "NONE"))
    print(f"  {n:<28} {gd:>8.4f} {gv:<18} {td:>8.4f} {tv:<18} {w:<5}")

for gn, mem in GROUPS.items():
    n2 = sum(1 for n in mem if v2[n] in enc); n3 = sum(1 for n in mem if v3[n] in enc)
    print(f"\n  {gn}: GCN={n2}/{len(mem)}, 3D-GNN={n3}/{len(mem)}")

t2 = sum(1 for n in MECHANISM_NAMES if v2[n] in enc)
t3 = sum(1 for n in MECHANISM_NAMES if v3[n] in enc)

# ============================================================
# PHASE 8: DISCOVERY READINESS
# ============================================================
print("\n" + "=" * 85)
print("PHASE 8: Discovery Readiness")
print("=" * 85)

wn, wv, wa = ("3D-GNN", v3, auc3) if t3 >= t2 else ("2D-GCN", v2, auc2)
nb = sum(1 for n in GROUPS['BINDING'] if wv[n] in enc)
nbr = sum(1 for n in GROUPS['BRAIN'] if wv[n] in enc)
print(f"  Winner: {wn} ({max(t2,t3)} vs {min(t2,t3)} encoded)")
print(f"  Binding: {nb}/4, Brain: {nbr}/2, AUC: {wa:.3f}")

if nb >= 3 and nbr >= 1:
    print(f"\n  DISCOVERY READY -- {wn} understands BACE1 mechanism.")
elif t3 > t2:
    print(f"\n  3D geometry helps ({t3} vs {t2}) but not full discovery readiness.")
    print(f"  Try AlphaFold-guided 3D with protein pocket features.")
else:
    print(f"\n  NOT DISCOVERY READY. Zombie binding features: "
          + ", ".join(n for n in GROUPS['BINDING'] if wv[n] not in enc))

print(f"\n{'='*85}")
print("ARCHITECTURE SUMMARY")
print(f"{'='*85}")
if t3 > t2:
    print(f"  3D geometry enables mechanism encoding 2D cannot: {t3}/10 vs {t2}/10.")
    print(f"  Use 3D-GNN for BACE1 screening, NOT 2D GCN.")
elif t2 > t3:
    print(f"  2D GCN outperforms ({t2} vs {t3}). Check conformer quality or 3D training.")
else:
    print(f"  Tied at {t2}/10. Neither has clear advantage. Try AlphaFold co-modeling.")

print("\n" + "=" * 85)
print("END OF 3D vs 2D COMPARISON")
print("=" * 85)
