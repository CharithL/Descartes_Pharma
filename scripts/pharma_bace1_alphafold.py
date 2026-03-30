#!/usr/bin/env python3
"""
DESCARTES-PHARMA: BACE1 Protein-Structure Interaction Probe
=============================================================
Instead of probing for molecular descriptors (what the molecule LOOKS
like), probe for protein-ligand INTERACTION features (how the molecule
INTERACTS with BACE1).

Uses BACE1 crystal structure PDB 4IVT (1.55A, hydroxyethylamine
inhibitor) to compute binding geometry ground truth.

Pipeline:
  Phase 1: Download BACE1 crystal structure, extract binding site
  Phase 2: Load BACE dataset + generate 3D conformers
  Phase 3: Compute 12 interaction features per molecule
  Phase 4: Train GCN, probe with interaction ground truth
  Phase 5: Statistical hardening (6 methods)
  Phase 6: Comparison -- molecular descriptors vs interaction features
"""

import subprocess, sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

def ensure_installed(pkg, pip_name=None):
    try: __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg, "-q"])

print("=" * 85)
print("DESCARTES-PHARMA: BACE1 Protein-Structure Interaction Probe")
print("=" * 85)

print("\nChecking dependencies...")
ensure_installed("tdc", "PyTDC")
ensure_installed("torch_geometric", "torch-geometric")
ensure_installed("rdkit", "rdkit")
ensure_installed("sklearn", "scikit-learn")
ensure_installed("Bio", "biopython")

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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from descartes_pharma.statistical.hardening import (
    fdr_correction, confound_removal, tost_equivalence_test, bayes_factor_null,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
os.makedirs('data', exist_ok=True)

INTERACTION_NAMES = [
    'dist_to_asp32', 'dist_to_asp228', 'hbond_count_catalytic', 'catalytic_triad_score',
    's1_pocket_contacts', 's1prime_contacts', 'total_pocket_contacts', 'buried_fraction',
    'docking_score', 'pose_rmsd_to_ref',
    'mw_raw', 'logp_raw',
]
GROUPS = {
    'CATALYTIC': ['dist_to_asp32', 'dist_to_asp228', 'hbond_count_catalytic', 'catalytic_triad_score'],
    'POCKET': ['s1_pocket_contacts', 's1prime_contacts', 'total_pocket_contacts', 'buried_fraction'],
    'QUALITY': ['docking_score', 'pose_rmsd_to_ref'],
    'CONFOUND': ['mw_raw', 'logp_raw'],
}
CONFOUND_IDX = [10, 11]
STANDARD_AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
S1_RESIDS = [30, 71, 108, 110]
S1P_RESIDS = [76, 118]


# ============================================================
# PHASE 1: GET BACE1 PROTEIN STRUCTURE
# ============================================================
print("\n" + "=" * 85)
print("PHASE 1: Download BACE1 crystal structure (PDB 4IVT)")
print("=" * 85)

PDB_FILE = 'data/4IVT.pdb'
if not os.path.exists(PDB_FILE):
    print("  Downloading PDB 4IVT (1.55A resolution)...")
    urllib.request.urlretrieve('https://files.rcsb.org/download/4IVT.pdb', PDB_FILE)
print(f"  Using {PDB_FILE}")

pdb_parser = PDBParser(QUIET=True)
structure = pdb_parser.get_structure('BACE1', PDB_FILE)
pdb_model = structure[0]

# Parse all protein atoms and residue map
protein_atoms = []
residue_map = {}
for chain in pdb_model:
    for res in chain:
        rid = res.get_id()[1]
        rname = res.get_resname()
        if rname in ('HOH', 'WAT'): continue
        alist = []
        for atom in res:
            coord = np.array(atom.get_vector().get_array(), dtype=np.float64)
            protein_atoms.append({'coord': coord, 'name': atom.get_name(),
                                  'resid': rid, 'resname': rname})
            alist.append((atom.get_name(), coord))
        if rid not in residue_map:
            residue_map[rid] = alist

# Find co-crystallized ligand
ligand_atoms = []
for chain in pdb_model:
    for res in chain:
        rname = res.get_resname()
        if rname not in STANDARD_AA and rname not in ('HOH','WAT') and res.get_id()[0] != ' ':
            for atom in res:
                ligand_atoms.append(np.array(atom.get_vector().get_array(), dtype=np.float64))

if ligand_atoms:
    ligand_center = np.mean(ligand_atoms, axis=0)
    print(f"  Ligand: {len(ligand_atoms)} atoms, center={ligand_center.round(1)}")
else:
    asp32_c = [c for _, c in residue_map.get(32, [])]
    asp228_c = [c for _, c in residue_map.get(228, [])]
    ligand_center = (np.mean(asp32_c, axis=0) + np.mean(asp228_c, axis=0))/2 if asp32_c and asp228_c else np.array([25.,25.,25.])
    print(f"  No ligand, using catalytic center: {ligand_center.round(1)}")

# Binding site residues
bs_resids = set()
for pa in protein_atoms:
    if np.linalg.norm(pa['coord'] - ligand_center) < 8.0:
        bs_resids.add(pa['resid'])
print(f"  Binding site: {len(bs_resids)} residues within 8A")

# Key atom coordinates
def get_res_atoms(rid, names=None):
    atoms = residue_map.get(rid, [])
    return {n: c for n, c in atoms if names is None or n in names}

asp32_ox = get_res_atoms(32, ['OD1', 'OD2'])
asp228_ox = get_res_atoms(228, ['OD1', 'OD2'])
print(f"  Asp32 OD: {list(asp32_ox.keys())}, Asp228 OD: {list(asp228_ox.keys())}")

# Pocket coordinate arrays
def collect_coords(resid_list):
    coords = []
    for rid in resid_list:
        for _, c in residue_map.get(rid, []):
            coords.append(c)
    return np.array(coords) if coords else np.zeros((0, 3))

s1_coords = collect_coords(S1_RESIDS)
s1p_coords = collect_coords(S1P_RESIDS)
bs_coords = collect_coords(list(bs_resids))
print(f"  S1: {len(s1_coords)} atoms, S1': {len(s1p_coords)} atoms, BS total: {len(bs_coords)} atoms")


# ============================================================
# PHASE 2: LOAD DATA + 3D CONFORMERS
# ============================================================
print("\n" + "=" * 85)
print("PHASE 2: Load BACE + generate 3D conformers")
print("=" * 85)

bace_loaded = False
for fn, label in [
    (lambda: __import__('tdc.single_pred', fromlist=['HTS']).HTS(name='BACE'), "HTS"),
    (lambda: __import__('tdc.single_pred', fromlist=['ADME']).ADME(name='BACE_Group'), "ADME"),
]:
    try: bace_data = fn(); split = bace_data.get_split(method='scaffold'); bace_loaded = True; print(f"  Loaded via {label}"); break
    except: pass
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

print("\n  Generating 3D conformers...")
ref_lig = np.array(ligand_atoms) if ligand_atoms else None

def gen_conf(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        mol3 = Chem.AddHs(mol)
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(mol3, p) == -1: return None
        AllChem.MMFFOptimizeMolecule(mol3, maxIters=200)
        pos = np.array(mol3.GetConformer().GetPositions(), dtype=np.float64)
        hi = [i for i, a in enumerate(mol3.GetAtoms()) if a.GetAtomicNum() > 1]
        hp = pos[hi]
        hp = hp - hp.mean(0) + ligand_center  # center on binding site
        hbd = [j for j, idx in enumerate(hi) if mol3.GetAtomWithIdx(idx).GetAtomicNum() in (7,8)
               and mol3.GetAtomWithIdx(idx).GetTotalNumHs() > 0]
        return {'pos': hp, 'hbd': hbd, 'n': len(hi), 'mol': mol}
    except: return None

all_smi = set()
for d in [train_df, val_df, test_df]: all_smi.update(d['Drug'].tolist())
cc = {}; nok = nf = 0; t0 = time.time()
for i, smi in enumerate(all_smi):
    r = gen_conf(smi)
    if r: cc[smi] = r; nok += 1
    else: nf += 1
    if (i+1) % 300 == 0: print(f"    {i+1}/{len(all_smi)}: {nok} ok, {nf} fail")
print(f"  Done: {nok} ok, {nf} fail in {time.time()-t0:.1f}s")


# ============================================================
# PHASE 3: COMPUTE INTERACTION FEATURES
# ============================================================
print("\n" + "=" * 85)
print("PHASE 3: Compute protein-ligand interaction features")
print("=" * 85)

def min_donor_dist(pos, donors, res_coords):
    if not donors or len(res_coords) == 0: return 20.0
    dp = pos[donors]
    return float(np.min(np.linalg.norm(dp[:, None, :] - res_coords[None, :, :], axis=2)))

def count_contacts(pos, pocket, cutoff=4.0):
    if len(pocket) == 0: return 0
    d = np.linalg.norm(pos[:, None, :] - pocket[None, :, :], axis=2)
    return int(np.sum(np.any(d < cutoff, axis=1)))

def interaction_features(smiles):
    if smiles not in cc: return None
    c = cc[smiles]; pos = c['pos']; hbd = c['hbd']; mol = c['mol']
    try:
        a32 = np.array(list(asp32_ox.values())) if asp32_ox else np.zeros((0,3))
        a228 = np.array(list(asp228_ox.values())) if asp228_ox else np.zeros((0,3))
        d32 = min_donor_dist(pos, hbd, a32)
        d228 = min_donor_dist(pos, hbd, a228)
        hbc = 0
        if hbd:
            for di in hbd:
                for oc in list(a32) + list(a228):
                    if np.linalg.norm(pos[di] - oc) < 3.5: hbc += 1
        cat = 1.0/max(d32,0.5) + 1.0/max(d228,0.5)
        s1c = count_contacts(pos, s1_coords)
        s1pc = count_contacts(pos, s1p_coords)
        tc = count_contacts(pos, bs_coords)
        bur = tc / max(len(pos), 1)
        md = np.mean([np.min(np.linalg.norm(p - bs_coords, axis=1)) for p in pos]) if len(bs_coords) > 0 else 10.0
        dock = -md
        if ref_lig is not None and len(ref_lig) > 0:
            nm = min(len(pos), len(ref_lig))
            rmsd = float(np.sqrt(np.mean(np.sum((pos[:nm]-ref_lig[:nm])**2, axis=1))))
        else: rmsd = 10.0
        mw = float(Descriptors.MolWt(mol)); logp = float(Descriptors.MolLogP(mol))
        return np.array([d32,d228,float(hbc),cat,float(s1c),float(s1pc),float(tc),bur,dock,rmsd,mw,logp], dtype=np.float32)
    except: return None

def smiles_to_graph(smi, lab):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    af = [[a.GetAtomicNum(),a.GetDegree(),a.GetFormalCharge(),int(a.GetHybridization()),
           int(a.GetIsAromatic()),a.GetTotalNumHs(),int(a.IsInRing())] for a in mol.GetAtoms()]
    x = torch.tensor(af, dtype=torch.float32)
    ei = []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx(); ei.extend([[i,j],[j,i]])
    if not ei: ei = [[0,0]]
    g = Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous())
    g.y = torch.tensor([lab], dtype=torch.float32)
    return g

def process(df):
    gs, ls, fs, ss = [], [], [], []
    for _, row in df.iterrows():
        smi, lab = row['Drug'], row['Y']
        g = smiles_to_graph(smi, lab); f = interaction_features(smi)
        if g is not None and f is not None and g.x.shape[0] > 0:
            gs.append(g); ls.append(lab); fs.append(f); ss.append(smi)
    return gs, np.array(ls), np.array(fs), ss

print("  Computing interaction features...")
t0 = time.time()
trg, trl, trf, trs = process(train_df)
vag, val_, vaf, vas = process(val_df)
teg, tel_, tef, tes = process(test_df)
print(f"  Done in {time.time()-t0:.1f}s: Train={len(trg)}, Val={len(vag)}, Test={len(teg)}")

for i, nm in enumerate(INTERACTION_NAMES):
    v = trf[:, i]; grp = [g for g,m in GROUPS.items() if nm in m][0]
    print(f"    [{grp[:4]}] {nm:<24}: mean={v.mean():.3f} std={v.std():.3f} [{v.min():.1f},{v.max():.1f}]")


# ============================================================
# PHASE 4: TRAIN GCN + PROBE
# ============================================================
print("\n" + "=" * 85)
print("PHASE 4: Train GCN + probe with interaction features")
print("=" * 85)

class ToxGCN(nn.Module):
    def __init__(self, hd=128, nl=3, do=0.2):
        super().__init__()
        self.cs = nn.ModuleList(); self.bs = nn.ModuleList()
        self.cs.append(GCNConv(7,hd)); self.bs.append(nn.BatchNorm1d(hd))
        for _ in range(nl-1): self.cs.append(GCNConv(hd,hd)); self.bs.append(nn.BatchNorm1d(hd))
        self.do = nn.Dropout(do)
        self.cl = nn.Sequential(nn.Linear(hd,hd//2),nn.ReLU(),nn.Dropout(do),nn.Linear(hd//2,1))
    def forward(self, data, return_embedding=False):
        x,ei,batch = data.x,data.edge_index,data.batch
        for c,b in zip(self.cs,self.bs): x = self.do(F.relu(b(c(x,ei))))
        emb = global_mean_pool(x,batch); logits = self.cl(emb).squeeze(-1)
        return (logits,emb) if return_embedding else logits

model = ToxGCN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
dtl = PyGDataLoader(trg, batch_size=64, shuffle=True)
dvl = PyGDataLoader(vag, batch_size=64, shuffle=False)
dtsl = PyGDataLoader(teg, batch_size=64, shuffle=False)
pw = torch.tensor([(1-np.mean(trl))/max(np.mean(trl),1e-6)]).to(device)

ba, bst = 0.0, None; t0 = time.time()
for ep in range(200):
    model.train(); ls = 0
    for b in dtl:
        b=b.to(device); opt.zero_grad()
        l=F.binary_cross_entropy_with_logits(model(b),b.y.squeeze(-1),pos_weight=pw)
        l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); ls+=l.item()*b.num_graphs
    if ep%5==0:
        model.eval(); vp,vt=[],[]
        with torch.no_grad():
            for b in dvl: b=b.to(device); vp.extend(torch.sigmoid(model(b)).cpu().numpy()); vt.extend(b.y.squeeze(-1).cpu().numpy())
        try: va=roc_auc_score(vt,vp)
        except: va=0.5
        sch.step(1-va)
        if va>ba: ba=va; bst={k:v.clone() for k,v in model.state_dict().items()}
        if ep%50==0: print(f"    Ep {ep:3d}: loss={ls/len(trg):.4f} val={va:.4f}")

model.load_state_dict(bst)
print(f"  Training: {time.time()-t0:.1f}s, best val AUC: {ba:.4f}")

model.eval(); tp,tt=[],[]
with torch.no_grad():
    for b in dtsl: b=b.to(device); tp.extend(torch.sigmoid(model(b)).cpu().numpy()); tt.extend(b.y.squeeze(-1).cpu().numpy())
try: test_auc=roc_auc_score(tt,tp)
except: test_auc=0.5
print(f"  Test AUC: {test_auc:.4f} {'PASS' if test_auc>=0.65 else 'WARN'}")

def get_emb(m,loader):
    m.eval(); o=[]
    with torch.no_grad():
        for b in loader: b=b.to(device); _,e=m(b,return_embedding=True); o.append(e.cpu().numpy())
    return np.concatenate(o)

et = get_emb(model, dtsl)
rm = ToxGCN().to(device); rm.eval()
er = get_emb(rm, dtsl)

sc = StandardScaler(); tfn = sc.fit_transform(tef)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Feature':<24} {'Ridge dR2':>10} {'Verdict':>18}")
print(f"  {'-'*55}")
raw = {}
for j,nm in enumerate(INTERACTION_NAMES):
    t = tfn[:,j]
    if np.std(t)<1e-10: raw[nm]={'ridge_delta_r2':0,'se':0.01}; continue
    st = cross_val_score(Ridge(alpha=1.0),et,t,cv=kf,scoring='r2')
    sr = cross_val_score(Ridge(alpha=1.0),er,t,cv=kf,scoring='r2')
    rd = np.mean(st)-np.mean(sr); se = np.sqrt(np.var(st)/len(st)+np.var(sr)/len(sr))
    raw[nm] = {'ridge_delta_r2':rd, 'se':max(se,1e-6)}
    print(f"  {nm:<24} {rd:>10.4f} {'ENCODED' if rd>0.05 else 'ZOMBIE':>18}")


# ============================================================
# PHASE 5: HARDENING
# ============================================================
print("\n" + "=" * 85)
print("PHASE 5: Statistical hardening")
print("=" * 85)

NP = 500; rng = np.random.default_rng(42)
scaff = []
for smi in tes:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        c = MurckoScaffold.GetScaffoldForMol(mol)
        g = MurckoScaffold.MakeScaffoldGeneric(c)
        scaff.append(Chem.MolToSmiles(g))
    else: scaff.append('UNK')
scaff = np.array(scaff)
sg = {s: np.where(scaff==s)[0] for s in np.unique(scaff)}

hd = {n:{} for n in INTERACTION_NAMES}

print(f"  Scaffold perm + y-scramble ({NP} perms)...")
for j,nm in enumerate(INTERACTION_NAMES):
    t = tfn[:,j]
    if np.std(t)<1e-10: hd[nm]['sp_p']=hd[nm]['ys_p']=1.0; continue
    obs = raw[nm]['ridge_delta_r2']; spn = np.zeros(NP); ysn = np.zeros(NP)
    for p in range(NP):
        pt = t.copy()
        for idx in sg.values():
            if len(idx)>1: pt[idx]=rng.permutation(pt[idx])
        spn[p] = np.mean(cross_val_score(Ridge(alpha=1.0),et,pt,cv=kf,scoring='r2'))-np.mean(cross_val_score(Ridge(alpha=1.0),er,pt,cv=kf,scoring='r2'))
        pt2 = rng.permutation(t)
        ysn[p] = np.mean(cross_val_score(Ridge(alpha=1.0),et,pt2,cv=kf,scoring='r2'))-np.mean(cross_val_score(Ridge(alpha=1.0),er,pt2,cv=kf,scoring='r2'))
    hd[nm]['sp_p']=float(np.mean(spn>=obs)); hd[nm]['ys_p']=float(np.mean(ysn>=obs))
    print(f"    {nm:<24}: sp={hd[nm]['sp_p']:.4f} ys={hd[nm]['ys_p']:.4f}")

print("  Confound regression...")
conf = tfn[:,CONFOUND_IDX]
tc = confound_removal(et,conf); rc = confound_removal(er,conf)
for j,nm in enumerate(INTERACTION_NAMES):
    t = tfn[:,j]
    if np.std(t)<1e-10: hd[nm]['cd']=0.0; continue
    lr = LinearRegression(); lr.fit(conf,t); t2 = t-lr.predict(conf)
    if np.std(t2)<1e-10: hd[nm]['cd']=0.0; continue
    hd[nm]['cd'] = np.mean(cross_val_score(Ridge(alpha=1.0),tc,t2,cv=kf,scoring='r2'))-np.mean(cross_val_score(Ridge(alpha=1.0),rc,t2,cv=kf,scoring='r2'))

print("  FDR + TOST + BF...")
rp = np.array([hd[n].get('sp_p',1.0) for n in INTERACTION_NAMES])
fdr = fdr_correction(rp, method='bh')
for j,n in enumerate(INTERACTION_NAMES): hd[n]['fdr_p'] = fdr['corrected_p'][j]
for n in INTERACTION_NAMES:
    d,se = raw[n]['ridge_delta_r2'],raw[n]['se']
    hd[n]['tost'] = False
    if d < 0.05:
        r = tost_equivalence_test(delta_r2=d,se=se,epsilon=0.05); hd[n]['tost'] = r['equivalent']
    bf = bayes_factor_null(delta_r2=d,se=se); hd[n]['bf01'] = bf['bf01']
print(f"  FDR survivors: {int(fdr['rejected'].sum())}/12")


# ============================================================
# PHASE 6: RESULTS
# ============================================================
print("\n" + "=" * 85)
print("PHASE 6: Hardened Results")
print("=" * 85)

def verd(hh):
    sp=hh.get('sp_p',1); fp=hh.get('fdr_p',1); cd=hh.get('cd',0); tost=hh.get('tost',False); bf=hh.get('bf01',1)
    if sp<0.05 and cd>0.05 and fp<0.05: return "CONFIRMED_ENCODED"
    elif sp<0.05 and cd<0.02: return "CONFOUND_DRIVEN"
    elif tost and bf>3: return "CONFIRMED_ZOMBIE"
    elif fp>=0.05: return "LIKELY_ZOMBIE"
    elif sp<0.05: return "CANDIDATE_ENCODED"
    return "LIKELY_ZOMBIE"

print(f"\n  {'Feature':<24} {'Raw dR2':>8} {'Scaf-p':>7} {'Clean dR2':>10} {'FDR-p':>7} {'BF01':>6} {'Verdict':<20}")
print(f"  {'-'*85}")
verdicts = {}
for nm in INTERACTION_NAMES:
    v = verd(hd[nm]); verdicts[nm] = v
    print(f"  {nm:<24} {raw[nm]['ridge_delta_r2']:>8.4f} {hd[nm].get('sp_p',1):>7.4f} "
          f"{hd[nm].get('cd',0):>10.4f} {hd[nm].get('fdr_p',1):>7.4f} "
          f"{hd[nm].get('bf01',1):>6.2f} {v:<20}")

enc = {'CONFIRMED_ENCODED','CANDIDATE_ENCODED'}

print(f"\n{'='*85}")
print("PROBE TARGET COMPARISON: Molecular Descriptors vs Interaction Features")
print(f"{'='*85}")
print(f"""
  Previous probes (molecular descriptors):
    "Does GCN encode what the molecule LOOKS like?"
    Result: 1/4 binding features, AUC=0.922, ZOMBIE

  New probes (interaction features):
    "Does GCN encode how the molecule INTERACTS with BACE1?"
""")
for gn,mem in GROUPS.items():
    ne = sum(1 for n in mem if verdicts[n] in enc)
    lb = {'CATALYTIC':'Catalytic site','POCKET':'Pocket shape','QUALITY':'Binding quality','CONFOUND':'Confounds'}
    print(f"    {lb[gn]}: {ne}/{len(mem)} encoded")
    for n in mem: print(f"      {n:<24}: {verdicts[n]}")

nc = sum(1 for n in GROUPS['CATALYTIC'] if verdicts[n] in enc)
np_ = sum(1 for n in GROUPS['POCKET'] if verdicts[n] in enc)

print(f"\n{'='*85}")
print("DISCOVERY READINESS (Interaction-Based)")
print(f"{'='*85}")
print(f"  Test AUC: {test_auc:.3f}, Catalytic: {nc}/4, Pocket: {np_}/4")

if nc>=2 and np_>=2:
    print(f"\n  DISCOVERY READY -- GCN implicitly learns interaction geometry!")
elif nc>=1 or np_>=2:
    print(f"\n  PARTIALLY INTERACTION-AWARE. Augment with protein pocket features.")
elif sum(1 for n in GROUPS['CONFOUND'] if verdicts[n] in enc)>0 and nc==0:
    print(f"\n  INTERACTION ZOMBIE -- encodes MW/LogP but NOT catalytic interactions.")
    print(f"  Architecture fundamentally limited. Need protein structure as input.")
else:
    print(f"\n  NOT INTERACTION-AWARE. 2D graph insufficient for binding geometry.")
    print(f"  Try: protein-ligand co-encoding or AlphaFold3 co-folding.")

print("\n" + "=" * 85)
print("NEXT STEPS")
print("=" * 85)
print("  1. If READY: validate on more targets (tau, gamma-secretase)")
print("  2. If ZOMBIE: co-encode protein pocket + ligand graph")
print("  3. Compare with actual Vina docking scores as ground truth")
print("  4. Test on second AD target (tau aggregation)")
print("=" * 85)
