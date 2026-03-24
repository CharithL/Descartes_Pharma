# DESCARTES-PHARMA v1.0

## Mechanistic Zombie Detection for Drug Discovery

### Adapted from the DESCARTES Enhanced Dual Factory v3.0 for Pharmaceutical Target Validation, AI Drug Discovery, and Preclinical-to-Clinical Translation

*March 2026*

---

## Table of Contents

1. [The Problem: $288 Billion in Pharmaceutical Zombies](#1-the-problem)
2. [Architecture Overview: From Neural Circuits to Drug Pipelines](#2-architecture-overview)
3. [Domain Translation Dictionary](#3-domain-translation)
4. [Recommended Test Datasets (Easiest → Hardest)](#4-test-datasets)
5. [C1 Mechanistic Probing Factory: Adapted Probe Taxonomy](#5-c1-probing-factory)
6. [HIGH PRIORITY: SAE Polypharmacology Decomposition](#6-sae)
7. [HIGH PRIORITY: MLP ΔR² Nonlinear Dose-Response Detection](#7-mlp)
8. [HIGH PRIORITY: Statistical Hardening Suite for Drug Data](#8-statistical-hardening)
9. [Tier 1 Joint Alignment Probes: Molecular Mechanism Matching](#9-joint-alignment)
10. [Tier 2 Dynamical Probes: Pharmacokinetic Trajectory Matching](#10-dynamical)
11. [Tier 3 Topological Probes: Chemical Space Topology](#11-topological)
12. [Tier 4 Causal Probes: Target Knockdown Validation](#12-causal)
13. [Tier 5 Information-Theoretic Probes: Binding Information Content](#13-information-theoretic)
14. [Tier 6 Temporal & Structural Probes: Dose-Time Encoding](#14-temporal-structural)
15. [C2 Drug Candidate Factory: Architecture with LLM Integration](#15-c2-drug-factory)
16. [Co-Evolution Protocol: The Full Nested Factory](#16-co-evolution)
17. [Zombie Verdict Generator: Pharma Edition](#17-zombie-verdict)
18. [Implementation Roadmap with Test Datasets](#18-implementation-roadmap)
19. [Compute Estimates](#19-compute)

---

## 1. The Problem: $288 Billion in Pharmaceutical Zombies

### 1.1 The Scale of Mechanistic Ignorance

The global pharmaceutical industry spends approximately **$288 billion per year** on R&D. Yet **90–92% of drugs entering clinical trials fail**. The cost per approved drug now exceeds **$2.23 billion** (Deloitte 2024). The single largest cause of failure: **lack of efficacy (40–50% of all failures)** — drugs that worked in preclinical models but failed in humans because the model was a pharmaceutical zombie.

**Definition — Pharmaceutical Zombie:** A preclinical model (animal model, cell assay, AI prediction system, or computational surrogate) that produces correct output predictions (drug response, toxicity, binding affinity) through internal computations that do NOT correspond to real human disease biology.

### 1.2 The Zombie Graveyard

| Domain | Investment | Failure Rate | Root Cause |
|--------|-----------|-------------|------------|
| Alzheimer's Disease | $42.5B since 1995 | 99.6% | Mouse amyloid models = zombie (right plaques, wrong biology) |
| Sepsis | Decades of research | 100% (150 drugs) | Mouse inflammatory response ≠ human response |
| CNS / Psychiatry | $Billions | 92% fail after animal tests | No mechanistic validation of animal models |
| AI Drug Discovery | $18B+ invested | 0 approved drugs | Black-box models predict without mechanism |
| Phase II Trials | $50-100M each | 72% failure rate | Proof-of-mechanism gap |

### 1.3 Why DESCARTES-PHARMA Is Necessary

Every existing validation framework tests only **input-output accuracy**: Does the model predict the right drug response? Does the animal model show tumor shrinkage?

Nobody systematically tests: **Does the model's INTERNAL computation correspond to real human disease biology?**

This is the pharmaceutical zombie question. DESCARTES-PHARMA provides the first systematic framework to answer it.

### 1.4 What DESCARTES-PHARMA Adapts

The original DESCARTES Dual Factory v3.0 (ARIA COGITO Programme) was designed to determine whether neural network surrogates of biological neural circuits are "computational zombies" — models whose hidden states don't correspond to biological gating variables. DESCARTES-PHARMA adapts this framework to drug discovery, where:

- **Hidden states** → model internal representations (GNN embeddings, attention weights, latent features)
- **Biological gating variables** → known molecular mechanisms (binding sites, pathway activations, pharmacophores)
- **Neural circuit surrogates** → drug discovery AI models (QSAR, GNN, transformer-based models)
- **Computational zombies** → pharmaceutical zombies (models that predict correctly for wrong reasons)

---

## 2. Architecture Overview

### 2.1 The Dual Factory Concept (Pharma Edition)

**C1 (Mechanistic Probing Factory):** Evolves measurement instruments (probes) that test whether a drug discovery model's internal representations correspond to known molecular biology. Instead of probing LSTM hidden states for neural gating variables, probes GNN embeddings for pharmacophore features, binding pocket geometry, and pathway activation signatures.

**C2 (Drug Candidate Factory):** Evolves model architectures and molecular candidates that attempt to predict drug properties while preserving internal mechanistic correspondence. Instead of evolving neural network surrogates, evolves molecular scaffolds, model architectures, and training strategies to find candidates that work for the right biological reasons.

**Co-evolution:** C1 discovers which molecular mechanisms are (or aren't) encoded in models → feeds back to C2 to guide architecture search toward mechanistically interpretable designs. C2 discovers which architectures naturally recover mechanism → feeds back to C1 to prioritize probing effort.

### 2.2 System Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                    DESCARTES-PHARMA v1.0                             ║
║         Mechanistic Zombie Detection for Drug Discovery              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌───────────────────────┐      ┌────────────────────────┐          ║
║  │  C1 MECHANISTIC       │◄────►│  C2 DRUG CANDIDATE     │          ║
║  │  PROBING FACTORY      │      │  FACTORY               │          ║
║  │                       │      │                        │          ║
║  │  43 Probe Methods     │      │  12 Model Architectures│          ║
║  │  7 Tiers              │      │  Molecular Scaffolds   │          ║
║  │  Pharma Adaptations   │      │  Thompson Sampling     │          ║
║  │  SAE Polypharmacology │      │  LLM Scaffold Design   │          ║
║  └──────────┬────────────┘      └──────────┬─────────────┘          ║
║             │                               │                        ║
║             └──────────┬────────────────────┘                        ║
║                        │                                             ║
║  ┌─────────────────────▼──────────────────────┐                     ║
║  │         TEST DATA TIERS                     │                     ║
║  │  Tier 1: HH Simulator (ground truth)        │                     ║
║  │  Tier 2: TDC ClinTox/BBBP (pharma bench)   │                     ║
║  │  Tier 3: Allen Brain (neuroscience)         │                     ║
║  │  Tier 4: RxRx3/GDSC (phenomics/genomics)   │                     ║
║  └─────────────────────┬──────────────────────┘                     ║
║                        │                                             ║
║             ┌──────────▼──────────┐                                  ║
║             │  STATISTICAL        │                                  ║
║             │  HARDENING SUITE    │                                  ║
║             │  13 Methods         │                                  ║
║             └──────────┬──────────┘                                  ║
║                        │                                             ║
║             ┌──────────▼──────────┐                                  ║
║             │  ZOMBIE VERDICT     │                                  ║
║             │  GENERATOR v1.0     │                                  ║
║             │  PHARMA EDITION     │                                  ║
║             │  8 Verdict Types    │                                  ║
║             └─────────────────────┘                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Domain Translation Dictionary

### 3.1 Concept Mapping: Neuroscience → Pharma

| DESCARTES (Neuroscience) | DESCARTES-PHARMA (Drug Discovery) |
|---|---|
| Neural circuit | Disease pathway / drug-target system |
| Biological gating variables (m, h, n) | Molecular mechanisms (binding affinity, pharmacophore, pathway flux) |
| LSTM hidden states | GNN/Transformer embeddings, latent representations |
| Surrogate model | Drug discovery AI model (QSAR, GNN, foundation model) |
| Input current (I) | Molecular structure / chemical descriptors |
| Output voltage (V) | Drug property prediction (efficacy, toxicity, ADMET) |
| Computational zombie | Pharmaceutical zombie (right prediction, wrong mechanism) |
| Superposition (polysemantic neurons) | Polypharmacology (multi-target drug effects) |
| Ridge ΔR² probing | Feature-mechanism correspondence testing |
| SAE decomposition | Polypharmacology decomposition |
| Resample ablation | Target knockdown / gene knockout validation |
| Mean-clamping (FORBIDDEN) | Mean-imputation of features (EQUALLY FORBIDDEN) |
| Koopman spectral analysis | PK/PD timescale matching |
| SINDy equation discovery | Mechanistic ODE recovery from drug response data |
| Gate-specific LSTM probing | Layer-specific GNN probing |
| Block permutation null | Scaffold-stratified permutation null |
| Cross-condition correlation | Cross-patient / cross-cell-line correlation |
| Zombie verdict | Mechanistic validation verdict |

### 3.2 The Superposition ↔ Polypharmacology Insight

**Original DESCARTES:** Large LSTMs (h=128/256) may encode biological variables in polysemantic superposition — multiple variables sharing the same hidden dimensions, invisible to linear probes.

**DESCARTES-PHARMA:** Drug discovery models may encode multiple target interactions in superposed embeddings. A drug that appears inactive against any single target may be potent through a **superposition of weak multi-target effects** — exactly the polypharmacology hypothesis. SAE decomposition can separate these entangled effects, potentially rescuing "failed" drug candidates.

### 3.3 The Forbidden Operation: Mean-Clamping ↔ Mean-Imputation

**DESCARTES rule:** NEVER use mean-clamping for causal ablation (produces z-scores of -3000 to -4000 that are OOD artifacts). ALWAYS use resample ablation.

**DESCARTES-PHARMA rule:** NEVER use mean-imputation of molecular features for causal testing. Replacing a molecular feature with its dataset mean creates molecules that don't exist in chemical space — out-of-distribution artifacts. ALWAYS resample from the empirical distribution of that feature (scaffold-matched permutation).

---

## 4. Recommended Test Datasets

### 4.1 Tiered Dataset Strategy

```
TIER 1 — SYNTHETIC GROUND TRUTH (Start Here — You Know the Answer)
├── Hodgkin-Huxley Simulator          [2 hours setup, perfect ground truth]
├── FitzHugh-Nagumo / Morris-Lecar    [1 hour setup, simplified]
└── Izhikevich Neuron Model           [1 hour setup, multi-regime]

TIER 2 — PHARMA BENCHMARKS (Known Mechanisms, Easy Access)
├── TDC ClinTox                       [30 min setup, FDA vs failed drugs]
├── TDC BBBP                          [30 min setup, BBB penetration]
├── MoleculeNet Tox21                 [30 min setup, 12 toxicity endpoints]
├── PharmaBench ADMET                 [1 hour setup, 11 ADMET properties]
└── MoleculeNet HIV                   [30 min setup, HIV inhibition]

TIER 3 — NEUROSCIENCE (Real Biology, Known Internals)
├── Allen Brain Observatory           [1 day setup, visual cortex responses]
├── CRCNS Datasets                    [1 day setup, 150 neural datasets]
└── OpenScope (Allen Institute)       [1 day setup, standardized NWB format]

TIER 4 — ADVANCED PHARMA (Complex, High-Dimensional)
├── RxRx3-core (Recursion Phenomics)  [1 day setup, 222K wells, Cell Painting]
├── GDSC (Drug Sensitivity in Cancer) [1 day setup, 1000 cell lines]
├── BELKA (Leash DEL data)            [1 day setup, 100M molecules]
├── PrimeKG (Knowledge Graph)         [1 day setup, drug-disease prediction]
└── ChEMBL / BindingDB               [variable, binding affinity data]
```

### 4.2 Tier 1: Hodgkin-Huxley Simulator (THE #1 Starting Point)

**Why start here:** You have COMPLETE ground truth for every biological intermediate variable (m, h, n, V). You KNOW what the zombie verdict should be. This is a unit test for your entire pipeline.

```python
"""
DESCARTES-PHARMA Tier 1: HH Simulator Ground Truth Generator

Generates training data with full biological intermediate variables.
Train an LSTM/GNN to replicate I→V mapping, then probe whether
hidden states recover m, h, n — the zombie test with known answer.
"""

import numpy as np
from scipy.integrate import odeint


class HodgkinHuxleySimulator:
    """
    Generate ground truth data for DESCARTES-PHARMA validation.
    
    This is the UNIT TEST for the entire framework:
    1. Simulate HH neuron → get (I, V, m, h, n) trajectories
    2. Train surrogate: I → V (input-output only)
    3. Probe surrogate hidden states for m, h, n
    4. Known answer: good surrogate MUST encode m, h, n
    5. If probes fail to find m, h, n → probe is broken
    6. If probes find m, h, n → probe is validated
    """
    
    # Standard HH parameters
    C_m = 1.0       # membrane capacitance (uF/cm^2)
    g_Na = 120.0    # sodium conductance (mS/cm^2)
    g_K = 36.0      # potassium conductance (mS/cm^2)
    g_L = 0.3       # leak conductance (mS/cm^2)
    E_Na = 50.0     # sodium reversal potential (mV)
    E_K = -77.0     # potassium reversal potential (mV)
    E_L = -54.387   # leak reversal potential (mV)
    
    @staticmethod
    def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    @staticmethod
    def beta_m(V): return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    @staticmethod
    def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    @staticmethod
    def beta_h(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    @staticmethod
    def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    @staticmethod
    def beta_n(V): return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    def derivatives(self, state, t, I_func):
        V, m, h, n = state
        I = I_func(t)
        
        dVdt = (I - self.g_Na * m**3 * h * (V - self.E_Na)
                  - self.g_K * n**4 * (V - self.E_K)
                  - self.g_L * (V - self.E_L)) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n
        
        return [dVdt, dmdt, dhdt, dndt]
    
    def simulate(self, I_func, T=100.0, dt=0.01):
        """
        Simulate HH neuron and return ALL biological variables.
        
        Returns:
            dict with keys:
                't': time vector
                'V': membrane voltage (OUTPUT — what surrogate must predict)
                'm': Na activation gate (BIOLOGICAL GROUND TRUTH)
                'h': Na inactivation gate (BIOLOGICAL GROUND TRUTH)
                'n': K activation gate (BIOLOGICAL GROUND TRUTH)
                'I': input current
                'I_Na': sodium current (DERIVED GROUND TRUTH)
                'I_K': potassium current (DERIVED GROUND TRUTH)
                'g_Na_eff': effective Na conductance (DERIVED GROUND TRUTH)
                'g_K_eff': effective K conductance (DERIVED GROUND TRUTH)
        """
        t = np.arange(0, T, dt)
        V0, m0, h0, n0 = -65.0, 0.05, 0.6, 0.32
        
        solution = odeint(self.derivatives, [V0, m0, h0, n0], t,
                         args=(I_func,), hmax=dt)
        
        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        I = np.array([I_func(ti) for ti in t])
        
        # Derived biological variables
        g_Na_eff = self.g_Na * m**3 * h
        g_K_eff = self.g_K * n**4
        I_Na = g_Na_eff * (V - self.E_Na)
        I_K = g_K_eff * (V - self.E_K)
        
        return {
            't': t, 'V': V, 'm': m, 'h': h, 'n': n, 'I': I,
            'I_Na': I_Na, 'I_K': I_K,
            'g_Na_eff': g_Na_eff, 'g_K_eff': g_K_eff
        }
    
    def generate_dataset(self, n_trials=100, T=100.0, dt=0.01, seed=42):
        """
        Generate a full training dataset with varied input currents.
        
        Each trial has a different current injection pattern, producing
        different firing patterns. The surrogate must learn the I→V mapping;
        the probing factory tests whether it also learns m, h, n internally.
        
        Returns:
            inputs: (n_trials, T_steps, 1) — input currents
            outputs: (n_trials, T_steps, 1) — voltage responses
            bio_targets: (n_trials, T_steps, 7) — ALL biological variables
            target_names: list of biological variable names
        """
        rng = np.random.default_rng(seed)
        t_steps = int(T / dt)
        
        inputs = np.zeros((n_trials, t_steps, 1))
        outputs = np.zeros((n_trials, t_steps, 1))
        bio_targets = np.zeros((n_trials, t_steps, 7))
        
        target_names = ['m', 'h', 'n', 'I_Na', 'I_K', 'g_Na_eff', 'g_K_eff']
        
        for i in range(n_trials):
            # Random current injection pattern
            pattern_type = rng.choice(['step', 'ramp', 'noisy', 'pulse_train'])
            
            if pattern_type == 'step':
                amplitude = rng.uniform(0, 20)
                onset = rng.uniform(10, 30)
                I_func = lambda t, a=amplitude, o=onset: a if t > o else 0.0
                
            elif pattern_type == 'ramp':
                rate = rng.uniform(0.1, 0.5)
                I_func = lambda t, r=rate: r * t
                
            elif pattern_type == 'noisy':
                mean_I = rng.uniform(5, 15)
                noise_std = rng.uniform(1, 5)
                noise = rng.normal(mean_I, noise_std, t_steps)
                I_func = lambda t, n=noise, d=dt: n[min(int(t/d), len(n)-1)]
                
            elif pattern_type == 'pulse_train':
                freq = rng.uniform(10, 100)  # Hz
                amplitude = rng.uniform(10, 30)
                duty = rng.uniform(0.1, 0.5)
                I_func = lambda t, f=freq, a=amplitude, d=duty: \
                    a if (t * f / 1000) % 1.0 < d else 0.0
            
            result = self.simulate(I_func, T, dt)
            
            inputs[i, :, 0] = result['I']
            outputs[i, :, 0] = result['V']
            bio_targets[i, :, 0] = result['m']
            bio_targets[i, :, 1] = result['h']
            bio_targets[i, :, 2] = result['n']
            bio_targets[i, :, 3] = result['I_Na']
            bio_targets[i, :, 4] = result['I_K']
            bio_targets[i, :, 5] = result['g_Na_eff']
            bio_targets[i, :, 6] = result['g_K_eff']
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'bio_targets': bio_targets,
            'target_names': target_names,
            'dt': dt,
            'T': T,
            'n_trials': n_trials
        }


# === USAGE ===
# hh = HodgkinHuxleySimulator()
# dataset = hh.generate_dataset(n_trials=200)
# 
# Then:
# 1. Train LSTM/GNN: inputs → outputs (I → V)
# 2. Extract hidden states from trained model
# 3. Run DESCARTES-PHARMA probes on hidden states vs bio_targets
# 4. The zombie verdict MUST be NON-ZOMBIE for a good surrogate
# 5. If verdict is ZOMBIE → your probes are broken, not the model
```

### 4.3 Tier 2: TDC ClinTox (30-Minute Setup)

**Why this dataset:** Directly tests the pharmaceutical zombie question — do models that predict clinical trial toxicity failure do so by encoding real toxicological mechanisms, or by finding statistical shortcuts?

```python
"""
DESCARTES-PHARMA Tier 2: TDC ClinTox Dataset Loader

ClinTox: 1,491 drugs with FDA approval status and 
clinical trial toxicity outcomes. The zombie question:
does a GNN predicting toxicity encode actual toxicological
features (reactive metabolites, hERG binding, hepatotoxicity
pathways) or dataset artifacts (molecular weight, logP)?
"""

def load_clintox():
    """
    Load ClinTox dataset from Therapeutics Data Commons.
    
    pip install PyTDC
    
    Returns molecular SMILES, binary labels, and 
    known mechanistic ground truth for probing.
    """
    from tdc.single_pred import Tox
    
    data = Tox(name='ClinTox')
    df = data.get_data()
    
    # Two tasks:
    # 1. CT_TOX: clinical trial toxicity (did drug fail for tox?)
    # 2. FDA_APPROVED: FDA approval status
    
    return {
        'smiles': df['Drug'].values,
        'labels': df['Y'].values,
        'task': 'binary_classification',
        'n_compounds': len(df),
        
        # KNOWN MECHANISTIC GROUND TRUTH for probing:
        # These are the "biological variables" we probe for
        'mechanism_targets': [
            'reactive_metabolite_alerts',    # Structural alerts for reactive metabolites
            'herg_pharmacophore',            # hERG channel binding features
            'hepatotoxicity_features',       # Known hepatotoxic substructures
            'mitochondrial_toxicity',        # Mitochondrial disruption features
            'phospholipidosis_risk',         # Cationic amphiphilic features
            'DNA_intercalation',             # Planar aromatic features
            'oxidative_stress_potential',    # Quinone/redox cycling features
        ]
    }


def load_bbbp():
    """
    Load Blood-Brain Barrier Penetration dataset.
    
    Known mechanistic features:
    - Molecular weight (MW < 450 for BBB crossing)
    - Polar surface area (PSA < 90 Å²)
    - Hydrogen bond donors (HBD ≤ 3)
    - logP (1.0-3.0 optimal range)
    - Presence of P-gp substrate features
    
    The zombie question: does the model encode these known
    biophysical determinants, or shortcuts like total atom count?
    """
    from tdc.single_pred import ADME
    
    data = ADME(name='BBB_Martins')
    df = data.get_data()
    
    return {
        'smiles': df['Drug'].values,
        'labels': df['Y'].values,
        'task': 'binary_classification',
        'n_compounds': len(df),
        
        'mechanism_targets': [
            'molecular_weight',
            'polar_surface_area',
            'hydrogen_bond_donors',
            'hydrogen_bond_acceptors',
            'logP',
            'rotatable_bonds',
            'pgp_substrate_features',
            'tight_junction_permeability',
        ]
    }


def compute_mechanistic_features(smiles_list):
    """
    Compute known mechanistic features from SMILES.
    These serve as "biological ground truth" for probing.
    
    pip install rdkit
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([np.nan] * 12)
            continue
        
        features.append([
            Descriptors.MolWt(mol),                          # Molecular weight
            Descriptors.MolLogP(mol),                        # LogP
            rdMolDescriptors.CalcTPSA(mol),                  # Polar surface area
            Descriptors.NumHDonors(mol),                     # H-bond donors
            Descriptors.NumHAcceptors(mol),                  # H-bond acceptors
            Descriptors.NumRotatableBonds(mol),              # Rotatable bonds
            Descriptors.NumAromaticRings(mol),               # Aromatic rings
            Descriptors.FractionCSP3(mol),                   # Fraction sp3 carbons
            rdMolDescriptors.CalcNumHeavyAtoms(mol),         # Heavy atom count
            Descriptors.NumAliphaticRings(mol),              # Aliphatic rings
            rdMolDescriptors.CalcNumAmideBonds(mol),         # Amide bonds
            Descriptors.PEOE_VSA1(mol),                      # Electrostatic features
        ])
    
    feature_names = [
        'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds',
        'AromaticRings', 'FractionCSP3', 'HeavyAtoms',
        'AliphaticRings', 'AmideBonds', 'PEOE_VSA1'
    ]
    
    return np.array(features), feature_names
```

### 4.4 Tier 3: Allen Brain Observatory

```python
"""
DESCARTES-PHARMA Tier 3: Allen Brain Observatory

Real neural data with known functional properties.
Train a model to predict neural responses to visual stimuli,
then probe whether hidden states encode known visual features.

pip install allensdk
"""

def load_allen_brain_observatory():
    """
    Load visual coding dataset from Allen Brain Observatory.
    
    Known biological ground truth:
    - Orientation selectivity (tuned to specific grating angles)
    - Spatial frequency preference
    - Temporal frequency preference
    - Direction selectivity
    - ON/OFF response types
    
    These serve as known "biological variables" for probing.
    """
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
    manifest_path = 'allen_brain_observatory_manifest.json'
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    
    # Get experiment containers for a specific visual area
    containers = boc.get_experiment_containers(
        targeted_structures=['VISp'],  # Primary visual cortex
        cre_lines=['Cux2-CreERT2']     # Layer 2/3 neurons
    )
    
    return {
        'containers': containers,
        'n_containers': len(containers),
        'mechanism_targets': [
            'orientation_selectivity_index',
            'preferred_orientation',
            'spatial_frequency_preference',
            'temporal_frequency_preference',
            'direction_selectivity_index',
            'on_off_ratio',
            'sustained_transient_ratio',
            'receptive_field_size',
        ],
        'note': 'Use AllenSDK to download actual neural traces per container'
    }
```

### 4.5 Tier 4: RxRx3-core (Phenomics)

```python
"""
DESCARTES-PHARMA Tier 4: RxRx3-core Phenomics Dataset

222,601 wells of Cell Painting data with:
- 735 genetic knockouts (KNOWN biological ground truth)
- 1,674 small-molecule perturbations
- 6-channel fluorescence images + precomputed embeddings

The zombie question: do models predicting perturbation 
identity encode actual cellular phenotypes (mitosis disruption,
ER stress, apoptosis) or imaging artifacts (plate effects,
edge effects, batch effects)?

Download: https://www.rxrx.ai/rxrx3
Size: < 18 GB
"""

def rxrx3_mechanism_targets():
    """
    Known biological mechanisms for genetic knockouts.
    These serve as ground truth for probing.
    """
    return {
        'mechanism_targets': [
            'cell_cycle_arrest',         # G1/S/G2/M checkpoint activation
            'apoptosis_markers',         # Caspase activation, nuclear fragmentation
            'er_stress_response',        # UPR pathway activation
            'mitotic_defects',           # Spindle abnormalities
            'cytoskeletal_disruption',   # Actin/tubulin reorganization
            'lipid_accumulation',        # Lipid droplet formation
            'dna_damage_response',       # γH2AX foci, repair pathway activation
            'autophagy_markers',         # LC3 puncta, lysosomal accumulation
            'senescence_markers',        # SA-β-gal, SASP factors
            'metabolic_shift',           # Glycolytic vs oxidative markers
        ],
        'confound_targets': [
            'plate_position',            # MUST NOT encode this
            'batch_id',                  # MUST NOT encode this  
            'well_edge_distance',        # MUST NOT encode this
            'cell_density_artifact',     # MUST NOT encode this
        ]
    }
```

---

## 5. C1 Mechanistic Probing Factory: Adapted Probe Taxonomy

### 5.1 Seven-Tier Organization (Pharma Edition)

```
TIER 0 — TRANSFORMS (preprocessing, applied before other probes)
├── SAE decomposition (polypharmacology → monosemantic target features)
├── PCA / z-score normalization
├── Molecular fingerprint decomposition
└── Scaffold-based stratification

TIER 1 — STATE-LEVEL PROBES [HIGH PRIORITY]
├── Ridge ΔR² (baseline: do embeddings encode mechanisms?)
├── Lasso regression (sparse mechanism selection)
├── MLP ΔR² (nonlinear mechanism encoding) [MANDATORY with every Ridge]
├── KNN probe (local structure)
├── SAE + Ridge [polypharmacology detection]
├── Kernel Ridge (nonlinear features)
└── Random kitchen sinks (scalable nonlinear baseline)

TIER 2 — JOINT ALIGNMENT PROBES
├── CCA (embedding-mechanism subspace alignment)
├── RSA (embedding geometry vs mechanism geometry)
├── CKA (nonlinear geometric alignment)
├── pi-VAE (identifiable latent recovery conditioned on scaffold)
├── CEBRA (joint structure-activity embedding)
└── Procrustes (rotation-invariant alignment)

TIER 3 — DYNAMICAL PROBES (for dose-response / PK data)
├── Koopman spectral analysis (PK/PD timescale matching)
├── SINDy symbolic regression (mechanistic ODE recovery)
├── DSA (dynamical similarity: model vs biological dynamics)
└── Trajectory matching (dose-response curve correspondence)

TIER 4 — TOPOLOGICAL PROBES
├── Persistent homology (chemical space topology matching)
└── Manifold dimension estimation (latent space complexity)

TIER 5 — CAUSAL PROBES
├── Resample ablation [CANONICAL — never mean-impute features]
├── DAS (distributed alignment search in embedding space)
├── Feature knockout (remove molecular substructure, test prediction)
├── Transfer entropy (directed information: structure → prediction)
└── Convergent cross-mapping (scaffold → activity causation)

TIER 6 — INFORMATION-THEORETIC PROBES
├── MINE / InfoNCE (mutual information: embedding ↔ mechanism)
├── MDL probing (minimum description length)
├── Scaffold-resolved R² (decompose by chemical scaffold)
└── Partial coherence (remove shared confounds)

TIER 7 — STRUCTURAL & LAYER-SPECIFIC PROBES
├── Layer-specific GNN probing (message passing layers probed separately)
├── Attention weight analysis (which molecular substructures attended to?)
├── Adversarial probes (discriminator-based mechanism detection)
├── Conditional probes (mechanism encoding varies by drug class?)
└── Scaffold-stratified probing (does encoding depend on chemical family?)
```

### 5.2 Probe Genome Specification (Pharma Edition)

```python
@dataclass
class PharmaProbeGenome:
    """Complete probe specification for drug discovery mechanistic validation."""
    
    # Identity
    genome_id: str
    tier: int                    # 0-7
    probe_type: str              # Method name from taxonomy
    
    # Tier 0 transform
    transform: str               # 'raw', 'sae', 'pca', 'zscore', 'fingerprint'
    transform_params: dict       # e.g., {'expansion': 4, 'k': 20} for SAE
    
    # Probe configuration
    decoder_type: str            # 'ridge', 'lasso', 'mlp_1', 'mlp_2', 'knn'
    decoder_params: dict
    
    # Target specification
    target_type: str             # 'single_mechanism', 'joint', 'conditional'
    target_names: List[str]      # Which mechanistic features to probe for
    condition: Optional[str]     # e.g., 'kinase_inhibitor_only'
    
    # Stratification (replaces temporal in neuroscience)
    stratification: str          # 'none', 'scaffold', 'target_class', 'assay_type'
    scaffold_family: Optional[str]
    
    # Statistical hardening
    null_method: str             # 'scaffold_permutation', 'random_smiles', 'y_scramble'
    n_permutations: int
    fdr_correction: bool
    
    # Model-specific
    layer_target: Optional[int]  # Which GNN layer to probe (None = final)
    attention_heads: Optional[List[int]]  # Which attention heads
    
    # Dataset-specific
    dataset: str                 # 'hh_simulator', 'clintox', 'bbbp', 'rxrx3', etc.
    split_strategy: str          # 'scaffold_split', 'random', 'temporal'
```

---

## 6. HIGH PRIORITY: SAE Polypharmacology Decomposition

### 6.1 Rationale

Linear probing of drug model embeddings assumes each embedding dimension encodes at most one molecular mechanism. But drugs routinely act through **polypharmacology** — simultaneous modulation of multiple targets. A kinase inhibitor may also bind hERG, inhibit CYP3A4, and activate an off-target GPCR. These multiple effects are **superposed** in the model's learned embedding.

An SAE trained on model embeddings decomposes this superposition into monosemantic features — each SAE feature should correspond to a single mechanistic effect.

### 6.2 Implementation

```python
class PharmaSAE(nn.Module):
    """
    Sparse Autoencoder for drug model embedding decomposition.
    
    Adapted from DESCARTES v3.0 SparseAutoencoder.
    Applied to GNN/Transformer embeddings instead of LSTM hidden states.
    
    Key adaptation: expansion factor may need to be higher (8-16×)
    for drug embeddings because the chemical mechanism space is
    larger than the neural gating variable space.
    """
    
    def __init__(self, input_dim, expansion_factor=8, k=30):
        super().__init__()
        n_features = expansion_factor * input_dim
        self.k = k
        self.input_dim = input_dim
        self.n_features = n_features
        
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)
        
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0)
    
    def encode(self, x):
        x_centered = x - self.decoder.bias
        pre_act = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return sparse
    
    def forward(self, x):
        sparse = self.encode(x)
        recon = self.decoder(sparse)
        return recon, sparse


def sae_probe_molecular_mechanisms(sae, model_embeddings, mechanism_features,
                                     mechanism_names, device='cpu'):
    """
    Probe SAE features for molecular mechanisms.
    
    Adapted from DESCARTES sae_probe_biological_variables().
    
    Instead of probing for HH gating variables (m, h, n),
    probes for molecular mechanisms (binding affinity, 
    pharmacophore features, pathway activation).
    
    Key output: monosemanticity scores.
    High monosemanticity → each SAE feature = one mechanism
    Low monosemanticity → polypharmacology superposition remains
    
    The "superposition_detected" flag means:
    SAE-Ridge R² >> raw-Ridge R² for a given mechanism,
    indicating the mechanism WAS encoded but was invisible
    to linear probing due to polypharmacological superposition.
    """
    with torch.no_grad():
        h_tensor = torch.tensor(model_embeddings, dtype=torch.float32, device=device)
        sae_features = sae.encode(h_tensor).cpu().numpy()
    
    n_features = sae_features.shape[1]
    n_mechanisms = mechanism_features.shape[1]
    
    # Correlation matrix: (n_sae_features, n_mechanisms)
    corr_matrix = np.zeros((n_features, n_mechanisms))
    for i in range(n_features):
        feat = sae_features[:, i]
        if feat.std() < 1e-10:
            continue
        for j in range(n_mechanisms):
            target = mechanism_features[:, j]
            if target.std() < 1e-10:
                continue
            corr_matrix[i, j] = np.corrcoef(feat, target)[0, 1]
    
    # Monosemanticity: does each SAE feature map to exactly one mechanism?
    monosemanticity = np.zeros(n_features)
    for i in range(n_features):
        abs_corr = np.abs(corr_matrix[i, :])
        total = abs_corr.sum()
        if total < 1e-10:
            monosemanticity[i] = 0.0
            continue
        probs = abs_corr / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(n_mechanisms)
        monosemanticity[i] = 1.0 - (entropy / max_entropy)
    
    # Compare SAE-Ridge vs raw-Ridge
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sae_r2 = {}
    raw_r2 = {}
    
    for j, name in enumerate(mechanism_names):
        target = mechanism_features[:, j]
        
        sae_scores = []
        raw_scores = []
        for train_idx, test_idx in kf.split(sae_features):
            ridge_sae = Ridge(alpha=1.0)
            ridge_sae.fit(sae_features[train_idx], target[train_idx])
            sae_scores.append(ridge_sae.score(sae_features[test_idx], target[test_idx]))
            
            ridge_raw = Ridge(alpha=1.0)
            ridge_raw.fit(model_embeddings[train_idx], target[train_idx])
            raw_scores.append(ridge_raw.score(model_embeddings[test_idx], target[test_idx]))
        
        sae_r2[name] = np.mean(sae_scores)
        raw_r2[name] = np.mean(raw_scores)
    
    return {
        'correlation_matrix': corr_matrix,
        'monosemanticity_scores': monosemanticity,
        'sae_r2': sae_r2,
        'raw_r2': raw_r2,
        'n_alive': int((np.abs(corr_matrix).max(axis=1) > 0.01).sum()),
        'mean_monosemanticity': float(monosemanticity[monosemanticity > 0].mean()),
        'polypharmacology_detected': {
            name: sae_r2[name] > raw_r2[name] + 0.05
            for name in mechanism_names
        }
    }
```

### 6.3 Interpretation Guide (Pharma Edition)

| SAE R² | Raw Ridge R² | Interpretation |
|:---:|:---:|---|
| High | High | **Monosemantic encoding** — mechanism directly readable from embeddings |
| High | Low | **POLYPHARMACOLOGY DETECTED** — mechanism encoded but entangled with others |
| Low | Low | **Genuinely not encoded** — pharmaceutical zombie confirmed for this mechanism |
| Low | High | Impossible (SAE features are a richer basis) — check SAE training |

### 6.4 Confound Warning: Dataset Artifacts

Analogous to DESCARTES's "grid-locked feature warning": check whether SAE features correlate with **dataset artifacts** rather than biology:
- Molecular weight distribution of training set
- Scaffold prevalence (common scaffolds dominate embeddings)
- Assay-specific batch effects
- Tautomer/stereoisomer inconsistencies

---

## 7. HIGH PRIORITY: MLP ΔR² Nonlinear Dose-Response Detection

### 7.1 Rationale

Ridge regression is linear. If molecular mechanisms are encoded as nonlinear functions of embedding dimensions (e.g., binding affinity = product of electrostatic + hydrophobic features), Ridge reports ΔR² ≈ 0 even though mechanism is present.

**Rule: every Ridge probe MUST have an MLP companion.**

If MLP ΔR² >> Ridge ΔR² → mechanism is **nonlinearly encoded**, not absent.
If MLP ΔR² ≈ Ridge ΔR² → linear probing is sufficient.

### 7.2 Implementation

```python
# IDENTICAL to DESCARTES v3.0 Section 4.2, with terminology changes:
# - hidden_trained → model_embeddings
# - hidden_untrained → random_embeddings (from untrained model)
# - targets → mechanism_features
# - target_names → mechanism_names
# 
# The code is the same. The interpretation changes.

def pharma_mlp_delta_r2(model_embeddings, random_embeddings, 
                          mechanism_features, mechanism_names,
                          hidden_dim=64, epochs=50, lr=1e-3,
                          n_splits=5, device='cpu'):
    """
    Compute MLP ΔR² alongside Ridge ΔR² for all mechanisms.
    
    ΔR² = R²(trained_model) - R²(untrained_model)
    
    The untrained model has the same architecture but random weights.
    If even random embeddings decode the mechanism, the "encoding"
    is trivially decodable from any projection — not meaningful.
    
    Encoding type classification:
    - LINEAR_ENCODED: Ridge finds it, MLP confirms
    - NONLINEAR_ONLY: MLP finds it, Ridge misses → nonlinear mechanism encoding
    - ZOMBIE: Neither finds it → mechanism genuinely not encoded
    """
    # [Same implementation as DESCARTES v3.0 mlp_delta_r2()]
    # See Section 4.2 of original guide
    pass
```

---

## 8. HIGH PRIORITY: Statistical Hardening Suite for Drug Data

### 8.1 The Problem (Pharma-Specific)

Drug discovery datasets have specific statistical pathologies that inflate probe results:

1. **Scaffold bias:** Molecules sharing scaffolds have correlated properties AND correlated embeddings → spurious R² from shared scaffolding, not mechanism encoding
2. **Activity cliffs:** Structurally similar molecules with dramatically different activities → probe overfits to cliff edges
3. **Assay noise:** Biological assays have 10-30% inherent variability → noise floor inflates null distribution
4. **Imbalanced classes:** Active compounds are typically 1-5% of screening data → accuracy metrics are misleading

### 8.2 Adapted 13-Method Suite

All 13 methods from DESCARTES v3.0 Section 5 are retained, with the following pharma-specific adaptations:

| # | DESCARTES Method | DESCARTES-PHARMA Adaptation |
|---|---|---|
| 1 | Block permutation | **Scaffold-stratified permutation** — permute within scaffold families |
| 2 | Phase-randomized surrogates | **Y-scramble** — shuffle activity labels preserving molecular feature distributions |
| 3 | Circular shift null | **Matched molecular pairs** — compare activity changes in MMP series |
| 4 | Effective degrees of freedom | **Scaffold-adjusted N_eff** — account for scaffold clustering |
| 5 | FDR correction (BH) | **Same** — applied across all mechanism × dataset combinations |
| 6 | Frequency-resolved R² | **Scaffold-resolved R²** — decompose by chemical family |
| 7 | Durbin-Watson | **Tanimoto-distance residual test** — check if residuals cluster by structure |
| 8 | Partial coherence | **Confound removal** — regress out MW, logP before probing |
| 9 | TOST equivalence testing | **Same** — formally confirm zombie status |
| 10 | Bayes factor for null | **Same** — BF01 > 3 = moderate zombie evidence |
| 11 | Gap CV | **Scaffold-split CV** — train/test never share scaffolds |
| 12 | Cluster permutation testing | **Activity-cliff-aware clustering** |
| 13 | Ljung-Box residual test | **Structural autocorrelation test** — residuals correlated with Tanimoto similarity? |

### 8.3 Scaffold-Stratified Permutation (Primary Null)

```python
def scaffold_stratified_permutation(smiles, mechanism_values, n_perms=1000, seed=42):
    """
    Permute mechanism values WITHIN scaffold families.
    
    Standard random permutation destroys scaffold structure and produces
    anti-conservative nulls. Scaffold-stratified permutation preserves
    the within-scaffold correlation structure while breaking the 
    embedding-mechanism association.
    
    This is the pharma equivalent of block permutation for time series.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    rng = np.random.default_rng(seed)
    
    # Assign scaffold families
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol))
            scaffolds.append(Chem.MolToSmiles(scaffold))
        else:
            scaffolds.append('UNKNOWN')
    
    scaffolds = np.array(scaffolds)
    unique_scaffolds = np.unique(scaffolds)
    
    null_values = np.zeros((n_perms, len(mechanism_values)))
    
    for p in range(n_perms):
        perm_values = mechanism_values.copy()
        for scaffold in unique_scaffolds:
            mask = scaffolds == scaffold
            idx = np.where(mask)[0]
            if len(idx) > 1:
                perm_idx = rng.permutation(idx)
                perm_values[idx] = mechanism_values[perm_idx]
        null_values[p] = perm_values
    
    return null_values
```

---

## 9-14. Tiers 1-6: Probe Implementations

*Each tier follows the identical mathematical formulation from DESCARTES v3.0 Sections 6-11, with the domain translations specified in Section 3.1 of this guide. Key adaptations per tier:*

### Tier 1 (Joint Alignment): CCA/RSA/CKA applied to **embedding space vs mechanism space** instead of hidden state space vs biological variable space. pi-VAE conditioned on **scaffold identity** instead of stimulus identity.

### Tier 2 (Dynamical): Koopman/SINDy/DSA applied to **dose-response time courses** and **pharmacokinetic trajectories** instead of neural dynamics. SINDy goal: recover known PK/PD ODEs (Michaelis-Menten, Hill equation) from model predictions.

### Tier 3 (Topological): TDA applied to **chemical space manifolds** — do model embeddings preserve the topological structure of known chemical series (e.g., SAR contours around a lead compound)?

### Tier 4 (Causal): Resample ablation applied to **molecular substructures** — remove pharmacophore features by resampling from scaffold-matched molecules, test if prediction degrades. DAS finds the **encoding direction** for each mechanism in embedding space.

### Tier 5 (Information-Theoretic): MINE estimates **mutual information between embeddings and mechanisms**. MDL measures how many bits are needed to describe the mechanism given the embedding. Transfer entropy measures **directed information flow** from structural features to model predictions.

### Tier 6 (Structural): Layer-specific GNN probing (equivalent to gate-specific LSTM probing). Attention weight analysis identifies which molecular substructures drive predictions. Scaffold-stratified probing tests whether mechanism encoding varies by drug class.

---

## 15. C2 Drug Candidate Factory

### 15.1 Architecture Registry (Pharma Edition)

```
ARCHITECTURE REGISTRY (by zombie risk)
│
├── TIER 1: MECHANISTIC BY CONSTRUCTION (risk: ZERO to VERY LOW)
│   ├── Physics-based docking (reference, not ML)
│   ├── QSP/PBPK models (mechanistic ODEs)
│   └── Matched Molecular Pair analysis (structure-based)
│
├── TIER 2: STRONG ANTI-ZOMBIE (risk: LOW)
│   ├── 3D-aware GNNs with physics priors (SchNet, DimeNet, GemNet)
│   ├── Equivariant neural networks (E(3)-equivariant)
│   ├── Neural ODE for PK/PD modeling
│   └── Mechanistic Neural Networks (gray-box)
│
├── TIER 3: PROMISING (risk: MEDIUM-LOW)
│   ├── Graph Neural Networks (GCN, GAT, MPNN)
│   ├── Transformer-based molecular models (MoLFormer)
│   ├── Molecular foundation models (fine-tuned)
│   └── Contrastive learning molecular encoders
│
├── TIER 4: FLEXIBLE BUT ZOMBIE-PRONE (risk: MEDIUM-HIGH)
│   ├── Morgan fingerprint + Random Forest
│   ├── Descriptor-based MLP
│   └── SMILES-based RNN/LSTM
│
└── TIER 5: MAXIMUM ZOMBIE RISK (risk: HIGH)
    ├── Molecular fingerprint + logistic regression
    ├── Naive descriptor stacking
    └── Black-box ensemble without interpretability
```

### 15.2 Drug Candidate Genome Specification

```python
@dataclass
class DrugCandidateGenome:
    """
    Specification of a drug discovery model for C2 factory evolution.
    
    Analogous to SurrogateGenome_v3 but for molecular property prediction.
    """
    
    # Identity
    genome_id: str = ''
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    
    # Model architecture
    architecture: str = 'gcn'        # gcn, gat, mpnn, schnet, dimenet,
                                      # transformer, fingerprint_mlp, neural_ode
    embedding_dim: int = 128
    n_layers: int = 4
    dropout: float = 0.1
    readout: str = 'mean'            # mean, sum, attention, set2set
    
    # Molecular representation
    representation: str = 'graph'     # graph, smiles, fingerprint, 3d_conformer
    fingerprint_type: Optional[str] = None   # morgan, rdkit, topological
    fingerprint_bits: int = 2048
    use_3d: bool = False
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 200
    optimizer: str = 'adam'
    split_strategy: str = 'scaffold' # scaffold, random, temporal
    
    # Loss
    primary_loss: str = 'bce'        # bce, mse, focal, contrastive
    auxiliary_mechanism_loss: bool = False
    aux_mechanism_weight: float = 0.0
    aux_mechanism_targets: List[str] = field(default_factory=list)
    
    # Regularization
    weight_decay: float = 0.0
    embedding_l1: float = 0.0
    information_bottleneck: bool = False
    ib_beta: float = 0.01
    
    # Inductive biases
    physics_priors: bool = False      # 3D geometry, electrostatics
    scaffold_awareness: bool = False  # Explicit scaffold decomposition
    pharmacophore_features: bool = False
    chirality_aware: bool = False
    
    # Dataset
    dataset: str = 'clintox'
    task: str = 'binary_classification'
```

---

## 16. Co-Evolution Protocol

### 16.1 The Full Nested Factory (Pharma Edition)

```
OUTER LOOP — DRUG CANDIDATE FACTORY (C2)
  Searches: architectures × representations × losses × regularizers
  Question: "Can ANY model predict drug properties NON-zombistically?"
  
  INNER LOOP — MECHANISTIC PROBING FACTORY (C1)
    43 probe methods × statistical hardening
    Question: "Does THIS model encode real molecular mechanisms?"
    Returns: MechanisticVerdict (fitness for outer loop)
  
  FITNESS = α·prediction_accuracy + β·mechanism_correspondence + γ·causal_necessity
  
  EVOLUTION: Mutation → Crossover → Thompson Sampling → LLM Balloon
  
  DREAMCODER: Extract reusable design patterns
    e.g., "3D_equivariant + pharmacophore_loss = low zombie risk"
```

---

## 17. Zombie Verdict Generator: Pharma Edition

### 17.1 Eight Verdict Types

```python
class PharmaZombieVerdictGenerator:
    """
    Generate definitive mechanistic validation verdict.
    
    VERDICT TYPES (adapted from DESCARTES v3.0):
    1. CONFIRMED_ZOMBIE — model predicts correctly but encodes NO real mechanisms
    2. LIKELY_ZOMBIE — most probes negative, statistics inconclusive
    3. SPURIOUS_SCAFFOLD — apparent encoding driven by scaffold similarity, not mechanism
    4. POLYPHARMACOLOGY_DETECTED — SAE reveals mechanisms invisible to linear probes
    5. NONLINEAR_MECHANISM — MLP finds mechanism Ridge misses
    6. CANDIDATE_MECHANISTIC — some positive evidence, needs more testing
    7. CONFIRMED_MECHANISTIC — multiple methods confirm mechanism encoding
    8. CAUSALLY_VALIDATED — mechanism encoded AND causally necessary for prediction
    """
    
    def generate_verdict(self, evidence_bundle):
        e = evidence_bundle
        
        # Check causal necessity first
        if e.get('resample_ablation', {}).get('causal', False):
            return {
                'verdict': 'CAUSALLY_VALIDATED',
                'confidence': 'HIGH',
                'evidence': 'Feature ablation degrades prediction (z < -2.0)',
                'recommendation': 'ADVANCE TO CLINICAL — mechanism validated'
            }
        
        # Check polypharmacology
        sae = e.get('polypharmacology_detected', False)
        ridge_low = e.get('ridge_delta_r2', 0) < 0.05
        if sae and ridge_low:
            return {
                'verdict': 'POLYPHARMACOLOGY_DETECTED',
                'confidence': 'HIGH',
                'evidence': 'SAE reveals multi-target encoding invisible to linear probes',
                'recommendation': 'Investigate multi-target profile before advancing'
            }
        
        # Check nonlinear encoding
        mlp_high = e.get('mlp_delta_r2', 0) > 0.1
        if mlp_high and ridge_low:
            return {
                'verdict': 'NONLINEAR_MECHANISM',
                'confidence': 'MEDIUM',
                'evidence': 'MLP ΔR² >> Ridge ΔR²',
                'recommendation': 'Mechanism present but nonlinearly encoded'
            }
        
        # Check scaffold confound
        scaffold_r2 = e.get('scaffold_resolved_r2', {})
        scaffold_only = (scaffold_r2.get('within_scaffold', 0) < 0.05 and 
                         scaffold_r2.get('between_scaffold', 0) > 0.1)
        if scaffold_only:
            return {
                'verdict': 'SPURIOUS_SCAFFOLD',
                'confidence': 'HIGH',
                'evidence': 'Encoding entirely driven by scaffold similarity',
                'recommendation': 'DO NOT ADVANCE — mechanism not validated'
            }
        
        # Check confirmed mechanistic
        significant = e.get('p_scaffold_permutation', 1.0) < 0.05
        ridge_high = e.get('ridge_delta_r2', 0) > 0.2
        cca_match = e.get('cca', {}).get('n_significant', 0) > 2
        rsa_match = e.get('rsa', {}).get('geometric_match', False)
        
        n_positive = sum([
            ridge_high, mlp_high, cca_match, rsa_match,
            e.get('tda', {}).get('topological_match', False),
            e.get('mdl', {}).get('encoded', False),
        ])
        
        if significant and n_positive >= 3:
            return {
                'verdict': 'CONFIRMED_MECHANISTIC',
                'confidence': 'HIGH',
                'evidence': f'{n_positive} methods confirm mechanism encoding',
                'recommendation': 'Strong candidate for clinical advancement'
            }
        
        if significant and n_positive >= 1:
            return {
                'verdict': 'CANDIDATE_MECHANISTIC',
                'confidence': 'MEDIUM',
                'evidence': f'{n_positive} methods positive',
                'recommendation': 'Additional mechanistic testing recommended'
            }
        
        # Confirmed zombie
        tost = e.get('tost_zombie', {}).get('zombie_confirmed', False)
        bf = e.get('bayes_factor', {}).get('bf01', 0) > 3
        
        if tost and bf and not significant:
            return {
                'verdict': 'CONFIRMED_ZOMBIE',
                'confidence': 'HIGH',
                'evidence': 'TOST + BF01 > 3 confirm no mechanism encoding',
                'recommendation': 'DO NOT ADVANCE — model is a pharmaceutical zombie'
            }
        
        if not significant and n_positive == 0:
            return {
                'verdict': 'LIKELY_ZOMBIE',
                'confidence': 'MEDIUM',
                'evidence': 'No method finds mechanism, not formally confirmed',
                'recommendation': 'Redesign model or investigate alternative mechanisms'
            }
        
        return {
            'verdict': 'AMBIGUOUS',
            'confidence': 'LOW',
            'evidence': 'Mixed signals',
            'recommendation': 'Extended probing campaign needed'
        }
```

---

## 18. Implementation Roadmap with Test Datasets

### Phase 1: GROUND TRUTH VALIDATION (Week 1-2)

```
Day 1-2: HH Simulator Pipeline
  - Generate HH dataset (200 trials, 100ms each, dt=0.01)
  - Train LSTM surrogate: I → V
  - Extract hidden states
  - Run Ridge ΔR² + MLP ΔR² for m, h, n, I_Na, I_K
  - EXPECTED RESULT: NON-ZOMBIE for well-trained model
  → This validates the entire probe stack on known ground truth

Day 3-4: Statistical Hardening on HH Data
  - Block permutation null (adaptive block size from ACF)
  - Phase-randomized surrogates (IAAFT)
  - FDR correction across all 7 biological variables
  - TOST equivalence testing for zombie variables
  - Bayes factor for null
  → Confirm that hardening doesn't reject TRUE encoding

Day 5-7: SAE Superposition on HH Data
  - Train SAE on LSTM hidden states (expansion 4×, 8×, 16×)
  - Compare SAE-Ridge vs raw-Ridge
  - Test zombie transition: h=16 (non-zombie) → h=256 (should detect superposition)
  → This validates SAE polypharmacology detection on known biology
```

### Phase 2: FIRST PHARMA TEST (Week 3-4)

```
Day 8-9: TDC ClinTox Pipeline
  - Load ClinTox (pip install PyTDC, one-line download)
  - Compute mechanistic features (RDKit descriptors)
  - Train GCN on SMILES → toxicity
  - Extract GNN embeddings
  - Run full Tier 1 probes: Ridge ΔR² + MLP ΔR² for ALL mechanistic features
  → First real pharma zombie detection

Day 10-11: TDC BBBP Pipeline
  - Load BBBP (one-line download)
  - Known mechanism: MW, TPSA, HBD, logP determine BBB penetration
  - Train GCN on SMILES → BBB
  - Probe: does GNN encode MW, TPSA, HBD, logP or shortcuts?
  → Direct zombie question on known biophysics

Day 12-14: Statistical Hardening for Pharma Data
  - Scaffold-stratified permutation nulls
  - Scaffold-split cross-validation
  - Confound removal (regress out MW, logP, HeavyAtoms)
  - Full 13-method suite on ClinTox + BBBP results
  → Hardens pharma results against scaffold and confound bias
```

### Phase 3: ADVANCED DATASETS (Week 5-6)

```
Day 15-16: Allen Brain Observatory (Neuroscience Validation)
  - Download via AllenSDK
  - Train model on visual stimuli → neural response
  - Probe for orientation selectivity, spatial frequency
  → Cross-validates probe stack on real neuroscience data

Day 17-18: SAE Polypharmacology on Tox21
  - Train SAE on GNN embeddings from Tox21 models
  - 12 toxicity endpoints = 12 mechanisms to probe
  - Test polypharmacology: do compounds that hit multiple endpoints
    show superposition in SAE features?
  → First real polypharmacology detection

Day 19-21: RxRx3-core Phenomics
  - Load 222K well embeddings
  - Known genetic knockouts = ground truth mechanisms
  - Probe for cell cycle, apoptosis, ER stress features
  - Test confound detection: plate position, batch effects
  → Largest-scale pharma zombie test
```

### Phase 4: INTEGRATION (Week 7)

```
Day 22-24: Verdict Generator
  - Integrate all evidence streams
  - Generate per-mechanism, per-dataset, per-architecture verdicts
  - Cross-dataset comparison table:
    HH (known answer) × ClinTox × BBBP × Tox21 × RxRx3

Day 25-26: Co-Evolution Campaign
  - Thompson sampling across GCN/GAT/SchNet/fingerprint architectures
  - DreamCoder pattern library from successful configurations
  - LLM balloon for novel architectures if search stalls

Day 27-28: Paper and Documentation
  - Generate publication-ready tables and figures
  - Demonstrate: DESCARTES-PHARMA catches zombie models
    that standard validation (AUC, accuracy) misses
  - Key metric: how many "high-accuracy" models are zombies?
```

---

## 19. Compute Estimates

### Per Model (Full v1.0 Campaign)

| Component | Time | Hardware |
|---|---|---|
| Statistical hardening (13 methods × all mechanisms) | ~1 hour | CPU |
| MLP ΔR² (all mechanisms × 5-fold × 50 epochs) | ~20 min | GPU |
| SAE training (3 expansion factors) | ~10 min | GPU |
| SAE + Ridge probing | ~5 min | CPU |
| CCA + RSA + CKA | ~10 min | CPU |
| TDA (chemical space topology) | ~10 min | CPU |
| Resample ablation (100 resamples × all mechanisms) | ~30 min | GPU |
| MINE (200 epochs × all mechanisms) | ~15 min | GPU |
| Layer-specific GNN probing | ~15 min | GPU |
| Scaffold-stratified analysis | ~10 min | CPU |
| **TOTAL per model** | **~2.5 hours** | **GPU recommended** |

### Full Cross-Dataset Campaign

| Dataset | Models | Total Time |
|---|---|---|
| HH Simulator (ground truth) | 3 arch × 4 hidden sizes | ~30 hours |
| TDC ClinTox | 5 architectures | ~12.5 hours |
| TDC BBBP | 5 architectures | ~12.5 hours |
| MoleculeNet Tox21 | 5 architectures | ~12.5 hours |
| Allen Brain Observatory | 3 architectures | ~7.5 hours |
| RxRx3-core | 3 architectures | ~7.5 hours |
| GDSC | 3 architectures | ~7.5 hours |
| **TOTAL** | | **~90 hours (~4 GPU-days)** |

Feasible on single RTX 4090 in ~1 week, or ~2 days on cloud with 4× parallel.

---

## Essential References

### Original DESCARTES Framework
1. DESCARTES Enhanced Dual Factory v3.0 — ARIA COGITO Programme (2026)

### Drug Discovery Benchmarks
2. Huang et al. (2021) — Therapeutics Data Commons (NeurIPS)
3. Wu et al. (2018) — MoleculeNet
4. Fang et al. (2024) — PharmaBench ADMET
5. Corsello et al. (2020) — Drug Repurposing Hub
6. Yang et al. (2019) — Analyzing Learned Molecular Representations (MPNN)

### Mechanistic Interpretability in Drug Discovery
7. Jiménez-Luna et al. (2020) — Drug Discovery with Explainable AI
8. Lavecchia (2025) — XAI in Drug Discovery (WIREs)
9. InterPLM (2025) — SAE decomposition of protein language models (Nature Methods)

### Clinical Translation Failure
10. Sun et al. (2022) — Why 90% of clinical drug development fails (Acta Pharm Sin B)
11. Begley & Ellis (2012) — Raise standards for preclinical research (Nature)
12. Freedman et al. (2015) — Economics of reproducibility (PLoS Biology)
13. Cummings et al. (2022) — Costs of developing AD treatments

### Statistical Methods (from DESCARTES v3.0)
14. Theiler (1986) — Surrogate data methods
15. Maris & Oostenveld (2007) — Cluster permutation testing
16. Voita & Titov (2020) — MDL probing

### SAE and Superposition
17. Bricken et al. (2023) — SAE for LLM interpretability
18. Gao et al. (2024) — SAE scaling laws, TopK activation
19. arXiv:2512.24440 — GraphCast SAE (weather model superposition)

### Neuroscience Test Datasets
20. Allen Brain Observatory (2023) — Sharing neurophysiology data (eLife)
21. CRCNS — Collaborative Research in Computational Neuroscience

---

## Non-Negotiable Methodological Requirements

*Carried over from DESCARTES v3.0 and adapted for pharmaceutical application:*

1. **Every probe result must pass the statistical hardening suite** before contributing to the zombie verdict. Scaffold-stratified permutation replaces block permutation as the primary null.

2. **Every Ridge probe must have an MLP companion.** If MLP ΔR² >> Ridge ΔR², the mechanism is nonlinearly encoded, not absent.

3. **Every causal claim must use resample ablation, never mean-imputation.** Replacing molecular features with dataset means creates molecules that don't exist — out-of-distribution artifacts identical to the mean-clamping problem in neuroscience.

4. **Start with HH simulator ground truth.** If your probes can't find m, h, n in a well-trained HH surrogate, the probes are broken. Fix probes before applying to drug discovery data.

5. **Scaffold-split cross-validation is mandatory for all pharma datasets.** Random splits allow scaffold memorization to masquerade as mechanism encoding.

6. **Confound regression (MW, logP, HeavyAtoms) must precede mechanism probing.** These trivially decodable features inflate R² without reflecting genuine mechanistic understanding.

7. **SAE polypharmacology detection must be attempted before declaring any mechanism ZOMBIE.** The mechanism may be encoded in superposition with other mechanisms.

8. **The verdict generator's CONFIRMED_ZOMBIE requires convergent evidence from ≥3 independent methods.** No single probe is sufficient to declare zombie status.

---

*This guide adapts the DESCARTES Enhanced Dual Factory v3.0 framework from computational neuroscience to pharmaceutical drug discovery. The core insight is that "Is this neural network a computational zombie?" and "Does this drug work for the right reason?" are structurally identical problems — both ask whether internal mechanism corresponds to external measurement, and both require the same multi-method, statistically hardened, causally validated approach to answer.*

*The $288 billion annual pharma R&D machine runs without a mechanistic validation operating system. DESCARTES-PHARMA provides one.*
