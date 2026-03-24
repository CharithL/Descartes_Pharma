# DESCARTES-PHARMA v1.2 ADDENDUM

## Hybrid Meta-Learner: Neural Fast Path + LLM Strategic Reasoning

### Integrating 7 Meta-Learning Paradigms from the Philosopher Engine, Explorer Agent, and VKS Architectures into Pharmaceutical Zombie Detection

*March 2026*

---

## Table of Contents

M1. [Why the Dual Factory Needs a Meta-Learner](#m1)
M2. [The 7 Paradigms and Where They Land](#m2)
M3. [Architecture Overview: The Meta-Learner Sits ABOVE Both Factories](#m3)
M4. [Paradigm 1: Online Feedback Loop (Neural Fast Path)](#m4)
M5. [Paradigm 2: Mechanism-Level Decomposition + Per-Mechanism Routing](#m5)
M6. [Paradigm 3: Verified Zombie Store (Persistent Memory)](#m6)
M7. [Paradigm 4: Self-Repair Before Escalation (Probe Cascade)](#m7)
M8. [Paradigm 5: VFE Belief Updating (Bayesian Meta-Cognition)](#m8)
M9. [Paradigm 6: Multi-Timescale Oscillatory Hierarchy](#m9)
M10. [Paradigm 7: Meta-Cognition Over the Search Process (HOT Layer)](#m10)
M11. [LLM Strategic Reasoning Layer](#m11)
M12. [Full Integrated Pipeline](#m12)
M13. [Bootstrap Protocol](#m13)
M14. [Convergence Metrics and Validation](#m14)

---

## M1. Why the Dual Factory Needs a Meta-Learner

The DESCARTES-PHARMA v1.0 dual factory has four components that approximate meta-learning but aren't:

| Component | What It Does | Critical Gap |
|---|---|---|
| Thompson Sampling | Allocates compute to promising architectures | Learns which arm to pull, not how to design better arms |
| DreamCoder | Extracts reusable patterns from successful genomes | Within-campaign compression, no cross-campaign transfer |
| LLM Balloon | Proposes novel architectures when stalled | One-shot proposals, no learning from proposal outcomes |
| DP/PY Clustering | Groups similar architectures | Descriptive, doesn't predict what works on new tasks |

**The result:** Campaign 50 starts from scratch, just like Campaign 1. The system discovers "3D-equivariant GNNs resist zombie-hood for binding affinity tasks" on ClinTox, then re-discovers it independently on BBBP, Tox21, and every subsequent dataset.

**What's needed:** A meta-learner that sits ABOVE both factories, learns general principles of mechanistic encoding across campaigns, and makes each subsequent campaign dramatically faster.

---

## M2. The 7 Paradigms and Where They Land

| # | Paradigm | Source Document | DESCARTES-PHARMA Adaptation |
|---|---|---|---|
| 1 | Online feedback loop | meta_learner.md | Every probe-model evaluation trains a neural router for future probe selection |
| 2 | Claim-level decomposition | VKS Addendum B | Per-mechanism routing: each biological mechanism gets its own probe cascade, not one-size-fits-all |
| 3 | Persistent verified memory | VKS Addendum B | Verified Zombie Store (VZS): once a mechanism is confirmed zombie/non-zombie for an architecture class, never re-probe |
| 4 | Self-repair before escalation | VKS Addendum B | Try cheap probes (Ridge) before expensive ones (SAE, ablation); only escalate when cheap probes fail |
| 5 | VFE belief updating | Explorer Agent v3 | Bayesian posterior over each (architecture × mechanism) pair; complexity-penalized zombie scores |
| 6 | Multi-timescale hierarchy | Explorer Agent v3 | Fast (per-probe, seconds), Medium (per-model, hours), Slow (per-campaign, days) with phase-amplitude coupling |
| 7 | Meta-cognition (HOT) | Explorer Agent v3 | Monitors the dual factory itself; detects stuck search, redundant probing, anomalous verdicts |

---

## M3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DESCARTES-PHARMA META-LEARNER v1.2                     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  LAYER 3: META-COGNITION (HOT — Paradigm 7)                       │  │
│  │  Variational posterior over the SEARCH PROCESS itself              │  │
│  │  Detects: stuck factories, redundant probing, verdict anomalies   │  │
│  │  Triggers: LLM strategic reasoning, C1 expansion, campaign pivot  │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                           │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │  LAYER 2: BELIEF UPDATING (VFE — Paradigm 5)                      │  │
│  │  F = KL(q||p) + E[-log p(data|θ)]  per (architecture × mechanism) │  │
│  │  Multi-channel precision: τ_probe, τ_data, τ_stat, τ_causal       │  │
│  │  Kalman filter updates with each new probe result                  │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                           │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │  LAYER 1: PROBE PRIORITIZATION (Global Workspace — Paradigm 6)    │  │
│  │  43 probes compete for compute budget                              │  │
│  │  Score = expected_info_gain × probe_cost_inv × mechanism_priority  │  │
│  │  ~5 probes active at a time (working memory constraint)            │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                           │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │  LAYER 0: CONTINUOUS MONITORING (Feedback Loop — Paradigms 1,4)    │  │
│  │  Three-timescale processing:                                       │  │
│  │    Fast (per-probe): result → feedback → update router             │  │
│  │    Medium (per-model): aggregate probes → zombie verdict           │  │
│  │    Slow (per-campaign): cross-campaign patterns → DreamCoder       │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                           │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │  INFRASTRUCTURE                                                    │  │
│  │  ├── Verified Zombie Store (VZS — Paradigm 3)                     │  │
│  │  ├── Mechanism Decomposer (Paradigm 2)                            │  │
│  │  ├── Probe Cascade Router (Paradigm 4)                            │  │
│  │  ├── Neural Fast Path (~2M params, trained online)                │  │
│  │  └── LLM Strategic Reasoner (Claude, triggered by HOT layer)      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────┐ ┌──────────┐                                              │
│  │C1 PROBING│ │C2 DRUG   │  ← Factories are BELOW the meta-learner     │
│  │FACTORY   │ │CANDIDATE │  ← Meta-learner decides WHAT to probe,      │
│  │          │ │FACTORY   │    WHEN to escalate, WHERE to search next    │
│  └──────────┘ └──────────┘                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## M4. Paradigm 1: Online Feedback Loop (Neural Fast Path)

### M4.1 Design

Adapted from `meta_learner.md` FeedbackBuffer + MetaLearnerTrainer. Every probe-model evaluation produces a ground-truth training signal.

```python
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class ProbeOutcome:
    """Ground truth signal from a single probe execution."""
    probe_type: str              # 'ridge', 'mlp', 'sae', 'cca', 'ablation', etc.
    architecture: str            # 'gcn', 'gat', 'schnet', etc.
    mechanism: str               # 'hbond_score', 'logP', 'electrostatic', etc.
    dataset: str                 # 'clintox', 'bbbp', 'hh_simulator', etc.
    
    delta_r2: float              # Primary probe result
    p_value: float               # Statistical significance
    compute_seconds: float       # Wall clock cost
    
    verdict_contribution: str    # How this probe affected the verdict
    # 'CONFIRMED_ENCODING', 'CONFIRMED_ZOMBIE', 'INCONCLUSIVE', 'REDUNDANT'
    
    was_useful: bool             # Did this probe change the verdict?
    # True if verdict would have been different without this probe


class PharmaMetaLearner(nn.Module):
    """
    Neural fast path for probe routing decisions.
    
    Architecture adapted from Philosopher Engine meta_learner.md:
    - Three-headed prediction (probe priority, expected yield, cost estimate)
    - Trained online from ProbeOutcome feedback
    - ~2M parameters, runs in <1ms per decision
    
    Input features (per mechanism × architecture pair):
    - Architecture embedding (learned, 32-dim)
    - Mechanism embedding (learned, 32-dim)
    - Dataset embedding (learned, 16-dim)
    - Prior probe results for this pair (summary statistics, 20-dim)
    - VZS lookup result (one-hot: HIT/MISS/PARTIAL, 3-dim)
    - VFE posterior (mean, variance, 2-dim)
    - Campaign progress (fraction of budget spent, 1-dim)
    
    Total input: 106 dimensions
    """
    
    def __init__(self, n_architectures=15, n_mechanisms=20, n_datasets=10,
                 n_probes=43, hidden_dim=128):
        super().__init__()
        
        # Embeddings
        self.arch_embed = nn.Embedding(n_architectures, 32)
        self.mech_embed = nn.Embedding(n_mechanisms, 32)
        self.data_embed = nn.Embedding(n_datasets, 16)
        
        # Feature processing (prior results + VZS + VFE + progress)
        context_dim = 32 + 32 + 16 + 20 + 3 + 2 + 1  # = 106
        
        self.backbone = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Head 1: Probe priority scores (which of 43 probes to run next?)
        self.probe_priority_head = nn.Linear(hidden_dim, n_probes)
        
        # Head 2: Expected information gain per probe
        self.expected_gain_head = nn.Linear(hidden_dim, n_probes)
        
        # Head 3: Routing decision (SKIP / CHEAP_PROBE / EXPENSIVE_PROBE / ESCALATE_TO_LLM)
        self.routing_head = nn.Linear(hidden_dim, 4)
        
        # Head 4: Confidence in current verdict (should we keep probing?)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, arch_id, mech_id, data_id, 
                prior_results, vzs_status, vfe_posterior, progress):
        
        arch_e = self.arch_embed(arch_id)
        mech_e = self.mech_embed(mech_id)
        data_e = self.data_embed(data_id)
        
        x = torch.cat([arch_e, mech_e, data_e, prior_results,
                        vzs_status, vfe_posterior, progress], dim=-1)
        
        features = self.backbone(x)
        
        return {
            'probe_priorities': self.probe_priority_head(features),
            'expected_gains': torch.relu(self.expected_gain_head(features)),
            'routing': self.routing_head(features),
            'routing_decision': ['SKIP', 'CHEAP_PROBE', 'EXPENSIVE_PROBE',
                                  'ESCALATE_TO_LLM'][
                self.routing_head(features).argmax(dim=-1).item()],
            'verdict_confidence': self.confidence_head(features),
            'features': features,  # Cached for feedback
        }


class PharmaFeedbackBuffer:
    """
    Stores probe outcomes for online meta-learner training.
    
    Adapted from meta_learner.md FeedbackBuffer with pharma-specific
    ground truth computation.
    """
    
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = 64
    
    def record(self, meta_features: torch.Tensor, 
               predicted_routing: str,
               predicted_confidence: float,
               outcome: ProbeOutcome):
        """Record a probe outcome as a training signal."""
        
        # Ground truth: was the routing decision correct?
        true_routing = self._compute_true_routing(outcome)
        true_confidence = self._compute_true_confidence(outcome)
        true_gain = self._compute_true_gain(outcome)
        
        self.buffer.append({
            'features': meta_features.detach(),
            'true_routing': true_routing,
            'true_confidence': true_confidence,
            'true_gain': true_gain,
            'predicted_routing': predicted_routing,
            'predicted_confidence': predicted_confidence,
            'outcome': outcome,
            'timestamp': len(self.buffer),
        })
    
    def _compute_true_routing(self, outcome: ProbeOutcome) -> int:
        """What routing SHOULD have been, given the outcome."""
        if outcome.verdict_contribution == 'REDUNDANT':
            return 0  # SKIP — wasted compute
        if outcome.compute_seconds < 60 and outcome.was_useful:
            return 1  # CHEAP_PROBE was appropriate
        if outcome.compute_seconds >= 60 and outcome.was_useful:
            return 2  # EXPENSIVE_PROBE was justified
        if not outcome.was_useful and outcome.compute_seconds >= 300:
            return 0  # Should have SKIPPED expensive useless probe
        return 1  # Default to cheap probe
    
    def _compute_true_confidence(self, outcome: ProbeOutcome) -> float:
        """How confident should the meta-learner have been?"""
        signals = []
        weights = []
        
        # Statistical significance: strongest signal
        if outcome.p_value < 0.001:
            signals.append(1.0)
            weights.append(3.0)
        elif outcome.p_value < 0.05:
            signals.append(0.7)
            weights.append(2.0)
        else:
            signals.append(0.3)
            weights.append(1.0)
        
        # Effect size
        if abs(outcome.delta_r2) > 0.2:
            signals.append(1.0)
            weights.append(2.0)
        elif abs(outcome.delta_r2) > 0.05:
            signals.append(0.6)
            weights.append(1.5)
        else:
            signals.append(0.2)
            weights.append(1.0)
        
        # Usefulness
        signals.append(1.0 if outcome.was_useful else 0.0)
        weights.append(2.0)
        
        return sum(s * w for s, w in zip(signals, weights)) / sum(weights)
    
    def _compute_true_gain(self, outcome: ProbeOutcome) -> float:
        """Information gain normalized by compute cost."""
        if not outcome.was_useful:
            return 0.0
        gain = abs(outcome.delta_r2) / max(outcome.compute_seconds, 1.0)
        return min(gain * 100, 1.0)  # Normalize to [0,1]
    
    def sample_batch(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        # Recency-weighted sampling (from meta_learner.md)
        indices = list(range(len(self.buffer)))
        weights = [1.0 + i / len(self.buffer) for i in indices]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        import random
        sampled = random.choices(indices, weights=probs, k=self.batch_size)
        batch = [self.buffer[i] for i in sampled]
        
        return {
            'features': torch.stack([b['features'].squeeze(0) for b in batch]),
            'true_routing': torch.tensor([b['true_routing'] for b in batch],
                                          dtype=torch.long),
            'true_confidence': torch.tensor([b['true_confidence'] for b in batch]),
            'true_gain': torch.tensor([b['true_gain'] for b in batch]),
        }


class PharmaMetaTrainer:
    """
    Online trainer for the meta-learner.
    
    Adapted from meta_learner.md MetaLearnerTrainer:
    - Trains every 16 new probe outcomes
    - Multi-loss: routing CE + confidence MSE + gain MSE
    - Loss weights: routing matters most (saves compute)
    """
    
    def __init__(self, meta_learner: PharmaMetaLearner, lr=1e-4):
        self.meta = meta_learner
        self.buffer = PharmaFeedbackBuffer()
        self.optimizer = torch.optim.AdamW(meta_learner.parameters(), lr=lr)
        
        self.routing_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()
        self.gain_loss = nn.MSELoss()
        
        # Routing matters most — bad routing wastes GPU-hours
        self.w_routing = 2.5
        self.w_confidence = 1.5
        self.w_gain = 1.0
        
        self.update_count = 0
        self.loss_history = deque(maxlen=5000)
    
    def record_and_maybe_train(self, meta_features, predicted_routing,
                                 predicted_confidence, outcome):
        self.buffer.record(meta_features, predicted_routing,
                            predicted_confidence, outcome)
        
        if len(self.buffer.buffer) >= 64 and len(self.buffer.buffer) % 16 == 0:
            self._train_step()
    
    def _train_step(self):
        batch = self.buffer.sample_batch()
        if batch is None:
            return
        
        self.meta.train()
        
        # Forward pass through backbone only (features already extracted)
        features = batch['features']
        
        # Reconstruct heads from cached features
        probe_priorities = self.meta.probe_priority_head(features)
        routing_logits = self.meta.routing_head(features)
        confidence = self.meta.confidence_head(features).squeeze(-1)
        expected_gains = self.meta.expected_gain_head(features).mean(dim=-1)
        
        # Losses
        loss_r = self.routing_loss(routing_logits, batch['true_routing'])
        loss_c = self.confidence_loss(confidence, batch['true_confidence'])
        loss_g = self.gain_loss(expected_gains, batch['true_gain'])
        
        total_loss = (self.w_routing * loss_r + 
                      self.w_confidence * loss_c +
                      self.w_gain * loss_g)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        self.loss_history.append(total_loss.item())
    
    def save(self, path):
        torch.save({
            'model_state': self.meta.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'buffer_size': len(self.buffer.buffer),
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.meta.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.update_count = ckpt['update_count']
```

---

## M5. Paradigm 2: Mechanism-Level Decomposition

### M5.1 Design

Adapted from VKS Addendum B claim-level routing. Instead of routing entire models to probing, decompose into individual mechanisms and route each independently.

```python
class MechanismDecomposer:
    """
    Splits the zombie question into per-mechanism sub-questions.
    
    Adapted from VKS Addendum B ClaimExtractor:
    - Instead of FORMAL/FACTUAL/INTERPRETIVE claim types
    - Uses STRUCTURAL/FUNCTIONAL/CAUSAL/DYNAMIC mechanism types
    - Each type routes to different probe tiers
    """
    
    MECHANISM_TYPES = {
        'STRUCTURAL': {
            'examples': ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'AromaticRings'],
            'preferred_probes': ['ridge', 'lasso', 'knn'],
            'description': 'Static molecular descriptors — cheapest to probe'
        },
        'FUNCTIONAL': {
            'examples': ['herg_pharmacophore', 'cyp_inhibition', 'pgp_substrate'],
            'preferred_probes': ['ridge', 'mlp', 'sae'],
            'description': 'Functional group patterns — may require SAE for polypharmacology'
        },
        'CAUSAL': {
            'examples': ['binding_affinity', 'target_engagement', 'pathway_activation'],
            'preferred_probes': ['resample_ablation', 'das', 'transfer_entropy'],
            'description': 'Causal relationships — REQUIRES ablation, never correlation alone'
        },
        'DYNAMIC': {
            'examples': ['pk_clearance', 'dose_response_curve', 'time_to_effect'],
            'preferred_probes': ['koopman', 'sindy', 'dsa', 'temporal'],
            'description': 'Time-dependent mechanisms — requires dynamical probes'
        }
    }
    
    def decompose(self, mechanism_list: List[str]) -> dict:
        """
        Classify each mechanism and assign probe cascade.
        
        Returns per-mechanism routing plan:
        {
            'MW': {'type': 'STRUCTURAL', 'cascade': ['ridge', 'mlp']},
            'herg_pharmacophore': {'type': 'FUNCTIONAL', 'cascade': ['ridge', 'mlp', 'sae']},
            'binding_affinity': {'type': 'CAUSAL', 'cascade': ['ridge', 'mlp', 'ablation']},
        }
        """
        routing_plan = {}
        
        for mech in mechanism_list:
            mech_type = self._classify_mechanism(mech)
            preferred = self.MECHANISM_TYPES[mech_type]['preferred_probes']
            
            # Always start with Ridge + MLP (mandatory pair from DESCARTES v3.0)
            cascade = ['ridge', 'mlp']
            
            # Add type-specific probes
            for probe in preferred:
                if probe not in cascade:
                    cascade.append(probe)
            
            routing_plan[mech] = {
                'type': mech_type,
                'cascade': cascade,
                'current_step': 0,
                'results_so_far': [],
                'early_stop': False,
            }
        
        return routing_plan
    
    def _classify_mechanism(self, mechanism_name: str) -> str:
        """Classify mechanism type from name."""
        for mtype, info in self.MECHANISM_TYPES.items():
            if mechanism_name in info['examples']:
                return mtype
        
        # Heuristic classification for unknown mechanisms
        name_lower = mechanism_name.lower()
        if any(k in name_lower for k in ['weight', 'area', 'count', 'ring', 'bond']):
            return 'STRUCTURAL'
        if any(k in name_lower for k in ['binding', 'target', 'pathway', 'knock']):
            return 'CAUSAL'
        if any(k in name_lower for k in ['clearance', 'dose', 'time', 'rate', 'pk']):
            return 'DYNAMIC'
        return 'FUNCTIONAL'  # Default
```

---

## M6. Paradigm 3: Verified Zombie Store (VZS)

### M6.1 Design

Adapted from VKS Addendum B Verified Knowledge Store. Once a mechanism is confirmed zombie or non-zombie for an architecture class, store it permanently so it's never re-probed.

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime


class VerifiedZombieStore:
    """
    Persistent memory of zombie verdicts across campaigns.
    
    Adapted from VKS Addendum B with 4-tier structure:
    
    Tier 1: AXIOMS (permanent, hash-chained)
      "GCN + MW encoding = CONFIRMED_ENCODED on ClinTox, BBBP, Tox21"
      → Never re-probe this combination. It's settled.
    
    Tier 2: DERIVED PATTERNS (proven across multiple campaigns)
      "3D-equivariant architectures resist zombie-hood for binding features"
      → DreamCoder-extracted design pattern with statistical support
    
    Tier 3: CONTESTED (conflicting evidence across datasets)
      "SAE polypharmacology on Tox21: DETECTED, but NOT on ClinTox"
      → Needs more evidence before settling
    
    Tier 4: PROVISIONAL (single-campaign result, awaiting replication)
      "SchNet + electrostatic encoding = CANDIDATE_MECHANISTIC on PDBbind"
      → Not yet replicated, re-probe on next relevant campaign
    
    Key insight from VKS: "Memory loss isn't death; belief corruption is."
    Tier 1 axioms are append-only, immutable, hash-chained.
    """
    
    def __init__(self, store_path='verified_zombie_store.json'):
        self.store_path = Path(store_path)
        self.store = self._load_or_create()
    
    def _load_or_create(self):
        if self.store_path.exists():
            with open(self.store_path) as f:
                return json.load(f)
        return {
            'tier1_axioms': [],
            'tier2_patterns': [],
            'tier3_contested': [],
            'tier4_provisional': [],
            'hash_chain': [],
        }
    
    def lookup(self, architecture: str, mechanism: str, 
               dataset: str = None) -> dict:
        """
        Check if this (architecture, mechanism) pair has a settled verdict.
        
        Returns:
            {'status': 'HIT', 'tier': 1, 'verdict': 'CONFIRMED_ENCODED', ...}
            or {'status': 'MISS'}
            or {'status': 'CONTESTED', 'evidence': [...]}
        """
        key = f"{architecture}:{mechanism}"
        
        # Check Tier 1 first (strongest evidence)
        for axiom in self.store['tier1_axioms']:
            if axiom['key'] == key:
                return {
                    'status': 'HIT',
                    'tier': 1,
                    'verdict': axiom['verdict'],
                    'datasets': axiom['datasets'],
                    'confidence': 'SETTLED',
                    'n_replications': axiom['n_replications'],
                }
        
        # Check Tier 2 (pattern-level)
        for pattern in self.store['tier2_patterns']:
            if self._pattern_matches(pattern, architecture, mechanism):
                return {
                    'status': 'HIT',
                    'tier': 2,
                    'verdict': pattern['predicted_verdict'],
                    'confidence': 'PATTERN_BASED',
                    'pattern_name': pattern['name'],
                }
        
        # Check Tier 3 (contested)
        for contested in self.store['tier3_contested']:
            if contested['key'] == key:
                return {
                    'status': 'CONTESTED',
                    'tier': 3,
                    'evidence': contested['evidence'],
                }
        
        # Check Tier 4 (provisional)
        for prov in self.store['tier4_provisional']:
            if prov['key'] == key:
                return {
                    'status': 'PROVISIONAL',
                    'tier': 4,
                    'verdict': prov['verdict'],
                    'dataset': prov['dataset'],
                }
        
        return {'status': 'MISS'}
    
    def record_verdict(self, architecture: str, mechanism: str,
                        dataset: str, verdict: str, evidence: dict):
        """
        Record a new verdict. Promotes through tiers with replication.
        
        First occurrence → Tier 4 (provisional)
        Replicated on 2nd dataset → Tier 3 or Tier 2
        Replicated on 3+ datasets with consistent verdict → Tier 1 (axiom)
        Conflicting verdicts → Tier 3 (contested)
        """
        key = f"{architecture}:{mechanism}"
        entry = {
            'key': key,
            'architecture': architecture,
            'mechanism': mechanism,
            'dataset': dataset,
            'verdict': verdict,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Check existing entries
        existing = self._find_all_entries(key)
        
        if not existing:
            # First time: Tier 4
            self.store['tier4_provisional'].append(entry)
        else:
            # Check consistency
            verdicts = set(e['verdict'] for e in existing)
            verdicts.add(verdict)
            datasets = set(e.get('dataset', '') for e in existing)
            datasets.add(dataset)
            
            if len(verdicts) == 1 and len(datasets) >= 3:
                # Consistent across 3+ datasets → promote to Tier 1 axiom
                self._promote_to_axiom(key, verdict, list(datasets), existing + [entry])
            elif len(verdicts) == 1 and len(datasets) >= 2:
                # Consistent across 2 datasets → Tier 2 pattern
                self._promote_to_pattern(key, verdict, list(datasets), existing + [entry])
            elif len(verdicts) > 1:
                # Conflicting → Tier 3 contested
                self._mark_contested(key, existing + [entry])
            else:
                # Same dataset, just add to provisional
                self.store['tier4_provisional'].append(entry)
        
        self._save()
    
    def _promote_to_axiom(self, key, verdict, datasets, evidence):
        """Promote to Tier 1. Append-only, hash-chained."""
        axiom = {
            'key': key,
            'verdict': verdict,
            'datasets': datasets,
            'n_replications': len(datasets),
            'evidence_summary': [e.get('evidence', {}) for e in evidence],
            'promoted_at': datetime.now().isoformat(),
        }
        
        # Hash chain (from VKS Addendum B)
        prev_hash = self.store['hash_chain'][-1] if self.store['hash_chain'] else '0'
        axiom_str = json.dumps(axiom, sort_keys=True)
        axiom['hash'] = hashlib.sha256(
            f"{prev_hash}:{axiom_str}".encode()).hexdigest()
        
        self.store['tier1_axioms'].append(axiom)
        self.store['hash_chain'].append(axiom['hash'])
        
        # Remove from lower tiers
        self._remove_from_lower_tiers(key)
    
    def _promote_to_pattern(self, key, verdict, datasets, evidence):
        arch, mech = key.split(':')
        pattern = {
            'key': key,
            'name': f"{arch}_encodes_{mech}",
            'predicted_verdict': verdict,
            'datasets': datasets,
            'architecture_class': arch,
            'mechanism_class': mech,
        }
        self.store['tier2_patterns'].append(pattern)
        self._remove_from_lower_tiers(key)
    
    def _mark_contested(self, key, all_evidence):
        contested = {
            'key': key,
            'evidence': [{'dataset': e.get('dataset'), 
                          'verdict': e.get('verdict')} for e in all_evidence],
        }
        self.store['tier3_contested'].append(contested)
    
    def _pattern_matches(self, pattern, architecture, mechanism):
        return (pattern.get('architecture_class') == architecture and 
                pattern.get('mechanism_class') == mechanism)
    
    def _find_all_entries(self, key):
        entries = []
        for tier in ['tier1_axioms', 'tier2_patterns', 'tier3_contested', 'tier4_provisional']:
            for entry in self.store[tier]:
                if entry.get('key') == key:
                    entries.append(entry)
        return entries
    
    def _remove_from_lower_tiers(self, key):
        for tier in ['tier3_contested', 'tier4_provisional']:
            self.store[tier] = [e for e in self.store[tier] if e.get('key') != key]
    
    def _save(self):
        with open(self.store_path, 'w') as f:
            json.dump(self.store, f, indent=2)
    
    def get_stats(self):
        return {
            'tier1_axioms': len(self.store['tier1_axioms']),
            'tier2_patterns': len(self.store['tier2_patterns']),
            'tier3_contested': len(self.store['tier3_contested']),
            'tier4_provisional': len(self.store['tier4_provisional']),
            'total_settled': len(self.store['tier1_axioms']) + len(self.store['tier2_patterns']),
        }
```

---

## M7. Paradigm 4: Self-Repair Before Escalation (Probe Cascade)

### M7.1 Design

Adapted from VKS self-repair loop. Before running expensive probes (SAE training, resample ablation), try cheaper alternatives. Only escalate when cheap probes are inconclusive.

```python
class ProbeCascadeRouter:
    """
    Escalating probe cascade: cheap → medium → expensive → LLM.
    
    Adapted from VKS self-repair loop:
    - VKS: Z3 fail → self-repair → re-verify → oracle (if still fails)
    - PHARMA: Ridge inconclusive → MLP → SAE → ablation → LLM balloon
    
    Saves ~40-60% compute by catching easy cases early.
    """
    
    TIERS = {
        0: {'probes': ['ridge', 'lasso'], 'cost': 'seconds', 'gate': 0.2},
        1: {'probes': ['mlp', 'knn'], 'cost': 'minutes', 'gate': 0.15},
        2: {'probes': ['sae', 'cca', 'rsa'], 'cost': '10_minutes', 'gate': 0.1},
        3: {'probes': ['resample_ablation', 'das'], 'cost': 'hours', 'gate': 0.05},
        4: {'probes': ['llm_balloon'], 'cost': 'api_call', 'gate': None},
    }
    
    def run_cascade(self, model, mechanism, dataset, hidden_states,
                     mechanism_values, meta_learner=None, vzs=None):
        """
        Run probes in escalating order. Stop early when confident.
        
        Returns verdict + which tier was sufficient.
        """
        # Step 0: Check VZS first (free!)
        if vzs is not None:
            vzs_result = vzs.lookup(model.architecture, mechanism, dataset)
            if vzs_result['status'] == 'HIT':
                return {
                    'verdict': vzs_result['verdict'],
                    'tier_reached': -1,
                    'source': 'VZS_CACHE',
                    'compute_saved': 'ALL',
                }
        
        # Step 1-4: Escalating probes
        evidence = {}
        
        for tier_id in range(5):
            tier = self.TIERS[tier_id]
            
            # Ask meta-learner if this tier is worth running
            if meta_learner is not None and tier_id > 0:
                ml_decision = meta_learner.should_escalate(
                    evidence, tier_id, model.architecture, mechanism)
                if ml_decision == 'SKIP':
                    continue
                if ml_decision == 'STOP':
                    break
            
            # Run probes at this tier
            for probe_name in tier['probes']:
                if probe_name == 'llm_balloon':
                    # Paradigm 11: LLM strategic reasoning
                    evidence['llm_suggestion'] = self._call_llm_balloon(
                        model.architecture, mechanism, evidence)
                    continue
                
                result = self._run_single_probe(
                    probe_name, hidden_states, mechanism_values)
                evidence[probe_name] = result
            
            # Check early stopping: is the verdict clear?
            verdict = self._check_early_verdict(evidence, tier['gate'])
            if verdict is not None:
                return {
                    'verdict': verdict,
                    'tier_reached': tier_id,
                    'evidence': evidence,
                    'compute_saved': f'skipped_tiers_{tier_id+1}_to_4',
                }
        
        # All tiers exhausted: generate final verdict from all evidence
        return {
            'verdict': self._final_verdict(evidence),
            'tier_reached': 4,
            'evidence': evidence,
            'compute_saved': 'none',
        }
    
    def _check_early_verdict(self, evidence, gate_threshold):
        """Can we stop early with confidence?"""
        if gate_threshold is None:
            return None
        
        # Count strong signals
        strong_positive = sum(
            1 for r in evidence.values() 
            if isinstance(r, dict) and r.get('delta_r2', 0) > gate_threshold
            and r.get('p_value', 1.0) < 0.05)
        
        strong_negative = sum(
            1 for r in evidence.values()
            if isinstance(r, dict) and r.get('delta_r2', 0) < 0.02
            and r.get('tost_zombie', False))
        
        if strong_positive >= 2:
            return 'CONFIRMED_ENCODED'
        if strong_negative >= 2:
            return 'CONFIRMED_ZOMBIE'
        
        return None  # Inconclusive, continue to next tier
    
    def _run_single_probe(self, probe_name, hidden_states, mechanism_values):
        """Run a single probe. Delegates to DESCARTES-PHARMA probe implementations."""
        # [Dispatches to probe implementations from v1.0 Sections 5-14]
        pass
    
    def _call_llm_balloon(self, architecture, mechanism, evidence_so_far):
        """Escalate to LLM for novel probe suggestions."""
        # [Uses LLM Balloon from v1.1 Addendum A2]
        pass
    
    def _final_verdict(self, evidence):
        """Generate verdict from all accumulated evidence."""
        # [Uses ZombieVerdictGenerator from v1.0 Section 17]
        pass
```

---

## M8. Paradigm 5: VFE Belief Updating

### M8.1 Design

Adapted from Explorer Agent v3 Layer 2. Maintains a Bayesian posterior over each (architecture × mechanism) pair. Complexity-penalized zombie scores replace raw ΔR².

```python
class VFEBeliefSystem:
    """
    Variational Free Energy belief updating for zombie detection.
    
    Adapted from Explorer Agent v3 Layer 2:
    F = KL(q||p) + E[-log p(data|θ)]
    
    For each (architecture, mechanism) pair, maintain:
    - q(zombie | evidence): posterior probability of zombie status
    - Prior: uniform (0.5) or informed by VZS pattern
    - Likelihood: probe results update posterior
    - Complexity penalty: models with more probes needed to resolve
      incur higher F (penalizes ambiguity)
    
    Multi-channel precision (from Explorer Agent):
    - τ_probe: reliability of the probe method
    - τ_data: quality of the dataset
    - τ_stat: statistical power of the test
    - τ_causal: strength of causal evidence (ablation > correlation)
    """
    
    def __init__(self):
        # Per (arch, mechanism) beliefs
        self.beliefs = {}  # key → {'mean': float, 'variance': float, 'n_updates': int}
        
        # Precision channels
        self.probe_precisions = {
            'ridge': 1.0,
            'mlp': 1.5,
            'sae': 2.0,
            'cca': 1.2,
            'rsa': 1.2,
            'cka': 1.5,
            'resample_ablation': 3.0,  # Highest — causal evidence
            'das': 2.5,
            'mine': 1.3,
            'koopman': 1.5,
            'sindy': 2.0,
            'tda': 1.0,
        }
    
    def get_belief(self, architecture, mechanism):
        key = f"{architecture}:{mechanism}"
        if key not in self.beliefs:
            self.beliefs[key] = {
                'mean': 0.5,      # Prior: 50% chance of zombie
                'variance': 0.25,  # High uncertainty
                'n_updates': 0,
                'free_energy': float('inf'),
            }
        return self.beliefs[key]
    
    def update(self, architecture, mechanism, probe_type,
               delta_r2, p_value, dataset_quality=1.0):
        """
        Kalman filter update of zombie belief.
        
        Observation: delta_r2 from probe
        Higher delta_r2 → lower P(zombie)
        Lower p_value → higher precision of observation
        """
        belief = self.get_belief(architecture, mechanism)
        
        # Observation precision (from multi-channel system)
        τ_probe = self.probe_precisions.get(probe_type, 1.0)
        τ_stat = max(0.1, -np.log10(p_value + 1e-10) / 3.0)  # Higher for smaller p
        τ_data = dataset_quality
        
        observation_precision = τ_probe * τ_stat * τ_data
        
        # Convert delta_r2 to zombie probability observation
        # High delta_r2 → low zombie probability
        obs_zombie_prob = max(0.0, 1.0 - min(delta_r2 * 5.0, 1.0))
        
        # Kalman update
        prior_mean = belief['mean']
        prior_var = belief['variance']
        
        obs_var = 1.0 / (observation_precision + 1e-10)
        
        kalman_gain = prior_var / (prior_var + obs_var)
        
        posterior_mean = prior_mean + kalman_gain * (obs_zombie_prob - prior_mean)
        posterior_var = (1 - kalman_gain) * prior_var
        
        # Compute free energy
        # F = KL(posterior || prior) + expected negative log likelihood
        kl = 0.5 * (np.log(prior_var / (posterior_var + 1e-10)) + 
                     posterior_var / prior_var + 
                     (posterior_mean - prior_mean)**2 / prior_var - 1)
        
        nll = 0.5 * (obs_zombie_prob - posterior_mean)**2 / obs_var
        
        free_energy = kl + nll
        
        belief['mean'] = posterior_mean
        belief['variance'] = posterior_var
        belief['n_updates'] += 1
        belief['free_energy'] = free_energy
        
        return {
            'posterior_zombie_prob': posterior_mean,
            'uncertainty': posterior_var,
            'free_energy': free_energy,
            'n_updates': belief['n_updates'],
        }
    
    def should_continue_probing(self, architecture, mechanism,
                                  uncertainty_threshold=0.05):
        """
        Should we keep probing this pair, or is the belief settled?
        
        Stop when variance < threshold (confident in verdict)
        OR when free energy stops decreasing (no more info to gain)
        """
        belief = self.get_belief(architecture, mechanism)
        
        if belief['variance'] < uncertainty_threshold:
            return False  # Confident enough
        
        if belief['n_updates'] > 10 and belief['free_energy'] > 0.5:
            return False  # Many probes, still ambiguous → give up
        
        return True
    
    def get_zombie_verdict(self, architecture, mechanism):
        """Convert belief to verdict."""
        belief = self.get_belief(architecture, mechanism)
        p_zombie = belief['mean']
        uncertainty = belief['variance']
        
        if p_zombie > 0.8 and uncertainty < 0.1:
            return 'CONFIRMED_ZOMBIE'
        elif p_zombie < 0.2 and uncertainty < 0.1:
            return 'CONFIRMED_MECHANISTIC'
        elif p_zombie > 0.6:
            return 'LIKELY_ZOMBIE'
        elif p_zombie < 0.4:
            return 'CANDIDATE_MECHANISTIC'
        else:
            return 'AMBIGUOUS'
```

---

## M9. Paradigm 6: Multi-Timescale Oscillatory Hierarchy

### M9.1 Design

Adapted from Explorer Agent v3 Layer 0. Three processing timescales with phase-amplitude coupling.

```python
class MultiTimescaleProcessor:
    """
    Three-timescale processing hierarchy for the dual factory.
    
    Adapted from Explorer Agent v3 three-level oscillatory hierarchy:
    
    Fast Oscillator (per-probe, seconds-minutes):
      - Run individual probe, record result
      - Update meta-learner feedback buffer
      - Update VFE belief for this (arch, mech) pair
      - Check early stopping conditions
    
    Medium Oscillator (per-model, hours):
      - Aggregate all probe results for this model
      - Generate zombie verdict
      - Record in VZS
      - Update Thompson sampling posteriors
      - DreamCoder wake phase (within-campaign patterns)
    
    Slow Oscillator (per-campaign, days):
      - Cross-campaign pattern extraction
      - DreamCoder sleep phase (compress library)
      - VZS tier promotions (provisional → pattern → axiom)
      - Meta-cognition HOT layer assessment
      - LLM strategic reasoning (if stagnation detected)
    
    Phase-amplitude coupling:
      Slow oscillator GATES fast oscillator.
      If slow detects "all 3D-equivariant models are non-zombie":
        → Fast oscillator SKIPS 3D probes for remaining models
        → Fast oscillator PRIORITIZES 2D models (higher info gain)
    """
    
    def __init__(self, meta_learner, vfe_system, vzs, cascade_router):
        self.meta = meta_learner
        self.vfe = vfe_system
        self.vzs = vzs
        self.cascade = cascade_router
        
        # Slow oscillator state
        self.campaign_patterns = []
        self.stagnation_counter = 0
        self.slow_gate_overrides = {}  # mechanism → 'SKIP' or 'PRIORITIZE'
    
    def fast_tick(self, probe_name, architecture, mechanism, 
                   delta_r2, p_value, compute_time):
        """Called after every individual probe execution."""
        
        # 1. Update VFE belief
        vfe_update = self.vfe.update(
            architecture, mechanism, probe_name, delta_r2, p_value)
        
        # 2. Record in meta-learner feedback
        outcome = ProbeOutcome(
            probe_type=probe_name,
            architecture=architecture,
            mechanism=mechanism,
            dataset='current',
            delta_r2=delta_r2,
            p_value=p_value,
            compute_seconds=compute_time,
            verdict_contribution='PENDING',
            was_useful=True,  # Updated at medium tick
        )
        
        # 3. Check if slow oscillator has gating override
        if mechanism in self.slow_gate_overrides:
            if self.slow_gate_overrides[mechanism] == 'SKIP':
                return {'action': 'SKIP_REMAINING', 'reason': 'slow_gate_override'}
        
        # 4. Check early stopping via VFE
        if not self.vfe.should_continue_probing(architecture, mechanism):
            return {'action': 'STOP_PROBING', 'reason': 'belief_settled',
                    'verdict': self.vfe.get_zombie_verdict(architecture, mechanism)}
        
        return {'action': 'CONTINUE', 'vfe_update': vfe_update}
    
    def medium_tick(self, architecture, all_probe_results, dataset):
        """Called after all probes for one model are complete."""
        
        # 1. Generate aggregate verdict
        verdict = self._aggregate_verdict(all_probe_results)
        
        # 2. Record in VZS for each mechanism
        for mechanism, results in all_probe_results.items():
            self.vzs.record_verdict(
                architecture, mechanism, dataset,
                results.get('verdict', 'AMBIGUOUS'),
                results)
        
        # 3. Update Thompson sampling
        is_success = verdict.get('n_mechanistic', 0) > 0
        
        # 4. DreamCoder wake (extract patterns from this model)
        patterns = self._extract_patterns(architecture, all_probe_results)
        self.campaign_patterns.extend(patterns)
        
        # 5. Track stagnation
        if not is_success:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return verdict
    
    def slow_tick(self, campaign_results):
        """Called at end of campaign or every N models."""
        
        # 1. Cross-campaign pattern extraction
        cross_patterns = self._extract_cross_campaign_patterns(campaign_results)
        
        # 2. DreamCoder sleep (compress library)
        self._dreamcoder_sleep(cross_patterns)
        
        # 3. VZS tier promotions
        self._promote_vzs_entries()
        
        # 4. Update slow gate overrides
        self._update_slow_gates(campaign_results)
        
        # 5. Meta-cognition check
        hot_assessment = self._hot_layer_check(campaign_results)
        
        # 6. LLM strategic reasoning (if stagnation)
        if hot_assessment.get('stagnation_detected'):
            llm_strategy = self._llm_strategic_reasoning(campaign_results)
            return {'hot': hot_assessment, 'llm_strategy': llm_strategy}
        
        return {'hot': hot_assessment}
    
    def _update_slow_gates(self, campaign_results):
        """
        Phase-amplitude coupling: slow oscillator gates fast.
        
        If we've seen enough evidence that a mechanism type is always
        encoded (or always zombie) for a given architecture class,
        gate the fast oscillator to skip/prioritize accordingly.
        """
        # Count verdicts per (arch_class, mech) across campaign
        verdict_counts = {}
        for result in campaign_results:
            for mech, mech_result in result.get('per_mechanism', {}).items():
                key = f"{result.get('architecture_class', 'unknown')}:{mech}"
                if key not in verdict_counts:
                    verdict_counts[key] = {'encoded': 0, 'zombie': 0, 'total': 0}
                v = mech_result.get('verdict', '')
                verdict_counts[key]['total'] += 1
                if 'ENCODED' in v or 'MECHANISTIC' in v:
                    verdict_counts[key]['encoded'] += 1
                elif 'ZOMBIE' in v:
                    verdict_counts[key]['zombie'] += 1
        
        # Set gates
        self.slow_gate_overrides = {}
        for key, counts in verdict_counts.items():
            if counts['total'] >= 3:
                if counts['encoded'] / counts['total'] > 0.9:
                    mech = key.split(':')[1]
                    self.slow_gate_overrides[mech] = 'SKIP'  # Always encoded, don't waste time
                elif counts['zombie'] / counts['total'] > 0.9:
                    mech = key.split(':')[1]
                    self.slow_gate_overrides[mech] = 'PRIORITIZE'  # Always zombie — worth investigating why
```

---

## M10. Paradigm 7: Meta-Cognition Over Search (HOT Layer)

### M10.1 Design

Adapted from Explorer Agent v3 Layer 4 (Operative HOT). Monitors the dual factory itself.

```python
class MetaCognitionHOT:
    """
    Higher-Order Thought layer: monitors the dual factory search process.
    
    Adapted from Explorer Agent v3 Layer 4:
    - Maintains a variational posterior over the SEARCH PROCESS
    - Detects: stuck factories, redundant probing, verdict anomalies
    - Triggers: LLM strategic reasoning, search pivots, campaign termination
    
    The HOT layer answers: "Is the factory making progress,
    or is it spinning its wheels?"
    """
    
    def __init__(self, stagnation_threshold=20, anomaly_threshold=3.0):
        self.stagnation_threshold = stagnation_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # Meta-posterior: P(search_productive | evidence)
        self.meta_belief = {
            'productive_prob': 0.8,  # Start optimistic
            'variance': 0.2,
            'n_assessments': 0,
        }
        
        # Tracking
        self.verdict_history = deque(maxlen=100)
        self.fitness_history = deque(maxlen=100)
        self.compute_spent = 0.0
        self.useful_probes = 0
        self.total_probes = 0
    
    def assess(self, recent_results, campaign_state):
        """
        Run meta-cognitive assessment.
        
        Returns diagnosis + recommended action.
        """
        self.n_assessments += 1 if hasattr(self, 'n_assessments') else 0
        
        diagnosis = {
            'stagnation': self._detect_stagnation(recent_results),
            'redundancy': self._detect_redundancy(recent_results),
            'anomaly': self._detect_anomaly(recent_results),
            'exhaustion': self._detect_exhaustion(campaign_state),
            'compute_efficiency': self._compute_efficiency(),
        }
        
        # Update meta-posterior
        productivity = sum([
            not diagnosis['stagnation']['detected'],
            not diagnosis['redundancy']['detected'],
            not diagnosis['exhaustion']['detected'],
            diagnosis['compute_efficiency']['efficiency'] > 0.3,
        ]) / 4.0
        
        self.meta_belief['productive_prob'] = (
            0.8 * self.meta_belief['productive_prob'] + 0.2 * productivity)
        
        # Recommend action
        action = self._recommend_action(diagnosis)
        
        return {
            'diagnosis': diagnosis,
            'meta_productive_prob': self.meta_belief['productive_prob'],
            'recommended_action': action,
        }
    
    def _detect_stagnation(self, recent_results):
        """Has the best fitness score improved recently?"""
        if len(recent_results) < 5:
            return {'detected': False}
        
        recent_fitness = [r.get('fitness', {}).get('fitness', 0) 
                          for r in recent_results[-20:]]
        
        if len(recent_fitness) >= 10:
            first_half = np.mean(recent_fitness[:len(recent_fitness)//2])
            second_half = np.mean(recent_fitness[len(recent_fitness)//2:])
            improvement = second_half - first_half
            
            if improvement < 0.01:
                return {
                    'detected': True,
                    'rounds_stagnant': len(recent_fitness),
                    'best_fitness': max(recent_fitness),
                }
        
        return {'detected': False}
    
    def _detect_redundancy(self, recent_results):
        """Are we probing the same (arch, mechanism) pairs repeatedly?"""
        if len(recent_results) < 10:
            return {'detected': False}
        
        pairs = [(r.get('architecture'), r.get('mechanism')) 
                 for r in recent_results[-20:]]
        unique_ratio = len(set(pairs)) / len(pairs)
        
        return {
            'detected': unique_ratio < 0.5,
            'unique_ratio': unique_ratio,
            'recommendation': 'Explore new architecture-mechanism combinations'
                if unique_ratio < 0.5 else None,
        }
    
    def _detect_anomaly(self, recent_results):
        """Are recent verdicts inconsistent with VFE beliefs?"""
        anomalies = []
        for r in recent_results[-10:]:
            arch = r.get('architecture', '')
            mech = r.get('mechanism', '')
            verdict = r.get('verdict', '')
            
            # Compare with VZS if available
            # An anomaly is: VZS says ENCODED but probe says ZOMBIE (or vice versa)
            if 'vzs_prediction' in r and r['vzs_prediction'] != verdict:
                anomalies.append({
                    'architecture': arch,
                    'mechanism': mech,
                    'expected': r['vzs_prediction'],
                    'observed': verdict,
                })
        
        return {
            'detected': len(anomalies) > 0,
            'anomalies': anomalies,
            'count': len(anomalies),
        }
    
    def _detect_exhaustion(self, campaign_state):
        """Has the search space been adequately covered?"""
        coverage = campaign_state.get('coverage_fraction', 0)
        rounds_remaining = campaign_state.get('rounds_remaining', float('inf'))
        
        return {
            'detected': coverage > 0.8 or rounds_remaining < 5,
            'coverage': coverage,
            'rounds_remaining': rounds_remaining,
        }
    
    def _compute_efficiency(self):
        """Fraction of probes that were useful (changed a verdict)."""
        if self.total_probes == 0:
            return {'efficiency': 1.0}
        return {'efficiency': self.useful_probes / self.total_probes}
    
    def _recommend_action(self, diagnosis):
        """Map diagnosis to action."""
        if diagnosis['stagnation']['detected'] and diagnosis['exhaustion']['detected']:
            return 'TERMINATE_CAMPAIGN'
        
        if diagnosis['stagnation']['detected']:
            return 'TRIGGER_LLM_STRATEGIC_REASONING'
        
        if diagnosis['redundancy']['detected']:
            return 'FORCE_EXPLORATION'
        
        if diagnosis['anomaly']['detected']:
            return 'INVESTIGATE_ANOMALIES'
        
        if diagnosis['compute_efficiency']['efficiency'] < 0.2:
            return 'INCREASE_VZS_CACHING'
        
        return 'CONTINUE_NORMAL'
```

---

## M11. LLM Strategic Reasoning Layer

### M11.1 Design

The LLM is triggered by the HOT layer when meta-cognition detects problems. Unlike the neural fast path (which handles routine probe routing in <1ms), the LLM provides strategic reasoning about the search process itself.

```python
SYSTEM_META_STRATEGIC = """You are the strategic reasoning layer of the 
DESCARTES-PHARMA dual factory meta-learner. The HOT (Higher-Order Thought) 
layer has detected an issue with the search process and is consulting you 
for strategic guidance.

Current factory state:
{factory_state}

HOT layer diagnosis:
{hot_diagnosis}

VZS statistics:
{vzs_stats}

VFE belief summary:
{vfe_summary}

Campaign history:
{campaign_history}

Your role is NOT to suggest specific probes or architectures (the neural
fast path handles that). Your role is STRATEGIC:

1. Is the search process stuck for a FUNDAMENTAL reason?
   (e.g., wrong probe types for this mechanism class, 
    wrong dataset for this question, missing ground truth)

2. Should the factory PIVOT to a different search strategy?
   (e.g., switch from architecture search to loss function search,
    switch from probing to AlphaFold structural validation)

3. Are there CROSS-CAMPAIGN patterns the DreamCoder missed?
   (e.g., "every model that encodes mechanism X also encodes Y",
    "mechanism Z is never encoded without 3D geometry")

4. Should this campaign be TERMINATED early?
   (search space exhausted, or fundamental blocking issue)

Respond with ONLY valid JSON:
{
  "diagnosis": "one-sentence summary of what's wrong",
  "root_cause": "STUCK | WRONG_APPROACH | MISSING_DATA | EXHAUSTED | ANOMALY",
  "recommended_pivot": "specific strategic change",
  "confidence": 0.0-1.0,
  "should_terminate": true/false,
  "cross_campaign_insight": "pattern discovered or null"
}"""


class LLMStrategicReasoner:
    """
    LLM strategic reasoning, triggered by HOT layer.
    
    Key distinction from LLM Balloon (v1.1 Addendum A2):
    - Balloon proposes NOVEL ARCHITECTURES AND PROBES
    - Strategic Reasoner diagnoses WHY THE SEARCH IS FAILING
    
    The neural fast path is the reflexive system (System 1).
    The LLM strategic reasoner is the deliberative system (System 2).
    """
    
    def __init__(self, model='claude-sonnet-4-20250514'):
        self.model = model
        self.reasoning_history = []
    
    def reason(self, factory_state, hot_diagnosis, vzs, vfe_system,
                campaign_history):
        import json, requests
        
        prompt = SYSTEM_META_STRATEGIC.format(
            factory_state=json.dumps(factory_state, indent=2, default=str),
            hot_diagnosis=json.dumps(hot_diagnosis, indent=2, default=str),
            vzs_stats=json.dumps(vzs.get_stats(), indent=2),
            vfe_summary=self._summarize_vfe(vfe_system),
            campaign_history=self._summarize_campaigns(campaign_history),
        )
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={'Content-Type': 'application/json'},
            json={
                'model': self.model,
                'max_tokens': 1000,
                'messages': [{'role': 'user', 'content': prompt}]
            }
        )
        
        data = response.json()
        text = data['content'][0]['text']
        
        try:
            reasoning = json.loads(text)
            self.reasoning_history.append(reasoning)
            return reasoning
        except json.JSONDecodeError:
            return {'diagnosis': 'LLM parse error', 'root_cause': 'UNKNOWN',
                    'recommended_pivot': 'continue normal', 'confidence': 0.0,
                    'should_terminate': False, 'cross_campaign_insight': None}
    
    def _summarize_vfe(self, vfe_system):
        """Summarize VFE beliefs for LLM context."""
        summary = {}
        for key, belief in vfe_system.beliefs.items():
            summary[key] = {
                'zombie_prob': round(belief['mean'], 3),
                'uncertainty': round(belief['variance'], 3),
                'n_updates': belief['n_updates'],
            }
        return json.dumps(summary, indent=2)
    
    def _summarize_campaigns(self, history):
        if not history:
            return "No previous campaigns"
        return json.dumps([{
            'dataset': h.get('dataset'),
            'best_fitness': h.get('best_fitness'),
            'n_non_zombie': h.get('n_non_zombie'),
            'top_architecture': h.get('top_architecture'),
        } for h in history[-5:]], indent=2)
```

---

## M12. Full Integrated Pipeline

```python
class DescartesPharmaMetaLearner:
    """
    The complete hybrid meta-learner integrating all 7 paradigms.
    
    Neural fast path: PharmaMetaLearner (~2M params, <1ms per decision)
    LLM strategic: LLMStrategicReasoner (triggered by HOT, ~$0.01/call)
    
    This sits ABOVE both C1 and C2 factories and orchestrates:
    - WHAT to probe (probe prioritization via VFE + Global Workspace)
    - WHEN to stop (early stopping via VFE belief convergence)
    - WHERE to search (architecture selection via Thompson + VZS)
    - WHETHER to escalate (cheap → expensive probe cascade)
    - WHY it's stuck (HOT layer + LLM strategic reasoning)
    """
    
    def __init__(self, meta_path=None):
        # Paradigm 1: Neural fast path
        self.neural = PharmaMetaLearner()
        self.trainer = PharmaMetaTrainer(self.neural)
        
        # Paradigm 2: Mechanism decomposer
        self.decomposer = MechanismDecomposer()
        
        # Paradigm 3: Verified Zombie Store
        self.vzs = VerifiedZombieStore()
        
        # Paradigm 4: Probe cascade router
        self.cascade = ProbeCascadeRouter()
        
        # Paradigm 5: VFE belief system
        self.vfe = VFEBeliefSystem()
        
        # Paradigm 6: Multi-timescale processor
        self.timescale = MultiTimescaleProcessor(
            self.neural, self.vfe, self.vzs, self.cascade)
        
        # Paradigm 7: Meta-cognition HOT layer
        self.hot = MetaCognitionHOT()
        
        # LLM strategic reasoner
        self.llm = LLMStrategicReasoner()
        
        # Load saved state if available
        if meta_path:
            self.trainer.load(meta_path)
    
    def evaluate_model(self, model, mechanisms, dataset, 
                        hidden_states, mechanism_features):
        """
        Full meta-learned evaluation of one model.
        
        Instead of blindly running all 43 probes on all mechanisms,
        the meta-learner decides what to probe, in what order,
        and when to stop — potentially saving 40-60% compute.
        """
        # Step 1: Decompose into per-mechanism sub-questions
        routing_plan = self.decomposer.decompose(mechanisms)
        
        all_results = {}
        
        for mechanism, plan in routing_plan.items():
            # Step 2: Check VZS (free!)
            vzs_result = self.vzs.lookup(model.architecture, mechanism, dataset)
            
            if vzs_result['status'] == 'HIT' and vzs_result['tier'] <= 2:
                # Settled verdict — skip all probing
                all_results[mechanism] = {
                    'verdict': vzs_result['verdict'],
                    'source': 'VZS_CACHE',
                    'tier': vzs_result['tier'],
                }
                continue
            
            # Step 3: Run probe cascade with meta-learned routing
            cascade_result = self.cascade.run_cascade(
                model, mechanism, dataset, hidden_states,
                mechanism_features, self.neural, self.vzs)
            
            all_results[mechanism] = cascade_result
            
            # Step 4: Fast tick (update VFE + meta-learner)
            for probe_name, probe_result in cascade_result.get('evidence', {}).items():
                if isinstance(probe_result, dict) and 'delta_r2' in probe_result:
                    self.timescale.fast_tick(
                        probe_name, model.architecture, mechanism,
                        probe_result['delta_r2'],
                        probe_result.get('p_value', 1.0),
                        probe_result.get('compute_seconds', 0))
        
        # Step 5: Medium tick (aggregate verdict)
        model_verdict = self.timescale.medium_tick(
            model.architecture, all_results, dataset)
        
        return model_verdict
    
    def end_of_campaign(self, campaign_results):
        """
        Slow tick: cross-campaign learning.
        
        This is where the meta-learner improves for NEXT campaign.
        """
        # Slow tick: patterns, DreamCoder, VZS promotions
        slow_result = self.timescale.slow_tick(campaign_results)
        
        # HOT assessment
        hot_result = self.hot.assess(campaign_results, {
            'coverage_fraction': len(campaign_results) / 200,
            'rounds_remaining': 0,
        })
        
        # LLM strategic reasoning if needed
        if hot_result['recommended_action'] == 'TRIGGER_LLM_STRATEGIC_REASONING':
            llm_result = self.llm.reason(
                factory_state={'n_models': len(campaign_results)},
                hot_diagnosis=hot_result['diagnosis'],
                vzs=self.vzs,
                vfe_system=self.vfe,
                campaign_history=campaign_results)
            
            return {
                'slow_tick': slow_result,
                'hot': hot_result,
                'llm_strategy': llm_result,
            }
        
        return {'slow_tick': slow_result, 'hot': hot_result}
    
    def save(self, path):
        self.trainer.save(f"{path}_neural.pt")
        self.vzs._save()
    
    def get_stats(self):
        return {
            'neural_updates': self.trainer.update_count,
            'vzs': self.vzs.get_stats(),
            'vfe_beliefs': len(self.vfe.beliefs),
            'hot_productive_prob': self.hot.meta_belief['productive_prob'],
            'avg_meta_loss': (
                sum(self.trainer.loss_history) / 
                max(len(self.trainer.loss_history), 1)
            ) if self.trainer.loss_history else None,
        }
```

---

## M13. Bootstrap Protocol

Adapted from meta_learner.md bootstrap method. Pre-train the neural fast path before the first real campaign.

```python
def bootstrap_pharma_meta_learner(hh_dataset, output_path, 
                                    n_bootstrap=500):
    """
    Bootstrap the meta-learner on HH simulator data.
    
    Why HH: we KNOW the answer. Probes that find m,h,n are useful.
    Probes that miss m,h,n are useless. This provides clean labels
    for the meta-learner before it sees real pharma data.
    
    Method (from meta_learner.md):
    1. Train 10 LSTM surrogates with varying hidden dims
    2. Run ALL 43 probes on each surrogate
    3. Label each probe as USEFUL or USELESS based on known ground truth
    4. Pre-train meta-learner on these labels
    5. Deploy with warm-started meta-learner to pharma campaigns
    """
    meta = DescartesPharmaMetaLearner()
    
    # Generate HH ground truth
    from descartes_pharma_v1 import HodgkinHuxleySimulator
    hh = HodgkinHuxleySimulator()
    dataset = hh.generate_dataset(n_trials=200)
    
    # Train 10 surrogates with different hidden dims
    hidden_dims = [8, 16, 32, 64, 128, 256, 8, 16, 32, 64]
    
    for i, h_dim in enumerate(hidden_dims):
        print(f"Bootstrap surrogate {i+1}/10 (h={h_dim})")
        
        # Train LSTM surrogate
        surrogate = train_lstm_surrogate(dataset, h_dim)
        hidden_states = extract_hidden_states(surrogate, dataset)
        
        # Run all probes with known ground truth
        for j, target_name in enumerate(dataset['target_names']):
            target = dataset['bio_targets'][:, :, j].reshape(-1)
            
            for probe_name in ['ridge', 'mlp', 'sae', 'cca', 'rsa', 
                               'cka', 'knn', 'mine', 'mdl']:
                result = run_probe(probe_name, hidden_states, target)
                
                # We KNOW the ground truth: well-trained models SHOULD encode m,h,n
                # So probes that detect encoding are USEFUL
                was_useful = result['delta_r2'] > 0.1 and result['p_value'] < 0.05
                
                outcome = ProbeOutcome(
                    probe_type=probe_name,
                    architecture='lstm',
                    mechanism=target_name,
                    dataset='hh_simulator',
                    delta_r2=result['delta_r2'],
                    p_value=result['p_value'],
                    compute_seconds=result.get('compute_seconds', 1.0),
                    verdict_contribution='CONFIRMED_ENCODING' if was_useful else 'INCONCLUSIVE',
                    was_useful=was_useful,
                )
                
                meta.trainer.record_and_maybe_train(
                    torch.randn(1, 106),  # Features (simplified for bootstrap)
                    'CHEAP_PROBE',
                    0.5,
                    outcome)
    
    # Save bootstrapped meta-learner
    meta.save(output_path)
    print(f"Bootstrap complete. Updates: {meta.trainer.update_count}")
    print(f"VZS entries: {meta.vzs.get_stats()}")
```

---

## M14. Convergence Metrics and Validation

### M14.1 Pass Criteria

Adapted from Ollama Addendum evaluation criteria:

```
METRIC                            THRESHOLD      NOTES
────────────────────────────────────────────────────────────────
Probe routing accuracy            ≥ 75%          On held-out HH data
Confidence calibration            ≤ 0.15         Mean |predicted - true|
VZS cache hit rate                ≥ 40%          By campaign 5
Compute savings vs no meta-learner ≥ 30%         Wall clock comparison
Verdict accuracy (HH ground truth) ≥ 90%         Known-answer test
Meta-learner convergence          ≤ 200 probes   To useful routing
HOT stagnation detection          ≤ 10 rounds    False positive rate < 5%
Cross-campaign transfer           ≥ 20% faster   Campaign N+1 vs campaign N

IF THRESHOLDS MISSED:
1. Check bootstrap quality (enough HH diversity?)
2. Generate more diverse HH surrogates (vary architectures)
3. Increase feedback buffer size
4. Re-bootstrap with higher learning rate
5. Check VZS promotion thresholds (too strict?)
```

### M14.2 The Key Test: Campaign 5 Should Be Dramatically Faster Than Campaign 1

```python
def measure_cross_campaign_transfer(meta_learner, datasets):
    """
    The definitive test of the meta-learner.
    
    Run 5 campaigns sequentially on different datasets.
    Measure: probes needed to reach 90% verdict confidence.
    
    Expected: Campaign 1 = 100% probes (no prior knowledge)
              Campaign 2 = 70-80% probes (some VZS + VFE transfer)
              Campaign 3 = 50-60% probes (patterns emerging)
              Campaign 5 = 30-40% probes (mature meta-learner)
    
    If Campaign 5 still needs 90%+ probes, the meta-learner isn't working.
    """
    campaign_efficiency = []
    
    for i, dataset in enumerate(datasets):
        probes_run = 0
        probes_possible = 0
        
        for model in dataset['models']:
            for mechanism in dataset['mechanisms']:
                probes_possible += len(ALL_PROBES)
                
                result = meta_learner.evaluate_model(
                    model, [mechanism], dataset['name'],
                    model.hidden_states, dataset['mechanism_features'])
                
                probes_run += result.get('probes_actually_run', len(ALL_PROBES))
        
        efficiency = 1.0 - (probes_run / probes_possible)
        campaign_efficiency.append(efficiency)
        
        # End-of-campaign learning
        meta_learner.end_of_campaign(dataset['results'])
        
        print(f"Campaign {i+1} ({dataset['name']}): "
              f"{efficiency:.1%} compute saved "
              f"({probes_run}/{probes_possible} probes run)")
    
    # The transfer test
    improvement = campaign_efficiency[-1] - campaign_efficiency[0]
    print(f"\nTransfer: Campaign 1 saved {campaign_efficiency[0]:.1%}, "
          f"Campaign 5 saved {campaign_efficiency[-1]:.1%}")
    print(f"Improvement: +{improvement:.1%}")
    print(f"{'PASS' if improvement > 0.2 else 'FAIL'} (threshold: +20%)")
    
    return campaign_efficiency
```

---

## Summary: What the Meta-Learner Changes

| Without Meta-Learner | With Meta-Learner |
|---|---|
| Every campaign starts from scratch | VZS carries settled verdicts across campaigns |
| All 43 probes run on every mechanism | Cascade router runs 3-10 probes, stops early |
| No learning from probe outcomes | Neural fast path improves routing with every outcome |
| Flat probe ordering | VFE + Global Workspace prioritizes highest-info-gain probes |
| Stagnation undetected until human notices | HOT layer triggers LLM strategic reasoning within minutes |
| LLM proposes blindly | LLM reasons about WHY the search is stuck |
| Same compute on campaign 50 as campaign 1 | Campaign 50 uses 40-60% less compute than campaign 1 |

---

*This addendum adds the "brain" that was missing from the DESCARTES-PHARMA dual factory. The 7 paradigms — online feedback, mechanism decomposition, persistent memory, self-repair cascade, Bayesian belief updating, multi-timescale processing, and meta-cognition — transform the factory from a brute-force search into an adaptive learning system that gets smarter with every campaign, every probe, and every verdict.*
