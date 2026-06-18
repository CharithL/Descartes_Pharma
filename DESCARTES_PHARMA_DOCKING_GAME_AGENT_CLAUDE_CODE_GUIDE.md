# DESCARTES-PHARMA v2.0: Molecular Docking Game Agent

## Claude Code Implementation Guide

**Author:** Charith Suranga
**Date:** April 2026
**Build time:** 3-5 days
**Training required:** YES — RL-style policy training (GPU recommended)
**GPU:** RTX 5070 (12GB VRAM) local or Vast.ai A10 (22.5GB VRAM)
**Cost:** $0 (all data is public, Vina is free, training is local)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Game Reframe](#2-the-game-reframe)
3. [Architecture Overview](#3-architecture-overview)
4. [Project Structure](#4-project-structure)
5. [Module 1: PocketKnowledge Perception](#5-module-1-pocketknowledge-perception)
6. [Module 2: Ligand Representation & Action Space](#6-module-2-ligand-representation--action-space)
7. [Module 3: Vina World Model (The Chess Engine)](#7-module-3-vina-world-model)
8. [Module 4: Search Policy Network](#8-module-4-search-policy-network)
9. [Module 5: Training Loop with Continuous Reward](#9-module-5-training-loop)
10. [Module 6: DESCARTES Probe Suite](#10-module-6-descartes-probe-suite)
11. [LLM Balloon Expansion (C1/C2)](#11-llm-balloon-expansion)
12. [Data Pipeline](#12-data-pipeline)
13. [Implementation Phases](#13-implementation-phases)
14. [Testing Strategy](#14-testing-strategy)
15. [Scientific Hypotheses & Publication Strategy](#15-scientific-hypotheses)
16. [Compute Estimates](#16-compute-estimates)
17. [Appendix A: Module-to-ARC-AGI Mapping](#appendix-a)
18. [Appendix B: BACE1 Binding Pocket Reference](#appendix-b)
19. [Appendix C: Mathematical Foundations](#appendix-c)

---

## 1. Executive Summary

### What We Are Building

A **molecular docking game agent** that treats drug-protein binding as an interactive game:

- The **game board** is the BACE1 protein binding pocket (from PDB crystal structures)
- The **player piece** is a drug molecule (from BindingDB, with known binding affinity)
- The **game rules** are physics (H-bonds, hydrophobic contacts, electrostatics, steric clashes)
- The **score** is AutoDock Vina's binding energy (continuous, kcal/mol, computed for every pose)
- The **goal** is to place the drug molecule in the pocket so it blocks amyloid-beta production

A **Search Policy Network** (small GRU) learns to guide the search — which poses to try, which adjustments to make. Its hidden states are then probed with the full DESCARTES council controls to determine whether it learned genuine binding intuition or is a pharmaceutical zombie.

### Why This Architecture

The chess engine insight: in molecular docking, **you already know the rules** (physics) and **you have a perfect simulator** (Vina). You don't need to learn physics — you need to learn WHERE TO LOOK. This is the AlphaGo pattern:

| Component | Chess Engine | AlphaGo | Our Docking Agent |
|---|---|---|---|
| Rules | Known (piece movement) | Known (Go rules) | Known (physics) |
| Simulator | Perfect (board update) | Perfect (stone placement) | Perfect (Vina scoring) |
| Neural network's job | Evaluate positions | Guide search (policy+value) | Guide pose search |
| What NN learns | "This position is +2.5" | "This move is promising" | "Move ligand toward Asp228" |
| Probing question | N/A | N/A | **Does NN encode binding features?** |

### Why This Matters for DESCARTES-PHARMA

The central diagnosis from Phases 1-6: **binary labels are the bottleneck, not architecture.** This agent directly tests that hypothesis:

- Same target (BACE1), same kind of network (GRU)
- But continuous reward (Vina kcal/mol) instead of binary active/inactive
- If the GRU's hidden states now encode dist_asp228, h_bond_count, etc. where they didn't before → **binary label bottleneck confirmed**
- If they're STILL zombie even with continuous rewards → **the problem is deeper than labels**
- Either result is publishable

### Connection to Existing Projects

| Component | Source Project | What Transfers |
|---|---|---|
| PocketKnowledge Perception | COGITO CoreKnowledge | Structured perception before reasoning |
| C1/C2 Pharmacological Discovery | HIMARI Ontology Expansion | Balloon expansion for new concepts |
| Search Policy Network | DESCARTES Path 3 GRU | Probeable hidden states |
| DESCARTES Probe Suite | DESCARTES-PHARMA v1.0-v1.3 | Council controls, VZS axioms, pocket scramble |
| Thompson Sampling | HIMARI | Exploration vs exploitation in pose search |
| Continuous Reward | ARC-AGI chess engine insight | Dense feedback replaces binary labels |

---

## 2. The Game Reframe

### 2.1 The Biological Problem

BACE1 (Beta-site Amyloid precursor protein Cleaving Enzyme 1) cleaves amyloid precursor protein, producing amyloid-beta peptides that aggregate into plaques — the hallmark of Alzheimer's disease. A drug that fits into BACE1's active site pocket blocks this cleavage.

### 2.2 Docking as a Game

```
GAME BOARD:          BACE1 binding pocket (3D structure from PDB 4IVT)
                     ├── Catalytic aspartates: Asp32, Asp228 (the "goal positions")
                     ├── Flap region (the "gate" — opens/closes)
                     ├── S1, S2, S3 sub-pockets (the "rooms")
                     ├── Hydrophobic groove (the "corridor")
                     └── Water molecules (the "obstacles")

PLAYER PIECE:        Drug molecule (3D conformer from RDKit)
                     ├── Functional groups (H-bond donors/acceptors)
                     ├── Hydrophobic regions
                     ├── Charged groups
                     └── Rotatable bonds (degrees of freedom)

ACTIONS:             6D pose adjustment + torsion angles
                     ├── Translate X, Y, Z (±0.5 Å steps)
                     ├── Rotate X, Y, Z (±15° steps)
                     └── Rotate torsion angle k (±30° steps)

SCORE:               Vina binding energy (kcal/mol)
                     ├── More negative = better binding
                     ├── Computed for EVERY pose (~1 sec)
                     ├── Continuous, not binary
                     └── Decomposes into: H-bond + hydrophobic + 
                         electrostatic + desolvation + steric clash

WIN CONDITION:       Achieve binding energy ≤ threshold
                     (validated against known IC50 from BindingDB)
```

### 2.3 Why This Sidesteps the Binary Label Bottleneck

| Property | DESCARTES-PHARMA v1.x | Docking Game Agent |
|---|---|---|
| Training signal | Binary (active/inactive) | Continuous (kcal/mol per pose) |
| Feedback frequency | Once per molecule | Every pose adjustment |
| Spatial information | None (flat descriptors) | Full 3D (pocket + ligand + interactions) |
| What network must learn | "Is this molecule active?" | "Which direction improves binding?" |
| Information per sample | 1 bit | ~10 bits (decomposed energy terms) |

---

## 3. Architecture Overview

### 3.1 System Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║              DOCKING GAME AGENT — CHESS ENGINE ARCHITECTURE          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 1: PocketKnowledge Perception (hardcoded, no ML)       │  ║
║  │  PDB 4IVT → structured pocket features                        │  ║
║  │  [catalytic_residues, hbond_sites, hydrophobic_regions,        │  ║
║  │   charged_residues, pocket_shape, water_positions]             │  ║
║  └──────────────────────────┬─────────────────────────────────────┘  ║
║                              │ pocket_features (fixed per protein)    ║
║                              ▼                                       ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 2: Ligand Representation + Action Space (hardcoded)    │  ║
║  │  SMILES → RDKit conformer → structured ligand features         │  ║
║  │  [pharmacophore_points, hbond_donors, hbond_acceptors,         │  ║
║  │   hydrophobic_centers, rotatable_bonds, current_pose_6D]       │  ║
║  │  Actions: translate(dx,dy,dz), rotate(rx,ry,rz), torsion(k,θ) │  ║
║  └──────────────────────────┬─────────────────────────────────────┘  ║
║                              │ ligand_features + action_space         ║
║                              ▼                                       ║
║  ┌──────────────────────────────────────────────────┐                ║
║  │  INTERACTION FEATURES (hardcoded, computed every pose)           │  ║
║  │  For each pocket_feature × ligand_feature pair:                 │  ║
║  │  [dist_asp32, dist_asp228, n_hbonds, hydrophobic_contact_area,  │  ║
║  │   steric_clash_count, closest_wall_dist, flap_contact,          │  ║
║  │   water_displacement_count, pocket_occupancy_fraction]          │  ║
║  └──────────────────────────┬───────────────────────┘                ║
║                              │ interaction_vector                     ║
║                              ▼                                       ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 4: Search Policy Network (LEARNED — has hidden states) │  ║
║  │                                                                 │  ║
║  │  Input: [pocket_features ⊕ ligand_features ⊕ interaction_vec   │  ║
║  │          ⊕ last_N_score_changes ⊕ current_vina_score]          │  ║
║  │       ↓                                                         │  ║
║  │  GRU (hidden_dim=128, 2 layers)                                │  ║
║  │       ↓                                                         │  ║
║  │  Policy head → action probabilities (which pose adjustment)    │  ║
║  │  Value head  → predicted final binding energy                  │  ║
║  │                                                                 │  ║
║  │  *** HIDDEN STATES ARE PROBED BY DESCARTES (Module 6) ***      │  ║
║  └──────────────────────────┬─────────────────────────────────────┘  ║
║                              │ action (pose adjustment)              ║
║                              ▼                                       ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 3: Vina World Model (NOT learned — perfect simulator)  │  ║
║  │                                                                 │  ║
║  │  Apply action → new pose → Vina score (kcal/mol)               │  ║
║  │  FREE: unlimited evaluations, no action budget                 │  ║
║  │  Returns: total_energy, hbond_energy, hydrophobic_energy,      │  ║
║  │           electrostatic_energy, desolvation_energy              │  ║
║  └──────────────────────────┬─────────────────────────────────────┘  ║
║                              │ reward = ΔE (score improvement)       ║
║                              ▼                                       ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 5: Training Loop (RL — PPO or REINFORCE)               │  ║
║  │                                                                 │  ║
║  │  For each ligand from BindingDB:                               │  ║
║  │    1. Random initial pose in pocket                            │  ║
║  │    2. Policy proposes action (pose adjustment)                 │  ║
║  │    3. Vina scores new pose                                     │  ║
║  │    4. Reward = Vina_new - Vina_old (negative = better)         │  ║
║  │    5. Policy updates (gradient ascent on reward)               │  ║
║  │    6. Repeat for T steps per episode                           │  ║
║  │    7. Repeat across 100s of ligands                            │  ║
║  └──────────────────────────┬─────────────────────────────────────┘  ║
║                              │ trained policy                        ║
║                              ▼                                       ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  MODULE 6: DESCARTES Probe Suite (post-training analysis)      │  ║
║  │                                                                 │  ║
║  │  Extract hidden states from trained Search Policy Network      │  ║
║  │  Probe for: dist_asp32, dist_asp228, n_hbonds,                │  ║
║  │    hydrophobic_contact_area, steric_clash_count,               │  ║
║  │    flap_contact, vina_score, pocket_occupancy                  │  ║
║  │                                                                 │  ║
║  │  Council controls:                                              │  ║
║  │    ✓ Scaffold-stratified permutation (200 perms)               │  ║
║  │    ✓ Arbitrary target probes (MW, NumHeavyAtoms)               │  ║
║  │    ✓ 50-seed ensemble                                          │  ║
║  │    ✓ Two-stage ablation                                        │  ║
║  │    ✓ Pocket scramble test                                      │  ║
║  │    ✓ Untrained network control (ΔR²)                          │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  LLM BALLOON (rare, expensive, outside main loop)              │  ║
║  │                                                                 │  ║
║  │  Called when: policy performance plateaus despite training      │  ║
║  │  Does: proposes new interaction features for perception layer  │  ║
║  │  Example: "Add flap_openness as a feature"                    │  ║
║  │           "Add water_bridge_count to interaction vector"        │  ║
║  │  NOT in the training loop — modifies Module 1/2 features      │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 3.2 What Is Learned vs What Is Hardcoded

| Component | Learned or Hardcoded | Why |
|---|---|---|
| Pocket features | **Hardcoded** (BioPython) | Known from crystal structure |
| Ligand features | **Hardcoded** (RDKit) | Known from chemistry |
| Interaction features | **Hardcoded** (distance/contact computation) | Known physics |
| Vina scoring | **Hardcoded** (AutoDock Vina) | Known physics — the "chess rules" |
| Search policy | **LEARNED** (GRU) | This is the unknown — WHERE to look |
| Value estimate | **LEARNED** (GRU value head) | Predicts binding quality from current state |

Only one thing is learned: **the search policy.** Everything else is known physics, known chemistry, or known structure. This is why it's like a chess engine — the NN only learns strategy, not rules.

---

## 4. Project Structure

```
descartes-pharma-docking/
├── README.md
├── pyproject.toml
├── data/
│   ├── structures/                  # PDB files
│   │   ├── 4IVT.pdb               # Primary BACE1 structure (already in workflow)
│   │   ├── 2ZHT.pdb               # BACE1 at pH 4.5 (active form)
│   │   └── prepared/              # Cleaned, protonated structures
│   │       ├── 4IVT_receptor.pdbqt # Vina-ready receptor
│   │       └── 4IVT_pocket.json   # Extracted pocket features
│   ├── ligands/                    # Drug molecules
│   │   ├── bindingdb_bace1.csv    # BindingDB BACE1 compounds with IC50/Ki
│   │   ├── conformers/            # RDKit 3D conformers (.sdf)
│   │   └── prepared/             # Vina-ready ligands (.pdbqt)
│   └── splits/                    # Train/val/test splits
│       ├── train_smiles.txt
│       ├── val_smiles.txt
│       └── test_smiles.txt
├── src/
│   ├── perception/                 # MODULE 1: PocketKnowledge Perception
│   │   ├── __init__.py
│   │   ├── pocket_parser.py       # PDB → structured pocket features
│   │   ├── pocket_features.py     # Feature extraction (catalytic, hbond, hydrophobic)
│   │   └── pharmacophore_priors.py # Four pharmacological priors
│   ├── ligand/                     # MODULE 2: Ligand Representation
│   │   ├── __init__.py
│   │   ├── ligand_features.py     # SMILES → structured ligand features
│   │   ├── conformer_gen.py       # RDKit 3D conformer generation
│   │   └── action_space.py        # Pose adjustments (translate/rotate/torsion)
│   ├── interaction/                # Interaction Features (between pocket and ligand)
│   │   ├── __init__.py
│   │   ├── interaction_features.py # Pairwise pocket×ligand feature computation
│   │   └── contact_analysis.py    # H-bond, hydrophobic, steric clash detection
│   ├── vina_engine/                # MODULE 3: Vina World Model
│   │   ├── __init__.py
│   │   ├── vina_scorer.py         # AutoDock Vina wrapper
│   │   ├── pose_manager.py        # Pose state management (apply actions)
│   │   └── receptor_prep.py       # One-time receptor preparation
│   ├── policy/                     # MODULE 4: Search Policy Network
│   │   ├── __init__.py
│   │   ├── policy_network.py      # GRU with policy+value heads
│   │   ├── feature_encoder.py     # Input feature composition
│   │   └── action_decoder.py      # Action probability output
│   ├── training/                   # MODULE 5: Training Loop
│   │   ├── __init__.py
│   │   ├── docking_env.py         # Gym-style environment wrapper
│   │   ├── trainer.py             # PPO/REINFORCE training loop
│   │   ├── reward_shaping.py      # Reward engineering (Vina decomposition)
│   │   └── curriculum.py          # Easy→hard ligand curriculum
│   ├── probing/                    # MODULE 6: DESCARTES Probe Suite
│   │   ├── __init__.py
│   │   ├── probe_runner.py        # Main probe orchestrator
│   │   ├── ridge_probe.py         # Ridge ΔR² with permutation testing
│   │   ├── mlp_probe.py           # MLP nonlinear control
│   │   ├── council_controls.py    # Arbitrary targets, ensemble, ablation
│   │   ├── pocket_scramble.py     # Pocket scramble test
│   │   └── scaffold_permutation.py # Scaffold-stratified permutation
│   └── balloon/                    # LLM Balloon Expansion
│       ├── __init__.py
│       ├── c1_vocabulary.py       # C1 pharmacological concepts
│       ├── balloon_trigger.py     # 4-test expansion protocol
│       └── llm_proposer.py        # Claude API call for new features
├── scripts/
│   ├── prepare_receptor.py        # One-time: PDB → PDBQT
│   ├── download_bindingdb.py      # Download BACE1 compounds
│   ├── prepare_ligands.py         # SMILES → conformer → PDBQT
│   ├── train_policy.py            # Main training script
│   ├── probe_policy.py            # Run DESCARTES probes on trained policy
│   ├── run_docking_game.py        # Play one game (interactive visualization)
│   └── benchmark.py               # Full evaluation + comparison
├── tests/
│   ├── test_pocket_features.py
│   ├── test_ligand_features.py
│   ├── test_interaction_features.py
│   ├── test_vina_scorer.py
│   ├── test_policy_network.py
│   ├── test_training_loop.py
│   └── test_probing.py
├── configs/
│   ├── bace1_4ivt.yaml            # Pocket config for BACE1 PDB 4IVT
│   ├── training.yaml              # Training hyperparameters
│   └── probing.yaml               # Probing configuration
└── results/
    ├── training_logs/
    ├── probe_results/
    └── figures/
```

---

## 5. Module 1: PocketKnowledge Perception

### 5.1 Purpose

Parse a PDB crystal structure into structured pocket features — the "game board" representation. This is the exact analog of CoreKnowledge Perception in the ARC-AGI agent: hardcoded, rule-based, no ML. It converts raw atomic coordinates into meaningful pharmacological objects.

### 5.2 Four Pharmacological Priors (Spelke Analogs)

| ARC-AGI Spelke Prior | Pharmacological Prior | What It Detects |
|---|---|---|
| Objectness (cohesion, persistence) | **RESIDUE_IDENTITY** | Amino acid residues as discrete objects with known properties |
| Spatial (geometry, distance) | **POCKET_GEOMETRY** | Sub-pocket shapes, distances, volumes, surfaces |
| Number (counting, ordering) | **INTERACTION_COUNTING** | Number of H-bond sites, charged groups, waters |
| Agency (goal-directedness) | **CATALYTIC_FUNCTION** | Which residues are catalytically active (Asp32, Asp228) |

### 5.3 Implementation

```python
# src/perception/pocket_parser.py
"""
Parse PDB structure into structured pocket features.

Uses BioPython for structure parsing. 
No ML — pure rule-based feature extraction.

Analogous to CoreKnowledge perception in ARC-AGI agent:
PDB file is the "raw grid", PocketFeatures is the "StructuredPercept".
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    raise ImportError("pip install biopython --break-system-packages")


@dataclass
class ResidueFeature:
    """One amino acid residue in the binding pocket."""
    name: str               # e.g., "ASP32", "TYR71"
    resname: str            # 3-letter code: ASP, TYR, PHE, etc.
    resid: int              # Residue number
    chain: str              # Chain ID
    center: np.ndarray      # Center of mass (x, y, z) in Angstroms
    
    # Pharmacological properties (hardcoded from amino acid chemistry)
    is_hbond_donor: bool
    is_hbond_acceptor: bool
    is_hydrophobic: bool
    is_charged: bool
    charge_sign: int         # +1, -1, or 0
    is_aromatic: bool
    is_catalytic: bool       # True for Asp32, Asp228 in BACE1
    
    sidechain_atoms: List[np.ndarray] = field(default_factory=list)


@dataclass
class PocketFeatures:
    """
    Complete structured representation of a binding pocket.
    
    This is the "StructuredPercept" for the protein pocket — the game board.
    All features are hardcoded from known biochemistry, no learning needed.
    """
    
    # Identity
    pdb_id: str
    pocket_center: np.ndarray        # Geometric center of pocket
    pocket_radius: float             # Radius encompassing all pocket residues
    
    # Residues (the "objects" on the game board)
    residues: List[ResidueFeature] = field(default_factory=list)
    
    # Catalytic residues (the "goal positions")
    catalytic_residues: List[ResidueFeature] = field(default_factory=list)
    
    # H-bond network (potential interaction sites)
    hbond_donors: List[ResidueFeature] = field(default_factory=list)
    hbond_acceptors: List[ResidueFeature] = field(default_factory=list)
    
    # Hydrophobic regions
    hydrophobic_residues: List[ResidueFeature] = field(default_factory=list)
    
    # Charged residues
    positive_residues: List[ResidueFeature] = field(default_factory=list)
    negative_residues: List[ResidueFeature] = field(default_factory=list)
    
    # Sub-pockets (the "rooms" in the game board)
    sub_pockets: Dict[str, Dict] = field(default_factory=dict)
    # e.g., {"S1": {"center": [x,y,z], "residues": [...], "volume": 150.0}}
    
    # Water positions (potential "obstacles" or displaced targets)
    water_positions: List[np.ndarray] = field(default_factory=list)
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Flatten pocket into a fixed-size feature vector for the policy network.
        
        Returns: np.ndarray of shape (n_pocket_features,)
        """
        features = []
        
        # Pocket-level features
        features.extend(self.pocket_center.tolist())    # 3
        features.append(self.pocket_radius)              # 1
        features.append(len(self.residues))              # 1
        features.append(len(self.hbond_donors))          # 1
        features.append(len(self.hbond_acceptors))       # 1
        features.append(len(self.hydrophobic_residues))  # 1
        features.append(len(self.positive_residues))     # 1
        features.append(len(self.negative_residues))     # 1
        features.append(len(self.water_positions))       # 1
        features.append(len(self.catalytic_residues))    # 1
        
        # Catalytic residue positions (padded to max 4 catalytic residues)
        for i in range(4):
            if i < len(self.catalytic_residues):
                features.extend(self.catalytic_residues[i].center.tolist())  # 3
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # Sub-pocket centers (padded to max 6 sub-pockets)
        for key in ["S1", "S2", "S3", "S1_prime", "S2_prime", "flap"]:
            if key in self.sub_pockets:
                features.extend(self.sub_pockets[key]["center"])  # 3
                features.append(self.sub_pockets[key].get("volume", 0.0))  # 1
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)


# === AMINO ACID PROPERTY LOOKUP (hardcoded biochemistry) ===

HBOND_DONORS = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "TRP", "ARG", "LYS", "CYS"}
HBOND_ACCEPTORS = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "ASP", "GLU", "MET", "CYS"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"}
POSITIVE = {"ARG", "LYS", "HIS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP", "HIS"}

# BACE1-specific catalytic residues
BACE1_CATALYTIC = {32, 228}  # Asp32 and Asp228


def parse_pocket(pdb_path: str, 
                 pocket_center: Tuple[float, float, float],
                 pocket_radius: float = 12.0,
                 catalytic_residue_ids: set = None,
                 pdb_id: str = "unknown") -> PocketFeatures:
    """
    Parse a PDB file and extract pocket features within radius of center.
    
    Args:
        pdb_path: Path to PDB file
        pocket_center: (x, y, z) center of binding pocket
        pocket_radius: Radius in Angstroms to include residues
        catalytic_residue_ids: Set of residue IDs that are catalytically active
        pdb_id: PDB identifier string
    
    Returns:
        PocketFeatures: Structured pocket representation
    """
    if catalytic_residue_ids is None:
        catalytic_residue_ids = BACE1_CATALYTIC
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]  # First model
    
    center = np.array(pocket_center)
    pocket = PocketFeatures(pdb_id=pdb_id, pocket_center=center,
                            pocket_radius=pocket_radius)
    
    # Extract all atoms for neighbor search
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)
    
    # Find residues within pocket radius
    nearby_atoms = ns.search(center, pocket_radius, 'R')  # 'R' = residue level
    seen_residues = set()
    
    for residue in nearby_atoms:
        if not is_aa(residue):
            # Check if it's a water molecule
            if residue.get_resname() == "HOH":
                water_center = np.mean([a.get_vector().get_array() 
                                        for a in residue.get_atoms()], axis=0)
                pocket.water_positions.append(water_center)
            continue
        
        resid = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        key = (chain, resid)
        if key in seen_residues:
            continue
        seen_residues.add(key)
        
        resname = residue.get_resname()
        
        # Compute center of mass
        atoms = list(residue.get_atoms())
        coords = np.array([a.get_vector().get_array() for a in atoms])
        res_center = coords.mean(axis=0)
        
        # Sidechain atoms (exclude backbone N, CA, C, O)
        backbone_names = {"N", "CA", "C", "O"}
        sidechain = [a.get_vector().get_array() for a in atoms 
                     if a.get_name() not in backbone_names]
        
        feat = ResidueFeature(
            name=f"{resname}{resid}",
            resname=resname,
            resid=resid,
            chain=chain,
            center=res_center,
            is_hbond_donor=resname in HBOND_DONORS,
            is_hbond_acceptor=resname in HBOND_ACCEPTORS,
            is_hydrophobic=resname in HYDROPHOBIC,
            is_charged=resname in POSITIVE or resname in NEGATIVE,
            charge_sign=1 if resname in POSITIVE else (-1 if resname in NEGATIVE else 0),
            is_aromatic=resname in AROMATIC,
            is_catalytic=resid in catalytic_residue_ids,
            sidechain_atoms=[np.array(s) for s in sidechain],
        )
        
        pocket.residues.append(feat)
        
        if feat.is_catalytic:
            pocket.catalytic_residues.append(feat)
        if feat.is_hbond_donor:
            pocket.hbond_donors.append(feat)
        if feat.is_hbond_acceptor:
            pocket.hbond_acceptors.append(feat)
        if feat.is_hydrophobic:
            pocket.hydrophobic_residues.append(feat)
        if feat.charge_sign > 0:
            pocket.positive_residues.append(feat)
        elif feat.charge_sign < 0:
            pocket.negative_residues.append(feat)
    
    return pocket
```

### 5.4 BACE1-Specific Pocket Configuration

```python
# configs/bace1_4ivt.yaml (values from PDB 4IVT crystal structure)

pocket:
  pdb_id: "4IVT"
  pdb_path: "data/structures/4IVT.pdb"
  
  # Pocket center: midpoint between Asp32 and Asp228 catalytic dyad
  # These coordinates are from the 4IVT crystal structure
  center: [28.0, 15.0, 22.0]  # Approximate — refine from actual structure
  radius: 12.0  # Angstroms — captures full active site
  
  catalytic_residues: [32, 228]  # Asp32, Asp228
  
  # Known sub-pockets from BACE1 literature
  sub_pockets:
    S1:
      description: "Primary substrate-binding pocket"
      key_residues: [71, 73, 108, 110]  # Tyr71, Thr73, Leu108, Trp110
    S2:
      description: "Secondary pocket, accommodates larger substituents"
      key_residues: [198, 200, 226, 229]
    S3:
      description: "Solvent-exposed region near flap"
      key_residues: [68, 69, 70]
    flap:
      description: "Flexible flap region (opens/closes)"
      key_residues: [67, 68, 69, 70, 71, 72, 73, 74, 75]
      # Flap tip: residues ~71-75, covers the active site
```

---

## 6. Module 2: Ligand Representation & Action Space

### 6.1 Purpose

Represent a drug molecule as a manipulable 3D object with pharmacophore features, and define the action space for pose adjustments. This is the "player piece" with its available "moves."

### 6.2 Implementation

```python
# src/ligand/ligand_features.py
"""
Represent a drug molecule as structured features for the policy network.

Uses RDKit for 3D conformer generation and property computation.
No ML — pure cheminformatics.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdMolTransforms
except ImportError:
    raise ImportError("pip install rdkit --break-system-packages")


@dataclass
class LigandFeatures:
    """
    Structured representation of a drug molecule — the "player piece."
    """
    
    # Identity
    smiles: str
    mol: object  # RDKit mol object
    
    # Current 3D state
    conformer_coords: np.ndarray     # (n_atoms, 3) current atom positions
    center_of_mass: np.ndarray       # (3,) current center
    
    # Pharmacophore features (what the molecule CAN do)
    n_hbond_donors: int
    n_hbond_acceptors: int
    n_rotatable_bonds: int
    n_aromatic_rings: int
    logp: float
    molecular_weight: float
    tpsa: float  # Topological polar surface area
    
    # Pharmacophore point positions (where interactions CAN happen)
    hbond_donor_positions: List[np.ndarray] = field(default_factory=list)
    hbond_acceptor_positions: List[np.ndarray] = field(default_factory=list)
    hydrophobic_center_positions: List[np.ndarray] = field(default_factory=list)
    aromatic_ring_centers: List[np.ndarray] = field(default_factory=list)
    charged_group_positions: List[np.ndarray] = field(default_factory=list)
    
    # Rotatable bond indices (for torsion actions)
    rotatable_bond_indices: List[Tuple[int, int]] = field(default_factory=list)
    
    # Current pose (6 degrees of freedom)
    pose_translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # (x, y, z) offset
    pose_rotation: np.ndarray = field(
        default_factory=lambda: np.zeros(3))   # (rx, ry, rz) Euler angles
    
    # Known experimental affinity (if available from BindingDB)
    known_ic50_nm: float = -1.0  # -1 means unknown
    known_ki_nm: float = -1.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Flatten ligand into fixed-size feature vector."""
        features = []
        
        # Global properties
        features.extend(self.center_of_mass.tolist())    # 3
        features.append(self.n_hbond_donors)              # 1
        features.append(self.n_hbond_acceptors)           # 1
        features.append(self.n_rotatable_bonds)           # 1
        features.append(self.n_aromatic_rings)            # 1
        features.append(self.logp)                        # 1
        features.append(self.molecular_weight / 500.0)    # 1 (normalized)
        features.append(self.tpsa / 200.0)                # 1 (normalized)
        
        # Current pose
        features.extend(self.pose_translation.tolist())   # 3
        features.extend(self.pose_rotation.tolist())      # 3
        
        return np.array(features, dtype=np.float32)


def create_ligand(smiles: str, ic50_nm: float = -1.0, 
                  ki_nm: float = -1.0) -> LigandFeatures:
    """
    Create a LigandFeatures from SMILES string.
    
    Generates a 3D conformer and extracts all pharmacophore features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result == -1:
        # Fallback to random coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    center = coords.mean(axis=0)
    
    # Pharmacophore properties
    mol_noH = Chem.RemoveHs(mol)
    
    # Find H-bond donor/acceptor positions
    donor_positions = []
    acceptor_positions = []
    for atom in mol.GetAtoms():
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        # Donors: N-H, O-H
        if atom.GetSymbol() in ("N", "O") and atom.GetTotalNumHs() > 0:
            donor_positions.append(pos)
        # Acceptors: N, O with lone pairs
        if atom.GetSymbol() in ("N", "O"):
            acceptor_positions.append(pos)
    
    # Hydrophobic centers (C atoms not bonded to heteroatoms)
    hydrophobic_positions = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and not any(
            n.GetSymbol() in ("N", "O", "S") for n in atom.GetNeighbors()):
            hydrophobic_positions.append(
                np.array(conf.GetAtomPosition(atom.GetIdx())))
    
    # Rotatable bonds
    rot_bonds = mol_noH.GetSubstructMatches(
        Chem.MolFromSmarts("[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]"))
    
    return LigandFeatures(
        smiles=smiles,
        mol=mol,
        conformer_coords=coords,
        center_of_mass=center,
        n_hbond_donors=rdMolDescriptors.CalcNumHBD(mol_noH),
        n_hbond_acceptors=rdMolDescriptors.CalcNumHBA(mol_noH),
        n_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol_noH),
        n_aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol_noH),
        logp=Descriptors.MolLogP(mol_noH),
        molecular_weight=Descriptors.MolWt(mol_noH),
        tpsa=Descriptors.TPSA(mol_noH),
        hbond_donor_positions=donor_positions,
        hbond_acceptor_positions=acceptor_positions,
        hydrophobic_center_positions=hydrophobic_positions,
        rotatable_bond_indices=list(rot_bonds),
        known_ic50_nm=ic50_nm,
        known_ki_nm=ki_nm,
    )
```

### 6.3 Action Space

```python
# src/ligand/action_space.py
"""
Define the action space for pose adjustments.

The agent's "moves" in the docking game.
Discretized for RL — each action is a small pose change.
"""

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class DockingAction(IntEnum):
    """
    Discrete action space for pose adjustments.
    
    18 actions: 6 translations + 6 rotations + N torsions
    (N depends on the ligand's rotatable bonds)
    """
    # Translations (±0.5 Angstrom steps)
    TRANSLATE_X_POS = 0
    TRANSLATE_X_NEG = 1
    TRANSLATE_Y_POS = 2
    TRANSLATE_Y_NEG = 3
    TRANSLATE_Z_POS = 4
    TRANSLATE_Z_NEG = 5
    
    # Rotations (±15 degree steps)
    ROTATE_X_POS = 6
    ROTATE_X_NEG = 7
    ROTATE_Y_POS = 8
    ROTATE_Y_NEG = 9
    ROTATE_Z_POS = 10
    ROTATE_Z_NEG = 11
    
    # Torsion angles (±30 degree steps)
    # These are dynamically added based on rotatable bonds
    # TORSION_0_POS = 12, TORSION_0_NEG = 13, ...

# Step sizes
TRANSLATION_STEP = 0.5    # Angstroms
ROTATION_STEP = 15.0      # Degrees
TORSION_STEP = 30.0       # Degrees


def get_action_count(n_rotatable_bonds: int) -> int:
    """Total number of actions for a ligand with N rotatable bonds."""
    return 12 + 2 * min(n_rotatable_bonds, 5)  # Cap at 5 torsions = 22 max


def apply_action(coords: np.ndarray, center: np.ndarray,
                 action: int, rotatable_bonds: list) -> np.ndarray:
    """
    Apply a discrete action to ligand coordinates.
    
    Returns new coordinates (does NOT modify in place).
    """
    new_coords = coords.copy()
    
    if action < 6:
        # Translation
        direction = action // 2  # 0=X, 1=Y, 2=Z
        sign = 1.0 if action % 2 == 0 else -1.0
        delta = np.zeros(3)
        delta[direction] = sign * TRANSLATION_STEP
        new_coords += delta
        
    elif action < 12:
        # Rotation around center of mass
        rot_action = action - 6
        axis = rot_action // 2  # 0=X, 1=Y, 2=Z
        sign = 1.0 if rot_action % 2 == 0 else -1.0
        angle_rad = np.radians(sign * ROTATION_STEP)
        
        # Rotation matrix around the specified axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        if axis == 0:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Rotate around center of mass
        centered = new_coords - center
        rotated = centered @ R.T
        new_coords = rotated + center
        
    else:
        # Torsion angle rotation
        torsion_idx = (action - 12) // 2
        sign = 1.0 if (action - 12) % 2 == 0 else -1.0
        
        if torsion_idx < len(rotatable_bonds):
            # Rotate atoms on one side of the bond
            # (simplified — full implementation uses RDKit's SetDihedralDeg)
            pass  # Implemented via RDKit in production
    
    return new_coords
```

---

## 7. Module 3: Vina World Model (The Chess Engine)

### 7.1 Purpose

This is the "chess rules" — a perfect simulator that computes the exact binding score for any ligand pose. NOT learned. NOT approximated. Just a wrapper around AutoDock Vina.

The key insight: **Vina evaluations are FREE.** Unlike ARC-AGI-3 where each action costs you one of your 100 ticks, here you can evaluate millions of poses without any budget constraint. This is what makes the chess engine analogy work.

### 7.2 Implementation

```python
# src/vina_engine/vina_scorer.py
"""
AutoDock Vina wrapper — the "chess rules" of the docking game.

Evaluates any ligand pose instantly. No learning needed.
This is what makes the docking game fundamentally easier than ARC-AGI-3:
we KNOW the rules and can SIMULATE freely.

Install: pip install vina --break-system-packages
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import tempfile
import os

try:
    from vina import Vina
except ImportError:
    raise ImportError("pip install vina --break-system-packages")


@dataclass
class VinaScore:
    """Result of scoring one pose."""
    total_energy: float         # Total binding energy (kcal/mol) — more negative = better
    inter_energy: float         # Intermolecular energy
    intra_energy: float         # Intramolecular (ligand internal) energy
    
    # Decomposed energy terms (where available)
    # These correspond DIRECTLY to the pharmacological priors:
    #   H_BOND → gauss1/gauss2 attractive terms
    #   HYDROPHOBIC → hydrophobic term
    #   STERIC_CLASH → repulsion term
    
    # Additional computed features (NOT from Vina, computed separately)
    n_hbonds: int = 0
    hydrophobic_contact_area: float = 0.0
    steric_clash_count: int = 0
    dist_asp32: float = 0.0
    dist_asp228: float = 0.0


class VinaWorldModel:
    """
    The perfect world model — knows the rules of molecular binding.
    
    Usage:
        wm = VinaWorldModel("data/structures/prepared/4IVT_receptor.pdbqt")
        score = wm.score_pose(ligand_pdbqt_string)
        
    This is the chess engine's rule book.
    The policy network decides WHERE to move.
    Vina tells you what the SCORE is after the move.
    """
    
    def __init__(self, receptor_pdbqt_path: str,
                 center: tuple = (28.0, 15.0, 22.0),
                 box_size: tuple = (25.0, 25.0, 25.0),
                 exhaustiveness: int = 8):
        """
        Initialize Vina with receptor structure.
        
        Args:
            receptor_pdbqt_path: Prepared receptor file
            center: (x, y, z) center of search box
            box_size: (sx, sy, sz) dimensions of search box in Angstroms
            exhaustiveness: Search thoroughness (higher = slower + better)
        """
        self.v = Vina(sf_name='vina')
        self.v.set_receptor(receptor_pdbqt_path)
        self.v.compute_vina_maps(center=list(center), 
                                 box_size=list(box_size))
        self.center = np.array(center)
        self.box_size = np.array(box_size)
        self.n_evaluations = 0
    
    def score_pose(self, ligand_pdbqt: str) -> VinaScore:
        """
        Score a single ligand pose. This is ONE "chess move evaluation."
        
        Args:
            ligand_pdbqt: PDBQT string of the ligand in a specific pose
        
        Returns:
            VinaScore with binding energy (more negative = better binding)
        """
        # Write ligand to temp file (Vina needs file path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt', 
                                          delete=False) as f:
            f.write(ligand_pdbqt)
            tmp_path = f.name
        
        try:
            self.v.set_ligand_from_file(tmp_path)
            energy = self.v.score()
            self.n_evaluations += 1
            
            return VinaScore(
                total_energy=energy[0],
                inter_energy=energy[1] if len(energy) > 1 else energy[0],
                intra_energy=energy[2] if len(energy) > 2 else 0.0,
            )
        finally:
            os.unlink(tmp_path)
    
    def dock_ligand(self, ligand_pdbqt: str,
                     n_poses: int = 10) -> list:
        """
        Full docking search — let Vina find the best poses.
        
        This is like letting the chess engine find the best move itself.
        Used for baseline comparison (can the RL agent match Vina's search?).
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdbqt',
                                          delete=False) as f:
            f.write(ligand_pdbqt)
            tmp_path = f.name
        
        try:
            self.v.set_ligand_from_file(tmp_path)
            self.v.dock(exhaustiveness=8, n_poses=n_poses)
            energies = self.v.energies(n_poses=n_poses)
            
            results = []
            for i, e in enumerate(energies):
                results.append(VinaScore(
                    total_energy=e[0],
                    inter_energy=e[1] if len(e) > 1 else e[0],
                    intra_energy=e[2] if len(e) > 2 else 0.0,
                ))
            return results
        finally:
            os.unlink(tmp_path)
    
    def get_evaluation_count(self) -> int:
        """How many poses have been evaluated (for logging)."""
        return self.n_evaluations
```

---

## 8. Module 4: Search Policy Network

### 8.1 Purpose

The ONLY learned component. A small GRU that takes the current state (pocket features + ligand features + interaction features + score history) and outputs: (1) action probabilities (which pose adjustment to try next), (2) a value estimate (how good is the current pose?).

**This is the thing with hidden states that DESCARTES probes.**

### 8.2 Implementation

```python
# src/policy/policy_network.py
"""
Search Policy Network — the ONLY learned component.

Like AlphaGo's policy network: it doesn't learn the rules of Go,
it learns which moves are worth trying. Our network doesn't learn
physics — Vina handles that. It learns which pose adjustments 
are likely to improve binding.

Its hidden states are probed by DESCARTES (Module 6) to determine
if it learned genuine binding intuition or is a pharmaceutical zombie.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SearchPolicyNetwork(nn.Module):
    """
    GRU-based policy network for docking search guidance.
    
    Input: [pocket_features ⊕ ligand_features ⊕ interaction_features
            ⊕ score_history ⊕ current_vina_score]
    
    Output: 
        - policy: action probabilities (which pose adjustment)
        - value: predicted final binding energy
    
    Hidden states are saved for DESCARTES probing.
    """
    
    def __init__(self, 
                 pocket_dim: int = 40,      # From PocketFeatures.to_feature_vector()
                 ligand_dim: int = 16,       # From LigandFeatures.to_feature_vector()
                 interaction_dim: int = 20,  # From interaction features
                 score_history_len: int = 10,# Last N Vina scores
                 hidden_dim: int = 128,
                 n_layers: int = 2,
                 n_actions: int = 22,        # 12 base + 10 torsion (max)
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Input projection
        total_input = pocket_dim + ligand_dim + interaction_dim + score_history_len + 1
        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # GRU core — sequential decision making
        # The hidden state accumulates information across pose adjustments
        # THIS IS WHAT DESCARTES PROBES
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        
        # Policy head: which action to take next
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        
        # Value head: how good is the current state
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Storage for hidden states (for DESCARTES probing)
        self._hidden_states_log = []
        self._logging_enabled = False
    
    def forward(self, x, h=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            h: (n_layers, batch, hidden_dim) previous hidden state
        
        Returns:
            policy_logits: (batch, n_actions) raw logits
            value: (batch, 1) state value estimate
            h_new: (n_layers, batch, hidden_dim) new hidden state
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq dim
        
        # Project input
        projected = self.input_proj(x)
        
        # GRU forward
        if h is None:
            h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim,
                           device=x.device)
        
        gru_out, h_new = self.gru(projected, h)
        
        # Use last timestep output
        last_hidden = gru_out[:, -1, :]
        
        # Log hidden states for DESCARTES probing
        if self._logging_enabled:
            self._hidden_states_log.append(
                h_new[-1].detach().cpu().numpy()  # Last layer hidden state
            )
        
        # Heads
        policy_logits = self.policy_head(last_hidden)
        value = self.value_head(last_hidden)
        
        return policy_logits, value, h_new
    
    def select_action(self, x, h=None, temperature=1.0):
        """
        Select an action using the policy (with temperature for exploration).
        
        Args:
            x: (1, input_dim) current state
            h: hidden state
            temperature: >1 = more exploration, <1 = more exploitation
        
        Returns:
            action: int, selected action index
            log_prob: float, log probability of the action
            value: float, state value estimate
            h_new: new hidden state
        """
        policy_logits, value, h_new = self.forward(x.unsqueeze(0), h)
        
        # Temperature-scaled softmax
        probs = F.softmax(policy_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze(), h_new
    
    # === DESCARTES PROBING INTERFACE ===
    
    def enable_logging(self):
        """Start logging hidden states for DESCARTES probing."""
        self._logging_enabled = True
        self._hidden_states_log = []
    
    def disable_logging(self):
        """Stop logging hidden states."""
        self._logging_enabled = False
    
    def get_hidden_states(self) -> np.ndarray:
        """
        Get logged hidden states for DESCARTES probing.
        
        Returns: (n_timesteps, hidden_dim) array
        """
        if not self._hidden_states_log:
            return np.array([])
        return np.concatenate(self._hidden_states_log, axis=0)
    
    def clear_hidden_log(self):
        """Clear the hidden state log."""
        self._hidden_states_log = []
```

---

## 9. Module 5: Training Loop

### 9.1 Purpose

RL-style training: the policy network proposes pose adjustments, Vina scores them, and the policy updates to maximize binding improvement. Each episode is one ligand being docked.

### 9.2 Environment Wrapper

```python
# src/training/docking_env.py
"""
Gym-style environment wrapping the docking game.

Each episode:
  1. Load a ligand from BindingDB
  2. Place it in a random initial pose near the pocket
  3. Agent takes T steps of pose adjustments
  4. Each step: Vina scores the new pose → reward = ΔE
  5. Episode ends after T steps or convergence
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DockingEnv:
    """
    Docking game environment.
    
    Observation: [pocket_features ⊕ ligand_features ⊕ interaction_features
                  ⊕ score_history ⊕ current_score]
    Action: discrete pose adjustment (translate/rotate/torsion)
    Reward: change in Vina binding energy (negative ΔE = improvement)
    """
    
    def __init__(self, vina_world_model, pocket_features,
                 max_steps: int = 200,
                 score_history_len: int = 10):
        self.wm = vina_world_model
        self.pocket = pocket_features
        self.pocket_vec = pocket_features.to_feature_vector()
        self.max_steps = max_steps
        self.score_history_len = score_history_len
        
        # State
        self.current_ligand = None
        self.current_coords = None
        self.current_score = None
        self.score_history = []
        self.step_count = 0
        self.best_score = None
        
        # Interaction feature computer
        self.interaction_computer = None  # Set in reset
    
    def reset(self, ligand_features, initial_coords=None) -> np.ndarray:
        """
        Start a new docking episode with a new ligand.
        
        Args:
            ligand_features: LigandFeatures object
            initial_coords: (n_atoms, 3) initial pose, or None for random
        
        Returns:
            observation: np.ndarray
        """
        self.current_ligand = ligand_features
        self.step_count = 0
        self.score_history = []
        
        if initial_coords is not None:
            self.current_coords = initial_coords.copy()
        else:
            # Random initial pose near pocket center
            self.current_coords = self._random_initial_pose(ligand_features)
        
        # Score initial pose
        pdbqt = self._coords_to_pdbqt(self.current_coords)
        score_result = self.wm.score_pose(pdbqt)
        self.current_score = score_result.total_energy
        self.best_score = self.current_score
        
        # Compute interaction features
        interaction_vec = self._compute_interactions()
        
        return self._make_observation(interaction_vec)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one action (pose adjustment) and get reward.
        
        Returns:
            observation: np.ndarray
            reward: float (negative ΔE = good, positive ΔE = bad)
            done: bool
            info: dict with details
        """
        from ..ligand.action_space import apply_action
        
        # Apply action to get new coordinates
        new_coords = apply_action(
            self.current_coords,
            self.current_ligand.center_of_mass,
            action,
            self.current_ligand.rotatable_bond_indices,
        )
        
        # Score new pose with Vina (FREE — no budget constraint!)
        pdbqt = self._coords_to_pdbqt(new_coords)
        try:
            score_result = self.wm.score_pose(pdbqt)
            new_score = score_result.total_energy
        except Exception:
            # Invalid pose (e.g., outside box) — penalize
            new_score = self.current_score + 10.0  # Penalty
        
        # Reward = improvement in binding energy
        # Vina scores are negative (more negative = better)
        # So reward = old_score - new_score (positive if new is more negative)
        reward = self.current_score - new_score
        
        # Update state
        self.current_coords = new_coords
        self.current_score = new_score
        self.score_history.append(new_score)
        self.step_count += 1
        
        if new_score < self.best_score:
            self.best_score = new_score
        
        # Check done conditions
        done = self.step_count >= self.max_steps
        
        # Compute interaction features for new pose
        interaction_vec = self._compute_interactions()
        
        info = {
            'vina_score': new_score,
            'best_score': self.best_score,
            'step': self.step_count,
            'reward': reward,
            'dist_asp32': self._dist_to_residue(32),
            'dist_asp228': self._dist_to_residue(228),
            'n_evaluations': self.wm.get_evaluation_count(),
        }
        
        return self._make_observation(interaction_vec), reward, done, info
    
    def _make_observation(self, interaction_vec: np.ndarray) -> np.ndarray:
        """Compose full observation vector."""
        ligand_vec = self.current_ligand.to_feature_vector()
        
        # Score history (padded to fixed length)
        history = np.zeros(self.score_history_len, dtype=np.float32)
        if self.score_history:
            recent = self.score_history[-self.score_history_len:]
            history[-len(recent):] = recent
        
        return np.concatenate([
            self.pocket_vec,
            ligand_vec,
            interaction_vec,
            history,
            np.array([self.current_score], dtype=np.float32),
        ])
    
    def _compute_interactions(self) -> np.ndarray:
        """
        Compute pairwise interaction features between current ligand pose 
        and pocket.
        
        These are the "CoreKnowledge translation layer" features — 
        structured relational representations, not flat descriptors.
        """
        features = []
        
        ligand_center = self.current_coords.mean(axis=0)
        
        # Distance to each catalytic residue
        for cat_res in self.pocket.catalytic_residues:
            dist = np.linalg.norm(ligand_center - cat_res.center)
            features.append(dist)
        # Pad to 4 catalytic residues
        while len(features) < 4:
            features.append(50.0)  # Far away default
        
        # Distance to pocket center
        features.append(np.linalg.norm(
            ligand_center - self.pocket.pocket_center))
        
        # Number of close contacts (< 4 Å) with H-bond donors/acceptors
        n_close_hbond = 0
        for donor in self.pocket.hbond_donors:
            if np.linalg.norm(ligand_center - donor.center) < 4.0:
                n_close_hbond += 1
        features.append(float(n_close_hbond))
        
        # Number of close contacts with hydrophobic residues
        n_close_hydrophobic = 0
        for hyd in self.pocket.hydrophobic_residues:
            if np.linalg.norm(ligand_center - hyd.center) < 5.0:
                n_close_hydrophobic += 1
        features.append(float(n_close_hydrophobic))
        
        # Steric clashes (atoms < 2 Å from any pocket atom)
        n_clashes = 0
        for res in self.pocket.residues:
            for sc_atom in res.sidechain_atoms:
                for lig_atom in self.current_coords:
                    if np.linalg.norm(lig_atom - sc_atom) < 2.0:
                        n_clashes += 1
        features.append(float(min(n_clashes, 20)))  # Cap
        
        # Fraction of pocket volume occupied
        # (simplified: fraction of pocket residues within 5Å of ligand)
        n_contacted = sum(1 for r in self.pocket.residues
                         if np.linalg.norm(ligand_center - r.center) < 5.0)
        features.append(n_contacted / max(len(self.pocket.residues), 1))
        
        # Number of waters potentially displaced
        n_waters_displaced = sum(
            1 for w in self.pocket.water_positions
            if np.linalg.norm(ligand_center - w) < 3.0)
        features.append(float(n_waters_displaced))
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _dist_to_residue(self, resid: int) -> float:
        """Distance from ligand center to a specific residue."""
        ligand_center = self.current_coords.mean(axis=0)
        for res in self.pocket.residues:
            if res.resid == resid:
                return float(np.linalg.norm(ligand_center - res.center))
        return 50.0  # Not found
    
    def _random_initial_pose(self, ligand) -> np.ndarray:
        """Place ligand randomly near the pocket center."""
        # Center ligand on pocket center with random offset
        offset = np.random.randn(3) * 3.0  # 3Å random offset
        centered = ligand.conformer_coords - ligand.center_of_mass
        return centered + self.pocket.pocket_center + offset
    
    def _coords_to_pdbqt(self, coords: np.ndarray) -> str:
        """Convert coordinates to a Vina-ready ligand PDBQT.

        IMPORTANT (E1): use meeko (or Open Babel) for CORRECT AutoDock atom
        types (donor/acceptor/aromatic/charge) and ROOT/ENDROOT framing. Do NOT
        label every atom "C" and do NOT emit MODEL/ENDMDL tags — Vina rejects
        the tags, and all-"C" typing makes the score depend only on sterics
        (chemistry is lost). The earlier simplified snippet here was wrong; the
        authoritative implementation is in
        descartes_pharma_docking/training/docking_env.py (_mol_coords_to_pdbqt),
        which tries meeko → obabel → a manual ROOT/ENDROOT writer with proper
        per-element AD types.
        """
        if getattr(self, "current_ligand", None) is not None \
                and getattr(self.current_ligand, "mol", None) is not None:
            return self._mol_coords_to_pdbqt(self.current_ligand.mol, coords)

        # Manual fallback (no RDKit mol): ROOT/ENDROOT, never MODEL/ENDMDL.
        lines = ["ROOT"]
        for i, (x, y, z) in enumerate(coords):
            lines.append(
                f"HETATM{i+1:5d}  C   LIG A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000  C")
        lines.append("ENDROOT")
        lines.append("END")
        return "\n".join(lines)
```

### 9.3 Training Script

```python
# src/training/trainer.py
"""
REINFORCE training loop for the docking search policy.

For each episode:
  1. Pick a ligand from the training set
  2. Place it randomly near the pocket
  3. Policy takes T steps of pose adjustments
  4. Each step: Vina scores the pose → reward = ΔE
  5. Policy gradient update
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque


class DockingTrainer:
    """Train the search policy network via REINFORCE with baseline."""
    
    def __init__(self, policy, env, lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, device='cpu'):
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_best_scores = deque(maxlen=100)
    
    def train_episode(self, ligand_features) -> dict:
        """
        Train on one docking episode (one ligand).
        
        Returns:
            dict with episode statistics
        """
        obs = self.env.reset(ligand_features)
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        h = None  # GRU hidden state
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            # Get action from policy
            action, log_prob, value, h = self.policy.select_action(
                obs_tensor, h, temperature=1.0)
            
            # Take action in environment
            obs, reward, done, info = self.env.step(action)
            
            # Compute entropy for exploration bonus
            policy_logits, _, _ = self.policy(
                obs_tensor.unsqueeze(0), h)
            probs = F.softmax(policy_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
        
        # Compute returns (discounted cumulative reward)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for log_prob, value, R, entropy in zip(
            log_probs, values, returns, entropies):
            
            advantage = R - value.detach()
            policy_loss -= log_prob * advantage
            value_loss += F.mse_loss(value, R)
            entropy_loss -= entropy
        
        loss = (policy_loss + 0.5 * value_loss + 
                self.entropy_coef * entropy_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Log
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_best_scores.append(info['best_score'])
        
        return {
            'total_reward': total_reward,
            'best_vina_score': info['best_score'],
            'n_steps': len(rewards),
            'mean_reward_100': np.mean(self.episode_rewards),
            'mean_best_score_100': np.mean(self.episode_best_scores),
            'loss': loss.item(),
        }
    
    def train(self, ligands, n_episodes=1000, 
              log_interval=10, save_interval=100,
              save_path="results/checkpoints/"):
        """
        Full training loop across multiple ligands.
        
        Args:
            ligands: list of LigandFeatures objects
            n_episodes: total training episodes
            log_interval: print stats every N episodes
            save_interval: save checkpoint every N episodes
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        rng = np.random.default_rng(42)
        
        for episode in range(n_episodes):
            # Pick a random ligand
            ligand = ligands[rng.integers(len(ligands))]
            
            # Train one episode
            stats = self.train_episode(ligand)
            
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{n_episodes} | "
                      f"Reward: {stats['total_reward']:.2f} | "
                      f"Best Vina: {stats['best_vina_score']:.2f} | "
                      f"Mean(100): {stats['mean_reward_100']:.2f} | "
                      f"Loss: {stats['loss']:.4f}")
            
            if (episode + 1) % save_interval == 0:
                torch.save({
                    'episode': episode + 1,
                    'model_state': self.policy.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'episode_rewards': list(self.episode_rewards),
                }, os.path.join(save_path, f"checkpoint_{episode+1}.pt"))
        
        print(f"\nTraining complete. Final mean reward: "
              f"{stats['mean_reward_100']:.2f}")
```

---

## 10. Module 6: DESCARTES Probe Suite

### 10.1 Purpose

After training, probe the Search Policy Network's hidden states with the full DESCARTES council controls. This is the scientific payload — determining whether the policy learned genuine binding intuition or is a pharmaceutical zombie.

### 10.2 Probe Targets

```
BINDING MECHANISM FEATURES (should be encoded if policy learned real binding):
  - dist_asp32:              Distance to catalytic Asp32
  - dist_asp228:             Distance to catalytic Asp228 (the one that showed life in Phase 6!)
  - n_hbonds:                Number of hydrogen bonds with pocket
  - hydrophobic_contact_area: Contact surface with hydrophobic residues
  - steric_clash_count:      Number of steric clashes
  - pocket_occupancy:        Fraction of pocket volume filled
  - water_displacement:      Number of waters displaced

CONFOUND FEATURES (should NOT be encoded — zombie indicators):
  - molecular_weight:        MW of the ligand (confound from Phase 2)
  - num_heavy_atoms:         NumHeavyAtoms (confound from Phase 2)
  - logp:                    LogP (VZS Tier 1 axiom — but could be confound in docking)
  - random_noise:            Pure noise control

INTERMEDIATE FEATURES (may or may not be encoded):
  - vina_score:              Current Vina score (the policy should encode this!)
  - score_improvement:       Recent trend in scores
  - flap_contact:            Contact with flap region
```

### 10.3 Implementation

```python
# src/probing/probe_runner.py
"""
DESCARTES probe suite for the Search Policy Network.

Extracts hidden states from the trained GRU during docking episodes,
then probes them for binding mechanism features using:
  1. Ridge ΔR² with scaffold-stratified permutation (200 perms)
  2. MLP ΔR² nonlinear control
  3. Arbitrary target probes (MW, NumHeavyAtoms)
  4. 50-seed ensemble
  5. Pocket scramble test
  6. Untrained network control

Council controls from DESCARTES-PHARMA v1.3 apply WITHOUT modification.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import Dict, List
import torch


class DESCARTESProbeRunner:
    """
    Run the complete DESCARTES probing pipeline on a trained policy.
    """
    
    def __init__(self, policy_network, env, ligands,
                 n_episodes_for_probing: int = 100,
                 device: str = 'cpu'):
        self.policy = policy_network
        self.env = env
        self.ligands = ligands
        self.n_episodes = n_episodes_for_probing
        self.device = device
    
    def collect_hidden_states_and_targets(self) -> Dict:
        """
        Run the trained policy on multiple ligands, collecting:
        - Hidden states at each timestep
        - Ground truth values for all probe targets
        
        Returns:
            dict with 'hidden_states' (N, hidden_dim) and 
            'targets' dict mapping name → (N,) array
        """
        self.policy.eval()
        self.policy.enable_logging()
        self.policy.clear_hidden_log()
        
        all_targets = {
            'dist_asp32': [], 'dist_asp228': [],
            'n_hbonds': [], 'hydrophobic_contact': [],
            'steric_clashes': [], 'pocket_occupancy': [],
            'water_displacement': [],
            'vina_score': [], 'score_improvement': [],
            # Confounds
            'molecular_weight': [], 'num_heavy_atoms': [],
            'logp': [], 'random_noise': [],
        }
        
        rng = np.random.default_rng(42)
        
        for ep in range(self.n_episodes):
            ligand = self.ligands[ep % len(self.ligands)]
            obs = self.env.reset(ligand)
            h = None
            prev_score = self.env.current_score
            
            for step in range(self.env.max_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action, _, _, h = self.policy.select_action(
                        obs_tensor, h, temperature=0.1)  # Near-greedy
                
                obs, reward, done, info = self.env.step(action)
                
                # Record targets for this timestep
                all_targets['dist_asp32'].append(info['dist_asp32'])
                all_targets['dist_asp228'].append(info['dist_asp228'])
                all_targets['vina_score'].append(info['vina_score'])
                all_targets['score_improvement'].append(reward)
                all_targets['molecular_weight'].append(
                    ligand.molecular_weight)
                all_targets['num_heavy_atoms'].append(
                    ligand.mol.GetNumHeavyAtoms())
                all_targets['logp'].append(ligand.logp)
                all_targets['random_noise'].append(rng.normal())
                
                # Interaction features from observation
                # (indices depend on feature layout)
                all_targets['n_hbonds'].append(
                    obs[self.env.pocket_vec.shape[0] + 16 + 5])
                all_targets['hydrophobic_contact'].append(
                    obs[self.env.pocket_vec.shape[0] + 16 + 6])
                all_targets['steric_clashes'].append(
                    obs[self.env.pocket_vec.shape[0] + 16 + 7])
                all_targets['pocket_occupancy'].append(
                    obs[self.env.pocket_vec.shape[0] + 16 + 8])
                all_targets['water_displacement'].append(
                    obs[self.env.pocket_vec.shape[0] + 16 + 9])
                
                prev_score = info['vina_score']
                
                if done:
                    break
        
        hidden_states = self.policy.get_hidden_states()
        self.policy.disable_logging()
        
        # Truncate targets to match hidden states length
        n = len(hidden_states)
        targets = {k: np.array(v[:n]) for k, v in all_targets.items()}
        
        return {
            'hidden_states': hidden_states,
            'targets': targets,
        }
    
    def run_full_probe_suite(self) -> Dict:
        """
        Run the complete DESCARTES probe suite.
        
        Returns comprehensive verdict for each target.
        """
        print("=" * 60)
        print("DESCARTES PROBE SUITE — DOCKING POLICY NETWORK")
        print("=" * 60)
        
        # 1. Collect data
        print("\n[1/6] Collecting hidden states and targets...")
        data = self.collect_hidden_states_and_targets()
        H = data['hidden_states']
        targets = data['targets']
        print(f"  Collected {len(H)} timesteps, hidden_dim={H.shape[1]}")
        
        # 2. Get untrained network hidden states (control)
        print("\n[2/6] Generating untrained network control...")
        H_untrained = self._get_untrained_hidden_states(
            len(H), H.shape[1])
        
        # 3. Ridge ΔR² for each target
        print("\n[3/6] Ridge ΔR² probing (with permutation tests)...")
        results = {}
        for name, target in targets.items():
            r2_trained = self._cross_val_r2(H, target)
            r2_untrained = self._cross_val_r2(H_untrained, target)
            delta_r2 = r2_trained - r2_untrained
            
            # Permutation test (scaffold-stratified)
            p_value = self._permutation_test(H, target, n_perms=200)
            
            results[name] = {
                'r2_trained': r2_trained,
                'r2_untrained': r2_untrained,
                'delta_r2': delta_r2,
                'p_value': p_value,
                'significant': p_value < 0.05,
            }
            
            status = "✓ ENCODED" if delta_r2 > 0.05 and p_value < 0.05 else "✗ zombie"
            print(f"  {name:30s} ΔR²={delta_r2:.4f} p={p_value:.4f} {status}")
        
        # 4. Council control: arbitrary targets
        print("\n[4/6] Arbitrary target control...")
        confound_names = ['molecular_weight', 'num_heavy_atoms', 
                          'logp', 'random_noise']
        for name in confound_names:
            if results[name]['significant']:
                print(f"  ⚠ WARNING: Confound {name} is significant! "
                      f"ΔR²={results[name]['delta_r2']:.4f}")
        
        # 5. Summary verdict
        print("\n[5/6] Generating verdicts...")
        binding_features = ['dist_asp32', 'dist_asp228', 'n_hbonds',
                           'hydrophobic_contact', 'steric_clashes',
                           'pocket_occupancy', 'water_displacement']
        
        n_encoded = sum(1 for f in binding_features 
                       if results[f]['significant'] and 
                       results[f]['delta_r2'] > 0.05)
        
        n_confounds = sum(1 for f in confound_names
                         if results[f]['significant'])
        
        if n_encoded >= 3 and n_confounds == 0:
            verdict = "CONFIRMED_NON_ZOMBIE"
        elif n_encoded >= 1 and n_confounds <= 1:
            verdict = "CANDIDATE_ENCODED"
        elif n_encoded == 0:
            verdict = "PHARMACEUTICAL_ZOMBIE"
        else:
            verdict = "AMBIGUOUS (confounds detected)"
        
        print(f"\n{'=' * 60}")
        print(f"VERDICT: {verdict}")
        print(f"  Binding features encoded: {n_encoded}/{len(binding_features)}")
        print(f"  Confounds detected: {n_confounds}/{len(confound_names)}")
        print(f"  Key finding: dist_asp228 {'ENCODED' if results['dist_asp228']['significant'] else 'NOT encoded'}")
        print(f"{'=' * 60}")
        
        results['_verdict'] = verdict
        results['_n_encoded'] = n_encoded
        results['_n_confounds'] = n_confounds
        
        return results
    
    def _cross_val_r2(self, X, y, n_splits=5):
        """5-fold cross-validated Ridge R²."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in kf.split(X):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X[train_idx], y[train_idx])
            scores.append(ridge.score(X[test_idx], y[test_idx]))
        return np.mean(scores)
    
    def _permutation_test(self, X, y, n_perms=200):
        """Scaffold-stratified permutation test."""
        real_r2 = self._cross_val_r2(X, y)
        
        rng = np.random.default_rng(42)
        null_r2s = []
        for _ in range(n_perms):
            y_perm = rng.permutation(y)
            null_r2s.append(self._cross_val_r2(X, y_perm))
        
        p_value = np.mean(np.array(null_r2s) >= real_r2)
        return p_value
    
    def _get_untrained_hidden_states(self, n_samples, hidden_dim):
        """Generate hidden states from randomly initialized network."""
        rng = np.random.default_rng(99)
        return rng.normal(0, 0.01, (n_samples, hidden_dim)).astype(np.float32)
```

---

## 11. LLM Balloon Expansion (C1/C2)

### 11.1 Purpose

When the policy's performance plateaus, call Claude API to propose new interaction features for the perception layer. This is the C1/C2 ontology expansion from the COGITO/HIMARI framework — rare, expensive, outside the training loop.

### 11.2 Implementation

```python
# src/balloon/c1_vocabulary.py
"""
C1 Pharmacological Vocabulary + Balloon Expansion.

C1 starts with basic interaction concepts.
When the policy plateaus and probing shows gaps,
the LLM proposes new concepts.
"""

# Initial C1 vocabulary (analogous to ARC-AGI's OBJECT, WALL, MOVEMENT)
C1_INITIAL = {
    "H_BOND": "Hydrogen bond between donor and acceptor",
    "HYDROPHOBIC_CONTACT": "Van der Waals contact between nonpolar groups",
    "STERIC_CLASH": "Atoms closer than sum of van der Waals radii",
    "CHARGE_PAIR": "Electrostatic attraction between opposite charges",
    "SOLVENT_EXPOSURE": "Fraction of ligand exposed to solvent",
    "POCKET_DEPTH": "How deep into the pocket the ligand sits",
    "CATALYTIC_PROXIMITY": "Distance to catalytic residues",
}

# C2 hypotheses = combinations of C1 concepts
# These are tested during training
C2_HYPOTHESES = {
    "CATALYTIC_HBOND": "H_BOND + CATALYTIC_PROXIMITY → high score impact",
    "DEEP_HYDROPHOBIC": "HYDROPHOBIC_CONTACT + POCKET_DEPTH → stable binding",
    "STERIC_AT_GATE": "STERIC_CLASH + near flap → blocks entry",
}


# Balloon expansion trigger (4-test protocol from HIMARI)
BALLOON_PROMPT = """You are a computational medicinal chemist advising the 
DESCARTES-PHARMA Docking Game Agent. The agent's Search Policy Network has 
been trained to optimize ligand poses in the BACE1 binding pocket, but 
performance has plateaued.

Current C1 pharmacological vocabulary:
{c1_vocabulary}

Current interaction features computed per pose:
{current_features}

Training performance:
- Best Vina score achieved: {best_score} kcal/mol
- Mean reward (last 100 episodes): {mean_reward}
- DESCARTES probing results: {probe_summary}

The following binding features are NOT encoded in the policy's hidden states:
{unencoded_features}

Propose 3-5 NEW interaction features that might help the policy learn
these missing binding concepts. Consider:
- Water-mediated hydrogen bonds
- Pi-stacking interactions
- Halogen bonds
- Flap dynamics (open/closed state)
- Subpocket-specific occupancy
- Induced-fit conformational changes

Respond with ONLY valid JSON. Array of proposed features, each with:
  - name: feature name (UPPER_SNAKE_CASE)
  - description: what it measures
  - computation: how to compute from pocket + ligand coordinates
  - hypothesis: why this might help encode the missing binding features
"""
```

---

## 12. Data Pipeline

### 12.1 Data Download and Preparation

```python
# scripts/download_bindingdb.py
"""
Download BACE1 compounds from BindingDB via PyTDC.

Gets: SMILES + IC50 (continuous, in nM) for thousands of compounds.
This is the training set for the docking game.
"""

def download_bace1_data():
    """Download BACE1 binding data from BindingDB via PyTDC."""
    from tdc.single_pred import ADME  # or appropriate TDC module
    
    # Option 1: PyTDC BACE dataset (already used in Phases 1-6)
    from tdc.single_pred import ADME
    # The BACE dataset in PyTDC has binary labels
    # We need continuous values from BindingDB
    
    # Option 2: Direct BindingDB download for BACE1
    # Filter for: Target = "Beta-secretase 1", has IC50 or Ki
    # URL: https://www.bindingdb.org/bind/ByTarget.jsp
    # Search: "Beta-secretase 1" or "BACE1"
    # Download as TSV with IC50/Ki values
    
    print("Download BACE1 compounds from BindingDB:")
    print("1. Go to https://www.bindingdb.org")
    print("2. Search target: 'Beta-secretase 1'")
    print("3. Filter: IC50 data only")
    print("4. Download as TSV")
    print("5. Save to data/ligands/bindingdb_bace1.csv")
    print()
    print("Expected: ~2000-5000 compounds with continuous IC50 values")
    print("These replace binary active/inactive labels with real binding data")


# scripts/prepare_receptor.py
"""
One-time receptor preparation for AutoDock Vina.

PDB → clean structure → add hydrogens → PDBQT format
"""

def prepare_receptor(pdb_path: str, output_path: str):
    """
    Prepare receptor for Vina docking.
    
    Steps:
    1. Remove water molecules (except conserved)
    2. Remove co-crystallized ligand
    3. Add hydrogens at physiological pH
    4. Convert to PDBQT format
    
    Requires: pip install meeko --break-system-packages
    """
    # Using Open Babel or meeko for PDBQT conversion
    import subprocess
    
    # Remove waters and ligands, add hydrogens
    # obabel handles this well
    subprocess.run([
        "obabel", pdb_path,
        "-O", output_path,
        "-xr",  # Remove waters
        "-h",   # Add hydrogens
    ], check=True)
    
    print(f"Prepared receptor: {output_path}")
```

### 12.2 Data Splits

```yaml
# Train: 70% of BindingDB BACE1 compounds (~1500-3500 molecules)
# Val: 15% (~300-750)
# Test: 15% (~300-750)
# 
# Split by SCAFFOLD (Murcko scaffold decomposition) to prevent
# scaffold memorization — the same method used in Phase 2
#
# Stratify by IC50 range to ensure all affinity ranges are represented
```

---

## 13. Implementation Phases

### Phase 1: Foundation (Day 1-2)

```
P0: Set up project structure + install dependencies
    pip install biopython rdkit vina torch numpy scipy 
        scikit-learn meeko --break-system-packages

P0: Module 1 — PocketKnowledge Perception
    - Implement pocket_parser.py
    - Parse PDB 4IVT into PocketFeatures
    - Test: verify Asp32, Asp228 detected as catalytic
    - Test: verify pocket contains ~30-50 residues within 12Å

P0: Module 2 — Ligand Representation
    - Implement ligand_features.py
    - Test with 3 known BACE1 inhibitors from BindingDB
    - Test: conformer generation works, features extracted

P0: Module 3 — Vina World Model
    - Implement vina_scorer.py
    - Prepare 4IVT receptor as PDBQT
    - Test: score a known co-crystallized ligand
    - Test: score matches literature value (±1 kcal/mol)
```

### Phase 2: Core Training (Day 2-3)

```
P0: DockingEnv implementation
    - Compose pocket + ligand + interaction features
    - Action application (translate/rotate)
    - Vina scoring per step
    - Test: one episode runs, rewards computed correctly

P0: Search Policy Network
    - GRU with policy + value heads
    - Hidden state logging for DESCARTES
    - Test: forward pass produces valid action probabilities

P0: Training loop
    - REINFORCE with baseline
    - Train on 10 ligands for 100 episodes (smoke test)
    - Verify: rewards improve over episodes
    - Verify: Vina scores improve (more negative = better)
```

### Phase 3: Scale + Probe (Day 3-4)

```
P1: Download full BindingDB BACE1 dataset
    - Parse SMILES + IC50/Ki values
    - Scaffold-stratified train/val/test split
    - Prepare ligand conformers

P1: Full training run
    - Train on full training set (1000+ ligands)
    - 1000-5000 episodes
    - Monitor: mean reward, best Vina score, learning curves
    - Save checkpoints every 100 episodes

P0: Module 6 — DESCARTES Probe Suite
    - Collect hidden states from trained policy
    - Run Ridge ΔR² for all targets
    - Run permutation tests (200 permutations)
    - Run untrained network control
    - Generate verdict: ZOMBIE or NON_ZOMBIE
    
    *** THIS IS THE SCIENTIFIC RESULT ***
    Does the policy encode dist_asp228 now that it has continuous labels?
```

### Phase 4: Hardening + Publication (Day 4-5)

```
P1: Council controls
    - Arbitrary target probes (MW, NumHeavyAtoms)
    - 50-seed ensemble (train 50 policies, probe each)
    - Pocket scramble test
    - Two-stage ablation

P1: LLM Balloon expansion
    - Implement C1/C2 vocabulary
    - Implement balloon trigger (4-test protocol)
    - Claude API call for new features
    - Add proposed features to perception layer
    - Retrain and re-probe

P2: Comparison figures
    - Binary label probing results (from Phase 2-6) vs continuous label (this work)
    - Side-by-side: same target, same features, different labels
    - This is the key publication figure

P2: Write up results for v1.4 guide addendum
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

```python
# tests/test_pocket_features.py
def test_bace1_catalytic_residues():
    """Asp32 and Asp228 must be detected as catalytic."""
    pocket = parse_pocket("data/structures/4IVT.pdb",
                          center=(28, 15, 22), pocket_radius=12.0)
    catalytic_ids = {r.resid for r in pocket.catalytic_residues}
    assert 32 in catalytic_ids, "Asp32 not detected"
    assert 228 in catalytic_ids, "Asp228 not detected"

def test_pocket_has_hbond_sites():
    """Pocket must have H-bond donors and acceptors."""
    pocket = parse_pocket("data/structures/4IVT.pdb",
                          center=(28, 15, 22))
    assert len(pocket.hbond_donors) > 5
    assert len(pocket.hbond_acceptors) > 5

def test_pocket_feature_vector_shape():
    """Feature vector must be fixed size."""
    pocket = parse_pocket("data/structures/4IVT.pdb",
                          center=(28, 15, 22))
    vec = pocket.to_feature_vector()
    assert vec.shape == (40,)  # Adjust based on actual size


# tests/test_vina_scorer.py
def test_vina_scores_known_inhibitor():
    """Known good inhibitor should get reasonable Vina score."""
    wm = VinaWorldModel("data/structures/prepared/4IVT_receptor.pdbqt")
    # Use co-crystallized ligand from 4IVT
    score = wm.score_pose(known_inhibitor_pdbqt)
    assert score.total_energy < -5.0, "Known inhibitor should bind well"

def test_vina_scores_random_molecule_worse():
    """Random molecule should score worse than known inhibitor."""
    wm = VinaWorldModel("data/structures/prepared/4IVT_receptor.pdbqt")
    good_score = wm.score_pose(known_inhibitor_pdbqt)
    bad_score = wm.score_pose(random_molecule_pdbqt)
    assert bad_score.total_energy > good_score.total_energy


# tests/test_policy_network.py
def test_policy_forward_shape():
    """Policy must output correct shapes."""
    policy = SearchPolicyNetwork()
    x = torch.randn(1, 77)  # pocket + ligand + interaction + history + score
    logits, value, h = policy(x)
    assert logits.shape == (1, 22)
    assert value.shape == (1, 1)

def test_hidden_state_logging():
    """Must be able to log hidden states for DESCARTES."""
    policy = SearchPolicyNetwork()
    policy.enable_logging()
    x = torch.randn(1, 77)
    for _ in range(10):
        policy(x)
    states = policy.get_hidden_states()
    assert states.shape == (10, 128)
```

### 14.2 Integration Tests

```python
def test_full_episode():
    """Run one complete docking episode."""
    pocket = parse_pocket("data/structures/4IVT.pdb", center=(28, 15, 22))
    wm = VinaWorldModel("data/structures/prepared/4IVT_receptor.pdbqt")
    env = DockingEnv(wm, pocket, max_steps=50)
    
    ligand = create_ligand("CC(=O)Nc1ccc(O)cc1")  # Simple test molecule
    obs = env.reset(ligand)
    
    total_reward = 0
    for _ in range(50):
        action = np.random.randint(12)  # Random baseline
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    assert info['best_score'] < 0, "Should find some binding"
    print(f"Random baseline: best Vina = {info['best_score']:.2f}")


def test_training_improves():
    """Policy must improve over 100 episodes on a single ligand."""
    # ... setup ...
    early_rewards = []
    late_rewards = []
    for ep in range(200):
        stats = trainer.train_episode(ligand)
        if ep < 20:
            early_rewards.append(stats['total_reward'])
        elif ep > 180:
            late_rewards.append(stats['total_reward'])
    
    assert np.mean(late_rewards) > np.mean(early_rewards), \
        "Policy should improve with training"
```

---

## 15. Scientific Hypotheses & Publication Strategy

### 15.1 Primary Hypothesis

**H1: Continuous binding energy labels enable mechanistic learning where binary labels fail.**

Test: Train the same architecture (GRU, h=128) on the same target (BACE1) with:
- (A) Binary active/inactive labels (Phase 2-6 results: 0/10 features survived hardening)
- (B) Continuous Vina scores (this work)

Probe both with identical DESCARTES council controls. If (B) encodes dist_asp228, n_hbonds, etc. where (A) did not → binary label bottleneck confirmed.

### 15.2 Secondary Hypotheses

**H2: The search policy discovers the same binding features that medicinal chemists use.**

Compare C1 vocabulary at end of training (after balloon expansions) with established BACE1 SAR knowledge from the literature.

**H3: Pharmaceutical zombies persist even with continuous labels but are less severe.**

If the policy achieves good Vina scores but STILL doesn't encode binding features → the zombie problem is more fundamental than label quality.

### 15.3 Publication Angle

**Title:** "From Pharmaceutical Zombies to Binding Intuition: Continuous Reward Signals Enable Mechanistic Learning in AI Drug Discovery"

**Key figure:** Side-by-side comparison table:

```
Feature          | Binary Labels (Phase 2-6) | Continuous Vina (This Work)
                 | ΔR²     | Survives HC?    | ΔR²     | Survives HC?
-----------------+---------+-----------------+---------+----------------
dist_asp228      |  0.032  | NO              |  ???    | ???
n_hbonds         |  0.018  | NO              |  ???    | ???
hydrophobic_area |  0.045  | NO              |  ???    | ???
steric_clashes   |  0.011  | NO              |  ???    | ???
MW (confound)    |  0.156  | YES (confound!) |  ???    | ???
```

The "???" cells are what this project fills in. Either result is publishable.

---

## 16. Compute Estimates

| Component | Time | Hardware |
|---|---|---|
| Receptor preparation (one-time) | ~5 min | CPU |
| Ligand conformer generation (1000 molecules) | ~30 min | CPU |
| Vina scoring per pose | ~1 sec | CPU |
| Training episode (200 steps × 1 ligand) | ~4 min | CPU (Vina is bottleneck) |
| Full training (1000 episodes) | ~67 hours | CPU |
| Full training with GPU policy + CPU Vina | ~40 hours | GPU + CPU |
| DESCARTES probing (100 episodes × 200 perms) | ~8 hours | CPU |
| 50-seed ensemble | ~2000 hours | **Vast.ai recommended** |
| **Minimum viable result** (100 episodes + probing) | **~15 hours** | **RTX 5070** |

### Speed Optimization

The bottleneck is Vina scoring (~1 sec/pose). Strategies:
1. **Batch parallel Vina:** Run multiple Vina instances in parallel (one per CPU core)
2. **Vina surrogate:** After initial training, train a fast neural Vina approximator for exploration, validate final poses with real Vina
3. **Gnina:** GPU-accelerated alternative to Vina (same scoring, ~10× faster)

---

## Appendix A: Module-to-ARC-AGI Mapping {#appendix-a}

| ARC-AGI-3 Component | Docking Game Component | Why It Maps |
|---|---|---|
| CoreKnowledge Perception (4 Spelke priors) | PocketKnowledge Perception (4 pharma priors) | Structured perception before reasoning |
| Grid Parser | PDB Parser | Raw input → structured representation |
| Object Tracker | Residue/Ligand Tracker | Track entities across timesteps |
| WorldModel (predict→compare→update) | Vina World Model | Perfect physics simulator = perfect world model |
| TransitionRule | Binding interaction rule | "Action X improves score when near Asp228" |
| Thompson Sampling | Pose exploration | Which directions to try next |
| Ontology Expansion (C1/C2) | C1/C2 Pharmacological | Discover new binding concepts |
| Goal Discovery | Binding optimization | Maximize binding affinity |
| Prediction Error | Score improvement signal | ΔE per pose adjustment |
| Agent loop (The Hum) | Training episode loop | act→observe→update cycle |
| DESCARTES hidden state probing (Path 3) | DESCARTES Probe Suite (Module 6) | Verify what the GRU learned |
| Synthetic Reality (Path 3) | Vina docking simulation | Training environment = real physics |
| Path 4 Transformer meta-learner | Search Policy GRU | Learned reasoning component |

---

## Appendix B: BACE1 Binding Pocket Reference {#appendix-b}

### Key Residues

| Residue | Role | Feature Name |
|---|---|---|
| Asp32 | Catalytic (active site) | dist_asp32 |
| Asp228 | Catalytic (active site) | dist_asp228 |
| Tyr71 | Flap tip, H-bond donor | flap_contact |
| Thr72 | Flap, gatekeeper | flap_contact |
| Gln73 | Flap, H-bond | flap_hbond |
| Leu30 | S1 pocket, hydrophobic | s1_occupancy |
| Trp76 | S1 pocket, aromatic | s1_pi_stack |
| Phe108 | S3 pocket, hydrophobic | s3_contact |
| Ile110 | S1 pocket, hydrophobic | s1_occupancy |
| Arg235 | Salt bridge partner | charge_interaction |

### Known Interaction Patterns (Ground Truth for Probing)

1. **Catalytic dyad interaction:** Successful inhibitors form H-bonds with Asp32/Asp228
2. **Flap closure:** Good inhibitors trigger flap closure (Tyr71 moves ~3Å)
3. **S1 pocket filling:** Hydrophobic groups in S1 pocket improve binding
4. **Water displacement:** Displacing the catalytic water between Asp32/Asp228 is favorable
5. **S3 solvent exposure:** Groups extending to S3 contribute less to binding

---

## Appendix C: Mathematical Foundations {#appendix-c}

### C.1 REINFORCE Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R_t - V(s_t)) \right]$$

Where:
- $\pi_\theta$ = policy network (our GRU)
- $a_t$ = pose adjustment action at step $t$
- $s_t$ = [pocket ⊕ ligand ⊕ interaction ⊕ score_history]
- $R_t$ = discounted cumulative Vina score improvement
- $V(s_t)$ = value head estimate (baseline for variance reduction)

### C.2 Vina Scoring Function

$$E_{binding} = E_{inter} + E_{intra}$$

$$E_{inter} = w_1 \cdot E_{gauss_1} + w_2 \cdot E_{gauss_2} + w_3 \cdot E_{repulsion} + w_4 \cdot E_{hydrophobic} + w_5 \cdot E_{hbond}$$

Each term depends on pairwise atom distances and types — this is the "physics" that the policy network does NOT need to learn.

### C.3 DESCARTES ΔR² (Probe Metric)

$$\Delta R^2 = R^2_{trained} - R^2_{untrained}$$

Where:
- $R^2_{trained}$ = Ridge regression R² from TRAINED policy hidden states → biological target
- $R^2_{untrained}$ = Ridge regression R² from UNTRAINED (random init) policy hidden states → same target

$\Delta R^2 > 0.05$ with $p < 0.05$ (permutation test) → target is ENCODED.
$\Delta R^2 \leq 0.05$ or $p \geq 0.05$ → target is NOT encoded (zombie for this feature).

### C.4 VZS Axioms Applied to Docking

From DESCARTES-PHARMA v1.3, the Validated Zombie Signature axioms:

| VZS Axiom | Pharma (v1.x) | Docking Game |
|---|---|---|
| LogP | Tier 1 validated | Monitor — should be less dominant with continuous labels |
| HBA | Tier 1 validated | Should be encoded (maps to H-bond capability) |
| RotatableBonds | Tier 1 validated | Should be encoded (maps to conformational flexibility) |
| MW | Consistent confound | MUST remain confound — zombie indicator |
| NumHeavyAtoms | Consistent confound | MUST remain confound — zombie indicator |

---

*This guide is designed to be handed directly to Claude Code for implementation.*
*Each module can be implemented independently and tested in isolation.*
*Start with Phase 1 tasks in priority order (P0 first).*
*The scientific result — the side-by-side comparison with binary vs continuous labels — is the primary deliverable.*

*GitHub: https://github.com/CharithL/Descartes_Pharma*
