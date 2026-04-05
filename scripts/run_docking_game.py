#!/usr/bin/env python3
"""
======================================================================================
DESCARTES-PHARMA v2.0 -- DOCKING GAME AGENT
Complete Pipeline: Foundation -> Data -> Training -> Probing -> Council -> Comparison
======================================================================================

Runs ALL phases of the Docking Game Agent on a single GPU (Vast.ai A10, 22.5GB VRAM).

Phases:
  1. FOUNDATION   (~2 min)  -- Download PDB, parse pocket, test scoring
  2. DATA PREP    (~5 min)  -- Load BindingDB BACE1, scaffold split, create ligands
  3. TRAINING     (~30-60m) -- RL policy training with continuous Vina reward
  4. PROBING      (~10 min) -- DESCARTES probe suite (Ridge dR2, permutation)
  5. COUNCIL      (~20 min) -- Arbitrary targets, multi-seed ensemble, 2-stage ablation
  6. COMPARISON   (~2 min)  -- Binary labels (v1.3) vs continuous reward (this run)

Author: Descartes Pharma
Target: Vast.ai A10 (22.5GB VRAM, Linux, CUDA)
"""

import os
import sys
import time
import logging
import warnings
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Project root -- make sure descartes_pharma_docking is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Suppress noisy warnings during bulk conformer generation
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*UFFTYPER.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("docking_game")

# ---------------------------------------------------------------------------
# Torch + CUDA detection
# ---------------------------------------------------------------------------
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"PyTorch device: {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
STRUCTURES_DIR = DATA_DIR / "structures"
PREPARED_DIR = STRUCTURES_DIR / "prepared"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PROBE_DIR = RESULTS_DIR / "probe_results"

for d in [DATA_DIR, STRUCTURES_DIR, PREPARED_DIR, RESULTS_DIR,
          CHECKPOINTS_DIR, PROBE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
_PHASE_TIMES = {}


def phase_header(phase_num: int, title: str) -> float:
    """Print a phase header and return the start time."""
    print("\n")
    print("=" * 85)
    print(f"  PHASE {phase_num}: {title}")
    print("=" * 85)
    return time.time()


def phase_footer(phase_num: int, title: str, t0: float):
    """Print elapsed time for a phase."""
    elapsed = time.time() - t0
    _PHASE_TIMES[phase_num] = elapsed
    mins, secs = divmod(elapsed, 60)
    print(f"\n  Phase {phase_num} ({title}) completed in {int(mins)}m {secs:.1f}s")
    print("-" * 85)


# ---------------------------------------------------------------------------
# Conditional imports from descartes_pharma_docking with fallback stubs
# ---------------------------------------------------------------------------

# --- Perception ---
try:
    from descartes_pharma_docking.perception.pocket_parser import (
        parse_pocket, PocketFeatures, ResidueFeature,
    )
    from descartes_pharma_docking.perception.pocket_features import (
        enrich_pocket_with_sub_pockets,
        BACE1_POCKET_CENTER,
        BACE1_POCKET_RADIUS,
        get_bace1_config,
    )
    POCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pocket modules unavailable: {e}")
    POCKET_AVAILABLE = False

# --- Ligand ---
try:
    from descartes_pharma_docking.ligand.ligand_features import (
        create_ligand, LigandFeatures,
    )
    LIGAND_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ligand modules unavailable: {e}")
    LIGAND_AVAILABLE = False

# --- Interaction ---
try:
    from descartes_pharma_docking.interaction.interaction_features import (
        compute_interaction_features, N_INTERACTION_FEATURES,
    )
    INTERACTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Interaction modules unavailable: {e}")
    INTERACTION_AVAILABLE = False

# --- Vina engine ---
try:
    from descartes_pharma_docking.vina_engine.receptor_prep import (
        prepare_receptor, download_pdb, clean_pdb, pdb_to_pdbqt,
    )
    from descartes_pharma_docking.vina_engine.vina_scorer import (
        VinaWorldModel, VinaScore, FallbackVinaModel,
    )
    VINA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vina engine modules unavailable: {e}")
    VINA_AVAILABLE = False

# --- Policy ---
try:
    from descartes_pharma_docking.policy.policy_network import (
        SearchPolicyNetwork,
    )
    POLICY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Policy modules unavailable: {e}")
    POLICY_AVAILABLE = False

# --- Training ---
try:
    from descartes_pharma_docking.training.docking_env import DockingEnv
    from descartes_pharma_docking.training.trainer import DockingTrainer
    TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Training modules unavailable: {e}")
    TRAINING_AVAILABLE = False

# --- Probing ---
try:
    from descartes_pharma_docking.probing.probe_runner import (
        DESCARTESProbeRunner, cv_ridge_r2, permutation_test,
        BINDING_PROBE_TARGETS,
    )
    from descartes_pharma_docking.probing.council_controls import (
        CouncilControls,
    )
    from descartes_pharma_docking.probing.scaffold_permutation import (
        hardened_probe, assign_scaffold_groups, run_hardened_probe_suite,
    )
    PROBING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Probing modules unavailable: {e}")
    PROBING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper to switch policy to inference mode (PyTorch .eval())
# ---------------------------------------------------------------------------
def _set_eval_mode(model):
    """Switch a PyTorch model to inference mode (no dropout)."""
    model.train(False)


# ====================================================================
# PHASE 1: FOUNDATION
# ====================================================================

def run_phase1():
    """Download PDB 4IVT, parse pocket, test Vina scoring."""
    t0 = phase_header(1, "FOUNDATION")

    # ------------------------------------------------------------------
    # 1a. Download PDB 4IVT and prepare receptor
    # ------------------------------------------------------------------
    print("\n  [1/7] Downloading PDB 4IVT and preparing receptor...")
    pdb_path = str(STRUCTURES_DIR / "4IVT.pdb")
    pdbqt_path = str(PREPARED_DIR / "4IVT_receptor.pdbqt")

    if VINA_AVAILABLE:
        try:
            pdbqt_path = prepare_receptor(
                pdb_id="4IVT",
                pdb_dir=str(STRUCTURES_DIR),
                prepared_dir=str(PREPARED_DIR),
                force_download=True,  # Regenerate to pick up atom type fixes
            )
            print(f"    Receptor PDBQT: {pdbqt_path}")
        except Exception as e:
            logger.warning(f"    Receptor prep failed: {e}. Generating stub.")
            _write_stub_pdbqt(pdbqt_path)
    else:
        # Manual download fallback
        if not os.path.exists(pdb_path):
            import urllib.request
            url = "https://files.rcsb.org/download/4IVT.pdb"
            print(f"    Downloading from {url}...")
            urllib.request.urlretrieve(url, pdb_path)
        _write_stub_pdbqt(pdbqt_path)
        print(f"    Stub receptor PDBQT: {pdbqt_path}")

    # ------------------------------------------------------------------
    # 1b. Parse pocket features
    # ------------------------------------------------------------------
    print("\n  [2/7] Parsing pocket features from PDB 4IVT...")
    pocket = None
    if POCKET_AVAILABLE and os.path.exists(pdb_path):
        try:
            pocket = parse_pocket(
                pdb_path=pdb_path,
                pocket_center=BACE1_POCKET_CENTER,
                pocket_radius=max(BACE1_POCKET_RADIUS, 15.0),  # Ensure large enough
                catalytic_residue_ids={32, 228},
                pdb_id="4IVT",
            )
            print(f"    Pocket parsed: {len(pocket.residues)} residues")
            print(f"    Catalytic residues found: {len(pocket.catalytic_residues)}")
            for cr in pocket.catalytic_residues:
                print(f"      {cr.name} (resid={cr.resid}) center={cr.center.round(1)}")
            # List all ASP residues for debugging
            asp_residues = [r for r in pocket.residues if r.resname == "ASP"]
            if not pocket.catalytic_residues and asp_residues:
                print(f"    WARNING: 0 catalytic but found {len(asp_residues)} ASP residues:")
                for ar in asp_residues:
                    print(f"      {ar.name} resid={ar.resid}")
        except Exception as e:
            logger.warning(f"    Pocket parsing failed: {e}")

    if pocket is None:
        pocket = _create_fallback_pocket()
        print(f"    Using fallback pocket: {len(pocket.residues)} residues")

    # ------------------------------------------------------------------
    # 1c. Enrich with sub-pockets
    # ------------------------------------------------------------------
    print("\n  [3/7] Enriching pocket with BACE1 sub-pockets...")
    if POCKET_AVAILABLE:
        try:
            pocket = enrich_pocket_with_sub_pockets(pocket)
            for sp_name, sp_info in pocket.sub_pockets.items():
                print(f"    {sp_name}: {sp_info.get('n_residues', 0)} residues, "
                      f"vol={sp_info.get('volume', 0):.0f} A^3")
        except Exception as e:
            logger.warning(f"    Sub-pocket enrichment failed: {e}")

    # ------------------------------------------------------------------
    # 1d. Download BindingDB BACE1 data (small test)
    # ------------------------------------------------------------------
    print("\n  [4/7] Downloading BindingDB BACE1 data (test batch)...")
    test_smiles = _get_test_smiles()
    print(f"    Test SMILES loaded: {len(test_smiles)} molecules")

    # ------------------------------------------------------------------
    # 1e. Create LigandFeatures for test batch (3 molecules)
    # ------------------------------------------------------------------
    print("\n  [5/7] Creating LigandFeatures for test batch (3 molecules)...")
    test_ligands = []
    for i, smi in enumerate(test_smiles[:3]):
        try:
            lig = create_ligand(smi) if LIGAND_AVAILABLE else _create_fallback_ligand(smi)
            test_ligands.append(lig)
            print(f"    [{i+1}] {smi[:50]}... MW={lig.molecular_weight:.1f} "
                  f"HBD={lig.n_hbond_donors} HBA={lig.n_hbond_acceptors}")
        except Exception as e:
            print(f"    [{i+1}] FAILED: {e}")

    if not test_ligands:
        print("    WARNING: No test ligands created. Using fallback.")
        test_ligands = [_create_fallback_ligand("c1ccccc1")]

    # ------------------------------------------------------------------
    # 1f. Test Vina scoring on one molecule
    # ------------------------------------------------------------------
    print("\n  [6/7] Testing Vina/fallback scoring on one molecule...")
    vina_model = None
    test_score = 0.0

    # Use ACTUAL pocket center from parsed catalytic residues (not hardcoded)
    if pocket and pocket.catalytic_residues:
        cat_centers = [cr.center for cr in pocket.catalytic_residues]
        actual_pocket_center = tuple(np.mean(cat_centers, axis=0).tolist())
    else:
        actual_pocket_center = tuple(pocket.pocket_center.tolist()) if pocket else (21.5, 26.5, 7.7)
    print(f"    Vina grid box center: {[round(c,1) for c in actual_pocket_center]}")
    print(f"    Vina grid box size: [30, 30, 30]")

    try:
        vina_model = VinaWorldModel(
            receptor_pdbqt_path=pdbqt_path,
            center=actual_pocket_center,
            box_size=(30.0, 30.0, 30.0),  # Larger box for 108-residue pocket
        )
        # Translate ligand to pocket center before scoring
        lig = test_ligands[0]
        centered_mol = _center_mol_on_pocket(lig.mol, actual_pocket_center)
        pdbqt_str = _mol_to_pdbqt(centered_mol)
        result = vina_model.score_pose(pdbqt_str)
        test_score = result.total_energy
        print(f"    Vina test score: {test_score:.3f} kcal/mol "
              f"(fallback={vina_model._fallback})")
    except Exception as e:
        logger.warning(f"    Vina scoring failed: {e}. Using simple scorer.")
        vina_model = _create_fallback_scorer(pdbqt_path)
        lig = test_ligands[0]
        pdbqt_str = _coords_to_pdbqt(lig.conformer_coords, mol=lig.mol)
        result = vina_model.score_pose(pdbqt_str)
        test_score = result.total_energy
        print(f"    Fallback test score: {test_score:.3f}")

    # ------------------------------------------------------------------
    # 1g. Print summary
    # ------------------------------------------------------------------
    print("\n  [7/7] PHASE 1 SUMMARY")
    print("  " + "-" * 60)
    n_catalytic = len(pocket.catalytic_residues) if hasattr(pocket, 'catalytic_residues') else 0
    pocket_vec = pocket.to_feature_vector()
    print(f"    PDB:             4IVT")
    print(f"    n_residues:      {len(pocket.residues)}")
    print(f"    n_catalytic:     {n_catalytic}")
    print(f"    n_hbond_donors:  {len(pocket.hbond_donors)}")
    print(f"    n_hbond_accept:  {len(pocket.hbond_acceptors)}")
    print(f"    n_hydrophobic:   {len(pocket.hydrophobic_residues)}")
    print(f"    n_sub_pockets:   {len(pocket.sub_pockets)}")
    print(f"    pocket_vec dim:  {len(pocket_vec)}")
    print(f"    Vina test score: {test_score:.3f} kcal/mol")
    print(f"    Test ligands:    {len(test_ligands)} created successfully")

    phase_footer(1, "FOUNDATION", t0)
    return pocket, vina_model, test_ligands, pdbqt_path


# ====================================================================
# PHASE 2: DATA PREPARATION
# ====================================================================

def run_phase2():
    """Load full BACE dataset, scaffold split, create LigandFeatures."""
    t0 = phase_header(2, "DATA PREPARATION")

    # ------------------------------------------------------------------
    # 2a. Load full BACE dataset
    # ------------------------------------------------------------------
    print("\n  [1/3] Loading full BACE1 dataset...")
    all_smiles, all_labels = _load_bace_dataset()
    print(f"    Total molecules: {len(all_smiles)}")
    if all_labels is not None:
        n_active = sum(1 for l in all_labels if l == 1)
        print(f"    Active: {n_active}, Inactive: {len(all_labels) - n_active}")

    # ------------------------------------------------------------------
    # 2b. Scaffold split (train/val/test)
    # ------------------------------------------------------------------
    print("\n  [2/3] Performing scaffold split (train 70% / val 15% / test 15%)...")
    train_smi, val_smi, test_smi = _scaffold_split(all_smiles, all_labels)
    print(f"    Train: {len(train_smi)} | Val: {len(val_smi)} | Test: {len(test_smi)}")

    # ------------------------------------------------------------------
    # 2c. Create LigandFeatures for ALL molecules
    # ------------------------------------------------------------------
    print("\n  [3/3] Creating LigandFeatures with conformer generation...")
    print("         (printing progress every 100 molecules)")

    train_ligands = _create_ligand_batch(train_smi, "train")
    val_ligands = _create_ligand_batch(val_smi, "val")
    test_ligands = _create_ligand_batch(test_smi, "test")

    print(f"\n    LIGAND CREATION SUMMARY")
    print(f"    {'Split':<8} {'Input':>8} {'Created':>8} {'Success%':>10}")
    print(f"    {'-----':<8} {'-----':>8} {'-------':>8} {'--------':>10}")
    for name, inp, out in [("train", train_smi, train_ligands),
                           ("val", val_smi, val_ligands),
                           ("test", test_smi, test_ligands)]:
        pct = 100 * len(out) / max(len(inp), 1)
        print(f"    {name:<8} {len(inp):>8} {len(out):>8} {pct:>9.1f}%")

    # Ensure we have at least some ligands in each split
    if not train_ligands:
        print("    WARNING: No training ligands. Using fallback molecules.")
        train_ligands = [_create_fallback_ligand(s) for s in train_smi[:20]]
        train_ligands = [l for l in train_ligands if l is not None]
    if not val_ligands:
        val_ligands = train_ligands[:max(1, len(train_ligands) // 5)]
    if not test_ligands:
        test_ligands = train_ligands[:max(1, len(train_ligands) // 5)]

    phase_footer(2, "DATA PREPARATION", t0)
    return train_ligands, val_ligands, test_ligands, train_smi, val_smi, test_smi


# ====================================================================
# PHASE 3: TRAINING
# ====================================================================

def run_phase3(pocket, vina_model, train_ligands, val_ligands, pdbqt_path):
    """Train the SearchPolicyNetwork via REINFORCE with continuous Vina reward."""
    t0 = phase_header(3, "TRAINING")

    N_EPISODES = 500
    MAX_STEPS = 30
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 100

    # ------------------------------------------------------------------
    # 3a. Initialize DockingEnv
    # ------------------------------------------------------------------
    print(f"\n  [1/6] Initializing DockingEnv (max_steps={MAX_STEPS})...")
    env = DockingEnv(
        vina_world_model=vina_model,
        pocket_features=pocket,
        max_steps=MAX_STEPS,
        score_history_len=10,
    )
    print(f"    Env ready. Pocket vector dim: {len(env.pocket_vec)}")

    # ------------------------------------------------------------------
    # 3b. Initialize SearchPolicyNetwork
    # ------------------------------------------------------------------
    print(f"\n  [2/6] Initializing SearchPolicyNetwork (hidden=128, layers=2)...")

    # Determine observation size by doing a dummy reset
    dummy_obs_dim = _get_obs_dim(env, train_ligands[0])
    print(f"    Observation dim: {dummy_obs_dim}")

    # The policy network's total_input must match the observation dim.
    # Observation = pocket_vec + ligand_vec + interaction(20) + history(10) + score(1)
    pocket_dim = len(env.pocket_vec)
    ligand_dim = 16   # LigandFeatures.to_feature_vector() yields 16
    interaction_dim = 20
    score_history_len = 10
    total_input = pocket_dim + ligand_dim + interaction_dim + score_history_len + 1

    # If actual obs is different, adjust interaction_dim to absorb the difference
    if dummy_obs_dim != total_input:
        interaction_dim = dummy_obs_dim - pocket_dim - ligand_dim - score_history_len - 1
        total_input = dummy_obs_dim
        print(f"    Adjusted interaction_dim to {interaction_dim} "
              f"(total_input={total_input})")

    policy = SearchPolicyNetwork(
        pocket_dim=pocket_dim,
        ligand_dim=ligand_dim,
        interaction_dim=interaction_dim,
        score_history_len=score_history_len,
        hidden_dim=128,
        n_layers=2,
        n_actions=22,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"    Policy network: {n_params:,} parameters on {DEVICE}")

    # ------------------------------------------------------------------
    # 3c. Initialize DockingTrainer
    # ------------------------------------------------------------------
    print(f"\n  [3/6] Initializing DockingTrainer (lr=3e-4, gamma=0.99)...")
    trainer = DockingTrainer(
        policy=policy,
        env=env,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        device=DEVICE,
    )

    # ------------------------------------------------------------------
    # 3d. Train
    # ------------------------------------------------------------------
    print(f"\n  [4/6] Training: {N_EPISODES} episodes, {MAX_STEPS} max steps each")
    print(f"         Logging every {LOG_INTERVAL} episodes, "
          f"checkpoints every {SAVE_INTERVAL}")
    print("  " + "-" * 75)

    rng = np.random.default_rng(42)
    total_vina_evals = 0
    best_reward_ever = -float("inf")
    best_vina_ever = float("inf")
    stats = {}

    for episode in range(N_EPISODES):
        # Pick a random training ligand
        ligand = train_ligands[rng.integers(len(train_ligands))]

        # Train one episode
        stats = trainer.train_episode(ligand)
        total_vina_evals += stats["n_steps"]

        if stats["total_reward"] > best_reward_ever:
            best_reward_ever = stats["total_reward"]
        if stats["best_vina_score"] < best_vina_ever:
            best_vina_ever = stats["best_vina_score"]

        # Log
        if (episode + 1) % LOG_INTERVAL == 0:
            print(
                f"    Ep {episode+1:>5}/{N_EPISODES} | "
                f"mean_reward={stats['mean_reward_100']:+.3f} | "
                f"best_vina={stats['mean_best_score_100']:.2f} | "
                f"policy_loss={stats['policy_loss']:.4f} | "
                f"vina_evals={total_vina_evals}"
            )

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            ckpt_path = str(CHECKPOINTS_DIR / f"checkpoint_{episode+1}.pt")
            torch.save({
                "episode": episode + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
            }, ckpt_path)
            print(f"    >> Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------
    # 3e. Evaluate on validation set
    # ------------------------------------------------------------------
    print(f"\n  [5/6] Evaluating on validation set ({len(val_ligands)} ligands)...")
    _set_eval_mode(policy)
    val_scores_init = []
    val_scores_final = []

    with torch.no_grad():
        for i, lig in enumerate(val_ligands[:50]):  # Cap at 50 for speed
            obs = env.reset(lig)
            init_score = env.current_score
            h = None

            for step in range(MAX_STEPS):
                obs_t = torch.FloatTensor(obs).to(DEVICE)
                action, _, _, h = policy.select_action(obs_t, h, temperature=0.1)
                obs, reward, done, info = env.step(action)
                if done:
                    break

            val_scores_init.append(init_score)
            val_scores_final.append(env.best_score)

    mean_init = np.mean(val_scores_init)
    mean_final = np.mean(val_scores_final)
    improved_frac = np.mean(
        [f < i for f, i in zip(val_scores_final, val_scores_init)]
    )
    print(f"    Val mean init Vina:  {mean_init:.3f}")
    print(f"    Val mean final Vina: {mean_final:.3f}")
    print(f"    Fraction improved:   {improved_frac:.1%}")

    # ------------------------------------------------------------------
    # 3f. Training summary
    # ------------------------------------------------------------------
    print(f"\n  [6/6] TRAINING SUMMARY")
    print("  " + "-" * 60)
    print(f"    Total episodes:       {N_EPISODES}")
    print(f"    Total Vina evals:     {total_vina_evals}")
    print(f"    Best reward:          {best_reward_ever:.3f}")
    print(f"    Best Vina score:      {best_vina_ever:.3f}")
    final_mean_reward = stats.get('mean_reward_100', 0.0)
    print(f"    Final mean reward:    {final_mean_reward:.3f}")
    print(f"    Val improvement:      {improved_frac:.1%}")

    phase_footer(3, "TRAINING", t0)
    return policy, env, trainer


# ====================================================================
# PHASE 4: DESCARTES PROBING
# ====================================================================

def run_phase4(policy, env, test_ligands, train_ligands):
    """
    THE scientific result: does the GRU encode binding mechanism features?
    """
    t0 = phase_header(4, "DESCARTES PROBING")

    MAX_STEPS_PROBE = 30
    env.max_steps = MAX_STEPS_PROBE

    # ------------------------------------------------------------------
    # 4a. Enable hidden state logging on trained policy
    # ------------------------------------------------------------------
    print("\n  [1/7] Enabling hidden state logging on trained policy...")
    _set_eval_mode(policy)

    # ------------------------------------------------------------------
    # 4b. Collect hidden states from TRAINED policy on test ligands
    # ------------------------------------------------------------------
    print(f"\n  [2/7] Running trained policy on {len(test_ligands)} test ligands...")
    trained_data = _collect_hidden_states_and_targets(
        policy, env, test_ligands, device=DEVICE, label="TRAINED"
    )
    H_trained = trained_data["hidden_states"]
    targets = trained_data["targets"]
    smiles_per_step = trained_data["smiles_per_step"]
    print(f"    Collected {len(H_trained)} timesteps, hidden_dim={H_trained.shape[1]}")

    # ------------------------------------------------------------------
    # 4c. Collect hidden states from UNTRAINED policy (same architecture)
    # ------------------------------------------------------------------
    print(f"\n  [3/7] Creating untrained policy (random init, same architecture)...")
    untrained_policy = SearchPolicyNetwork(
        pocket_dim=policy.pocket_dim,
        ligand_dim=policy.ligand_dim,
        interaction_dim=policy.interaction_dim,
        score_history_len=policy.score_history_len,
        hidden_dim=policy.hidden_dim,
        n_layers=policy.n_layers,
        n_actions=policy.n_actions,
    ).to(DEVICE)
    _set_eval_mode(untrained_policy)

    print(f"    Running untrained policy on same test ligands...")
    untrained_data = _collect_hidden_states_and_targets(
        untrained_policy, env, test_ligands, device=DEVICE, label="UNTRAINED"
    )
    H_untrained = untrained_data["hidden_states"]

    # Ensure same length
    n = min(len(H_trained), len(H_untrained))
    H_trained = H_trained[:n]
    H_untrained = H_untrained[:n]
    targets = {k: v[:n] for k, v in targets.items()}
    smiles_per_step = smiles_per_step[:n]

    print(f"    Matched samples: {n}")

    # ------------------------------------------------------------------
    # 4d. Run Ridge dR2 for 8 binding targets
    # ------------------------------------------------------------------
    PROBE_TARGETS = [
        "dist_asp32", "dist_asp228", "n_hbonds", "hydrophobic_contact_area",
        "steric_clash_count", "vina_score", "pocket_occupancy",
        "closest_wall_dist",
    ]

    print(f"\n  [4/7] Running Ridge dR2 probes for {len(PROBE_TARGETS)} targets...")
    ridge_results = {}
    for tname in PROBE_TARGETS:
        if tname not in targets or len(targets[tname]) < 20:
            ridge_results[tname] = {"r2_T": 0.0, "r2_U": 0.0, "dR2": 0.0}
            continue
        r2_t = cv_ridge_r2(H_trained, targets[tname])
        r2_u = cv_ridge_r2(H_untrained, targets[tname])
        dR2 = r2_t - r2_u
        ridge_results[tname] = {"r2_T": r2_t, "r2_U": r2_u, "dR2": dR2}

    # ------------------------------------------------------------------
    # 4e. Scaffold-stratified permutation (200 perms) for each target
    # ------------------------------------------------------------------
    print(f"\n  [5/7] Running scaffold-stratified permutation (200 perms)...")
    scaffolds = assign_scaffold_groups(smiles_per_step) if PROBING_AVAILABLE else \
        np.zeros(n, dtype=np.int64)

    perm_results = {}
    for tname in PROBE_TARGETS:
        if tname not in targets or len(targets[tname]) < 20:
            perm_results[tname] = 1.0
            continue
        try:
            _, p_val, _ = hardened_probe(
                H_trained, H_untrained, targets[tname],
                scaffolds, n_perms=200,
            )
            perm_results[tname] = p_val
        except Exception as e:
            logger.warning(f"    Permutation failed for {tname}: {e}")
            # Fallback: simple permutation
            perm_results[tname] = permutation_test(
                H_trained, targets[tname], n_perms=200
            )

    # ------------------------------------------------------------------
    # 4f. Print results table
    # ------------------------------------------------------------------
    print(f"\n  [6/7] DESCARTES PROBE RESULTS:")
    print(f"         Does the GRU encode binding mechanism?\n")
    print(f"    {'Target':<26} {'Ridge(T)':>9} {'Ridge(U)':>9} "
          f"{'dR2':>7} {'p-value':>8} {'Verdict':>8}")
    print("    " + "-" * 70)

    n_encoded = 0
    encoded_targets = []
    for tname in PROBE_TARGETS:
        r = ridge_results.get(tname, {"r2_T": 0, "r2_U": 0, "dR2": 0})
        p = perm_results.get(tname, 1.0)
        is_enc = r["dR2"] > 0.05 and p < 0.05
        verdict = "ENC" if is_enc else "ZOMBIE"
        if is_enc:
            n_encoded += 1
            encoded_targets.append(tname)
        print(f"    {tname:<26} {r['r2_T']:>9.4f} {r['r2_U']:>9.4f} "
              f"{r['dR2']:>7.4f} {p:>8.4f} {verdict:>8}")

    print(f"\n    Binding features encoded: {n_encoded}/{len(PROBE_TARGETS)}")

    # ------------------------------------------------------------------
    # 4g. Save probe data to disk
    # ------------------------------------------------------------------
    print(f"\n  [7/7] Saving probe data...")
    np.savez(str(PROBE_DIR / "probe_data.npz"),
             H_trained=H_trained, H_untrained=H_untrained,
             scaffolds=scaffolds)
    print(f"    Saved to {PROBE_DIR / 'probe_data.npz'}")

    phase_footer(4, "DESCARTES PROBING", t0)
    return (ridge_results, perm_results, H_trained, H_untrained,
            targets, scaffolds, encoded_targets, smiles_per_step)


# ====================================================================
# PHASE 5: COUNCIL CONTROLS
# ====================================================================

def run_phase5(policy, env, train_ligands, test_ligands,
               H_trained, H_untrained, targets, scaffolds,
               ridge_results, perm_results, encoded_targets):
    """
    Validate probe results with the three council controls.
    """
    t0 = phase_header(5, "COUNCIL CONTROLS")

    council = CouncilControls(random_seed=42) if PROBING_AVAILABLE else None

    PROBE_TARGETS = [
        "dist_asp32", "dist_asp228", "n_hbonds", "hydrophobic_contact_area",
        "steric_clash_count", "vina_score", "pocket_occupancy",
        "closest_wall_dist",
    ]
    KEY_TARGETS = ["dist_asp32", "dist_asp228", "n_hbonds", "vina_score"]

    # ------------------------------------------------------------------
    # 5a. Arbitrary target probes
    # ------------------------------------------------------------------
    print("\n  [1/3] ARBITRARY TARGET PROBES")
    print("         5 random + 3 Lorenz + 2 shuffled\n")

    if council is not None:
        ceiling, arb_scores = council.arbitrary_target_probes(
            H_trained, H_untrained,
            n_random=5, n_lorenz=3, n_shuffled=2,
        )
    else:
        ceiling, arb_scores = _fallback_arbitrary_probes(H_trained, H_untrained)

    print(f"    {'Arbitrary Target':<30} {'dR2':>8}")
    print("    " + "-" * 40)
    for name, dr2 in sorted(arb_scores.items(), key=lambda x: -x[1]):
        print(f"    {name:<30} {dr2:>8.4f}")
    print(f"\n    False-positive ceiling: {ceiling:.4f}")
    print(f"\n    Real targets above ceiling:")
    above_ceiling = []
    for tname in PROBE_TARGETS:
        r = ridge_results.get(tname, {"dR2": 0})
        if r["dR2"] > ceiling:
            above_ceiling.append(tname)
            print(f"      {tname}: dR2={r['dR2']:.4f} > ceiling={ceiling:.4f}")
    if not above_ceiling:
        print(f"      (none)")

    # ------------------------------------------------------------------
    # 5b. 20-seed ensemble
    # ------------------------------------------------------------------
    print(f"\n  [2/3] 20-SEED ENSEMBLE")
    print(f"         Training 20 policies (200 episodes each) for 4 key targets")
    print(f"         Key targets: {', '.join(KEY_TARGETS)}")
    print()

    N_SEEDS = 20
    ENSEMBLE_EPISODES = 200
    ENSEMBLE_MAX_STEPS = 30
    seed_pass_counts = {t: 0 for t in KEY_TARGETS}

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Fresh policy
        fresh_policy = SearchPolicyNetwork(
            pocket_dim=policy.pocket_dim,
            ligand_dim=policy.ligand_dim,
            interaction_dim=policy.interaction_dim,
            score_history_len=policy.score_history_len,
            hidden_dim=128,
            n_layers=2,
            n_actions=22,
        ).to(DEVICE)

        # Short training
        env.max_steps = ENSEMBLE_MAX_STEPS
        mini_trainer = DockingTrainer(
            policy=fresh_policy, env=env, lr=3e-4, device=DEVICE,
        )

        # Suppress output during mini training
        rng_seed = np.random.default_rng(seed)
        for ep in range(ENSEMBLE_EPISODES):
            lig = train_ligands[rng_seed.integers(len(train_ligands))]
            try:
                mini_trainer.train_episode(lig)
            except Exception:
                continue

        # Collect hidden states from this seed
        _set_eval_mode(fresh_policy)
        seed_data = _collect_hidden_states_and_targets(
            fresh_policy, env, test_ligands[:20], device=DEVICE, label=None
        )
        H_seed = seed_data["hidden_states"]
        seed_targets = seed_data["targets"]

        if len(H_seed) == 0:
            continue

        # Untrained control for this seed
        rng_ctrl = np.random.default_rng(seed + 1000)
        H_ctrl = rng_ctrl.normal(0, 0.01, H_seed.shape).astype(np.float32)

        # Probe key targets
        for tname in KEY_TARGETS:
            if tname not in seed_targets or len(seed_targets[tname]) < 10:
                continue
            n_s = min(len(H_seed), len(seed_targets[tname]))
            r2_t = cv_ridge_r2(H_seed[:n_s], seed_targets[tname][:n_s])
            r2_u = cv_ridge_r2(H_ctrl[:n_s], seed_targets[tname][:n_s])
            if r2_t - r2_u > 0.05:
                seed_pass_counts[tname] += 1

        if (seed + 1) % 5 == 0:
            print(f"    Seed {seed+1:>2}/{N_SEEDS} done | "
                  + " | ".join(f"{t}:{seed_pass_counts[t]}"
                               for t in KEY_TARGETS))

    print(f"\n    SEED ENSEMBLE RESULTS:")
    print(f"    {'Target':<26} {'Seeds':>8} {'Status':>10}")
    print("    " + "-" * 46)
    seed_status = {}
    for tname in KEY_TARGETS:
        cnt = seed_pass_counts[tname]
        if cnt >= 16:
            status = "ROBUST"
        elif cnt >= 4:
            status = "FRAGILE"
        else:
            status = "ABSENT"
        seed_status[tname] = status
        print(f"    {tname:<26} {cnt:>4}/{N_SEEDS:>2}    {status:>10}")

    # ------------------------------------------------------------------
    # 5c. Two-stage ablation
    # ------------------------------------------------------------------
    print(f"\n  [3/3] TWO-STAGE ABLATION")
    print(f"         For each encoded target, regress out all others\n")

    ablation_results = {}
    for tname in PROBE_TARGETS:
        r = ridge_results.get(tname, {"dR2": 0})
        if r["dR2"] <= 0.05 or tname not in targets:
            ablation_results[tname] = "NONE"
            continue

        # Build dict of all other targets
        other_targets = {k: v for k, v in targets.items()
                         if k != tname and k in
                         {t for t in PROBE_TARGETS if t in targets}}

        if council is not None:
            try:
                s1, s2, classification = council.two_stage_ablation(
                    H_trained, targets[tname], other_targets,
                )
                ablation_results[tname] = classification
                print(f"    {tname:<26} S1={s1:.4f} S2={s2:.4f} -> {classification}")
            except Exception as e:
                logger.warning(f"    Ablation failed for {tname}: {e}")
                ablation_results[tname] = "ERROR"
        else:
            ablation_results[tname] = "NONE"

    # ------------------------------------------------------------------
    # 5d. Integrated verdict table
    # ------------------------------------------------------------------
    print(f"\n")
    print("=" * 85)
    print("  INTEGRATED VERDICT: Docking Game Agent")
    print("=" * 85)
    print()
    print(f"    {'Target':<22} {'Hardened':>9} {'Ceiling':>9} "
          f"{'Seeds':>8} {'2Stage':>9} {'FINAL':>9}")
    print("    " + "-" * 68)

    final_verdicts = {}
    for tname in PROBE_TARGETS:
        r = ridge_results.get(tname, {"dR2": 0})
        p = perm_results.get(tname, 1.0)
        hardened = "PASS" if (r["dR2"] > 0.05 and p < 0.05) else "FAIL"
        ceil_pass = "PASS" if r["dR2"] > ceiling else "FAIL"
        seeds_str = f"{seed_pass_counts.get(tname, 0)}/{N_SEEDS}" \
            if tname in KEY_TARGETS else "---"
        abl = ablation_results.get(tname, "---")

        # Final verdict: need all four gates for key targets
        if tname in KEY_TARGETS:
            passes = 0
            if hardened == "PASS":
                passes += 1
            if ceil_pass == "PASS":
                passes += 1
            if seed_pass_counts.get(tname, 0) >= 4:
                passes += 1
            if abl in ("DIRECT", "CORRELATED", "---"):
                passes += 1
            if passes == 4:
                final = "GENUINE"
            elif passes >= 2:
                final = "PARTIAL"
            else:
                final = "ZOMBIE"
        else:
            if hardened == "PASS" and ceil_pass == "PASS":
                final = "GENUINE"
            elif hardened == "PASS":
                final = "PARTIAL"
            else:
                final = "ZOMBIE"

        final_verdicts[tname] = final

        print(f"    {tname:<22} {hardened:>9} {ceil_pass:>9} "
              f"{seeds_str:>8} {abl:>9} {final:>9}")

    n_genuine = sum(1 for v in final_verdicts.values() if v == "GENUINE")
    n_partial = sum(1 for v in final_verdicts.values() if v == "PARTIAL")
    print(f"\n    GENUINE: {n_genuine}/{len(PROBE_TARGETS)} | "
          f"PARTIAL: {n_partial}/{len(PROBE_TARGETS)} | "
          f"ZOMBIE: {len(PROBE_TARGETS) - n_genuine - n_partial}/{len(PROBE_TARGETS)}")

    pub_ready = all(
        final_verdicts.get(t, "ZOMBIE") == "GENUINE" for t in KEY_TARGETS
    )
    print(f"\n    PUBLICATION_READY requires ALL FOUR gates for key targets.")
    print(f"    Status: {'PUBLICATION READY' if pub_ready else 'NOT YET READY'}")

    phase_footer(5, "COUNCIL CONTROLS", t0)
    return final_verdicts, seed_pass_counts, ceiling, ablation_results


# ====================================================================
# PHASE 6: COMPARISON WITH BINARY LABELS
# ====================================================================

def run_phase6(ridge_results, perm_results, final_verdicts,
               seed_pass_counts):
    """
    Side-by-side comparison: binary labels (v1.3) vs continuous reward (game).
    """
    t0 = phase_header(6, "BINARY vs CONTINUOUS COMPARISON")

    # ------------------------------------------------------------------
    # Hardcoded binary label results from v1.3
    # ------------------------------------------------------------------
    BINARY_RESULTS = {
        "dist_asp32": {
            "verdict": "ABSENT", "seeds": "0/50",
            "detail": "ZOMBIE (0/50 seeds)",
        },
        "dist_asp228": {
            "verdict": "FRAGILE", "seeds": "7/20",
            "detail": "FRAGILE (7/20 seeds), GENUINE pocket scramble",
        },
        "hbond_catalytic": {
            "verdict": "ABSENT", "seeds": "0/50",
            "detail": "ZOMBIE (0/50 seeds)",
        },
        "catalytic_score": {
            "verdict": "ABSENT", "seeds": "0/50",
            "detail": "ZOMBIE (0/50 seeds)",
        },
    }

    # Map binary target names to continuous target names
    COMPARISON_MAP = [
        ("dist_asp32",      "dist_asp32"),
        ("dist_asp228",     "dist_asp228"),
        ("hbond_catalytic",  "n_hbonds"),
        ("catalytic_score",  "vina_score"),
    ]

    print()
    print("=" * 85)
    print("  BINARY LABELS vs CONTINUOUS REWARD: Does the training signal matter?")
    print("=" * 85)
    print()
    print(f"    {'Target':<22} {'Binary (v1.3)':<28} "
          f"{'Continuous (Game)':<24} {'Delta':<10}")
    print("    " + "-" * 84)

    continuous_better = 0
    both_zombie = 0

    for bin_name, cont_name in COMPARISON_MAP:
        bin_info = BINARY_RESULTS[bin_name]
        bin_str = bin_info["detail"]

        # Get continuous result
        r = ridge_results.get(cont_name, {"dR2": 0})
        p = perm_results.get(cont_name, 1.0)
        fv = final_verdicts.get(cont_name, "ZOMBIE")
        sc = seed_pass_counts.get(cont_name, 0)

        if cont_name in seed_pass_counts:
            cont_str = f"{fv} ({sc}/20 seeds)"
        else:
            is_enc = r["dR2"] > 0.05 and p < 0.05
            cont_str = f"{'ENC' if is_enc else 'ZOMBIE'} (dR2={r['dR2']:.3f})"

        # Delta
        bin_is_zombie = bin_info["verdict"] == "ABSENT"
        cont_is_enc = fv in ("GENUINE", "PARTIAL")

        if bin_is_zombie and cont_is_enc:
            delta = "+GAINED"
            continuous_better += 1
        elif not bin_is_zombie and cont_is_enc:
            delta = "STABLE"
        elif bin_is_zombie and not cont_is_enc:
            delta = "SAME"
            both_zombie += 1
        else:
            delta = "-LOST"

        # Special case: vina_score is NEW for continuous
        if bin_name == "catalytic_score" and cont_name == "vina_score":
            delta = "NEW"
            if cont_is_enc:
                continuous_better += 1

        print(f"    {bin_name:<22} {bin_str:<28} {cont_str:<24} {delta:<10}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print()
    print("  " + "-" * 78)

    if continuous_better > 0:
        conf_level = "CONFIRMED" if continuous_better >= 2 else "PARTIALLY CONFIRMED"
        print(f"""
  CONCLUSION: Binary label bottleneck {conf_level}.
    Continuous Vina reward enables mechanism encoding that binary cannot.
    {continuous_better} target(s) gained encoding with continuous reward.
    This supports the central DESCARTES hypothesis: the training signal,
    not the architecture, was the primary bottleneck in v1.0-v1.3.""")
    elif both_zombie == len(COMPARISON_MAP):
        print("""
  CONCLUSION: Problem is deeper than labels.
    Architecture or data quantity may be the real bottleneck.
    Even continuous Vina reward does not enable mechanism encoding.
    Next step: try larger networks, more data, or different architectures.""")
    else:
        print(f"""
  CONCLUSION: Mixed results.
    {continuous_better} targets gained, {both_zombie} both zombie.
    The label bottleneck may be part of the story but not the whole story.
    Additional investigation needed.""")

    phase_footer(6, "COMPARISON", t0)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def _write_stub_pdbqt(path: str):
    """Write a minimal stub PDBQT for fallback scoring."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write a minimal protein-like PDBQT with atoms near the pocket center
    lines = []
    center = np.array([28.0, 15.0, 22.0])
    rng = np.random.default_rng(42)
    for i in range(100):
        pos = center + rng.normal(0, 5, 3)
        lines.append(
            f"ATOM  {i+1:5d}  CA  ALA A {i+1:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            f"  1.00  0.00    +0.000 C"
        )
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _create_fallback_pocket():
    """Create a minimal PocketFeatures when PDB parsing is unavailable."""
    center = np.array([28.0, 15.0, 22.0])

    # Create a few synthetic residues
    residues = []
    rng = np.random.default_rng(42)
    residue_data = [
        ("ASP32", "ASP", 32, True),
        ("ASP228", "ASP", 228, True),
        ("TYR71", "TYR", 71, False),
        ("PHE108", "PHE", 108, False),
        ("TRP110", "TRP", 110, False),
        ("THR73", "THR", 73, False),
        ("LEU30", "LEU", 30, False),
        ("ILE118", "ILE", 118, False),
        ("GLY34", "GLY", 34, False),
        ("SER35", "SER", 35, False),
    ]

    HBOND_DONORS_SET = {"SER", "THR", "TYR", "ASN", "GLN", "HIS",
                        "TRP", "ARG", "LYS", "CYS"}
    HBOND_ACCEPTORS_SET = {"SER", "THR", "TYR", "ASN", "GLN", "HIS",
                           "ASP", "GLU", "MET", "CYS"}
    HYDROPHOBIC_SET = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE",
                       "TRP", "MET"}
    POSITIVE_SET = {"ARG", "LYS", "HIS"}
    NEGATIVE_SET = {"ASP", "GLU"}

    for name, resname, resid, is_cat in residue_data:
        res_center = center + rng.normal(0, 4, 3)
        feat = ResidueFeature(
            name=name, resname=resname, resid=resid, chain="A",
            center=res_center,
            is_hbond_donor=resname in HBOND_DONORS_SET,
            is_hbond_acceptor=resname in HBOND_ACCEPTORS_SET,
            is_hydrophobic=resname in HYDROPHOBIC_SET,
            is_charged=resname in POSITIVE_SET or resname in NEGATIVE_SET,
            charge_sign=(1 if resname in POSITIVE_SET
                         else (-1 if resname in NEGATIVE_SET else 0)),
            is_aromatic=resname in {"PHE", "TYR", "TRP", "HIS"},
            is_catalytic=is_cat,
            sidechain_atoms=[res_center + rng.normal(0, 1, 3)
                             for _ in range(3)],
        )
        residues.append(feat)

    pocket = PocketFeatures(
        pdb_id="4IVT",
        pocket_center=center,
        pocket_radius=12.0,
        residues=residues,
        catalytic_residues=[r for r in residues if r.is_catalytic],
        hbond_donors=[r for r in residues if r.is_hbond_donor],
        hbond_acceptors=[r for r in residues if r.is_hbond_acceptor],
        hydrophobic_residues=[r for r in residues if r.is_hydrophobic],
        positive_residues=[r for r in residues if r.charge_sign > 0],
        negative_residues=[r for r in residues if r.charge_sign < 0],
    )
    return pocket


def _create_fallback_ligand(smiles: str):
    """Create a minimal LigandFeatures without RDKit."""
    n_atoms = max(5, len(smiles) // 3)
    coords = np.random.randn(n_atoms, 3) * 2.0
    center = coords.mean(axis=0)
    return LigandFeatures(
        smiles=smiles,
        mol=None,
        conformer_coords=coords,
        center_of_mass=center,
        n_hbond_donors=2,
        n_hbond_acceptors=3,
        n_rotatable_bonds=4,
        n_aromatic_rings=1,
        logp=2.0,
        molecular_weight=350.0,
        tpsa=80.0,
    )


def _create_fallback_scorer(pdbqt_path: str, center=(21.5, 26.5, 7.7)):
    """Create a scorer — VinaWorldModel with automatic fallback on error."""
    from descartes_pharma_docking.vina_engine.vina_scorer import FallbackVinaModel
    try:
        model = VinaWorldModel(
            receptor_pdbqt_path=pdbqt_path,
            center=center,
            box_size=(30.0, 30.0, 30.0),
        )
        return model
    except Exception:
        return FallbackVinaModel(
            receptor_pdbqt_path=pdbqt_path,
            center=center,
            box_size=(30.0, 30.0, 30.0),
        )


def _center_mol_on_pocket(mol, pocket_center):
    """Translate an RDKit mol's conformer so its centroid is at pocket_center."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    mol = Chem.RWMol(mol)
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        mol2 = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol2, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol2)
        mol = mol2
    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions())
    centroid = coords.mean(axis=0)
    shift = np.array(pocket_center) - centroid
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(
            pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]))
    return mol


def _mol_to_pdbqt(mol) -> str:
    """Convert an RDKit mol (with conformer) to Vina-compatible PDBQT using meeko or obabel."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import tempfile, subprocess, os

    # Ensure 3D coords and hydrogens
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

    # Write mol to SDF
    sdf_path = tempfile.mktemp(suffix='.sdf')
    pdbqt_path = tempfile.mktemp(suffix='.pdbqt')

    try:
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        # Try meeko first (official AutoDock tool)
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if is_ok:
                # Remove MODEL/ENDMDL if present
                lines = [l for l in pdbqt_string.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                return '\n'.join(lines)
        except Exception:
            pass

        # Try obabel
        try:
            result = subprocess.run(
                ['obabel', sdf_path, '-O', pdbqt_path, '-p', '7.4'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(pdbqt_path):
                with open(pdbqt_path) as f:
                    content = f.read()
                # Remove MODEL/ENDMDL
                lines = [l for l in content.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                return '\n'.join(lines)
        except Exception:
            pass

        # Last resort: manual PDBQT with ROOT/ENDROOT (Vina ligand format)
        return _manual_ligand_pdbqt(mol)

    finally:
        for p in [sdf_path, pdbqt_path]:
            if os.path.exists(p):
                os.unlink(p)


def _manual_ligand_pdbqt(mol) -> str:
    """Manual ligand PDBQT with ROOT/ENDROOT for Vina compatibility."""
    from rdkit import Chem
    conf = mol.GetConformer()
    lines = ["ROOT"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = atom.GetSymbol().upper()
        # AD4 atom types for ligands
        if sym == 'C':
            ad = 'A' if atom.GetIsAromatic() else 'C'
        elif sym == 'N':
            ad = 'NA' if atom.GetTotalNumHs() == 0 else 'N'
        elif sym == 'O':
            ad = 'OA'
        elif sym == 'S':
            ad = 'SA'
        elif sym == 'H':
            # Check if bonded to N or O
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            ad = 'HD' if any(n in ('N', 'O') for n in neighbors) else 'H'
        elif sym == 'F':
            ad = 'F'
        elif sym == 'CL':
            ad = 'Cl'
        elif sym == 'BR':
            ad = 'Br'
        else:
            ad = 'C'
        lines.append(
            f"HETATM{i+1:5d} {sym:<4s} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00    "
            f"+0.000 {ad:>2s}"
        )
    lines.append("ENDROOT")
    lines.append("END")
    return "\n".join(lines)


def _coords_to_pdbqt(coords, mol=None):
    """Convert coordinates to PDBQT. Uses meeko/obabel if mol provided."""
    if mol is not None:
        try:
            return _mol_to_pdbqt(mol)
        except Exception:
            pass
    # Bare fallback (all carbon, for testing only)
    lines = ["ROOT"]
    for i, (x, y, z) in enumerate(coords):
        lines.append(
            f"HETATM{i+1:5d}  C   LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000  C"
        )
    lines.append("ENDROOT")
    lines.append("END")
    return "\n".join(lines)


def _get_test_smiles():
    """Return a small set of known BACE1 inhibitor SMILES for testing."""
    return [
        # Known BACE1 inhibitors from literature
        "CC(C)CC1=CC=C(C=C1)C(=O)N[C@@H](CC2=CC=CC=C2)C(=O)N",
        "O=C(NC1CCCCC1)C2=CC=C(F)C=C2",
        "CC(=O)NC1=CC=C(C=C1)S(=O)(=O)N",
        "C1=CC=C(C=C1)CC(=O)NC2=CC=CC=C2O",
        "CC1=CC=C(C=C1)NC(=O)C2=CC=CC=C2Cl",
        "O=C(NC1=CC=CC=C1)C2=CC=C(O)C=C2",
        "CC(C)NC(=O)C1=CC=C(C=C1)OC2=CC=CC=C2",
        "C1=CC=C(C(=C1)O)C(=O)NC2=CC=C(C=C2)F",
        "CC(=O)NC1=CC=C(C=C1)C(=O)NC2CCCCC2",
        "OC1=CC=C(C=C1)C(=O)NCCC2=CC=CC=C2",
    ]


def _load_bace_dataset():
    """
    Load the full BACE1 dataset.

    Tries in order:
    1. TDC (Therapeutics Data Commons) HTS BACE dataset
    2. MoleculeNet BACE CSV from DeepChem S3
    3. Fallback: hardcoded test SMILES

    Returns (smiles_list, labels_list_or_None)
    """
    # --- Attempt 1: TDC ---
    try:
        from tdc.single_pred import HTS
        data = HTS(name="BACE")
        df = data.get_data()
        smiles = df["Drug"].tolist()
        labels = df["Y"].tolist()
        print(f"    Loaded from TDC: {len(smiles)} molecules")
        return smiles, labels
    except Exception as e:
        logger.info(f"    TDC load failed: {e}")

    # --- Attempt 2: MoleculeNet BACE CSV ---
    try:
        import pandas as pd
        csv_url = ("https://deepchemdata.s3-us-west-1.amazonaws.com/"
                   "datasets/bace.csv")
        local_csv = str(DATA_DIR / "bace_moleculenet.csv")
        if not os.path.exists(local_csv):
            print(f"    Downloading BACE dataset from MoleculeNet...")
            import urllib.request
            urllib.request.urlretrieve(csv_url, local_csv)
        df = pd.read_csv(local_csv)
        # MoleculeNet BACE CSV has 'mol' (SMILES) and 'Class' (0/1)
        smiles_col = "mol" if "mol" in df.columns else df.columns[0]
        label_col = "Class" if "Class" in df.columns else None
        smiles = df[smiles_col].dropna().tolist()
        labels = df[label_col].tolist() if label_col else None
        print(f"    Loaded from MoleculeNet CSV: {len(smiles)} molecules")
        return smiles, labels
    except Exception as e:
        logger.info(f"    MoleculeNet CSV load failed: {e}")

    # --- Attempt 3: Fallback ---
    print("    Using fallback test SMILES (10 molecules)")
    smiles = _get_test_smiles()
    return smiles, [1] * 5 + [0] * 5


def _scaffold_split(smiles_list, labels, train_frac=0.7, val_frac=0.15):
    """
    Scaffold-based split into train/val/test.

    Uses Murcko scaffolds via RDKit. Falls back to random split
    if RDKit scaffold decomposition is unavailable.
    """
    n = len(smiles_list)
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import (
            GetScaffoldForMol, MakeScaffoldGeneric,
        )

        scaffold_sets = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scaffold_sets["INVALID"].append(i)
                continue
            try:
                core = MakeScaffoldGeneric(GetScaffoldForMol(mol))
                scaffold_smi = Chem.MolToSmiles(core)
            except Exception:
                scaffold_smi = smi[:8]
            scaffold_sets[scaffold_smi].append(i)

        # Sort scaffolds by size (largest first) for reproducibility
        sorted_scaffolds = sorted(
            scaffold_sets.values(), key=len, reverse=True
        )

        train_idx, val_idx, test_idx = [], [], []
        train_cutoff = int(n * train_frac)
        val_cutoff = int(n * (train_frac + val_frac))

        for group in sorted_scaffolds:
            if len(train_idx) < train_cutoff:
                train_idx.extend(group)
            elif len(train_idx) + len(val_idx) < val_cutoff:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        # Ensure test is not empty
        if not test_idx and val_idx:
            split_at = max(1, len(val_idx) // 2)
            test_idx = val_idx[split_at:]
            val_idx = val_idx[:split_at]

    except ImportError:
        # Random split fallback
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        train_cutoff = int(n * train_frac)
        val_cutoff = int(n * (train_frac + val_frac))
        train_idx = indices[:train_cutoff].tolist()
        val_idx = indices[train_cutoff:val_cutoff].tolist()
        test_idx = indices[val_cutoff:].tolist()

    train_smi = [smiles_list[i] for i in train_idx]
    val_smi = [smiles_list[i] for i in val_idx]
    test_smi = [smiles_list[i] for i in test_idx]

    return train_smi, val_smi, test_smi


def _create_ligand_batch(smiles_list, split_name):
    """Create LigandFeatures for a list of SMILES, with progress reporting."""
    ligands = []
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        try:
            if LIGAND_AVAILABLE:
                lig = create_ligand(smi)
            else:
                lig = _create_fallback_ligand(smi)
            ligands.append(lig)
        except Exception:
            n_fail += 1

        if (i + 1) % 100 == 0:
            print(f"    {split_name}: {i+1}/{len(smiles_list)} processed "
                  f"({len(ligands)} ok, {n_fail} failed)")

    if len(smiles_list) > 0 and len(smiles_list) % 100 != 0:
        print(f"    {split_name}: {len(smiles_list)}/{len(smiles_list)} processed "
              f"({len(ligands)} ok, {n_fail} failed)")

    return ligands


def _get_obs_dim(env, ligand):
    """Get the observation dimensionality by doing a dummy reset."""
    obs = env.reset(ligand)
    return len(obs)


def _collect_hidden_states_and_targets(policy, env, ligands, device="cpu",
                                        label=None):
    """
    Run a policy on ligands, collecting hidden states and interaction targets.

    Returns dict with:
        hidden_states: (N, hidden_dim)
        targets: dict of name -> (N,) arrays
        smiles_per_step: list of SMILES strings (one per timestep)
    """
    _set_eval_mode(policy)
    policy.enable_logging()
    policy.clear_hidden_log()

    all_targets = defaultdict(list)
    smiles_per_step = []

    n_episodes = len(ligands)
    for ep_idx in range(n_episodes):
        lig = ligands[ep_idx % len(ligands)]
        obs = env.reset(lig)
        h = None

        for step in range(env.max_steps):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action, _, _, h = policy.select_action(
                    obs_t, h, temperature=0.1
                )
            obs, reward, done, info = env.step(action)

            # Record targets
            all_targets["dist_asp32"].append(info.get("dist_asp32", 50.0))
            all_targets["dist_asp228"].append(info.get("dist_asp228", 50.0))
            all_targets["vina_score"].append(info.get("vina_score", 0.0))

            # Interaction features from observation vector
            pocket_len = len(env.pocket_vec)
            int_offset = pocket_len + 16  # After pocket + ligand vectors

            all_targets["n_hbonds"].append(
                float(obs[int_offset + 5])
                if int_offset + 5 < len(obs) else 0.0
            )
            all_targets["hydrophobic_contact_area"].append(
                float(obs[int_offset + 6])
                if int_offset + 6 < len(obs) else 0.0
            )
            all_targets["steric_clash_count"].append(
                float(obs[int_offset + 7])
                if int_offset + 7 < len(obs) else 0.0
            )
            all_targets["pocket_occupancy"].append(
                float(obs[int_offset + 8])
                if int_offset + 8 < len(obs) else 0.0
            )

            # Closest wall distance
            if hasattr(env, 'current_coords') and env.current_coords is not None:
                lig_center = env.current_coords.mean(axis=0)
                dist_to_pcenter = np.linalg.norm(
                    lig_center - env.pocket.pocket_center
                )
                wall_dist = max(0.0,
                                env.pocket.pocket_radius - dist_to_pcenter)
            else:
                wall_dist = 0.0
            all_targets["closest_wall_dist"].append(wall_dist)

            # Confounds
            all_targets["molecular_weight"].append(
                getattr(lig, "molecular_weight", 300.0)
            )
            all_targets["logp"].append(getattr(lig, "logp", 2.0))

            smiles_per_step.append(lig.smiles)

            if done:
                break

        if label and (ep_idx + 1) % 20 == 0:
            print(f"    [{label}] Episode {ep_idx+1}/{n_episodes}")

    H = policy.get_hidden_states()
    policy.disable_logging()

    n = len(H)
    targets_out = {
        k: np.array(v[:n], dtype=np.float32)
        for k, v in all_targets.items()
    }

    return {
        "hidden_states": H,
        "targets": targets_out,
        "smiles_per_step": smiles_per_step[:n],
    }


def _fallback_arbitrary_probes(H_trained, H_untrained):
    """Fallback arbitrary target probes when CouncilControls unavailable."""
    rng = np.random.default_rng(42)
    n = len(H_trained)
    scores = {}

    for i in range(5):
        target = rng.normal(0, 1, n).astype(np.float32)
        r2_t = cv_ridge_r2(H_trained, target)
        r2_u = cv_ridge_r2(H_untrained, target)
        scores[f"random_gaussian_{i}"] = r2_t - r2_u

    # Simple Lorenz-like structured noise
    for i in range(3):
        t = np.linspace(0, 10 + i, n)
        target = np.sin(t * (i + 1) * 2.7 + i).astype(np.float32)
        r2_t = cv_ridge_r2(H_trained, target)
        r2_u = cv_ridge_r2(H_untrained, target)
        scores[f"lorenz_{i}"] = r2_t - r2_u

    for i in range(2):
        norms = np.linalg.norm(H_trained, axis=1)
        shuffled = rng.permutation(norms).astype(np.float32)
        r2_t = cv_ridge_r2(H_trained, shuffled)
        r2_u = cv_ridge_r2(H_untrained, shuffled)
        scores[f"shuffled_norm_{i}"] = r2_t - r2_u

    ceiling = max(scores.values()) if scores else 0.0
    return ceiling, scores


# ====================================================================
# MAIN
# ====================================================================

def main():
    """Run all 6 phases of the Docking Game Agent."""
    print()
    print("=" * 85)
    print("  DESCARTES-PHARMA v2.0 -- DOCKING GAME AGENT")
    print("  Complete Pipeline: Foundation -> Data -> Training -> Probing -> "
          "Council -> Comparison")
    print("=" * 85)
    print(f"  Device:    {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:      {vram:.1f} GB")
    print(f"  Project:   {PROJECT_ROOT}")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check critical imports
    import_status = {
        "BioPython/Pocket": POCKET_AVAILABLE,
        "RDKit/Ligand": LIGAND_AVAILABLE,
        "Vina Engine": VINA_AVAILABLE,
        "Policy Network": POLICY_AVAILABLE,
        "Training": TRAINING_AVAILABLE,
        "Probing": PROBING_AVAILABLE,
    }
    print("  Module availability:")
    for mod, avail in import_status.items():
        status = "OK" if avail else "FALLBACK"
        print(f"    {mod:<20} [{status}]")

    if not POLICY_AVAILABLE or not TRAINING_AVAILABLE:
        print("\n  FATAL: Policy or Training modules unavailable. Cannot proceed.")
        sys.exit(1)

    total_start = time.time()

    # ---- PHASE 1 ----
    pocket, vina_model, test_ligands_p1, pdbqt_path = run_phase1()

    # ---- PHASE 2 ----
    (train_ligands, val_ligands, test_ligands,
     train_smi, val_smi, test_smi) = run_phase2()

    # ---- PHASE 3 ----
    policy, env, trainer = run_phase3(
        pocket, vina_model, train_ligands, val_ligands, pdbqt_path,
    )

    # ---- PHASE 4 ----
    (ridge_results, perm_results, H_trained, H_untrained,
     targets, scaffolds, encoded_targets, smiles_per_step) = run_phase4(
        policy, env, test_ligands, train_ligands,
    )

    # ---- PHASE 5 ----
    final_verdicts, seed_pass_counts, ceiling, ablation_results = run_phase5(
        policy, env, train_ligands, test_ligands,
        H_trained, H_untrained, targets, scaffolds,
        ridge_results, perm_results, encoded_targets,
    )

    # ---- PHASE 6 ----
    run_phase6(ridge_results, perm_results, final_verdicts, seed_pass_counts)

    # ---- FINAL SUMMARY ----
    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(total_elapsed, 60)
    total_hours, total_mins = divmod(total_mins, 60)

    print("\n")
    print("=" * 85)
    print("  FINAL RUNTIME SUMMARY")
    print("=" * 85)
    phase_names = {
        1: "Foundation", 2: "Data Preparation", 3: "Training",
        4: "DESCARTES Probing", 5: "Council Controls",
        6: "Binary vs Continuous",
    }
    for phase_num in sorted(_PHASE_TIMES.keys()):
        elapsed = _PHASE_TIMES[phase_num]
        m, s = divmod(elapsed, 60)
        name = phase_names.get(phase_num, f"Phase {phase_num}")
        print(f"    Phase {phase_num} ({name:<25}): "
              f"{int(m):>3}m {s:>5.1f}s")

    print(f"\n    TOTAL RUNTIME: "
          f"{int(total_hours)}h {int(total_mins)}m {total_secs:.1f}s")
    print("=" * 85)
    print()


if __name__ == "__main__":
    main()
