"""
Pocket Scramble Test.

A negative control that tests whether the policy's learned representations
depend on the ACTUAL pocket structure or just generic spatial patterns.

Protocol:
  1. Take the real pocket features
  2. Generate a scrambled pocket: permute residue features (positions,
     identities, properties) so the spatial structure is destroyed but
     the marginal distributions are preserved
  3. Train a fresh policy on the scrambled pocket
  4. Probe the scrambled-pocket policy for the same binding features
  5. Features that are STILL "encoded" in the scrambled policy are
     TRIVIAL (generic spatial, not pocket-specific)
  6. Features encoded in the real policy but NOT in the scrambled policy
     are GENUINE pocket-specific representations

Classification per feature:
  - GENUINE: Encoded in real policy, NOT in scrambled policy
  - TRIVIAL: Encoded in BOTH real and scrambled policies
  - NEITHER: Not encoded in either policy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

from descartes_pharma_docking.probing.probe_runner import cv_ridge_r2


def scramble_pocket(pocket_features, random_seed: int = 42) -> object:
    """
    Create a scrambled version of pocket features.

    Permutes residue-level features so the spatial structure is
    destroyed but marginal distributions (element types, charge
    distributions, etc.) are preserved.

    Args:
        pocket_features: PocketFeatures object with residue data.
        random_seed: Random seed for reproducibility.

    Returns:
        Scrambled pocket features object (same type as input).
    """
    rng = np.random.default_rng(random_seed)

    # Deep copy the pocket features
    import copy
    scrambled = copy.deepcopy(pocket_features)

    # Scramble the feature vector
    if hasattr(scrambled, "to_feature_vector"):
        vec = scrambled.to_feature_vector()
        if vec is not None and len(vec) > 0:
            # Permute the feature vector in chunks of 4
            # (preserves local structure within residue features
            # but destroys inter-residue spatial relationships)
            vec_copy = vec.copy()
            chunk_size = 4
            n_chunks = len(vec_copy) // chunk_size
            if n_chunks > 1:
                chunk_order = rng.permutation(n_chunks)
                for i, new_i in enumerate(chunk_order):
                    start_old = new_i * chunk_size
                    start_new = i * chunk_size
                    end_old = min(start_old + chunk_size, len(vec_copy))
                    end_new = min(start_new + chunk_size, len(vec_copy))
                    n_copy = min(end_old - start_old, end_new - start_new)
                    vec_copy[start_new:start_new + n_copy] = vec[start_old:start_old + n_copy]

            # Override the feature vector method
            scrambled._scrambled_vec = vec_copy
            original_method = scrambled.to_feature_vector

            def scrambled_to_feature_vector():
                return scrambled._scrambled_vec

            scrambled.to_feature_vector = scrambled_to_feature_vector

    # Scramble residue positions if available
    if hasattr(scrambled, "residues") and scrambled.residues:
        residue_indices = list(range(len(scrambled.residues)))
        rng.shuffle(residue_indices)
        scrambled.residues = [scrambled.residues[i] for i in residue_indices]

    # Scramble catalytic residue assignments
    if hasattr(scrambled, "catalytic_residues") and scrambled.catalytic_residues:
        if hasattr(scrambled, "residues") and len(scrambled.residues) >= 2:
            # Pick random residues as "catalytic" (same count, wrong residues)
            n_cat = len(scrambled.catalytic_residues)
            cat_indices = rng.choice(
                len(scrambled.residues),
                size=min(n_cat, len(scrambled.residues)),
                replace=False,
            )
            scrambled.catalytic_residues = [
                scrambled.residues[i] for i in cat_indices
            ]

    # Scramble pocket center (shift randomly)
    if hasattr(scrambled, "pocket_center") and scrambled.pocket_center is not None:
        scrambled.pocket_center = (
            scrambled.pocket_center + rng.normal(0, 5.0, 3)
        )

    return scrambled


def pocket_scramble_test(
    policy_class,
    env_class_or_factory,
    real_pocket,
    vina_world_model,
    test_ligands: list,
    targets: Dict[str, np.ndarray],
    training_episodes: int = 300,
    probing_episodes: int = 50,
    n_scrambles: int = 3,
    device: str = "cpu",
) -> Dict[str, str]:
    """
    Run the pocket scramble test to classify features as GENUINE,
    TRIVIAL, or NEITHER.

    Args:
        policy_class: SearchPolicyNetwork class for instantiation.
        env_class_or_factory: Callable that creates a DockingEnv given
            (vina_world_model, pocket_features).
        real_pocket: Real PocketFeatures object.
        vina_world_model: VinaWorldModel instance.
        test_ligands: List of LigandFeatures for probing.
        targets: Dict of target name -> (N,) ground truth arrays.
        training_episodes: Episodes to train each policy.
        probing_episodes: Episodes for collecting hidden states.
        n_scrambles: Number of scrambled pockets to test.
        device: Torch device.

    Returns:
        Dict mapping target name -> "GENUINE", "TRIVIAL", or "NEITHER".
    """
    from descartes_pharma_docking.training.trainer import DockingTrainer

    # First: get real policy probe results (assumes already collected)
    # We re-probe to be consistent
    print("Pocket Scramble Test")
    print("=" * 50)

    print("\n[1/3] Training and probing on REAL pocket...")
    real_results = _train_and_probe(
        policy_class, env_class_or_factory,
        real_pocket, vina_world_model,
        test_ligands, targets,
        training_episodes, probing_episodes,
        seed=42, device=device,
    )

    # Now train on scrambled pockets
    scramble_results_all = []
    for scr_idx in range(n_scrambles):
        print(f"\n[2/3] Training on SCRAMBLED pocket {scr_idx+1}/{n_scrambles}...")
        scrambled = scramble_pocket(real_pocket, random_seed=100 + scr_idx)
        scr_results = _train_and_probe(
            policy_class, env_class_or_factory,
            scrambled, vina_world_model,
            test_ligands, targets,
            training_episodes, probing_episodes,
            seed=200 + scr_idx, device=device,
        )
        scramble_results_all.append(scr_results)

    # Classify each feature
    print("\n[3/3] Classifying features...")
    classifications = {}
    threshold = 0.05

    for name in targets:
        real_dr2 = real_results.get(name, 0.0)
        real_encoded = real_dr2 > threshold

        # Average scrambled delta-R2
        scr_dr2s = [sr.get(name, 0.0) for sr in scramble_results_all]
        mean_scr_dr2 = np.mean(scr_dr2s) if scr_dr2s else 0.0
        scr_encoded = mean_scr_dr2 > threshold

        if real_encoded and not scr_encoded:
            classification = "GENUINE"
        elif real_encoded and scr_encoded:
            classification = "TRIVIAL"
        else:
            classification = "NEITHER"

        classifications[name] = classification
        print(
            f"  {name:30s} real_dR2={real_dr2:.4f} "
            f"scr_dR2={mean_scr_dr2:.4f} -> {classification}"
        )

    return classifications


def _train_and_probe(
    policy_class,
    env_factory,
    pocket,
    vina_world_model,
    test_ligands,
    targets,
    training_episodes,
    probing_episodes,
    seed,
    device,
) -> Dict[str, float]:
    """Train a policy and probe it, returning delta-R2 per target."""
    from descartes_pharma_docking.training.trainer import DockingTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment with this pocket
    env = env_factory(vina_world_model, pocket)

    # Create and train policy
    policy = policy_class().to(device)
    trainer = DockingTrainer(policy=policy, env=env, lr=3e-4, device=device)
    trainer.train(
        test_ligands,
        n_episodes=training_episodes,
        log_interval=training_episodes + 1,
        save_interval=training_episodes + 1,
    )

    # Collect hidden states
    policy.train(False)
    policy.enable_logging()
    policy.clear_hidden_log()

    for ep in range(probing_episodes):
        ligand = test_ligands[ep % len(test_ligands)]
        obs = env.reset(ligand)
        h = None
        for _ in range(env.max_steps):
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action, _, _, h = policy.select_action(obs_t, h, temperature=0.1)
            obs, _, done, _ = env.step(action)
            if done:
                break

    H = policy.get_hidden_states()
    policy.disable_logging()

    if len(H) == 0:
        return {name: 0.0 for name in targets}

    # Untrained control
    rng = np.random.default_rng(seed + 999)
    H_ctrl = rng.normal(0, 0.01, H.shape).astype(np.float32)

    # Probe each target
    results = {}
    for name, target_full in targets.items():
        target = target_full[:len(H)]
        if len(target) != len(H):
            results[name] = 0.0
            continue
        r2_trained = cv_ridge_r2(H, target)
        r2_ctrl = cv_ridge_r2(H_ctrl, target)
        results[name] = r2_trained - r2_ctrl

    return results
