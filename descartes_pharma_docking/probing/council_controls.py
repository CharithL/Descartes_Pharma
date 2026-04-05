"""
Three council controls from DESCARTES Cogito.

These controls validate that probing results reflect genuine learned
representations rather than statistical artifacts:

1. Arbitrary Target Probes -- probe for random/Lorenz/shuffled targets
   that the network could not have learned. Establishes a ceiling for
   spurious R2.

2. Multi-Seed Ensemble -- train N networks with different seeds and
   check which features are stably encoded across seeds. Features that
   appear in only 1-2 seeds are likely flukes.

3. Two-Stage Ablation -- first remove a feature's variance, then check
   if other features remain decodable. Distinguishes direct encoding
   from correlation-based leakage.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional

from descartes_pharma_docking.probing.probe_runner import cv_ridge_r2


class CouncilControls:
    """
    Three council controls from DESCARTES Cogito for validating probe results.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def arbitrary_target_probes(
        self,
        embeddings: np.ndarray,
        untrained_embeddings: np.ndarray,
        n_random: int = 5,
        n_lorenz: int = 3,
        n_shuffled: int = 2,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Probe for targets the network could not have learned.

        Generates three types of arbitrary targets:
        1. Random Gaussian vectors (pure noise)
        2. Lorenz attractor trajectories (structured but irrelevant)
        3. Shuffled real hidden state norms (preserves distribution,
           destroys correspondence)

        Args:
            embeddings: (N, hidden_dim) trained hidden states.
            untrained_embeddings: (N, hidden_dim) untrained hidden states.
            n_random: Number of random Gaussian targets to probe.
            n_lorenz: Number of Lorenz attractor targets.
            n_shuffled: Number of shuffled-norm targets.

        Returns:
            ceiling: float -- maximum delta-R2 across all arbitrary targets.
                Any real feature with delta-R2 below this is suspicious.
            scores: dict mapping arbitrary target name -> delta-R2.
        """
        n_samples = len(embeddings)
        scores = {}

        # 1. Random Gaussian targets
        for i in range(n_random):
            target = self.rng.normal(0, 1, n_samples).astype(np.float32)
            r2_trained = cv_ridge_r2(embeddings, target)
            r2_untrained = cv_ridge_r2(untrained_embeddings, target)
            delta_r2 = r2_trained - r2_untrained
            scores[f"random_gaussian_{i}"] = delta_r2

        # 2. Lorenz attractor targets (deterministic chaos -- structured noise)
        lorenz_trajectories = _generate_lorenz_targets(n_samples, n_lorenz)
        for i, traj in enumerate(lorenz_trajectories):
            r2_trained = cv_ridge_r2(embeddings, traj)
            r2_untrained = cv_ridge_r2(untrained_embeddings, traj)
            delta_r2 = r2_trained - r2_untrained
            scores[f"lorenz_{i}"] = delta_r2

        # 3. Shuffled embedding norms
        emb_norms = np.linalg.norm(embeddings, axis=1)
        for i in range(n_shuffled):
            shuffled = self.rng.permutation(emb_norms).astype(np.float32)
            r2_trained = cv_ridge_r2(embeddings, shuffled)
            r2_untrained = cv_ridge_r2(untrained_embeddings, shuffled)
            delta_r2 = r2_trained - r2_untrained
            scores[f"shuffled_norm_{i}"] = delta_r2

        # Ceiling: maximum delta-R2 across all arbitrary targets
        ceiling = max(scores.values()) if scores else 0.0

        return ceiling, scores

    def multi_seed_ensemble(
        self,
        policy_class,
        env,
        ligands: list,
        targets: Dict[str, np.ndarray],
        n_seeds: int = 20,
        threshold: float = 0.05,
        training_episodes: int = 200,
        device: str = "cpu",
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Train N networks with different seeds and check which features
        are stably encoded.

        For each seed:
          1. Initialize a fresh policy network
          2. Train for a short period
          3. Collect hidden states
          4. Probe for each target
          5. Count how many seeds encode each feature

        Args:
            policy_class: Class to instantiate (SearchPolicyNetwork).
            env: DockingEnv instance.
            ligands: Training ligands.
            targets: Dict of target name -> ground truth array.
            n_seeds: Number of random seeds to test.
            threshold: delta-R2 threshold for "encoded".
            training_episodes: Episodes per seed (short training).
            device: Torch device.

        Returns:
            pass_counts: dict mapping target name -> number of seeds
                where the feature was significantly encoded.
            stability: dict mapping target name -> fraction of seeds
                (0.0 to 1.0) where the feature was encoded.
        """
        import torch
        from descartes_pharma_docking.training.trainer import DockingTrainer

        pass_counts = {name: 0 for name in targets}
        all_delta_r2s = {name: [] for name in targets}

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Fresh network
            policy = policy_class().to(device)

            # Short training
            trainer = DockingTrainer(
                policy=policy, env=env, lr=3e-4, device=device
            )
            trainer.train(
                ligands, n_episodes=training_episodes,
                log_interval=training_episodes + 1,  # Suppress output
                save_interval=training_episodes + 1,
            )

            # Collect hidden states
            policy.train(False)
            policy.enable_logging()
            policy.clear_hidden_log()

            for ep in range(min(20, len(ligands))):
                ligand = ligands[ep % len(ligands)]
                obs = env.reset(ligand)
                h = None
                for _ in range(env.max_steps):
                    obs_t = torch.FloatTensor(obs).to(device)
                    with torch.no_grad():
                        action, _, _, h = policy.select_action(
                            obs_t, h, temperature=0.1
                        )
                    obs, _, done, _ = env.step(action)
                    if done:
                        break

            H = policy.get_hidden_states()
            policy.disable_logging()

            if len(H) == 0:
                continue

            # Untrained control
            rng = np.random.default_rng(seed + 1000)
            H_ctrl = rng.normal(0, 0.01, H.shape).astype(np.float32)

            # Probe each target
            for name, target_full in targets.items():
                target = target_full[:len(H)]
                if len(target) != len(H):
                    continue

                r2_trained = cv_ridge_r2(H, target)
                r2_ctrl = cv_ridge_r2(H_ctrl, target)
                delta_r2 = r2_trained - r2_ctrl
                all_delta_r2s[name].append(delta_r2)

                if delta_r2 > threshold:
                    pass_counts[name] += 1

        stability = {}
        for name in targets:
            stability[name] = pass_counts[name] / max(n_seeds, 1)

        return pass_counts, stability

    def two_stage_ablation(
        self,
        embeddings: np.ndarray,
        target: np.ndarray,
        all_other_targets: Dict[str, np.ndarray],
    ) -> Tuple[float, float, str]:
        """
        Two-stage ablation to distinguish direct encoding from
        correlation-based leakage.

        Stage 1: Probe embeddings for the target (baseline R2).
        Stage 2: Regress out all other targets from embeddings,
                 then re-probe the residuals for the target.

        If Stage 2 R2 is still significant, the feature is directly
        encoded (not just correlated with other encoded features).

        Args:
            embeddings: (N, hidden_dim) trained hidden states.
            target: (N,) target values to probe for.
            all_other_targets: Dict of other target name -> (N,) arrays.

        Returns:
            stage1: R2 from direct probing (baseline).
            stage2: R2 after regressing out all other targets.
            classification: "DIRECT", "CORRELATED", or "NOT_ENCODED".
        """
        n = len(embeddings)

        # Stage 1: Direct probe
        stage1 = cv_ridge_r2(embeddings, target)

        # Stage 2: Regress out other targets from embeddings
        if all_other_targets:
            # Build a matrix of all other targets
            other_matrix = np.column_stack(
                [v[:n] for v in all_other_targets.values()]
            )

            # For each embedding dimension, regress out other targets
            residual_embeddings = np.zeros_like(embeddings)
            for dim in range(embeddings.shape[1]):
                ridge = Ridge(alpha=1.0)
                ridge.fit(other_matrix, embeddings[:, dim])
                predicted = ridge.predict(other_matrix)
                residual_embeddings[:, dim] = embeddings[:, dim] - predicted

            stage2 = cv_ridge_r2(residual_embeddings, target)
        else:
            stage2 = stage1

        # Classify
        if stage1 < 0.05:
            classification = "NOT_ENCODED"
        elif stage2 > 0.03:
            classification = "DIRECT"
        else:
            classification = "CORRELATED"

        return stage1, stage2, classification


# =====================================================================
# Lorenz attractor generator for arbitrary target probes
# =====================================================================


def _generate_lorenz_targets(
    n_samples: int, n_trajectories: int = 3
) -> List[np.ndarray]:
    """
    Generate Lorenz attractor trajectories as structured-but-irrelevant
    probe targets.

    The Lorenz system is deterministic chaos: highly structured output
    that has absolutely nothing to do with protein-ligand binding.
    If a probe decodes Lorenz trajectories from hidden states, something
    is wrong with the probing methodology.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    trajectories = []

    for traj_idx in range(n_trajectories):
        # Different initial conditions for each trajectory
        x, y, z = 1.0 + traj_idx * 0.1, 1.0, 1.0
        values = []

        # Generate enough points with downsampling
        total_steps = n_samples * 10
        for step in range(total_steps):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz

            if step % 10 == 0:
                # Use different coordinates for different trajectories
                if traj_idx == 0:
                    values.append(x)
                elif traj_idx == 1:
                    values.append(y)
                else:
                    values.append(z)

        arr = np.array(values[:n_samples], dtype=np.float32)
        # Normalize to zero mean, unit variance
        if arr.std() > 1e-8:
            arr = (arr - arr.mean()) / arr.std()
        trajectories.append(arr)

    return trajectories
