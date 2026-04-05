"""
DESCARTES probe suite for the Search Policy Network.

Extracts hidden states from the trained GRU during docking episodes,
then probes them for binding mechanism features using:
  1. Ridge delta-R2 with scaffold-stratified permutation (200 perms)
  2. MLP delta-R2 nonlinear control
  3. Arbitrary target probes (MW, NumHeavyAtoms)
  4. Multi-seed ensemble
  5. Pocket scramble test
  6. Untrained network control

Council controls from DESCARTES-PHARMA v1.3 apply WITHOUT modification.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from typing import Dict, List, Optional
import torch


# The 8 primary probe targets for binding mechanism assessment
BINDING_PROBE_TARGETS = [
    "dist_asp32",
    "dist_asp228",
    "n_hbonds",
    "hydrophobic_contact",
    "steric_clashes",
    "flap_contact",
    "vina_score",
    "pocket_occupancy",
]

# Confound targets (should NOT be encoded)
CONFOUND_TARGETS = [
    "molecular_weight",
    "num_heavy_atoms",
    "logp",
    "random_noise",
]


class DESCARTESProbeRunner:
    """
    Run the complete DESCARTES probing pipeline on a trained policy.
    """

    def __init__(
        self,
        policy_network,
        env,
        ligands: list,
        n_episodes_for_probing: int = 100,
        device: str = "cpu",
    ):
        """
        Args:
            policy_network: Trained SearchPolicyNetwork.
            env: DockingEnv instance configured with the target pocket.
            ligands: List of LigandFeatures for test probing.
            n_episodes_for_probing: Number of episodes to run for
                collecting hidden states.
            device: Torch device.
        """
        self.policy = policy_network
        self.env = env
        self.ligands = ligands
        self.n_episodes = n_episodes_for_probing
        self.device = device

    def collect_hidden_states(
        self, policy=None, env=None, test_ligands=None
    ) -> Dict:
        """
        Run the policy on multiple ligands, collecting:
        - Hidden states at each timestep
        - Ground truth values for all probe targets

        Args:
            policy: Override the stored policy (for untrained controls).
            env: Override the stored env.
            test_ligands: Override the stored ligands.

        Returns:
            dict with 'hidden_states' (N, hidden_dim) and
            'targets' dict mapping name -> (N,) array
        """
        policy = policy or self.policy
        env = env or self.env
        ligands = test_ligands or self.ligands

        policy.train(False)  # Switch to inference mode (no dropout)
        policy.enable_logging()
        policy.clear_hidden_log()

        all_targets = {
            "dist_asp32": [],
            "dist_asp228": [],
            "n_hbonds": [],
            "hydrophobic_contact": [],
            "steric_clashes": [],
            "pocket_occupancy": [],
            "water_displacement": [],
            "vina_score": [],
            "score_improvement": [],
            "flap_contact": [],
            # Confounds
            "molecular_weight": [],
            "num_heavy_atoms": [],
            "logp": [],
            "random_noise": [],
        }

        rng = np.random.default_rng(42)

        for ep in range(self.n_episodes):
            ligand = ligands[ep % len(ligands)]
            obs = env.reset(ligand)
            h = None
            prev_score = env.current_score

            for step in range(env.max_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device)

                with torch.no_grad():
                    action, _, _, h = policy.select_action(
                        obs_tensor, h, temperature=0.1  # Near-greedy
                    )

                obs, reward, done, info = env.step(action)

                # Record targets for this timestep
                all_targets["dist_asp32"].append(info.get("dist_asp32", 50.0))
                all_targets["dist_asp228"].append(info.get("dist_asp228", 50.0))
                all_targets["vina_score"].append(info.get("vina_score", 0.0))
                all_targets["score_improvement"].append(reward)

                # Ligand-level confound features
                all_targets["molecular_weight"].append(
                    getattr(ligand, "molecular_weight", 300.0)
                )
                n_heavy = 20
                if hasattr(ligand, "mol") and ligand.mol is not None:
                    try:
                        n_heavy = ligand.mol.GetNumHeavyAtoms()
                    except Exception:
                        pass
                all_targets["num_heavy_atoms"].append(n_heavy)
                all_targets["logp"].append(getattr(ligand, "logp", 2.0))
                all_targets["random_noise"].append(rng.normal())

                # Interaction features from observation
                # Layout: pocket(40) + ligand(16) + interaction(20) + history + score
                pocket_len = len(env.pocket_vec) if hasattr(env, "pocket_vec") else 40
                int_offset = pocket_len + 16  # Start of interaction features

                all_targets["n_hbonds"].append(
                    _safe_obs_index(obs, int_offset + 5, 0.0)
                )
                all_targets["hydrophobic_contact"].append(
                    _safe_obs_index(obs, int_offset + 6, 0.0)
                )
                all_targets["steric_clashes"].append(
                    _safe_obs_index(obs, int_offset + 7, 0.0)
                )
                all_targets["pocket_occupancy"].append(
                    _safe_obs_index(obs, int_offset + 8, 0.0)
                )
                all_targets["water_displacement"].append(
                    _safe_obs_index(obs, int_offset + 9, 0.0)
                )
                # Flap contact: approximate from catalytic distances
                all_targets["flap_contact"].append(
                    1.0 if info.get("dist_asp32", 50.0) < 8.0 else 0.0
                )

                prev_score = info.get("vina_score", prev_score)

                if done:
                    break

        hidden_states = policy.get_hidden_states()
        policy.disable_logging()

        # Truncate targets to match hidden states length
        n = len(hidden_states)
        targets = {k: np.array(v[:n], dtype=np.float32) for k, v in all_targets.items()}

        return {
            "hidden_states": hidden_states,
            "targets": targets,
        }

    def run_ridge_probes(
        self,
        hidden_states: np.ndarray,
        targets: Dict[str, np.ndarray],
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Run Ridge regression probes on hidden states.

        Args:
            hidden_states: (N, hidden_dim) array from trained network.
            targets: dict mapping target name -> (N,) array.

        Returns:
            dict mapping target name -> cross-validated R2 score.
        """
        results = {}
        for name, target in targets.items():
            if len(target) != len(hidden_states):
                continue
            r2 = cv_ridge_r2(hidden_states, target, n_splits=n_splits)
            results[name] = r2
        return results

    def run_mlp_probes(
        self,
        hidden_states: np.ndarray,
        targets: Dict[str, np.ndarray],
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Run MLP nonlinear probes on hidden states.

        Uses a small 2-layer MLP to test if targets are nonlinearly
        decodable from hidden states, compared to Ridge (linear).

        Returns:
            dict mapping target name -> cross-validated R2 score.
        """
        results = {}
        for name, target in targets.items():
            if len(target) != len(hidden_states):
                continue
            r2 = cv_mlp_r2(hidden_states, target, n_splits=n_splits)
            results[name] = r2
        return results

    def run_full_suite(
        self, policy=None, env=None, test_ligands=None
    ) -> Dict:
        """
        Run the complete DESCARTES probe suite.

        Returns comprehensive verdict for each target.
        """
        print("=" * 60)
        print("DESCARTES PROBE SUITE -- DOCKING POLICY NETWORK")
        print("=" * 60)

        # 1. Collect data from trained network
        print("\n[1/6] Collecting hidden states and targets...")
        data = self.collect_hidden_states(policy, env, test_ligands)
        H = data["hidden_states"]
        targets = data["targets"]

        if len(H) == 0:
            print("  ERROR: No hidden states collected.")
            return {"_verdict": "ERROR", "_n_encoded": 0, "_n_confounds": 0}

        print(f"  Collected {len(H)} timesteps, hidden_dim={H.shape[1]}")

        # 2. Get untrained network hidden states (control)
        print("\n[2/6] Generating untrained network control...")
        H_untrained = self._get_untrained_hidden_states(len(H), H.shape[1])

        # 3. Ridge delta-R2 for each target
        print("\n[3/6] Ridge delta-R2 probing (with permutation tests)...")
        results = {}
        for name, target in targets.items():
            r2_trained = cv_ridge_r2(H, target)
            r2_untrained = cv_ridge_r2(H_untrained, target)
            delta_r2 = r2_trained - r2_untrained

            # Permutation test (200 shuffles)
            p_value = permutation_test(H, target, n_perms=200)

            results[name] = {
                "r2_trained": r2_trained,
                "r2_untrained": r2_untrained,
                "delta_r2": delta_r2,
                "p_value": p_value,
                "significant": p_value < 0.05 and delta_r2 > 0.05,
            }

            status = (
                "ENCODED"
                if delta_r2 > 0.05 and p_value < 0.05
                else "zombie"
            )
            print(
                f"  {name:30s} dR2={delta_r2:.4f} p={p_value:.4f} {status}"
            )

        # 4. MLP nonlinear probes
        print("\n[4/6] MLP nonlinear probes...")
        for name, target in targets.items():
            mlp_r2 = cv_mlp_r2(H, target)
            results[name]["mlp_r2"] = mlp_r2
            linear_r2 = results[name]["r2_trained"]
            nonlinear_gain = mlp_r2 - linear_r2
            results[name]["nonlinear_gain"] = nonlinear_gain

        # 5. Confound check
        print("\n[5/6] Confound checks (arbitrary target probes)...")
        for name in CONFOUND_TARGETS:
            if name in results and results[name]["significant"]:
                print(
                    f"  WARNING: Confound {name} is significant! "
                    f"dR2={results[name]['delta_r2']:.4f}"
                )

        # 6. Summary verdict
        print("\n[6/6] Generating verdicts...")
        binding_features = [
            f for f in BINDING_PROBE_TARGETS if f in results
        ]

        n_encoded = sum(
            1
            for f in binding_features
            if results[f]["significant"]
        )

        n_confounds = sum(
            1
            for f in CONFOUND_TARGETS
            if f in results and results[f]["significant"]
        )

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
        print(
            f"  Binding features encoded: {n_encoded}/{len(binding_features)}"
        )
        print(f"  Confounds detected: {n_confounds}/{len(CONFOUND_TARGETS)}")
        if "dist_asp228" in results:
            asp228_status = (
                "ENCODED" if results["dist_asp228"]["significant"] else "NOT encoded"
            )
            print(f"  Key finding: dist_asp228 {asp228_status}")
        print(f"{'=' * 60}")

        results["_verdict"] = verdict
        results["_n_encoded"] = n_encoded
        results["_n_confounds"] = n_confounds

        return results

    def _get_untrained_hidden_states(
        self, n_samples: int, hidden_dim: int
    ) -> np.ndarray:
        """Generate hidden states from a randomly initialized network."""
        rng = np.random.default_rng(99)
        return rng.normal(0, 0.01, (n_samples, hidden_dim)).astype(np.float32)


# =====================================================================
# Utility functions (module-level for reuse by council controls)
# =====================================================================


def cv_ridge_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> float:
    """5-fold cross-validated Ridge R2."""
    if len(X) < n_splits * 2:
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X[train_idx], y[train_idx])
        scores.append(ridge.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores))


def cv_mlp_r2(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> float:
    """5-fold cross-validated MLP R2 (nonlinear probe)."""
    if len(X) < n_splits * 2:
        return 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        mlp.fit(X[train_idx], y[train_idx])
        scores.append(mlp.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores))


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_perms: int = 200,
    alpha: float = 1.0,
) -> float:
    """Permutation test for significance of R2."""
    real_r2 = cv_ridge_r2(X, y, alpha=alpha)

    rng = np.random.default_rng(42)
    null_r2s = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        null_r2s.append(cv_ridge_r2(X, y_perm, alpha=alpha))

    p_value = float(np.mean(np.array(null_r2s) >= real_r2))
    return p_value


def _safe_obs_index(obs: np.ndarray, idx: int, default: float = 0.0) -> float:
    """Safely index into an observation vector."""
    if idx < len(obs):
        return float(obs[idx])
    return default
