"""
REINFORCE training loop for the docking search policy.

For each episode:
  1. Pick a ligand from the training set
  2. Place it randomly near the pocket
  3. Policy takes T steps of pose adjustments
  4. Each step: Vina scores the pose -> reward = delta-E
  5. Policy gradient update
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Optional, Dict


class DockingTrainer:
    """Train the search policy network via REINFORCE with baseline."""

    def __init__(
        self,
        policy,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            policy: SearchPolicyNetwork instance.
            env: DockingEnv instance.
            lr: Learning rate for Adam optimizer.
            gamma: Discount factor for returns.
            entropy_coef: Weight for entropy bonus (encourages exploration).
            value_coef: Weight for value loss in total loss.
            max_grad_norm: Max gradient norm for clipping.
            device: Torch device ('cpu' or 'cuda').
        """
        self.policy = policy.to(device)
        self.env = env
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        # Logging
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_best_scores: deque = deque(maxlen=100)
        self.training_log: List[Dict] = []

    def train_episode(self, ligand_features) -> dict:
        """
        Train on one docking episode (one ligand).

        Args:
            ligand_features: LigandFeatures for the ligand to dock.

        Returns:
            dict with episode statistics.
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
                obs_tensor, h, temperature=1.0
            )

            # Take action in environment
            obs, reward, done, info = self.env.step(action)

            # Compute entropy for exploration bonus
            with torch.no_grad():
                policy_logits, _, _ = self.policy(
                    obs_tensor.unsqueeze(0), h
                )
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
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)

        for log_prob, value, R, entropy in zip(
            log_probs, values, returns, entropies
        ):
            advantage = R - value.detach()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + F.mse_loss(value, R)
            entropy_loss = entropy_loss - entropy

        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # Log
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        self.episode_best_scores.append(info["best_score"])

        stats = {
            "total_reward": total_reward,
            "best_vina_score": info["best_score"],
            "n_steps": len(rewards),
            "mean_reward_100": float(np.mean(self.episode_rewards)),
            "mean_best_score_100": float(np.mean(self.episode_best_scores)),
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }
        self.training_log.append(stats)

        return stats

    def train(
        self,
        ligands: list,
        n_episodes: int = 1000,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = "results/checkpoints/",
        reward_shaper=None,
    ) -> List[Dict]:
        """
        Full training loop across multiple ligands.

        Args:
            ligands: list of LigandFeatures objects.
            n_episodes: total training episodes.
            log_interval: print stats every N episodes.
            save_interval: save checkpoint every N episodes.
            save_path: directory for checkpoint files.
            reward_shaper: optional RewardShaper for decomposed rewards.

        Returns:
            training_log: list of per-episode stat dicts.
        """
        os.makedirs(save_path, exist_ok=True)

        rng = np.random.default_rng(42)
        stats = {}

        for episode in range(n_episodes):
            # Pick a random ligand
            ligand = ligands[rng.integers(len(ligands))]

            # Train one episode
            stats = self.train_episode(ligand)

            if (episode + 1) % log_interval == 0:
                print(
                    f"Episode {episode+1}/{n_episodes} | "
                    f"Reward: {stats['total_reward']:.2f} | "
                    f"Best Vina: {stats['best_vina_score']:.2f} | "
                    f"Mean(100): {stats['mean_reward_100']:.2f} | "
                    f"Loss: {stats['loss']:.4f}"
                )

            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1, save_path)

        if stats:
            print(
                f"\nTraining complete. Final mean reward: "
                f"{stats['mean_reward_100']:.2f}"
            )

        return self.training_log

    def _save_checkpoint(self, episode: int, save_path: str):
        """Save a training checkpoint."""
        torch.save(
            {
                "episode": episode,
                "model_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "episode_rewards": list(self.episode_rewards),
                "episode_best_scores": list(self.episode_best_scores),
            },
            os.path.join(save_path, f"checkpoint_{episode}.pt"),
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a training checkpoint.

        Returns:
            episode number from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "episode_rewards" in checkpoint:
            self.episode_rewards.extend(checkpoint["episode_rewards"])
        if "episode_best_scores" in checkpoint:
            self.episode_best_scores.extend(checkpoint["episode_best_scores"])
        return checkpoint.get("episode", 0)

    def get_training_log(self) -> List[Dict]:
        """Return the full training log."""
        return self.training_log

    def is_plateaued(self, window: int = 100, threshold: float = 0.01) -> bool:
        """
        Check if training has plateaued.

        Returns True if the mean reward over the last `window` episodes
        has not improved by more than `threshold`.
        """
        if len(self.training_log) < 2 * window:
            return False

        recent = [d["total_reward"] for d in self.training_log[-window:]]
        prev = [d["total_reward"] for d in self.training_log[-2 * window:-window]]

        return abs(np.mean(recent) - np.mean(prev)) < threshold
