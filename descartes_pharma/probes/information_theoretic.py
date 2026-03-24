"""
DESCARTES-PHARMA Tier 6: Information-Theoretic Probes (MINE, MDL).
"""

import numpy as np
import torch
import torch.nn as nn


class MINENetwork(nn.Module):
    """MINE: Mutual Information Neural Estimation network."""

    def __init__(self, x_dim, y_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def mine_probe(embeddings, mechanism_values, hidden_dim=64, epochs=200,
               lr=1e-3, device='cpu'):
    """
    Estimate mutual information I(embeddings; mechanism) using MINE.
    """
    N = len(embeddings)
    x_dim = embeddings.shape[1]

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    Y = torch.tensor(mechanism_values.reshape(-1, 1), dtype=torch.float32, device=device)

    mine = MINENetwork(x_dim, 1, hidden_dim).to(device)
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)

    mi_estimates = []
    for epoch in range(epochs):
        # Joint samples
        joint_score = mine(X, Y)

        # Marginal samples (shuffle Y)
        perm = torch.randperm(N, device=device)
        Y_shuffled = Y[perm]
        marginal_score = mine(X, Y_shuffled)

        # MINE loss: -MI lower bound
        loss = -(joint_score.mean() - torch.logsumexp(marginal_score, 0) + np.log(N))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mi_estimates.append(-loss.item())

    final_mi = np.mean(mi_estimates[-20:])

    return {
        'mutual_information': final_mi,
        'mi_history': mi_estimates,
        'encoded': final_mi > 0.1,
    }


def mdl_probe(embeddings, mechanism_values, n_splits=5):
    """
    Minimum Description Length probing.

    Measures how many bits are needed to describe the mechanism
    given the embeddings. Fewer bits = better encoding.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Online code length (from Voita & Titov 2020)
    total_codelength = 0.0
    uniform_codelength = 0.0

    target_var = np.var(mechanism_values)
    N = len(mechanism_values)

    for train_idx, test_idx in kf.split(embeddings):
        ridge = Ridge(alpha=1.0)
        ridge.fit(embeddings[train_idx], mechanism_values[train_idx])
        preds = ridge.predict(embeddings[test_idx])

        residual_var = np.var(mechanism_values[test_idx] - preds) + 1e-10
        # Codelength under model
        total_codelength += len(test_idx) * 0.5 * np.log2(residual_var)
        # Codelength under uniform
        uniform_codelength += len(test_idx) * 0.5 * np.log2(target_var + 1e-10)

    compression = 1.0 - (total_codelength / (uniform_codelength + 1e-10))

    return {
        'model_codelength': total_codelength,
        'uniform_codelength': uniform_codelength,
        'compression_ratio': compression,
        'encoded': compression > 0.1,
    }
