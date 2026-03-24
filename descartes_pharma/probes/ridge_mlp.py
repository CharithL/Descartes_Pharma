"""
DESCARTES-PHARMA Tier 1 Probes: Ridge and MLP delta-R^2.

RULE: Every Ridge probe MUST have an MLP companion.
If MLP dR2 >> Ridge dR2 -> mechanism is nonlinearly encoded, not absent.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor


def ridge_delta_r2(trained_embeddings, random_embeddings, targets,
                   alpha=1.0, n_splits=5, seed=42):
    """
    Compute Ridge delta-R^2 = R^2(trained) - R^2(random).

    Args:
        trained_embeddings: (N, D) from trained model
        random_embeddings: (N, D) from untrained model (same architecture)
        targets: (N,) mechanism feature values
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    trained_scores = []
    random_scores = []

    for train_idx, test_idx in kf.split(trained_embeddings):
        ridge_t = Ridge(alpha=alpha)
        ridge_t.fit(trained_embeddings[train_idx], targets[train_idx])
        trained_scores.append(ridge_t.score(trained_embeddings[test_idx], targets[test_idx]))

        ridge_r = Ridge(alpha=alpha)
        ridge_r.fit(random_embeddings[train_idx], targets[train_idx])
        random_scores.append(ridge_r.score(random_embeddings[test_idx], targets[test_idx]))

    trained_r2 = np.mean(trained_scores)
    random_r2 = np.mean(random_scores)
    delta = trained_r2 - random_r2

    return {
        'trained_r2': trained_r2,
        'random_r2': random_r2,
        'delta_r2': delta,
        'trained_scores': trained_scores,
        'random_scores': random_scores,
    }


class MLPProbe(nn.Module):
    """MLP probe with controlled capacity."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_delta_r2(trained_embeddings, random_embeddings, targets,
                 hidden_dim=64, epochs=50, lr=1e-3, n_splits=5,
                 device='cpu', seed=42):
    """
    Compute MLP delta-R^2 alongside Ridge delta-R^2.

    Encoding type classification:
    - LINEAR_ENCODED: Ridge finds it, MLP confirms
    - NONLINEAR_ONLY: MLP finds it, Ridge misses
    - ZOMBIE: Neither finds it
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    input_dim = trained_embeddings.shape[1]

    trained_scores = []
    random_scores = []

    for train_idx, test_idx in kf.split(trained_embeddings):
        for emb, scores_list in [(trained_embeddings, trained_scores),
                                  (random_embeddings, random_scores)]:
            mlp = MLPProbe(input_dim, hidden_dim).to(device)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
            criterion = nn.MSELoss()

            X_train = torch.tensor(emb[train_idx], dtype=torch.float32).to(device)
            y_train = torch.tensor(targets[train_idx], dtype=torch.float32).to(device)
            X_test = torch.tensor(emb[test_idx], dtype=torch.float32).to(device)
            y_test = torch.tensor(targets[test_idx], dtype=torch.float32).to(device)

            mlp.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = criterion(mlp(X_train), y_train)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                mlp.eval()  # noqa: this is intentional placement
                preds = mlp(X_test).cpu().numpy()
                y_true = y_test.cpu().numpy()
                ss_res = np.sum((y_true - preds) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                scores_list.append(r2)

    trained_r2 = np.mean(trained_scores)
    random_r2 = np.mean(random_scores)
    delta = trained_r2 - random_r2

    return {
        'trained_r2': trained_r2,
        'random_r2': random_r2,
        'delta_r2': delta,
    }


def pharma_mlp_delta_r2(model_embeddings, random_embeddings,
                         mechanism_features, mechanism_names,
                         hidden_dim=64, epochs=50, lr=1e-3,
                         n_splits=5, device='cpu'):
    """
    Compute both Ridge and MLP delta-R^2 for all mechanisms.

    Returns per-mechanism encoding classification.
    """
    results = {}

    for j, name in enumerate(mechanism_names):
        targets = mechanism_features[:, j]
        if np.std(targets) < 1e-10:
            results[name] = {'encoding_type': 'CONSTANT_TARGET', 'delta_r2': 0.0}
            continue

        ridge = ridge_delta_r2(model_embeddings, random_embeddings, targets,
                               n_splits=n_splits)
        mlp = mlp_delta_r2(model_embeddings, random_embeddings, targets,
                           hidden_dim=hidden_dim, epochs=epochs, lr=lr,
                           n_splits=n_splits, device=device)

        r_dr2 = ridge['delta_r2']
        m_dr2 = mlp['delta_r2']

        if r_dr2 > 0.05 and m_dr2 > 0.05:
            enc_type = 'LINEAR_ENCODED' if m_dr2 < r_dr2 + 0.1 else 'NONLINEAR_ENCODED'
        elif m_dr2 > 0.1:
            enc_type = 'NONLINEAR_ONLY'
        else:
            enc_type = 'ZOMBIE'

        results[name] = {
            'ridge_delta_r2': r_dr2,
            'mlp_delta_r2': m_dr2,
            'ridge_trained_r2': ridge['trained_r2'],
            'mlp_trained_r2': mlp['trained_r2'],
            'encoding_type': enc_type,
        }

    return results
