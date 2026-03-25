"""
DESCARTES-PHARMA: Sparse Autoencoder for polypharmacology decomposition.

Decomposes superposed drug embeddings into monosemantic features.
Key: SAE R^2 >> raw Ridge R^2 -> polypharmacology detected.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class PharmaSAE(nn.Module):
    """
    Sparse Autoencoder for drug model embedding decomposition.

    Expansion factor may need to be higher (8-16x) for drug embeddings
    because the chemical mechanism space is larger than neural gating space.
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


def train_sae(embeddings_list, input_dim, expansion_factor=8, k=30,
              epochs=200, lr=1e-3, l1_weight=1e-4, device='cpu',
              batch_size=4096):
    """Train SAE on model embeddings with mini-batch training."""
    sae = PharmaSAE(input_dim, expansion_factor, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    all_emb = np.concatenate(embeddings_list, axis=0)
    n_samples = all_emb.shape[0]

    loss_history = []
    for epoch in range(epochs):
        # Shuffle each epoch
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            batch = torch.tensor(
                all_emb[batch_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            recon, sparse = sae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            l1_loss = l1_weight * sparse.abs().mean()
            total_loss = recon_loss + l1_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0)

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if epoch % 50 == 0:
            print(f"SAE Epoch {epoch}: loss={avg_loss:.6f}")

    return sae, loss_history


def sae_probe_molecular_mechanisms(sae, model_embeddings, mechanism_features,
                                   mechanism_names, device='cpu'):
    """
    Probe SAE features for molecular mechanisms.

    Key output: monosemanticity scores.
    polypharmacology_detected flag: SAE-Ridge R^2 >> raw-Ridge R^2.
    """
    with torch.no_grad():
        h_tensor = torch.tensor(model_embeddings, dtype=torch.float32, device=device)
        sae_features = sae.encode(h_tensor).cpu().numpy()

    n_features = sae_features.shape[1]
    n_mechanisms = mechanism_features.shape[1]

    # Correlation matrix
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

    # Monosemanticity scores
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
        'mean_monosemanticity': float(monosemanticity[monosemanticity > 0].mean())
            if (monosemanticity > 0).any() else 0.0,
        'polypharmacology_detected': {
            name: sae_r2[name] > raw_r2[name] + 0.05
            for name in mechanism_names
        }
    }
