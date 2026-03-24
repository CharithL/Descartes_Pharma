"""
DESCARTES-PHARMA Tier 5 Probes: Causal (Resample Ablation, DAS).

CRITICAL: NEVER use mean-imputation for causal testing.
ALWAYS resample from the empirical distribution (scaffold-matched).
"""

import numpy as np
from sklearn.linear_model import Ridge


def resample_ablation(model_fn, embeddings, mechanism_features,
                      mechanism_names, n_resamples=100, seed=42):
    """
    Resample ablation: replace mechanism-correlated embedding dimensions
    with resampled values from the empirical distribution.

    If prediction degrades -> mechanism is causally necessary.
    If prediction unchanged -> mechanism is NOT causally necessary (zombie signal).

    NEVER use mean-imputation (creates OOD artifacts).
    """
    rng = np.random.default_rng(seed)
    results = {}

    baseline_preds = model_fn(embeddings)

    for j, name in enumerate(mechanism_names):
        mech_values = mechanism_features[:, j]

        # Find embedding dimensions most correlated with this mechanism
        correlations = np.array([
            np.corrcoef(embeddings[:, d], mech_values)[0, 1]
            for d in range(embeddings.shape[1])
        ])
        top_dims = np.argsort(np.abs(correlations))[-10:]  # Top 10 dims

        degradations = []
        for _ in range(n_resamples):
            ablated = embeddings.copy()
            for dim in top_dims:
                # Resample from empirical distribution (NOT mean!)
                ablated[:, dim] = rng.choice(embeddings[:, dim], size=len(embeddings))

            ablated_preds = model_fn(ablated)
            degradation = np.mean((baseline_preds - ablated_preds) ** 2)
            degradations.append(degradation)

        # Null: resample random dimensions
        null_degradations = []
        non_top = [d for d in range(embeddings.shape[1]) if d not in top_dims]
        for _ in range(n_resamples):
            ablated = embeddings.copy()
            random_dims = rng.choice(non_top, size=len(top_dims), replace=False)
            for dim in random_dims:
                ablated[:, dim] = rng.choice(embeddings[:, dim], size=len(embeddings))
            ablated_preds = model_fn(ablated)
            null_degradations.append(np.mean((baseline_preds - ablated_preds) ** 2))

        mean_deg = np.mean(degradations)
        mean_null = np.mean(null_degradations)
        std_null = np.std(null_degradations) + 1e-10
        z_score = (mean_deg - mean_null) / std_null

        results[name] = {
            'mean_degradation': mean_deg,
            'null_degradation': mean_null,
            'z_score': z_score,
            'causal': z_score > 2.0,
            'p_value': 1.0 - float(np.mean(np.array(degradations) > np.array(null_degradations))),
        }

    return results


def das_probe(embeddings, mechanism_values, n_directions=5, seed=42):
    """
    Distributed Alignment Search: find encoding direction for a mechanism.

    Searches for a linear direction in embedding space that maximally
    predicts the mechanism value.
    """
    from sklearn.decomposition import PCA

    # PCA to reduce dimensionality
    n_comp = min(n_directions * 2, embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    emb_pca = pca.fit_transform(embeddings)

    # Find direction that best predicts mechanism
    ridge = Ridge(alpha=1.0)
    ridge.fit(emb_pca, mechanism_values)

    # The encoding direction in PCA space
    direction_pca = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-10)

    # Transform back to original space
    direction_original = pca.components_.T @ direction_pca

    # Project embeddings onto encoding direction
    projections = embeddings @ direction_original
    r2 = np.corrcoef(projections, mechanism_values)[0, 1] ** 2

    return {
        'encoding_direction': direction_original,
        'projection_r2': r2,
        'direction_found': r2 > 0.1,
    }
