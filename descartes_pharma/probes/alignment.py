"""
DESCARTES-PHARMA Tier 2 Probes: Joint Alignment (CCA, RSA, CKA).
"""

import numpy as np
from sklearn.cross_decomposition import CCA as SkCCA


def cca_probe(embeddings, mechanism_features, n_components=5):
    """
    Canonical Correlation Analysis: alignment between
    embedding space and mechanism space.
    """
    n_comp = min(n_components, embeddings.shape[1], mechanism_features.shape[1])
    cca = SkCCA(n_components=n_comp)
    cca.fit(embeddings, mechanism_features)

    X_c, Y_c = cca.transform(embeddings, mechanism_features)

    correlations = []
    for i in range(n_comp):
        r = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        correlations.append(r)

    return {
        'canonical_correlations': correlations,
        'mean_correlation': np.mean(correlations),
        'n_significant': sum(1 for r in correlations if abs(r) > 0.3),
        'n_components': n_comp,
    }


def rsa_probe(embeddings, mechanism_features, metric='correlation'):
    """
    Representational Similarity Analysis: does embedding geometry
    match mechanism geometry?
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr

    emb_dist = pdist(embeddings, metric=metric)
    mech_dist = pdist(mechanism_features, metric=metric)

    rho, p_value = spearmanr(emb_dist, mech_dist)

    return {
        'spearman_rho': rho,
        'p_value': p_value,
        'geometric_match': rho > 0.3 and p_value < 0.05,
    }


def cka_probe(embeddings, mechanism_features):
    """
    Centered Kernel Alignment: nonlinear geometric alignment.
    """
    def _center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K_emb = embeddings @ embeddings.T
    K_mech = mechanism_features @ mechanism_features.T

    K_emb_c = _center_gram(K_emb)
    K_mech_c = _center_gram(K_mech)

    hsic = np.sum(K_emb_c * K_mech_c)
    norm1 = np.sqrt(np.sum(K_emb_c * K_emb_c))
    norm2 = np.sqrt(np.sum(K_mech_c * K_mech_c))

    cka = hsic / (norm1 * norm2 + 1e-10)

    return {
        'cka_score': cka,
        'alignment': 'HIGH' if cka > 0.5 else 'MEDIUM' if cka > 0.2 else 'LOW',
    }
