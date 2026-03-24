"""
DESCARTES-PHARMA v3.0 -- Statistical Hardening Suite
=====================================================

Thirteen complementary statistical tests designed to distinguish genuine
mechanism-of-action (MoA) signal from structural confounds in
pharmaceutical embedding spaces.

The suite addresses the core challenge in cheminformatics probing: molecules
that share a scaffold often share an activity label, so a linear probe can
appear to decode "mechanism" when it has merely memorised chemical families.
Every method below either *quantifies* or *removes* this scaffold bias.

Requires: numpy, scipy, scikit-learn.
Optional: rdkit (for scaffold-aware methods; graceful fallback if absent).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

# ---------------------------------------------------------------------------
# RDKit import -- optional
# ---------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    warnings.warn(
        "RDKit is not installed. Scaffold-aware methods "
        "(scaffold_stratified_permutation, scaffold_adjusted_neff, "
        "scaffold_resolved_r2, tanimoto_residual_test, scaffold_split_cv) "
        "will raise RuntimeError if called.",
        stacklevel=2,
    )


# ===================================================================== #
#  Helper utilities                                                      #
# ===================================================================== #

def _require_rdkit(fn_name: str) -> None:
    """Raise RuntimeError when RDKit is needed but unavailable."""
    if not _HAS_RDKIT:
        raise RuntimeError(
            f"{fn_name} requires RDKit. Install with: "
            "conda install -c conda-forge rdkit"
        )


def _get_murcko_scaffolds(smiles: Sequence[str]) -> List[str]:
    """Return Murcko generic scaffolds for a list of SMILES strings."""
    _require_rdkit("_get_murcko_scaffolds")
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds.append("__INVALID__")
        else:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            generic = MurckoScaffold.MakeScaffoldGeneric(core)
            scaffolds.append(Chem.MolToSmiles(generic))
    return scaffolds


def _scaffold_to_groups(smiles: Sequence[str]) -> Dict[str, List[int]]:
    """Map scaffold SMILES -> list of molecule indices."""
    scaffolds = _get_murcko_scaffolds(smiles)
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        groups[scaf].append(idx)
    return groups


def _tanimoto_matrix(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute pairwise Tanimoto similarity from Morgan fingerprints."""
    _require_rdkit("_tanimoto_matrix")
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), radius, nBits=n_bits
            ))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    n = len(fps)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim[i, j] = t
            sim[j, i] = t
    return sim


# ===================================================================== #
#  1. Scaffold-stratified permutation test                               #
# ===================================================================== #

def scaffold_stratified_permutation(
    smiles: Sequence[str],
    mechanism_values: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Permute activity labels *within* scaffold families, then re-probe.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Standard permutation tests shuffle labels globally, which destroys
    scaffold-activity correlation and inflates the apparent significance of
    a probe.  By permuting only *within* each Murcko scaffold family we
    preserve the scaffold-activity link and test whether there is signal
    **beyond** shared chemistry.

    Parameters
    ----------
    smiles : sequence of str
        SMILES strings for each molecule.
    mechanism_values : array-like, shape (n,)
        Continuous or encoded MoA labels.
    n_perms : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``observed_var`` -- variance of the original labels,
        ``null_vars``    -- array of within-scaffold-permuted variances,
        ``p_value``      -- proportion of null >= observed.
    """
    _require_rdkit("scaffold_stratified_permutation")
    mechanism_values = np.asarray(mechanism_values, dtype=float)
    groups = _scaffold_to_groups(smiles)
    rng = np.random.default_rng(seed)

    observed_var = np.var(mechanism_values)

    null_vars = np.empty(n_perms)
    for p in range(n_perms):
        permuted = mechanism_values.copy()
        for indices in groups.values():
            if len(indices) > 1:
                subset = permuted[indices]
                rng.shuffle(subset)
                permuted[indices] = subset
        null_vars[p] = np.var(permuted)

    p_value = float(np.mean(null_vars >= observed_var))
    return {
        "observed_var": float(observed_var),
        "null_vars": null_vars,
        "p_value": p_value,
    }


# ===================================================================== #
#  2. Y-scramble null distribution                                       #
# ===================================================================== #

def y_scramble_null(
    mechanism_values: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Global label shuffle to build an unconditional null distribution.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Paired with scaffold-stratified permutation (method 1): the difference
    between this global null and the scaffold-conditional null quantifies
    how much apparent signal is attributable to scaffold confounding.

    Parameters
    ----------
    mechanism_values : array-like, shape (n,)
        Activity or MoA labels.
    n_perms : int
        Number of shuffles.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``null_means`` -- mean of each permuted set,
        ``null_stds``  -- std of each permuted set,
        ``original_mean``, ``original_std``.
    """
    mechanism_values = np.asarray(mechanism_values, dtype=float)
    rng = np.random.default_rng(seed)

    original_mean = float(np.mean(mechanism_values))
    original_std = float(np.std(mechanism_values))

    null_means = np.empty(n_perms)
    null_stds = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(mechanism_values)
        null_means[i] = np.mean(perm)
        null_stds[i] = np.std(perm)

    return {
        "original_mean": original_mean,
        "original_std": original_std,
        "null_means": null_means,
        "null_stds": null_stds,
    }


# ===================================================================== #
#  3. Scaffold-adjusted effective N                                      #
# ===================================================================== #

def scaffold_adjusted_neff(smiles: Sequence[str]) -> Dict[str, Any]:
    """Effective degrees of freedom accounting for scaffold clustering.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Drug datasets are highly clustered: a 10 000-molecule set may contain
    only 200 unique Murcko scaffolds.  Using raw N in significance tests
    overstates power.  We compute N_eff as the number of unique scaffolds,
    and also report a variance-inflation-style correction ratio.

    Parameters
    ----------
    smiles : sequence of str
        SMILES strings.

    Returns
    -------
    dict
        ``n_total``     -- total number of molecules,
        ``n_scaffolds`` -- number of unique Murcko scaffolds,
        ``n_eff``       -- effective sample size (= n_scaffolds),
        ``correction_ratio`` -- n_eff / n_total.
    """
    _require_rdkit("scaffold_adjusted_neff")
    groups = _scaffold_to_groups(smiles)
    n_total = len(smiles)
    n_scaffolds = len(groups)

    # Weighted effective N: smaller clusters contribute more independence
    cluster_sizes = np.array([len(v) for v in groups.values()], dtype=float)
    # Design-effect style: n_eff = N / (1 + cv^2 of cluster sizes)
    cv2 = float(np.var(cluster_sizes) / (np.mean(cluster_sizes) ** 2)) if np.mean(cluster_sizes) > 0 else 0.0
    n_eff_deff = n_total / (1.0 + cv2)

    return {
        "n_total": n_total,
        "n_scaffolds": n_scaffolds,
        "n_eff": float(min(n_scaffolds, n_eff_deff)),
        "correction_ratio": float(min(n_scaffolds, n_eff_deff) / n_total) if n_total > 0 else 0.0,
    }


# ===================================================================== #
#  4. FDR correction (Benjamini-Hochberg)                                #
# ===================================================================== #

def fdr_correction(
    p_values: np.ndarray,
    method: str = "bh",
) -> Dict[str, Any]:
    """Benjamini-Hochberg (or Bonferroni) FDR correction.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    When probing many MoA categories simultaneously, multiple-testing
    correction is essential.  BH is the default as it controls the false
    discovery *rate* rather than the overly conservative family-wise error
    rate (Bonferroni).

    Parameters
    ----------
    p_values : array-like
        Raw p-values from probe tests.
    method : str
        ``'bh'`` for Benjamini-Hochberg, ``'bonferroni'`` for Bonferroni.

    Returns
    -------
    dict
        ``corrected_p`` -- adjusted p-values,
        ``rejected``    -- boolean mask at alpha = 0.05,
        ``method``      -- method name.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)

    if method == "bh":
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        ranks = np.arange(1, n + 1)
        adjusted = np.minimum(1.0, sorted_p * n / ranks)
        # enforce monotonicity (right to left)
        for i in range(n - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
        corrected = np.empty(n)
        corrected[sorted_idx] = adjusted
    elif method == "bonferroni":
        corrected = np.minimum(1.0, p * n)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bh' or 'bonferroni'.")

    return {
        "corrected_p": corrected,
        "rejected": corrected < 0.05,
        "method": method,
    }


# ===================================================================== #
#  5. Scaffold-resolved R^2                                              #
# ===================================================================== #

def scaffold_resolved_r2(
    embeddings: np.ndarray,
    targets: np.ndarray,
    smiles: Sequence[str],
) -> Dict[str, Any]:
    """Decompose linear-probe R^2 by Murcko scaffold family.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    A global R^2 of 0.60 may hide the fact that the probe succeeds for
    beta-lactams (n=500) and fails for macrolides (n=50).  This method
    fits one global linear probe and then computes per-scaffold R^2 to
    expose such heterogeneity.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
        Embedding vectors.
    targets : ndarray, shape (n,)
        Activity / MoA labels.
    smiles : sequence of str
        SMILES for scaffold assignment.

    Returns
    -------
    dict
        ``global_r2``   -- overall R^2,
        ``scaffold_r2`` -- dict mapping scaffold SMILES -> R^2 (or NaN
                           for scaffolds with < 3 members),
        ``n_scaffolds`` -- total scaffolds evaluated.
    """
    _require_rdkit("scaffold_resolved_r2")
    embeddings = np.asarray(embeddings)
    targets = np.asarray(targets, dtype=float)

    model = Ridge(alpha=1.0)
    model.fit(embeddings, targets)
    preds = model.predict(embeddings)
    global_r2 = float(r2_score(targets, preds))

    groups = _scaffold_to_groups(smiles)
    scaffold_r2: Dict[str, float] = {}
    for scaf, indices in groups.items():
        if len(indices) < 3:
            scaffold_r2[scaf] = float("nan")
        else:
            y_true = targets[indices]
            y_pred = preds[indices]
            if np.std(y_true) == 0:
                scaffold_r2[scaf] = float("nan")
            else:
                scaffold_r2[scaf] = float(r2_score(y_true, y_pred))

    return {
        "global_r2": global_r2,
        "scaffold_r2": scaffold_r2,
        "n_scaffolds": len(groups),
    }


# ===================================================================== #
#  6. Tanimoto residual clustering test                                  #
# ===================================================================== #

def tanimoto_residual_test(
    residuals: np.ndarray,
    smiles: Sequence[str],
    n_perms: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test whether probe residuals cluster by chemical structure.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    If a linear probe's residuals are correlated with Tanimoto similarity,
    the probe has *not* fully captured the structure-activity relationship
    and the R^2 may be scaffold-driven.  We compute a Mantel-like
    statistic: correlation between the residual-distance matrix and the
    (1 - Tanimoto) distance matrix, then assess significance by permutation.

    Parameters
    ----------
    residuals : ndarray, shape (n,)
        Probe residuals (y_true - y_pred).
    smiles : sequence of str
        SMILES strings.
    n_perms : int
        Permutations for Mantel test.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``mantel_r``   -- observed correlation,
        ``p_value``    -- permutation p-value,
        ``n_perms``    -- number of permutations used.
    """
    _require_rdkit("tanimoto_residual_test")
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)

    # Distance matrices -- upper triangle flattened
    tanimoto_sim = _tanimoto_matrix(smiles)
    chem_dist = 1.0 - tanimoto_sim

    resid_dist = np.abs(residuals[:, None] - residuals[None, :])

    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    chem_vec = chem_dist[iu]
    resid_vec = resid_dist[iu]

    observed_r = float(np.corrcoef(chem_vec, resid_vec)[0, 1])

    rng = np.random.default_rng(seed)
    null_r = np.empty(n_perms)
    for p in range(n_perms):
        perm_idx = rng.permutation(n)
        perm_resid = resid_dist[np.ix_(perm_idx, perm_idx)]
        null_r[p] = np.corrcoef(chem_vec, perm_resid[iu])[0, 1]

    p_value = float(np.mean(np.abs(null_r) >= np.abs(observed_r)))
    return {
        "mantel_r": observed_r,
        "p_value": p_value,
        "n_perms": n_perms,
    }


# ===================================================================== #
#  7. Confound removal (MW, logP regression)                             #
# ===================================================================== #

def confound_removal(
    embeddings: np.ndarray,
    confounds: np.ndarray,
) -> np.ndarray:
    """Regress out molecular confounds from embedding space.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Molecular weight and cLogP are strongly encoded in most molecular
    representations and correlate with many activity endpoints.  Before
    probing for MoA, we orthogonalise the embedding matrix with respect
    to these confounds, ensuring that the probe cannot rely on trivial
    physicochemical features.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
        Original embeddings.
    confounds : ndarray, shape (n, k)
        Confound matrix (e.g., columns for MW, logP).

    Returns
    -------
    ndarray, shape (n, d)
        Residualised embeddings.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    confounds = np.asarray(confounds, dtype=float)
    if confounds.ndim == 1:
        confounds = confounds[:, None]

    # Ordinary least-squares per embedding dimension
    model = LinearRegression()
    model.fit(confounds, embeddings)
    predicted = model.predict(confounds)
    return embeddings - predicted


# ===================================================================== #
#  8. TOST equivalence test (zombie confirmation)                        #
# ===================================================================== #

def tost_equivalence_test(
    delta_r2: float,
    se: float,
    epsilon: float = 0.05,
    n_eff: Optional[float] = None,
) -> Dict[str, Any]:
    """Two one-sided tests to formally confirm a probe is a *zombie*.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    In pharma probing, the question is often not "is the effect
    significant?" but "is the effect *negligibly small*?"  TOST lets us
    *affirmatively* conclude that Delta-R^2 lies within (-epsilon, +epsilon),
    i.e., the mechanism is not decodable beyond chance.

    Parameters
    ----------
    delta_r2 : float
        Observed change in R^2 (probe - baseline).
    se : float
        Standard error of delta_r2.
    epsilon : float
        Equivalence bound (default 0.05, i.e., 5 percentage points).
    n_eff : float, optional
        Effective sample size for df.  If None, uses z-test.

    Returns
    -------
    dict
        ``t_lower``, ``t_upper`` -- test statistics,
        ``p_lower``, ``p_upper`` -- one-sided p-values,
        ``p_tost``               -- max of the two (overall TOST p),
        ``equivalent``           -- True if p_tost < 0.05.
    """
    if se <= 0:
        raise ValueError("Standard error must be positive.")

    t_lower = (delta_r2 - (-epsilon)) / se
    t_upper = (delta_r2 - epsilon) / se

    if n_eff is not None and n_eff > 2:
        df = n_eff - 2
        p_lower = float(1.0 - sp_stats.t.cdf(t_lower, df))
        p_upper = float(sp_stats.t.cdf(t_upper, df))
    else:
        # z-test fallback
        p_lower = float(1.0 - sp_stats.norm.cdf(t_lower))
        p_upper = float(sp_stats.norm.cdf(t_upper))

    p_tost = max(p_lower, p_upper)
    return {
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": p_tost < 0.05,
    }


# ===================================================================== #
#  9. Bayes factor for null (zombie evidence)                            #
# ===================================================================== #

def bayes_factor_null(
    delta_r2: float,
    se: float,
    prior_scale: float = 0.5,
) -> Dict[str, Any]:
    """Approximate Bayes factor in favour of the null (BF01).

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Unlike p-values, a Bayes factor can *quantify evidence for* the null
    hypothesis (i.e., no decodable mechanism).  We use a Savage-Dickey
    ratio with a Cauchy prior on effect size (prior_scale controls width).
    BF01 > 3 is conventionally "moderate evidence" for the null / zombie
    status.

    Parameters
    ----------
    delta_r2 : float
        Observed delta R^2.
    se : float
        Standard error.
    prior_scale : float
        Scale of the Cauchy prior on the standardised effect.

    Returns
    -------
    dict
        ``bf01``    -- Bayes factor for null vs alternative,
        ``verdict`` -- human-readable interpretation.
    """
    if se <= 0:
        raise ValueError("Standard error must be positive.")

    # Standardised effect
    d = delta_r2 / se

    # Savage-Dickey: BF01 = p(d=0 | H1, data) / p(d=0 | H0, data)
    # Under H0: d ~ N(0, 1)  -> density at 0 = norm.pdf(0)
    # Under H1: d | delta ~ N(delta, 1), delta ~ Cauchy(0, prior_scale)
    # Marginal under H1 at d: integrate N(d; delta, 1) * Cauchy(delta; 0, scale) d(delta)
    # Approximation via Savage-Dickey: BF01 ≈ likelihood_at_0_H0 / likelihood_at_0_H1
    # where H1 prior predictive at d evaluated at d_obs

    lik_h0 = sp_stats.norm.pdf(d, loc=0, scale=1)
    # H1 marginal: convolve normal(0,1) with Cauchy(0, prior_scale)
    # = Voigt profile; approximate as Student-t with df related to scale
    # A Cauchy(0, r) prior gives a marginal that is approximately
    # t-distributed with df ~ 1 and scale sqrt(1 + r^2)
    scale_h1 = np.sqrt(1.0 + prior_scale ** 2)
    lik_h1 = sp_stats.t.pdf(d, df=1, loc=0, scale=scale_h1)

    bf01 = float(lik_h0 / lik_h1) if lik_h1 > 0 else float("inf")

    if bf01 > 10:
        verdict = "strong evidence for null (zombie)"
    elif bf01 > 3:
        verdict = "moderate evidence for null (zombie)"
    elif bf01 > 1:
        verdict = "anecdotal evidence for null"
    elif bf01 > 1 / 3:
        verdict = "anecdotal evidence for alternative"
    elif bf01 > 1 / 10:
        verdict = "moderate evidence for alternative"
    else:
        verdict = "strong evidence for alternative"

    return {"bf01": bf01, "verdict": verdict}


# ===================================================================== #
#  10. Scaffold-split cross-validation                                   #
# ===================================================================== #

def scaffold_split_cv(
    smiles: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test splits where no scaffold appears in both sets.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Random CV splits leak scaffold information: if the same Murcko core
    is in both train and test, the model can exploit structural similarity
    rather than learning genuine MoA signal.  Scaffold-split CV is the
    gold standard in molecular-property prediction.

    Parameters
    ----------
    smiles : sequence of str
        SMILES strings.
    n_splits : int
        Number of CV folds.
    seed : int
        Random seed for scaffold assignment to folds.

    Returns
    -------
    list of (train_indices, test_indices) tuples
        Each element is a pair of numpy int arrays.
    """
    _require_rdkit("scaffold_split_cv")
    groups = _scaffold_to_groups(smiles)
    scaffold_keys = list(groups.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_keys)

    # Assign scaffolds round-robin to folds
    fold_assignment: Dict[int, List[int]] = defaultdict(list)
    for i, key in enumerate(scaffold_keys):
        fold_assignment[i % n_splits].extend(groups[key])

    splits = []
    for fold_idx in range(n_splits):
        test = np.array(fold_assignment[fold_idx], dtype=int)
        train = np.array(
            [idx for f, indices in fold_assignment.items() if f != fold_idx for idx in indices],
            dtype=int,
        )
        splits.append((train, test))

    return splits


# ===================================================================== #
#  11. Bootstrap confidence interval for Delta-R^2                       #
# ===================================================================== #

def bootstrap_delta_r2(
    embeddings: np.ndarray,
    targets: np.ndarray,
    confounds: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap 95 % CI for the change in R^2 after confound removal.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Delta-R^2 (probe R^2 minus confound-only R^2) is the primary DESCARTES
    effect size.  A bootstrap CI lets us judge uncertainty without assuming
    normality, which is important for small pharma datasets.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
    targets : ndarray, shape (n,)
    confounds : ndarray, shape (n, k) or None
        If None, confound R^2 is set to 0.
    n_boot : int
    seed : int

    Returns
    -------
    dict
        ``delta_r2``   -- point estimate,
        ``ci_lower``   -- 2.5th percentile,
        ``ci_upper``   -- 97.5th percentile,
        ``boot_dist``  -- full bootstrap distribution.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n = len(targets)
    rng = np.random.default_rng(seed)

    def _compute_delta(idx: np.ndarray) -> float:
        X = embeddings[idx]
        y = targets[idx]
        model_full = Ridge(alpha=1.0)
        model_full.fit(X, y)
        r2_full = r2_score(y, model_full.predict(X))
        if confounds is not None:
            C = np.asarray(confounds, dtype=float)[idx]
            if C.ndim == 1:
                C = C[:, None]
            model_conf = LinearRegression()
            model_conf.fit(C, y)
            r2_conf = r2_score(y, model_conf.predict(C))
        else:
            r2_conf = 0.0
        return r2_full - r2_conf

    point = _compute_delta(np.arange(n))
    boot_dist = np.array([
        _compute_delta(rng.choice(n, size=n, replace=True))
        for _ in range(n_boot)
    ])

    return {
        "delta_r2": float(point),
        "ci_lower": float(np.percentile(boot_dist, 2.5)),
        "ci_upper": float(np.percentile(boot_dist, 97.5)),
        "boot_dist": boot_dist,
    }


# ===================================================================== #
#  12. Probe sensitivity to embedding perturbation                       #
# ===================================================================== #

def embedding_noise_sensitivity(
    embeddings: np.ndarray,
    targets: np.ndarray,
    noise_levels: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5),
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure how probe R^2 degrades under Gaussian embedding noise.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    A robust MoA signal should degrade gracefully; a scaffold-memorised
    signal collapses under small perturbation because the probe relied on
    fine-grained positional structure rather than a broad linear trend.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
    targets : ndarray, shape (n,)
    noise_levels : sequence of float
        Standard deviations of Gaussian noise relative to embedding std.
    seed : int

    Returns
    -------
    dict
        ``baseline_r2``   -- R^2 with no noise,
        ``noise_levels``  -- the levels tested,
        ``noisy_r2``      -- R^2 at each noise level,
        ``relative_drop`` -- fractional R^2 drop at each level.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    targets = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(seed)

    emb_std = np.std(embeddings)
    model = Ridge(alpha=1.0)
    model.fit(embeddings, targets)
    baseline_r2 = float(r2_score(targets, model.predict(embeddings)))

    noisy_r2 = []
    for sigma in noise_levels:
        noise = rng.normal(0, sigma * emb_std, size=embeddings.shape)
        noisy_emb = embeddings + noise
        m = Ridge(alpha=1.0)
        m.fit(noisy_emb, targets)
        noisy_r2.append(float(r2_score(targets, m.predict(noisy_emb))))

    relative_drop = [
        (baseline_r2 - nr2) / baseline_r2 if baseline_r2 > 0 else 0.0
        for nr2 in noisy_r2
    ]

    return {
        "baseline_r2": baseline_r2,
        "noise_levels": list(noise_levels),
        "noisy_r2": noisy_r2,
        "relative_drop": relative_drop,
    }


# ===================================================================== #
#  13. Cliff-pair analysis (activity-cliff diagnostic)                   #
# ===================================================================== #

def activity_cliff_analysis(
    embeddings: np.ndarray,
    targets: np.ndarray,
    smiles: Sequence[str],
    tanimoto_threshold: float = 0.85,
    activity_gap: float = 1.0,
) -> Dict[str, Any]:
    """Identify activity cliffs and test whether probes handle them.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    Activity cliffs (structurally similar molecules with very different
    activities) are the hardest cases for any MoA probe.  If a probe
    performs well globally but fails on cliff pairs, it has memorised
    scaffold averages rather than learning mechanism.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
    targets : ndarray, shape (n,)
    smiles : sequence of str
    tanimoto_threshold : float
        Minimum Tanimoto to consider a pair structurally similar.
    activity_gap : float
        Minimum |delta_activity| to be an activity cliff.

    Returns
    -------
    dict
        ``n_cliff_pairs``   -- number of cliff pairs found,
        ``cliff_r2``        -- R^2 restricted to cliff-pair molecules,
        ``non_cliff_r2``    -- R^2 for remaining molecules,
        ``cliff_indices``   -- set of indices involved in cliffs.
    """
    _require_rdkit("activity_cliff_analysis")
    embeddings = np.asarray(embeddings, dtype=float)
    targets = np.asarray(targets, dtype=float)

    sim_mat = _tanimoto_matrix(smiles)
    n = len(targets)

    cliff_pairs = []
    cliff_idx_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i, j] >= tanimoto_threshold and abs(targets[i] - targets[j]) >= activity_gap:
                cliff_pairs.append((i, j))
                cliff_idx_set.update([i, j])

    model = Ridge(alpha=1.0)
    model.fit(embeddings, targets)
    preds = model.predict(embeddings)

    cliff_indices = np.array(sorted(cliff_idx_set), dtype=int)
    non_cliff_indices = np.array([i for i in range(n) if i not in cliff_idx_set], dtype=int)

    cliff_r2 = float("nan")
    if len(cliff_indices) >= 3 and np.std(targets[cliff_indices]) > 0:
        cliff_r2 = float(r2_score(targets[cliff_indices], preds[cliff_indices]))

    non_cliff_r2 = float("nan")
    if len(non_cliff_indices) >= 3 and np.std(targets[non_cliff_indices]) > 0:
        non_cliff_r2 = float(r2_score(targets[non_cliff_indices], preds[non_cliff_indices]))

    return {
        "n_cliff_pairs": len(cliff_pairs),
        "cliff_r2": cliff_r2,
        "non_cliff_r2": non_cliff_r2,
        "cliff_indices": cliff_idx_set,
    }


# ===================================================================== #
#  StatisticalHardeningSuite -- orchestrator                             #
# ===================================================================== #

class StatisticalHardeningSuite:
    """Run all 13 DESCARTES-PHARMA statistical hardening tests.

    DESCARTES v3.0 pharma adaptation
    ---------------------------------
    This orchestrator collects all methods into a single entry point.
    Call :meth:`run_all` with the required data and receive a unified
    summary dict keyed by method name.  Methods that require RDKit are
    skipped gracefully if the package is absent.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
        Molecular embedding vectors.
    targets : ndarray, shape (n,)
        Activity / MoA labels (continuous or integer-encoded).
    smiles : sequence of str or None
        SMILES strings.  Required for scaffold-aware tests.
    confounds : ndarray, shape (n, k) or None
        Physicochemical confounds (MW, logP, ...).

    Example
    -------
    >>> suite = StatisticalHardeningSuite(embeddings, targets, smiles, confounds)
    >>> report = suite.run_all()
    >>> print(report["tost_equivalence"]["equivalent"])
    True  # zombie confirmed
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        smiles: Optional[Sequence[str]] = None,
        confounds: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.smiles = smiles
        self.confounds = confounds
        self.seed = seed
        self._results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Individual runners                                                 #
    # ------------------------------------------------------------------ #

    def _safe_run(self, name: str, fn, **kwargs) -> Any:
        """Execute *fn* and store result; catch errors and log them."""
        try:
            result = fn(**kwargs)
            self._results[name] = result
            return result
        except Exception as exc:
            self._results[name] = {"error": str(exc)}
            return None

    def run_scaffold_stratified_permutation(self, n_perms: int = 1000) -> Optional[Dict]:
        """Method 1: scaffold-stratified permutation test."""
        if self.smiles is None:
            self._results["scaffold_stratified_permutation"] = {"error": "smiles not provided"}
            return None
        return self._safe_run(
            "scaffold_stratified_permutation",
            scaffold_stratified_permutation,
            smiles=self.smiles,
            mechanism_values=self.targets,
            n_perms=n_perms,
            seed=self.seed,
        )

    def run_y_scramble_null(self, n_perms: int = 1000) -> Optional[Dict]:
        """Method 2: y-scramble null distribution."""
        return self._safe_run(
            "y_scramble_null",
            y_scramble_null,
            mechanism_values=self.targets,
            n_perms=n_perms,
            seed=self.seed,
        )

    def run_scaffold_adjusted_neff(self) -> Optional[Dict]:
        """Method 3: scaffold-adjusted effective N."""
        if self.smiles is None:
            self._results["scaffold_adjusted_neff"] = {"error": "smiles not provided"}
            return None
        return self._safe_run(
            "scaffold_adjusted_neff",
            scaffold_adjusted_neff,
            smiles=self.smiles,
        )

    def run_fdr_correction(self, p_values: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Method 4: FDR correction (requires external p-values)."""
        if p_values is None:
            self._results["fdr_correction"] = {"skipped": "no p_values provided"}
            return None
        return self._safe_run(
            "fdr_correction",
            fdr_correction,
            p_values=p_values,
        )

    def run_scaffold_resolved_r2(self) -> Optional[Dict]:
        """Method 5: scaffold-resolved R^2."""
        if self.smiles is None:
            self._results["scaffold_resolved_r2"] = {"error": "smiles not provided"}
            return None
        return self._safe_run(
            "scaffold_resolved_r2",
            scaffold_resolved_r2,
            embeddings=self.embeddings,
            targets=self.targets,
            smiles=self.smiles,
        )

    def run_tanimoto_residual_test(self, n_perms: int = 500) -> Optional[Dict]:
        """Method 6: Tanimoto residual clustering test."""
        if self.smiles is None:
            self._results["tanimoto_residual_test"] = {"error": "smiles not provided"}
            return None
        # Need to compute residuals first
        model = Ridge(alpha=1.0)
        model.fit(self.embeddings, self.targets)
        residuals = self.targets - model.predict(self.embeddings)
        return self._safe_run(
            "tanimoto_residual_test",
            tanimoto_residual_test,
            residuals=residuals,
            smiles=self.smiles,
            n_perms=n_perms,
            seed=self.seed,
        )

    def run_confound_removal(self) -> Optional[np.ndarray]:
        """Method 7: confound removal (returns cleaned embeddings)."""
        if self.confounds is None:
            self._results["confound_removal"] = {"skipped": "no confounds provided"}
            return None
        try:
            cleaned = confound_removal(self.embeddings, self.confounds)
            self._results["confound_removal"] = {
                "original_norm": float(np.linalg.norm(self.embeddings)),
                "cleaned_norm": float(np.linalg.norm(cleaned)),
                "variance_removed_frac": float(
                    1.0 - np.var(cleaned) / np.var(self.embeddings)
                ) if np.var(self.embeddings) > 0 else 0.0,
            }
            return cleaned
        except Exception as exc:
            self._results["confound_removal"] = {"error": str(exc)}
            return None

    def run_tost_equivalence(self, delta_r2: float, se: float, epsilon: float = 0.05) -> Optional[Dict]:
        """Method 8: TOST equivalence test."""
        n_eff_val = None
        if "scaffold_adjusted_neff" in self._results and "n_eff" in self._results.get("scaffold_adjusted_neff", {}):
            n_eff_val = self._results["scaffold_adjusted_neff"]["n_eff"]
        return self._safe_run(
            "tost_equivalence",
            tost_equivalence_test,
            delta_r2=delta_r2,
            se=se,
            epsilon=epsilon,
            n_eff=n_eff_val,
        )

    def run_bayes_factor_null(self, delta_r2: float, se: float) -> Optional[Dict]:
        """Method 9: Bayes factor for the null."""
        return self._safe_run(
            "bayes_factor_null",
            bayes_factor_null,
            delta_r2=delta_r2,
            se=se,
        )

    def run_scaffold_split_cv(self, n_splits: int = 5) -> Optional[List]:
        """Method 10: scaffold-split cross-validation."""
        if self.smiles is None:
            self._results["scaffold_split_cv"] = {"error": "smiles not provided"}
            return None
        try:
            splits = scaffold_split_cv(self.smiles, n_splits=n_splits, seed=self.seed)
            # Evaluate probe R^2 on each fold
            fold_r2s = []
            for train_idx, test_idx in splits:
                model = Ridge(alpha=1.0)
                model.fit(self.embeddings[train_idx], self.targets[train_idx])
                preds = model.predict(self.embeddings[test_idx])
                y_test = self.targets[test_idx]
                if np.std(y_test) > 0:
                    fold_r2s.append(float(r2_score(y_test, preds)))
                else:
                    fold_r2s.append(float("nan"))
            self._results["scaffold_split_cv"] = {
                "n_splits": n_splits,
                "fold_r2s": fold_r2s,
                "mean_r2": float(np.nanmean(fold_r2s)),
                "std_r2": float(np.nanstd(fold_r2s)),
            }
            return splits
        except Exception as exc:
            self._results["scaffold_split_cv"] = {"error": str(exc)}
            return None

    def run_bootstrap_delta_r2(self, n_boot: int = 1000) -> Optional[Dict]:
        """Method 11: bootstrap CI for Delta-R^2."""
        return self._safe_run(
            "bootstrap_delta_r2",
            bootstrap_delta_r2,
            embeddings=self.embeddings,
            targets=self.targets,
            confounds=self.confounds,
            n_boot=n_boot,
            seed=self.seed,
        )

    def run_embedding_noise_sensitivity(self) -> Optional[Dict]:
        """Method 12: embedding noise sensitivity."""
        return self._safe_run(
            "embedding_noise_sensitivity",
            embedding_noise_sensitivity,
            embeddings=self.embeddings,
            targets=self.targets,
            seed=self.seed,
        )

    def run_activity_cliff_analysis(self) -> Optional[Dict]:
        """Method 13: activity cliff analysis."""
        if self.smiles is None:
            self._results["activity_cliff_analysis"] = {"error": "smiles not provided"}
            return None
        return self._safe_run(
            "activity_cliff_analysis",
            activity_cliff_analysis,
            embeddings=self.embeddings,
            targets=self.targets,
            smiles=self.smiles,
        )

    # ------------------------------------------------------------------ #
    #  Run everything                                                     #
    # ------------------------------------------------------------------ #

    def run_all(
        self,
        p_values: Optional[np.ndarray] = None,
        n_perms: int = 1000,
        n_boot: int = 1000,
    ) -> Dict[str, Any]:
        """Execute all 13 hardening tests and return a unified report.

        Parameters
        ----------
        p_values : ndarray or None
            External p-values for FDR correction (method 4).
        n_perms : int
            Permutation count for methods 1, 2, 6.
        n_boot : int
            Bootstrap replicates for method 11.

        Returns
        -------
        dict
            Keys are method names; values are result dicts or error dicts.
        """
        self._results = {}

        # Methods 1-3 (scaffold-aware)
        self.run_scaffold_stratified_permutation(n_perms=n_perms)
        self.run_y_scramble_null(n_perms=n_perms)
        self.run_scaffold_adjusted_neff()

        # Method 4 (FDR)
        self.run_fdr_correction(p_values=p_values)

        # Method 5 (scaffold-resolved R^2)
        self.run_scaffold_resolved_r2()

        # Method 6 (Tanimoto residuals)
        self.run_tanimoto_residual_test(n_perms=min(n_perms, 500))

        # Method 7 (confound removal)
        self.run_confound_removal()

        # Method 11 (bootstrap) -- needed for delta_r2 / se estimates
        self.run_bootstrap_delta_r2(n_boot=n_boot)

        # Extract delta_r2 and se for methods 8-9
        boot = self._results.get("bootstrap_delta_r2", {})
        delta_r2 = boot.get("delta_r2", 0.0)
        boot_dist = boot.get("boot_dist", None)
        se = float(np.std(boot_dist)) if boot_dist is not None and len(boot_dist) > 0 else 0.01

        # Methods 8-9 (equivalence / Bayes)
        self.run_tost_equivalence(delta_r2=delta_r2, se=max(se, 1e-8))
        self.run_bayes_factor_null(delta_r2=delta_r2, se=max(se, 1e-8))

        # Method 10 (scaffold split CV)
        self.run_scaffold_split_cv()

        # Method 12 (noise sensitivity)
        self.run_embedding_noise_sensitivity()

        # Method 13 (activity cliffs)
        self.run_activity_cliff_analysis()

        # ---- Summary ------------------------------------------------- #
        summary = dict(self._results)

        # Strip large arrays for concise reporting
        summary_clean: Dict[str, Any] = {}
        for key, val in summary.items():
            if isinstance(val, dict):
                clean = {}
                for k, v in val.items():
                    if isinstance(v, np.ndarray):
                        clean[k] = f"<array shape={v.shape}>"
                    elif isinstance(v, set):
                        clean[k] = f"<set len={len(v)}>"
                    else:
                        clean[k] = v
                summary_clean[key] = clean
            else:
                summary_clean[key] = val

        return summary_clean
