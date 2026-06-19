"""Module S: SINDy on the GRU update -- symbolic dynamics of the hidden state.

Recovers an approximate symbolic law for how the hidden state evolves,
h_{k+1} = f(h_k, u_k), then relates the reduced coordinates back to mechanism.
This reads the function's DYNAMICS, not just its static representation.

CORROBORATING EVIDENCE ONLY. A low-dimensional symbolic fit is an interpretive
lens, never proof -- if the one-step prediction R2 is low the equations are not
trustworthy and we say so loudly.

Pipeline:
  S1  PCA-reduce the 128-d hidden state to k PCs (~90% variance).
  S2  Discrete-time SINDy *with control* (the update depends on h AND the input):
        h_{k+1} = f(h_k, u_k),  u = mechanistic subset of the observation.
      Trajectories are split by episode so no transition crosses an episode
      boundary (same grouping discipline as GroupKFold).
  S3  Correlate each PC with each mechanistic feature (incl. held-out) so the
      equation means something: "PC1 ~ dist_asp228 and the dynamics drive PC1 to
      a fixed point" = the policy implements a controller toward catalytic contact.
  S4  Report equations, active terms, sparsity, held-out one-step prediction R2,
      and the PC<->feature map.
"""
import numpy as np
from sklearn.decomposition import PCA

try:
    import pysindy as ps
    _PYSINDY = True
except Exception:
    _PYSINDY = False


def _split_by_episode(arr, episode_ids):
    """List of per-episode contiguous sub-arrays, preserving timestep order.

    The collector appends all timesteps of episode 0, then episode 1, ... so
    episodes are contiguous runs -- exactly what a trajectory needs.
    """
    episode_ids = np.asarray(episode_ids)
    out = []
    if len(arr) == 0:
        return out
    start = 0
    for i in range(1, len(episode_ids) + 1):
        if i == len(episode_ids) or episode_ids[i] != episode_ids[start]:
            out.append(arr[start:i])
            start = i
    return out


def fit_sindy_dynamics(
    hidden_states,
    episode_ids,
    targets,
    control_features=("dist_asp32", "dist_asp228", "vina_score", "n_hbonds"),
    variance_kept=0.90,
    max_pcs=10,
    poly_degree=2,
    stlsq_threshold=0.05,
    test_frac=0.2,
    seed=0,
):
    """Fit reduced symbolic dynamics of the GRU hidden state and map PCs to features.

    Returns a dict with: sindy_available, k_pcs, pca_variance, control_features,
    pc_top_feature ({PCi: (feature, corr)}), pc_feature_corr (full matrix),
    and -- when pysindy is installed and the fit succeeds -- equations,
    n_active_terms, sparsity, pred_r2, plus a `warning` string.
    """
    H = np.asarray(hidden_states, dtype=np.float64)
    n = len(H)
    result = {"sindy_available": _PYSINDY, "warning": ""}
    if n < 10:
        result["warning"] = "too few timesteps for SINDy"
        return result

    # ---- S1: PCA reduce ------------------------------------------------------
    pca = PCA().fit(H)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance_kept) + 1)
    k = max(2, min(k, max_pcs, H.shape[1]))
    Xpc = pca.transform(H)[:, :k]
    result["k_pcs"] = k
    result["pca_variance"] = float(cum[k - 1])

    # control input u = mechanistic subset of the observation/targets
    feats = [f for f in control_features if f in targets]
    if feats:
        U = np.column_stack(
            [np.asarray(targets[f], dtype=np.float64)[:n] for f in feats]
        )
    else:
        U = np.zeros((n, 1))
    result["control_features"] = feats

    # ---- S3: PC <-> mechanistic feature correlation (incl. held-out) ---------
    corr, top = {}, {}
    for j in range(k):
        pc = Xpc[:, j]
        row, best_f, best_c = {}, None, 0.0
        for fname, fvals in targets.items():
            fv = np.asarray(fvals, dtype=np.float64)[:n]
            if fv.std() < 1e-9 or pc.std() < 1e-9:
                c = 0.0
            else:
                c = float(np.corrcoef(pc, fv)[0, 1])
            row[fname] = c
            if abs(c) > abs(best_c):
                best_c, best_f = c, fname
        corr[f"PC{j + 1}"] = row
        top[f"PC{j + 1}"] = (best_f, round(best_c, 3))
    result["pc_feature_corr"] = corr
    result["pc_top_feature"] = top

    # ---- S2/S4: discrete-time SINDy with control -----------------------------
    X_list = _split_by_episode(Xpc, episode_ids)
    U_list = _split_by_episode(U, episode_ids)
    pairs = [(x, u) for x, u in zip(X_list, U_list) if len(x) >= 3]

    if not _PYSINDY:
        result["warning"] = "pysindy not installed -- PCA + PC/feature map only"
        return result
    if len(pairs) < 2:
        result["warning"] = "too few multi-step episodes for a SINDy fit"
        return result

    X_list = [p[0] for p in pairs]
    U_list = [p[1] for p in pairs]
    try:
        # Honest one-step prediction: hold out whole episodes.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X_list))
        n_test = max(1, int(len(X_list) * test_frac))
        test_i = set(order[:n_test].tolist())
        Xtr = [X_list[i] for i in range(len(X_list)) if i not in test_i]
        Utr = [U_list[i] for i in range(len(X_list)) if i not in test_i]
        Xte = [X_list[i] for i in range(len(X_list)) if i in test_i]
        Ute = [U_list[i] for i in range(len(X_list)) if i in test_i]

        # NOTE: pysindy control API -- discrete_time + multiple_trajectories + u.
        # If a future pysindy changes this signature, adapt here.
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=poly_degree),
            optimizer=ps.STLSQ(threshold=stlsq_threshold),
            discrete_time=True,
        )
        model.fit(Xtr, u=Utr, multiple_trajectories=True)
        result["equations"] = list(model.equations())
        coef = np.asarray(model.coefficients())
        result["n_active_terms"] = int(np.count_nonzero(coef))
        result["sparsity"] = float(np.mean(coef == 0))
        try:
            r2 = float(model.score(Xte, u=Ute, multiple_trajectories=True))
        except Exception:
            r2 = float("nan")
        result["pred_r2"] = r2
        if not np.isfinite(r2) or r2 < 0.5:
            result["warning"] = (
                f"low one-step prediction R2={r2:.3f} -- reduced model "
                f"inadequate; equations NOT trustworthy"
            )
    except Exception as e:
        result["warning"] = f"SINDy fit failed: {type(e).__name__}: {e}"

    return result


def format_sindy_report(result) -> str:
    """Human-readable Module S summary block."""
    lines = ["  [Module S] SINDy hidden-state dynamics (CORROBORATING only):"]
    if "k_pcs" in result:
        lines.append(f"    PCA: {result['k_pcs']} PCs kept "
                     f"({result.get('pca_variance', 0):.2f} variance)")
    for pc, (feat, c) in result.get("pc_top_feature", {}).items():
        lines.append(f"    {pc} ~ {feat} (r={c})")
    if result.get("sindy_available"):
        if "pred_r2" in result:
            lines.append(f"    one-step prediction R2: {result['pred_r2']:.3f} "
                         f"| active terms: {result.get('n_active_terms', '?')}")
        for eq in result.get("equations", [])[:6]:
            lines.append(f"      {eq}")
    if result.get("warning"):
        lines.append(f"    WARNING: {result['warning']}")
    return "\n".join(lines)
