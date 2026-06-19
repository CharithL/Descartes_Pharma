"""Module M: MDL / prequential codelength probing.

A capacity-controlled encoding measure that backs up Delta-R2 against the
"your probe is too powerful" critique (Hewitt & Liang 2019). Instead of asking
"can a probe fit y?", it asks "how many bits does it take to TRANSMIT y given the
hidden states, paying as you go?" -- a powerful probe that merely memorizes does
not compress.

Prequential (online) code: walk through the data in ~10 increasing portions; at
each step train Ridge on everything seen so far, predict the next portion, and
pay a Gaussian codelength for the residual. Compare the total to the uniform
code that knows only the marginal variance of y.

  compression_ratio = 1 - total_codelength / uniform_codelength
  "encoded/compressed" if compression_ratio > ~0.1

Portions respect episode/trajectory boundaries (whole trajectories move together),
matching the GroupKFold discipline so memorized within-trajectory structure can't
inflate compression.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _grouped_portions(n, episode_ids, n_portions=10):
    """Increasing index portions that keep whole episodes together, in order."""
    if episode_ids is None:
        bounds = np.linspace(0, n, n_portions + 1).astype(int)
        return [np.arange(bounds[i], bounds[i + 1])
                for i in range(n_portions) if bounds[i + 1] > bounds[i]]
    episode_ids = np.asarray(episode_ids)
    uniq = list(dict.fromkeys(episode_ids.tolist()))   # first-appearance order
    portions = []
    for g in np.array_split(uniq, min(n_portions, len(uniq))):
        idx = np.where(np.isin(episode_ids, list(g)))[0]
        if len(idx) > 0:
            portions.append(np.sort(idx))
    return portions


def _gaussian_codelength(mse, m):
    """Bits (nats) to code m residuals under a Gaussian of variance mse."""
    mse = max(float(mse), 1e-12)
    return 0.5 * m * np.log(2 * np.pi * np.e * mse)


def mdl_codelength(X, y, episode_ids=None, n_portions=10, alpha=1.0):
    """Prequential description length of y given X.

    Returns dict: codelength, uniform_codelength, compression_ratio, compressed.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    portions = _grouped_portions(n, episode_ids, n_portions)
    if len(portions) < 2 or y.std() < 1e-9:
        return {"codelength": float("nan"), "uniform_codelength": float("nan"),
                "compression_ratio": 0.0, "compressed": False}

    # First portion has no prior model: code it under its own variance.
    seen = portions[0]
    total = _gaussian_codelength(max(np.var(y[seen]), 1e-12), len(seen))
    for p in range(1, len(portions)):
        nxt = portions[p]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X[seen], y[seen])
        pred = model.predict(X[nxt])
        mse = float(np.mean((y[nxt] - pred) ** 2))
        total += _gaussian_codelength(mse, len(nxt))
        seen = np.concatenate([seen, nxt])

    uniform = _gaussian_codelength(max(float(np.var(y)), 1e-12), n)
    ratio = 1.0 - total / uniform if uniform != 0 else 0.0
    return {"codelength": float(total), "uniform_codelength": float(uniform),
            "compression_ratio": float(ratio), "compressed": bool(ratio > 0.1)}


def mdl_for_targets(H, targets, episode_ids=None, n_portions=10):
    """Run mdl_codelength for every target -> {name: result-dict}."""
    out = {}
    for name, y in targets.items():
        y = np.asarray(y)
        if len(y) != len(H):
            continue
        out[name] = mdl_codelength(H, y, episode_ids, n_portions)
    return out
