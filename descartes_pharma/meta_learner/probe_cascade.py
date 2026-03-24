"""
Paradigm 4 -- Probe Cascade Router
Routes evaluation through 5 tiers of increasing computational cost,
with early-exit when a confident verdict is reached.

Tier 0: ridge / lasso          (seconds)
Tier 1: mlp / knn              (seconds)
Tier 2: sae / cca / rsa        (minutes)
Tier 3: ablation / das         (minutes-hours)
Tier 4: llm_balloon            (minutes, LLM-based)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    """Configuration for a single probe."""
    name: str
    tier: int
    estimator_factory: Optional[Callable] = None  # returns a fitted probe
    timeout_seconds: float = 300.0
    min_samples: int = 50


@dataclass
class CascadeResult:
    """Result from running a single probe in the cascade."""
    probe_name: str
    tier: int
    delta_r2: float
    p_value: float
    compute_seconds: float
    raw_output: Any = None
    error: Optional[str] = None


@dataclass
class CascadeVerdict:
    """Aggregate verdict after running the cascade."""
    is_zombie: Optional[bool]      # True = zombie, False = real, None = inconclusive
    confidence: float              # 0-1
    tier_reached: int              # how deep we went
    results: List[CascadeResult] = field(default_factory=list)
    early_exit: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Default probe runners (lightweight implementations / placeholders)
# ---------------------------------------------------------------------------

def _run_ridge(activations: np.ndarray, labels: np.ndarray, **kw) -> Dict[str, float]:
    """Ridge regression probe."""
    from numpy.linalg import lstsq
    n, d = activations.shape
    # Add L2 regularisation via augmentation
    alpha = kw.get("alpha", 1.0)
    A = np.vstack([activations, np.sqrt(alpha) * np.eye(d)])
    b = np.concatenate([labels, np.zeros(d)])
    coef, _, _, _ = lstsq(A, b, rcond=None)
    pred = activations @ coef
    ss_res = np.sum((labels - pred) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"r2": float(r2), "p_value": 0.05 if r2 > 0.1 else 0.5}


def _run_lasso(activations: np.ndarray, labels: np.ndarray, **kw) -> Dict[str, float]:
    """Lasso (L1) regression probe via coordinate descent."""
    n, d = activations.shape
    alpha = kw.get("alpha", 0.01)
    coef = np.zeros(d)
    for _ in range(100):
        for j in range(d):
            residual = labels - activations @ coef + activations[:, j] * coef[j]
            rho = activations[:, j] @ residual / n
            coef[j] = np.sign(rho) * max(abs(rho) - alpha, 0.0)
    pred = activations @ coef
    ss_res = np.sum((labels - pred) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    sparsity = np.mean(np.abs(coef) < 1e-6)
    return {"r2": float(r2), "p_value": 0.05 if r2 > 0.1 else 0.5, "sparsity": float(sparsity)}


def _run_mlp_probe(activations: np.ndarray, labels: np.ndarray, **kw) -> Dict[str, float]:
    """Simple 1-hidden-layer MLP probe using numpy."""
    n, d = activations.shape
    h = kw.get("hidden", 64)
    lr = kw.get("lr", 0.01)
    epochs = kw.get("epochs", 50)
    rng = np.random.RandomState(42)

    W1 = rng.randn(d, h) * 0.01
    b1 = np.zeros(h)
    W2 = rng.randn(h, 1) * 0.01
    b2 = np.zeros(1)

    for _ in range(epochs):
        z1 = activations @ W1 + b1
        a1 = np.maximum(z1, 0)  # ReLU
        pred = (a1 @ W2 + b2).flatten()
        err = pred - labels

        dW2 = (a1.T @ err.reshape(-1, 1)) / n
        db2 = np.mean(err)
        da1 = err.reshape(-1, 1) @ W2.T
        da1[z1 <= 0] = 0
        dW1 = (activations.T @ da1) / n
        db1 = np.mean(da1, axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    pred = (np.maximum(activations @ W1 + b1, 0) @ W2 + b2).flatten()
    ss_res = np.sum((labels - pred) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"r2": float(r2), "p_value": 0.03 if r2 > 0.15 else 0.4}


def _run_knn_probe(activations: np.ndarray, labels: np.ndarray, **kw) -> Dict[str, float]:
    """K-nearest-neighbours probe."""
    k = kw.get("k", 5)
    n = activations.shape[0]
    # Leave-one-out cross-validation
    preds = np.zeros(n)
    for i in range(n):
        dists = np.sum((activations - activations[i]) ** 2, axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        preds[i] = np.mean(labels[neighbors])
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"r2": float(r2), "p_value": 0.04 if r2 > 0.12 else 0.45}


def _run_placeholder(activations: np.ndarray, labels: np.ndarray, **kw) -> Dict[str, float]:
    """Placeholder for expensive probes (SAE, CCA, RSA, ablation, DAS, LLM)."""
    # Return a synthetic result based on simple correlation
    if activations.shape[1] > 0:
        corr = np.corrcoef(activations[:, 0], labels)[0, 1]
        r2 = corr ** 2
    else:
        r2 = 0.0
    return {"r2": float(r2), "p_value": 0.02 if r2 > 0.2 else 0.3}


# ---------------------------------------------------------------------------
# Default tier configuration
# ---------------------------------------------------------------------------

DEFAULT_TIERS: Dict[int, List[ProbeConfig]] = {
    0: [
        ProbeConfig("ridge", 0, timeout_seconds=30),
        ProbeConfig("lasso", 0, timeout_seconds=30),
    ],
    1: [
        ProbeConfig("mlp", 1, timeout_seconds=60),
        ProbeConfig("knn", 1, timeout_seconds=60),
    ],
    2: [
        ProbeConfig("sae", 2, timeout_seconds=300),
        ProbeConfig("cca", 2, timeout_seconds=300),
        ProbeConfig("rsa", 2, timeout_seconds=300),
    ],
    3: [
        ProbeConfig("ablation", 3, timeout_seconds=600),
        ProbeConfig("das", 3, timeout_seconds=600),
    ],
    4: [
        ProbeConfig("llm_balloon", 4, timeout_seconds=300),
    ],
}

_PROBE_RUNNERS: Dict[str, Callable] = {
    "ridge": _run_ridge,
    "lasso": _run_lasso,
    "mlp": _run_mlp_probe,
    "knn": _run_knn_probe,
    "sae": _run_placeholder,
    "cca": _run_placeholder,
    "rsa": _run_placeholder,
    "ablation": _run_placeholder,
    "das": _run_placeholder,
    "llm_balloon": _run_placeholder,
}


# ---------------------------------------------------------------------------
# Cascade router
# ---------------------------------------------------------------------------

class ProbeCascadeRouter:
    """
    Runs probes in tiered order, checking for early exit after each tier.

    Tier 0 (fast):     ridge, lasso
    Tier 1 (medium):   mlp, knn
    Tier 2 (slow):     sae, cca, rsa
    Tier 3 (expensive): ablation, das
    Tier 4 (LLM):     llm_balloon
    """

    def __init__(
        self,
        tiers: Optional[Dict[int, List[ProbeConfig]]] = None,
        probe_runners: Optional[Dict[str, Callable]] = None,
        early_exit_confidence: float = 0.85,
        early_exit_agreement: float = 0.80,
        min_probes_for_exit: int = 2,
    ):
        self.tiers = tiers or DEFAULT_TIERS
        self.probe_runners = probe_runners or _PROBE_RUNNERS
        self.early_exit_confidence = early_exit_confidence
        self.early_exit_agreement = early_exit_agreement
        self.min_probes_for_exit = min_probes_for_exit

    def run_cascade(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        max_tier: int = 4,
        probe_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> CascadeVerdict:
        """
        Run the probe cascade up to max_tier, with early exit.

        Parameters
        ----------
        activations : array (n_samples, n_features)
        labels : array (n_samples,)
        max_tier : stop after this tier
        probe_kwargs : per-probe extra keyword arguments

        Returns
        -------
        CascadeVerdict with all results and final verdict.
        """
        probe_kwargs = probe_kwargs or {}
        all_results: List[CascadeResult] = []

        for tier_idx in sorted(self.tiers.keys()):
            if tier_idx > max_tier:
                break

            for probe_cfg in self.tiers[tier_idx]:
                runner = self.probe_runners.get(probe_cfg.name)
                if runner is None:
                    continue

                kw = probe_kwargs.get(probe_cfg.name, {})
                t0 = time.time()
                try:
                    out = runner(activations, labels, **kw)
                    elapsed = time.time() - t0
                    result = CascadeResult(
                        probe_name=probe_cfg.name,
                        tier=tier_idx,
                        delta_r2=out.get("r2", 0.0),
                        p_value=out.get("p_value", 1.0),
                        compute_seconds=elapsed,
                        raw_output=out,
                    )
                except Exception as e:
                    elapsed = time.time() - t0
                    result = CascadeResult(
                        probe_name=probe_cfg.name,
                        tier=tier_idx,
                        delta_r2=0.0,
                        p_value=1.0,
                        compute_seconds=elapsed,
                        error=str(e),
                    )

                all_results.append(result)

            # Check early exit after each tier
            verdict = self._check_early_verdict(all_results, tier_idx)
            if verdict is not None:
                return verdict

        # No early exit -- produce final verdict from all evidence
        return self._make_final_verdict(all_results, max_tier)

    def _check_early_verdict(
        self, results: List[CascadeResult], current_tier: int
    ) -> Optional[CascadeVerdict]:
        """
        Check if we can issue an early verdict.
        Requires:
          - At least min_probes_for_exit successful results
          - High agreement among probes (all say zombie or all say real)
          - Sufficient statistical significance
        """
        valid = [r for r in results if r.error is None]
        if len(valid) < self.min_probes_for_exit:
            return None

        # Count how many probes say "significant" vs "not significant"
        significant = [r for r in valid if r.p_value < 0.05 and r.delta_r2 > 0.1]
        not_significant = [r for r in valid if r.p_value >= 0.05 or r.delta_r2 <= 0.05]

        n_valid = len(valid)
        sig_ratio = len(significant) / n_valid
        not_sig_ratio = len(not_significant) / n_valid

        # Strong agreement that mechanism IS real (not zombie)
        if sig_ratio >= self.early_exit_agreement:
            confidence = sig_ratio * (1.0 - np.mean([r.p_value for r in significant]))
            if confidence >= self.early_exit_confidence:
                return CascadeVerdict(
                    is_zombie=False,
                    confidence=float(confidence),
                    tier_reached=current_tier,
                    results=results,
                    early_exit=True,
                    reason=f"Strong agreement ({sig_ratio:.0%}) that mechanism is real",
                )

        # Strong agreement that mechanism IS zombie
        if not_sig_ratio >= self.early_exit_agreement:
            confidence = not_sig_ratio
            if confidence >= self.early_exit_confidence:
                return CascadeVerdict(
                    is_zombie=True,
                    confidence=float(confidence),
                    tier_reached=current_tier,
                    results=results,
                    early_exit=True,
                    reason=f"Strong agreement ({not_sig_ratio:.0%}) that mechanism is zombie",
                )

        return None

    def _make_final_verdict(
        self, results: List[CascadeResult], max_tier: int
    ) -> CascadeVerdict:
        """Produce a final verdict from all accumulated evidence."""
        valid = [r for r in results if r.error is None]
        if not valid:
            return CascadeVerdict(
                is_zombie=None,
                confidence=0.0,
                tier_reached=max_tier,
                results=results,
                reason="No valid probe results",
            )

        significant = [r for r in valid if r.p_value < 0.05 and r.delta_r2 > 0.1]
        sig_ratio = len(significant) / len(valid)

        if sig_ratio >= 0.5:
            is_zombie = False
            confidence = sig_ratio
        elif sig_ratio <= 0.2:
            is_zombie = True
            confidence = 1.0 - sig_ratio
        else:
            is_zombie = None
            confidence = 0.5

        return CascadeVerdict(
            is_zombie=is_zombie,
            confidence=float(confidence),
            tier_reached=max_tier,
            results=results,
            reason=f"Final verdict after tier {max_tier}: {sig_ratio:.0%} probes significant",
        )
