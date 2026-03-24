"""
Paradigm 5 -- Variational Free-Energy (VFE) Belief System
Uses a Kalman-filter-style belief update with precision-weighted channels
for each probe type. The system accumulates evidence and decides when
enough probing has been done to issue a confident verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProbeChannel:
    """Precision channel for a single probe type."""
    name: str
    base_precision: float = 1.0      # prior precision (inverse variance)
    current_precision: float = 1.0   # updated precision after observations
    observation_count: int = 0
    cumulative_evidence: float = 0.0  # sum of precision-weighted observations
    last_observation: float = 0.0


DEFAULT_CHANNELS: Dict[str, float] = {
    "ridge": 0.5,
    "lasso": 0.5,
    "mlp": 0.8,
    "knn": 0.6,
    "sae": 1.5,
    "cca": 1.2,
    "rsa": 1.2,
    "ablation": 2.0,
    "das": 2.5,
    "llm_balloon": 1.0,
}


# ---------------------------------------------------------------------------
# VFE Belief System
# ---------------------------------------------------------------------------

class VFEBeliefSystem:
    """
    Maintains a Gaussian belief state about mechanism validity.

    State: N(mu, sigma^2)
      - mu > 0.5  => likely real mechanism
      - mu < 0.5  => likely zombie
      - sigma     => uncertainty

    Each probe observation updates the belief via Kalman-filter equations:
      precision_posterior = precision_prior + channel_precision
      mu_posterior = (precision_prior * mu_prior + channel_precision * obs) / precision_posterior

    The system tracks free energy F = -log p(observations | model) and uses
    it to decide when to stop probing.
    """

    def __init__(
        self,
        prior_mu: float = 0.5,
        prior_sigma: float = 0.3,
        channel_precisions: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.85,
        min_observations: int = 3,
        max_free_energy_delta: float = 0.01,
    ):
        self.mu = prior_mu
        self.sigma = prior_sigma
        self.precision = 1.0 / (prior_sigma ** 2)  # tau = 1/sigma^2

        self.confidence_threshold = confidence_threshold
        self.min_observations = min_observations
        self.max_free_energy_delta = max_free_energy_delta

        # Per-channel state
        self.channels: Dict[str, ProbeChannel] = {}
        precisions = channel_precisions or DEFAULT_CHANNELS
        for name, base_prec in precisions.items():
            self.channels[name] = ProbeChannel(
                name=name,
                base_precision=base_prec,
                current_precision=base_prec,
            )

        # History
        self._observations: List[Tuple[str, float, float]] = []  # (channel, obs, precision)
        self._free_energy_history: List[float] = []
        self._mu_history: List[float] = [prior_mu]
        self._sigma_history: List[float] = [prior_sigma]

    @property
    def total_observations(self) -> int:
        return len(self._observations)

    def update(self, probe_type: str, observation: float, p_value: float = 0.05) -> Dict[str, float]:
        """
        Update beliefs with a new probe observation.

        Parameters
        ----------
        probe_type : str
            Which probe channel produced this observation.
        observation : float
            The observed evidence value (0 = strong zombie, 1 = strong real).
            Typically derived from R^2 and significance of the probe.
        p_value : float
            Statistical significance; used to modulate channel precision.

        Returns
        -------
        dict with updated mu, sigma, free_energy, confidence.
        """
        # Get or create channel
        if probe_type not in self.channels:
            self.channels[probe_type] = ProbeChannel(
                name=probe_type,
                base_precision=0.5,
                current_precision=0.5,
            )
        channel = self.channels[probe_type]

        # Modulate precision by p-value (lower p = higher precision)
        significance_boost = max(0.1, -math.log10(max(p_value, 1e-10)) / 5.0)
        effective_precision = channel.base_precision * significance_boost

        # Kalman update
        old_precision = self.precision
        old_mu = self.mu

        self.precision = old_precision + effective_precision
        self.mu = (old_precision * old_mu + effective_precision * observation) / self.precision
        self.sigma = 1.0 / math.sqrt(self.precision)

        # Update channel state
        channel.current_precision += effective_precision
        channel.observation_count += 1
        channel.cumulative_evidence += effective_precision * observation
        channel.last_observation = observation

        # Record history
        self._observations.append((probe_type, observation, effective_precision))
        self._mu_history.append(self.mu)
        self._sigma_history.append(self.sigma)

        # Compute free energy
        fe = self._compute_free_energy()
        self._free_energy_history.append(fe)

        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "free_energy": fe,
            "confidence": self._compute_confidence(),
            "effective_precision": effective_precision,
        }

    def _compute_free_energy(self) -> float:
        """
        Variational free energy F = KL(q || p) - E_q[log p(data | theta)]
        Approximated as negative log-evidence under the current Gaussian belief.
        """
        # KL divergence from prior N(0.5, 0.3^2) to posterior N(mu, sigma^2)
        prior_mu, prior_sigma = 0.5, 0.3
        prior_precision = 1.0 / (prior_sigma ** 2)

        kl = 0.5 * (
            math.log(prior_sigma / self.sigma)
            + (self.sigma ** 2 + (self.mu - prior_mu) ** 2) / (prior_sigma ** 2)
            - 1.0
        )

        # Negative expected log-likelihood (lower = better fit)
        neg_ll = 0.0
        for _, obs, prec in self._observations:
            neg_ll += 0.5 * prec * (obs - self.mu) ** 2

        return float(kl + neg_ll)

    def _compute_confidence(self) -> float:
        """
        Confidence = how far mu is from 0.5 (the undecided boundary),
        scaled by precision.
        """
        distance = abs(self.mu - 0.5)
        precision_factor = min(self.precision / 10.0, 1.0)
        return min(2.0 * distance * precision_factor + 0.5 * precision_factor, 1.0)

    def should_continue_probing(self) -> Tuple[bool, str]:
        """
        Decide whether more probes are needed.

        Returns (should_continue, reason).
        """
        if self.total_observations < self.min_observations:
            return True, f"Need at least {self.min_observations} observations (have {self.total_observations})"

        confidence = self._compute_confidence()
        if confidence >= self.confidence_threshold:
            return False, f"Confidence {confidence:.3f} >= threshold {self.confidence_threshold}"

        # Check if free energy has plateaued
        if len(self._free_energy_history) >= 2:
            delta_fe = abs(self._free_energy_history[-1] - self._free_energy_history[-2])
            if delta_fe < self.max_free_energy_delta:
                return False, f"Free energy plateaued (delta={delta_fe:.4f} < {self.max_free_energy_delta})"

        return True, "Uncertainty remains high; continue probing"

    def get_zombie_verdict(self) -> Dict[str, object]:
        """
        Issue a verdict based on current beliefs.

        Returns dict with is_zombie, confidence, mu, sigma, evidence_summary.
        """
        confidence = self._compute_confidence()

        if self.mu > 0.6:
            is_zombie = False
            verdict_str = "REAL"
        elif self.mu < 0.4:
            is_zombie = True
            verdict_str = "ZOMBIE"
        else:
            is_zombie = None
            verdict_str = "INCONCLUSIVE"

        # Channel summary
        channel_summary = {}
        for name, ch in self.channels.items():
            if ch.observation_count > 0:
                channel_summary[name] = {
                    "observations": ch.observation_count,
                    "effective_precision": ch.current_precision,
                    "last_observation": ch.last_observation,
                }

        return {
            "is_zombie": is_zombie,
            "verdict": verdict_str,
            "confidence": confidence,
            "mu": self.mu,
            "sigma": self.sigma,
            "total_observations": self.total_observations,
            "free_energy": self._free_energy_history[-1] if self._free_energy_history else 0.0,
            "channel_summary": channel_summary,
        }

    def reset(self, prior_mu: float = 0.5, prior_sigma: float = 0.3) -> None:
        """Reset to prior beliefs."""
        self.mu = prior_mu
        self.sigma = prior_sigma
        self.precision = 1.0 / (prior_sigma ** 2)
        for ch in self.channels.values():
            ch.current_precision = ch.base_precision
            ch.observation_count = 0
            ch.cumulative_evidence = 0.0
            ch.last_observation = 0.0
        self._observations.clear()
        self._free_energy_history.clear()
        self._mu_history = [prior_mu]
        self._sigma_history = [prior_sigma]
