"""
Paradigm 6 -- Multi-Timescale Processor
Manages three processing timescales inspired by neural oscillation theory:

  fast_tick   (~probe level)   : immediate probe results, reactive adjustments
  medium_tick (~campaign level): cross-probe patterns, within-model summaries
  slow_tick   (~portfolio level): cross-model learning, strategy updates

Phase-amplitude coupling: slow gates modulate which fast signals get amplified.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Timescale state containers
# ---------------------------------------------------------------------------

@dataclass
class FastState:
    """Per-probe-tick state (gamma band analogy)."""
    recent_r2: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_p_values: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_compute: deque = field(default_factory=lambda: deque(maxlen=20))
    momentum: float = 0.0          # EMA of delta_r2
    surprise: float = 0.0          # deviation from expected
    tick_count: int = 0


@dataclass
class MediumState:
    """Per-campaign-tick state (theta band analogy)."""
    probe_type_scores: Dict[str, List[float]] = field(default_factory=dict)
    cumulative_r2: float = 0.0
    best_r2: float = 0.0
    stagnation_counter: int = 0
    tick_count: int = 0


@dataclass
class SlowState:
    """Cross-campaign state (delta band analogy)."""
    architecture_priors: Dict[str, float] = field(default_factory=dict)
    mechanism_difficulty: Dict[str, float] = field(default_factory=dict)
    global_learning_rate: float = 1.0
    slow_gates: Dict[str, float] = field(default_factory=dict)  # modulates fast signals
    tick_count: int = 0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class MultiTimescaleProcessor:
    """
    Three-timescale processing with phase-amplitude coupling.

    The slow_gates (updated at slow_tick) modulate the impact of fast_tick
    signals, implementing top-down attentional control over which probes
    and mechanisms get priority.
    """

    def __init__(
        self,
        fast_ema_alpha: float = 0.3,
        medium_decay: float = 0.95,
        slow_learning_rate: float = 0.01,
        stagnation_threshold: int = 5,
    ):
        self.fast = FastState()
        self.medium = MediumState()
        self.slow = SlowState()

        self.fast_ema_alpha = fast_ema_alpha
        self.medium_decay = medium_decay
        self.slow_learning_rate = slow_learning_rate
        self.stagnation_threshold = stagnation_threshold

    # -----------------------------------------------------------------------
    # Fast tick: individual probe results
    # -----------------------------------------------------------------------

    def fast_tick(
        self,
        probe_type: str,
        delta_r2: float,
        p_value: float,
        compute_seconds: float,
    ) -> Dict[str, float]:
        """
        Process a single probe result at the fastest timescale.

        Returns fast-level signals: momentum, surprise, gated_impact.
        """
        self.fast.tick_count += 1

        # Record raw values
        self.fast.recent_r2.append(delta_r2)
        self.fast.recent_p_values.append(p_value)
        self.fast.recent_compute.append(compute_seconds)

        # Update momentum (exponential moving average of delta_r2)
        self.fast.momentum = (
            self.fast_ema_alpha * delta_r2
            + (1 - self.fast_ema_alpha) * self.fast.momentum
        )

        # Surprise: how far is this from the running mean?
        if len(self.fast.recent_r2) > 1:
            mean_r2 = np.mean(list(self.fast.recent_r2)[:-1])
            std_r2 = max(np.std(list(self.fast.recent_r2)[:-1]), 1e-6)
            self.fast.surprise = abs(delta_r2 - mean_r2) / std_r2
        else:
            self.fast.surprise = 0.0

        # Phase-amplitude coupling: slow gates modulate fast impact
        gate = self.slow.slow_gates.get(probe_type, 1.0)
        gated_impact = delta_r2 * gate

        return {
            "momentum": self.fast.momentum,
            "surprise": self.fast.surprise,
            "gated_impact": gated_impact,
            "gate_value": gate,
            "running_mean_r2": float(np.mean(list(self.fast.recent_r2))),
        }

    # -----------------------------------------------------------------------
    # Medium tick: campaign-level aggregation
    # -----------------------------------------------------------------------

    def medium_tick(
        self,
        probe_type: str,
        delta_r2: float,
        mechanism_name: str = "",
    ) -> Dict[str, Any]:
        """
        Process at the campaign level (called less frequently than fast_tick,
        typically after a batch of probes or at tier boundaries).

        Tracks per-probe-type performance and detects stagnation.
        """
        self.medium.tick_count += 1

        # Record per-probe-type scores
        if probe_type not in self.medium.probe_type_scores:
            self.medium.probe_type_scores[probe_type] = []
        self.medium.probe_type_scores[probe_type].append(delta_r2)

        # Update cumulative and best R2
        self.medium.cumulative_r2 = (
            self.medium_decay * self.medium.cumulative_r2 + delta_r2
        )
        if delta_r2 > self.medium.best_r2:
            self.medium.best_r2 = delta_r2
            self.medium.stagnation_counter = 0
        else:
            self.medium.stagnation_counter += 1

        # Compute probe-type rankings
        probe_rankings = {}
        for pt, scores in self.medium.probe_type_scores.items():
            probe_rankings[pt] = {
                "mean": float(np.mean(scores)),
                "max": float(np.max(scores)),
                "count": len(scores),
            }

        is_stagnating = self.medium.stagnation_counter >= self.stagnation_threshold

        return {
            "cumulative_r2": self.medium.cumulative_r2,
            "best_r2": self.medium.best_r2,
            "stagnation_counter": self.medium.stagnation_counter,
            "is_stagnating": is_stagnating,
            "probe_rankings": probe_rankings,
        }

    # -----------------------------------------------------------------------
    # Slow tick: portfolio-level learning
    # -----------------------------------------------------------------------

    def slow_tick(
        self,
        architecture: str,
        mechanism_name: str,
        campaign_r2: float,
        campaign_confidence: float,
    ) -> Dict[str, Any]:
        """
        Process at the portfolio level (called at the end of each campaign
        or periodically across campaigns).

        Updates architecture priors, mechanism difficulty estimates, and
        slow gates that modulate fast-tick processing.
        """
        self.slow.tick_count += 1

        # Update architecture priors (running average of campaign quality)
        if architecture not in self.slow.architecture_priors:
            self.slow.architecture_priors[architecture] = 0.5
        self.slow.architecture_priors[architecture] = (
            (1 - self.slow_learning_rate) * self.slow.architecture_priors[architecture]
            + self.slow_learning_rate * campaign_r2
        )

        # Update mechanism difficulty (inverse of confidence achieved)
        if mechanism_name not in self.slow.mechanism_difficulty:
            self.slow.mechanism_difficulty[mechanism_name] = 0.5
        difficulty = 1.0 - campaign_confidence
        self.slow.mechanism_difficulty[mechanism_name] = (
            (1 - self.slow_learning_rate) * self.slow.mechanism_difficulty[mechanism_name]
            + self.slow_learning_rate * difficulty
        )

        # Update slow gates
        self._update_slow_gates()

        # Adjust global learning rate based on overall performance
        all_priors = list(self.slow.architecture_priors.values())
        if all_priors:
            mean_prior = np.mean(all_priors)
            # If we are consistently getting low R2, speed up learning
            if mean_prior < 0.2:
                self.slow.global_learning_rate = min(self.slow.global_learning_rate * 1.1, 5.0)
            elif mean_prior > 0.5:
                self.slow.global_learning_rate = max(self.slow.global_learning_rate * 0.95, 0.1)

        return {
            "architecture_priors": dict(self.slow.architecture_priors),
            "mechanism_difficulty": dict(self.slow.mechanism_difficulty),
            "global_learning_rate": self.slow.global_learning_rate,
            "slow_gates": dict(self.slow.slow_gates),
        }

    def _update_slow_gates(self) -> None:
        """
        Phase-amplitude coupling: compute slow gates that modulate fast signals.

        Probes that have historically been more informative get higher gates.
        This implements the slow oscillation modulating the amplitude of fast
        gamma-band activity.
        """
        if not self.medium.probe_type_scores:
            return

        # Compute information value for each probe type
        probe_values: Dict[str, float] = {}
        for probe_type, scores in self.medium.probe_type_scores.items():
            if scores:
                # Value = mean R2 * consistency (low variance = reliable)
                mean_score = np.mean(scores)
                std_score = max(np.std(scores), 1e-6)
                consistency = 1.0 / (1.0 + std_score)
                probe_values[probe_type] = mean_score * consistency
            else:
                probe_values[probe_type] = 0.5

        # Normalize to [0.1, 2.0] range for gating
        if probe_values:
            max_val = max(probe_values.values()) or 1.0
            for probe_type, val in probe_values.items():
                normalized = 0.1 + 1.9 * (val / max_val)
                # Smooth update
                old_gate = self.slow.slow_gates.get(probe_type, 1.0)
                self.slow.slow_gates[probe_type] = (
                    (1 - self.slow_learning_rate) * old_gate
                    + self.slow_learning_rate * normalized
                )

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of all three timescale states."""
        return {
            "fast": {
                "tick_count": self.fast.tick_count,
                "momentum": self.fast.momentum,
                "surprise": self.fast.surprise,
                "recent_r2_mean": float(np.mean(list(self.fast.recent_r2))) if self.fast.recent_r2 else 0.0,
            },
            "medium": {
                "tick_count": self.medium.tick_count,
                "cumulative_r2": self.medium.cumulative_r2,
                "best_r2": self.medium.best_r2,
                "stagnation_counter": self.medium.stagnation_counter,
                "probe_types_seen": list(self.medium.probe_type_scores.keys()),
            },
            "slow": {
                "tick_count": self.slow.tick_count,
                "global_learning_rate": self.slow.global_learning_rate,
                "n_architectures": len(self.slow.architecture_priors),
                "n_mechanisms": len(self.slow.mechanism_difficulty),
                "slow_gates": dict(self.slow.slow_gates),
            },
        }

    def reset(self) -> None:
        """Reset all timescale states."""
        self.fast = FastState()
        self.medium = MediumState()
        self.slow = SlowState()
