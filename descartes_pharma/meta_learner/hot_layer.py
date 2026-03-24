"""
Paradigm 7a -- Higher-Order Thought (HOT) Meta-Cognition Layer
Monitors the evaluation process itself and detects pathological states:
  - Stagnation:  no progress across recent probes
  - Redundancy:  probes returning similar information
  - Anomaly:     unexpected outlier results
  - Exhaustion:  diminishing returns from additional probing

Recommends corrective actions: skip tier, switch probe, halt, escalate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Assessment result
# ---------------------------------------------------------------------------

@dataclass
class HOTAssessment:
    """Result of a meta-cognitive assessment."""
    stagnation_score: float       # 0-1, 1 = total stagnation
    redundancy_score: float       # 0-1, 1 = fully redundant
    anomaly_score: float          # 0-1, 1 = extreme anomaly
    exhaustion_score: float       # 0-1, 1 = fully exhausted
    overall_health: float         # 0-1, 1 = healthy process
    recommended_action: str       # "continue", "skip_tier", "switch_probe", "halt", "escalate"
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HOT meta-cognition
# ---------------------------------------------------------------------------

class MetaCognitionHOT:
    """
    Higher-Order Thought layer that monitors the evaluation campaign
    and recommends meta-level actions.

    Maintains a sliding window of recent probe results and computes
    four diagnostic scores that together determine process health.
    """

    def __init__(
        self,
        window_size: int = 15,
        stagnation_threshold: float = 0.6,
        redundancy_threshold: float = 0.7,
        anomaly_threshold: float = 0.8,
        exhaustion_threshold: float = 0.7,
    ):
        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.redundancy_threshold = redundancy_threshold
        self.anomaly_threshold = anomaly_threshold
        self.exhaustion_threshold = exhaustion_threshold

        # Sliding windows
        self._r2_window: deque = deque(maxlen=window_size)
        self._p_window: deque = deque(maxlen=window_size)
        self._probe_type_window: deque = deque(maxlen=window_size)
        self._compute_window: deque = deque(maxlen=window_size)
        self._gain_window: deque = deque(maxlen=window_size)  # incremental gains

        self._prev_best_r2 = 0.0
        self._total_probes = 0
        self._assessment_history: List[HOTAssessment] = []

    def assess(
        self,
        probe_type: str,
        delta_r2: float,
        p_value: float,
        compute_seconds: float,
    ) -> HOTAssessment:
        """
        Perform a meta-cognitive assessment after receiving a probe result.

        Returns an HOTAssessment with scores and recommended action.
        """
        self._total_probes += 1

        # Track incremental gain
        gain = max(delta_r2 - self._prev_best_r2, 0.0)
        if delta_r2 > self._prev_best_r2:
            self._prev_best_r2 = delta_r2

        self._r2_window.append(delta_r2)
        self._p_window.append(p_value)
        self._probe_type_window.append(probe_type)
        self._compute_window.append(compute_seconds)
        self._gain_window.append(gain)

        # Compute diagnostic scores
        stagnation = self._detect_stagnation()
        redundancy = self._detect_redundancy()
        anomaly = self._detect_anomaly(delta_r2)
        exhaustion = self._detect_exhaustion()

        # Overall health: inverse of worst pathology
        worst = max(stagnation, redundancy, exhaustion)
        overall_health = 1.0 - worst

        # Decide action
        action, reason = self._recommend_action(
            stagnation, redundancy, anomaly, exhaustion
        )

        assessment = HOTAssessment(
            stagnation_score=stagnation,
            redundancy_score=redundancy,
            anomaly_score=anomaly,
            exhaustion_score=exhaustion,
            overall_health=overall_health,
            recommended_action=action,
            reason=reason,
            details={
                "total_probes": self._total_probes,
                "best_r2": self._prev_best_r2,
                "window_mean_r2": float(np.mean(list(self._r2_window))),
                "window_mean_gain": float(np.mean(list(self._gain_window))),
            },
        )
        self._assessment_history.append(assessment)
        return assessment

    def _detect_stagnation(self) -> float:
        """
        Stagnation = no improvement in recent probes.
        Measured as the fraction of recent gains that are near zero.
        """
        if len(self._gain_window) < 3:
            return 0.0

        gains = list(self._gain_window)
        near_zero = sum(1 for g in gains if g < 0.01)
        return near_zero / len(gains)

    def _detect_redundancy(self) -> float:
        """
        Redundancy = recent probes giving very similar R^2 values.
        Measured as 1 - coefficient of variation of recent R^2.
        """
        if len(self._r2_window) < 3:
            return 0.0

        r2_vals = list(self._r2_window)
        mean_r2 = np.mean(r2_vals)
        std_r2 = np.std(r2_vals)

        if mean_r2 < 1e-6:
            return 0.8  # All near zero = high redundancy

        cv = std_r2 / (abs(mean_r2) + 1e-6)
        # Low CV = high redundancy
        redundancy = max(0.0, 1.0 - cv * 2.0)

        # Also check probe type diversity
        recent_types = list(self._probe_type_window)
        unique_types = len(set(recent_types))
        type_diversity = unique_types / max(len(recent_types), 1)
        # Low diversity increases redundancy
        redundancy = redundancy * (1.0 + (1.0 - type_diversity)) / 2.0

        return min(redundancy, 1.0)

    def _detect_anomaly(self, latest_r2: float) -> float:
        """
        Anomaly = latest result is an outlier relative to the window.
        Uses modified z-score.
        """
        if len(self._r2_window) < 5:
            return 0.0

        r2_vals = list(self._r2_window)[:-1]  # exclude latest
        median = np.median(r2_vals)
        mad = np.median(np.abs(np.array(r2_vals) - median))
        mad = max(mad, 1e-6)

        modified_z = 0.6745 * abs(latest_r2 - median) / mad
        # Map z-score to 0-1
        anomaly = min(modified_z / 3.5, 1.0)
        return float(anomaly)

    def _detect_exhaustion(self) -> float:
        """
        Exhaustion = diminishing returns over time.
        Measured as the ratio of recent gains to early gains.
        """
        if len(self._gain_window) < 6:
            return 0.0

        gains = list(self._gain_window)
        half = len(gains) // 2
        early_gains = np.mean(gains[:half]) + 1e-6
        recent_gains = np.mean(gains[half:]) + 1e-6

        if early_gains > recent_gains:
            exhaustion = 1.0 - (recent_gains / early_gains)
        else:
            exhaustion = 0.0

        # Also consider total compute spent
        total_compute = sum(self._compute_window)
        if total_compute > 600:  # more than 10 minutes
            exhaustion = min(exhaustion + 0.2, 1.0)

        return float(exhaustion)

    def _recommend_action(
        self,
        stagnation: float,
        redundancy: float,
        anomaly: float,
        exhaustion: float,
    ) -> Tuple[str, str]:
        """
        Based on diagnostic scores, recommend a meta-level action.

        Priority order:
        1. High anomaly -> escalate (something unexpected, need expert review)
        2. High exhaustion -> halt (diminishing returns)
        3. High stagnation + redundancy -> skip_tier (try more powerful probes)
        4. High redundancy alone -> switch_probe
        5. Otherwise -> continue
        """
        if anomaly >= self.anomaly_threshold:
            return "escalate", (
                f"Anomaly detected (score={anomaly:.2f}): "
                "latest result is a significant outlier. Consider expert review."
            )

        if exhaustion >= self.exhaustion_threshold:
            return "halt", (
                f"Exhaustion detected (score={exhaustion:.2f}): "
                "diminishing returns from additional probing."
            )

        if stagnation >= self.stagnation_threshold and redundancy >= self.redundancy_threshold:
            return "skip_tier", (
                f"Stagnation ({stagnation:.2f}) + Redundancy ({redundancy:.2f}): "
                "current tier is not adding value. Skip to next tier."
            )

        if redundancy >= self.redundancy_threshold:
            return "switch_probe", (
                f"Redundancy detected ({redundancy:.2f}): "
                "try a different probe type for diversity."
            )

        if stagnation >= self.stagnation_threshold:
            return "skip_tier", (
                f"Stagnation detected ({stagnation:.2f}): "
                "no recent improvement. Try escalating to next tier."
            )

        return "continue", "Process is healthy. Continue current trajectory."

    def get_history(self) -> List[HOTAssessment]:
        return list(self._assessment_history)

    def reset(self) -> None:
        self._r2_window.clear()
        self._p_window.clear()
        self._probe_type_window.clear()
        self._compute_window.clear()
        self._gain_window.clear()
        self._prev_best_r2 = 0.0
        self._total_probes = 0
        self._assessment_history.clear()
