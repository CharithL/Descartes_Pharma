"""
Paradigm 2 -- Mechanism Decomposer
Classifies claimed mechanisms into orthogonal categories (STRUCTURAL,
FUNCTIONAL, CAUSAL, DYNAMIC) so the probe cascade can target each
dimension independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mechanism taxonomy
# ---------------------------------------------------------------------------

MECHANISM_TYPES: Dict[str, Dict[str, object]] = {
    "STRUCTURAL": {
        "description": "Claims about which components (heads, layers, neurons) are involved",
        "keywords": [
            "head", "layer", "neuron", "mlp", "attention", "residual",
            "embedding", "position", "weight", "parameter", "circuit",
            "subnetwork", "pathway", "component", "module",
        ],
        "probe_affinity": ["ridge", "lasso", "sae", "ablation"],
    },
    "FUNCTIONAL": {
        "description": "Claims about what computation a component performs (e.g. copying, matching)",
        "keywords": [
            "copy", "match", "inhibit", "suppress", "amplify", "gate",
            "route", "select", "compose", "detect", "classify", "predict",
            "transform", "encode", "decode", "retrieve", "store",
            "induction", "duplicate", "move", "shift",
        ],
        "probe_affinity": ["mlp", "knn", "cca", "rsa", "das"],
    },
    "CAUSAL": {
        "description": "Claims about necessity/sufficiency -- removing X breaks Y",
        "keywords": [
            "necessary", "sufficient", "causal", "ablation", "knockout",
            "intervention", "counterfactual", "patch", "activation_patch",
            "interchange", "denoising", "noising", "mean_ablation",
            "zero_ablation", "resample", "indirect_effect", "direct_effect",
            "mediation", "path_specific",
        ],
        "probe_affinity": ["ablation", "das", "llm_balloon"],
    },
    "DYNAMIC": {
        "description": "Claims about how processing unfolds over layers/time",
        "keywords": [
            "progressive", "iterative", "refinement", "cascade", "early",
            "late", "layer_by_layer", "sequential", "parallel", "phase",
            "stage", "transition", "emergence", "convergence", "divergence",
            "bottleneck", "information_flow", "residual_stream",
        ],
        "probe_affinity": ["cca", "rsa", "sae", "das"],
    },
}


@dataclass
class MechanismClassification:
    """Result of classifying a single mechanism name."""
    name: str
    category: str           # one of STRUCTURAL, FUNCTIONAL, CAUSAL, DYNAMIC
    confidence: float       # 0-1 score based on keyword overlap
    matched_keywords: List[str] = field(default_factory=list)
    recommended_probes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class MechanismDecomposer:
    """
    Takes a list of mechanism names/descriptions and decomposes them into
    the four orthogonal categories. This guides which probes should target
    which aspects of the claim.
    """

    def __init__(self, custom_types: Optional[Dict[str, Dict]] = None):
        self.mechanism_types = custom_types or MECHANISM_TYPES
        # Pre-compile keyword patterns for efficient matching
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for cat, info in self.mechanism_types.items():
            self._patterns[cat] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in info["keywords"]
            ]

    def _classify_mechanism(self, name: str) -> MechanismClassification:
        """Classify a single mechanism name into the best-matching category."""
        scores: Dict[str, Tuple[float, List[str]]] = {}

        for cat, patterns in self._patterns.items():
            matched = []
            for pat in patterns:
                if pat.search(name):
                    matched.append(pat.pattern.strip(r"\b"))
            n_keywords = len(self.mechanism_types[cat]["keywords"])
            score = len(matched) / max(n_keywords, 1)
            scores[cat] = (score, matched)

        # Pick best category
        best_cat = max(scores, key=lambda c: scores[c][0])
        best_score, best_matched = scores[best_cat]

        # If nothing matched, default to FUNCTIONAL with low confidence
        if best_score == 0:
            best_cat = "FUNCTIONAL"
            best_score = 0.1
            best_matched = []

        recommended = list(self.mechanism_types[best_cat].get("probe_affinity", []))

        return MechanismClassification(
            name=name,
            category=best_cat,
            confidence=min(best_score * 5.0, 1.0),  # scale up for usability
            matched_keywords=best_matched,
            recommended_probes=recommended,
        )

    def decompose(self, mechanism_list: List[str]) -> Dict[str, List[MechanismClassification]]:
        """
        Decompose a list of mechanism descriptions into categorized groups.

        Returns a dict keyed by category with lists of classified mechanisms.
        """
        result: Dict[str, List[MechanismClassification]] = {
            cat: [] for cat in self.mechanism_types
        }

        for mech_name in mechanism_list:
            classification = self._classify_mechanism(mech_name)
            result[classification.category].append(classification)

        return result

    def get_probe_plan(self, mechanism_list: List[str]) -> Dict[str, List[str]]:
        """
        Given mechanism names, return a mapping of probe_type -> mechanisms
        that should be tested with that probe.
        """
        decomposed = self.decompose(mechanism_list)
        probe_plan: Dict[str, List[str]] = {}

        for cat, classifications in decomposed.items():
            for cls in classifications:
                for probe in cls.recommended_probes:
                    if probe not in probe_plan:
                        probe_plan[probe] = []
                    if cls.name not in probe_plan[probe]:
                        probe_plan[probe].append(cls.name)

        return probe_plan

    def summary(self, mechanism_list: List[str]) -> str:
        """Human-readable decomposition summary."""
        decomposed = self.decompose(mechanism_list)
        lines = ["=== Mechanism Decomposition ==="]
        for cat, classifications in decomposed.items():
            if classifications:
                lines.append(f"\n[{cat}] ({self.mechanism_types[cat]['description']})")
                for c in classifications:
                    kw_str = ", ".join(c.matched_keywords) if c.matched_keywords else "no keyword match"
                    lines.append(f"  - {c.name} (conf={c.confidence:.2f}, keywords: {kw_str})")
                    lines.append(f"    recommended probes: {c.recommended_probes}")
        return "\n".join(lines)
