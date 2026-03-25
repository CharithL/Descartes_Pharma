"""
Paradigm 7b -- LLM Strategic Reasoner
Provides high-level strategic reasoning about the evaluation campaign
when the HOT layer detects problems that require deliberative thinking.

This is a placeholder that structures the prompt and response format.
In production, it would call an LLM API (OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System prompt for meta-strategic reasoning
# ---------------------------------------------------------------------------

SYSTEM_META_STRATEGIC = """You are the strategic reasoning layer of the Descartes-Pharma
mechanistic interpretability evaluation system. You are called when
the meta-cognition (HOT) layer detects a problem with the current
evaluation campaign.

Your role:
1. Analyze the current state of the evaluation campaign.
2. Identify why progress has stalled, results are anomalous, or
   resources are being wasted.
3. Recommend a concrete action plan.

Context you will receive:
- mechanism_name: the mechanism being evaluated
- architecture: the model architecture
- current_tier: which probe cascade tier we are on
- evidence_so_far: list of (probe_type, delta_r2, p_value) tuples
- hot_assessment: the HOT layer's diagnostic scores
- belief_state: the VFE belief system's current mu, sigma, confidence
- timescale_state: multi-timescale processor summary

Your response must be structured JSON with these fields:
{
  "diagnosis": "1-2 sentence diagnosis of what is going wrong",
  "root_cause": "STAGNATION | REDUNDANCY | ANOMALY | EXHAUSTION | CONFLICT | UNKNOWN",
  "recommended_actions": [
    {"action": "skip_to_tier_3", "reason": "..."},
    {"action": "add_probe_type_X", "reason": "..."}
  ],
  "confidence_in_diagnosis": 0.0-1.0,
  "should_halt": false,
  "reasoning": "detailed chain-of-thought reasoning"
}

Guidelines:
- Be conservative: do not recommend halting unless evidence is overwhelming.
- Prefer escalation (trying more powerful probes) over halting.
- If results conflict, recommend targeted probes to resolve the conflict.
- Consider compute budget: do not recommend expensive probes if cheap ones
  have not been exhausted.
- If the mechanism is likely a zombie, recommend the minimum additional
  evidence needed to reach VERIFIED tier in the zombie store.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategicRecommendation:
    """Output from the LLM strategic reasoner."""
    diagnosis: str
    root_cause: str
    recommended_actions: List[Dict[str, str]]
    confidence_in_diagnosis: float
    should_halt: bool
    reasoning: str
    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Strategic reasoner
# ---------------------------------------------------------------------------

class LLMStrategicReasoner:
    """
    Invokes an LLM for strategic reasoning about the evaluation campaign.

    In production, this would call an actual LLM API. Here it provides
    a structured placeholder that returns heuristic-based recommendations.
    """

    def __init__(
        self,
        system_prompt: str = SYSTEM_META_STRATEGIC,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
    ):
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def reason(
        self,
        mechanism_name: str,
        architecture: str,
        current_tier: int,
        evidence_so_far: List[Dict[str, Any]],
        hot_assessment: Optional[Dict[str, Any]] = None,
        belief_state: Optional[Dict[str, Any]] = None,
        timescale_state: Optional[Dict[str, Any]] = None,
    ) -> StrategicRecommendation:
        """
        Perform strategic reasoning about the current campaign.

        In production, this would:
        1. Format the context into a prompt
        2. Call the LLM API
        3. Parse the structured response

        For now, it returns a heuristic-based recommendation.
        """
        # Build the user prompt
        user_prompt = self._build_prompt(
            mechanism_name, architecture, current_tier,
            evidence_so_far, hot_assessment, belief_state, timescale_state,
        )

        # --- Placeholder: heuristic-based reasoning ---
        # In production, replace this block with:
        #   response = openai.ChatCompletion.create(
        #       model=self.model,
        #       messages=[
        #           {"role": "system", "content": self.system_prompt},
        #           {"role": "user", "content": user_prompt},
        #       ],
        #       temperature=self.temperature,
        #   )
        #   return self._parse_response(response)

        return self._heuristic_reason(
            mechanism_name, architecture, current_tier,
            evidence_so_far, hot_assessment, belief_state,
        )

    def _build_prompt(
        self,
        mechanism_name: str,
        architecture: str,
        current_tier: int,
        evidence_so_far: List[Dict[str, Any]],
        hot_assessment: Optional[Dict[str, Any]],
        belief_state: Optional[Dict[str, Any]],
        timescale_state: Optional[Dict[str, Any]],
    ) -> str:
        """Format all context into a structured prompt for the LLM."""
        lines = [
            f"## Current Campaign State",
            f"- Mechanism: {mechanism_name}",
            f"- Architecture: {architecture}",
            f"- Current Tier: {current_tier}",
            f"- Evidence count: {len(evidence_so_far)}",
            "",
            "## Evidence So Far",
        ]
        for i, ev in enumerate(evidence_so_far):
            lines.append(
                f"  {i+1}. probe={ev.get('probe_type', '?')}, "
                f"delta_r2={ev.get('delta_r2', 0):.4f}, "
                f"p_value={ev.get('p_value', 1):.4f}"
            )

        if hot_assessment:
            lines.append("")
            lines.append("## HOT Assessment")
            for k, v in hot_assessment.items():
                lines.append(f"  - {k}: {v}")

        if belief_state:
            lines.append("")
            lines.append("## Belief State (VFE)")
            for k, v in belief_state.items():
                lines.append(f"  - {k}: {v}")

        if timescale_state:
            lines.append("")
            lines.append("## Timescale State")
            for k, v in timescale_state.items():
                lines.append(f"  - {k}: {v}")

        lines.append("")
        lines.append("Please analyze and provide your structured recommendation.")
        return "\n".join(lines)

    def _heuristic_reason(
        self,
        mechanism_name: str,
        architecture: str,
        current_tier: int,
        evidence_so_far: List[Dict[str, Any]],
        hot_assessment: Optional[Dict[str, Any]],
        belief_state: Optional[Dict[str, Any]],
    ) -> StrategicRecommendation:
        """
        Fallback heuristic reasoning when no LLM API is available.
        Analyzes the evidence pattern and produces a recommendation.
        """
        n_evidence = len(evidence_so_far)

        # Analyze evidence pattern
        if n_evidence == 0:
            return StrategicRecommendation(
                diagnosis="No evidence collected yet.",
                root_cause="UNKNOWN",
                recommended_actions=[
                    {"action": "start_tier_0", "reason": "Begin with fast linear probes"}
                ],
                confidence_in_diagnosis=0.9,
                should_halt=False,
                reasoning="Campaign has not started. Begin with tier 0 probes.",
            )

        significant = [e for e in evidence_so_far if e.get("p_value", 1) < 0.05]
        mean_r2 = sum(e.get("delta_r2", 0) for e in evidence_so_far) / n_evidence
        sig_ratio = len(significant) / n_evidence

        # Check HOT assessment
        hot = hot_assessment or {}
        stagnation = hot.get("stagnation_score", 0)
        redundancy = hot.get("redundancy_score", 0)
        anomaly = hot.get("anomaly_score", 0)
        exhaustion = hot.get("exhaustion_score", 0)

        actions = []
        root_cause = "UNKNOWN"
        diagnosis = ""

        if anomaly > 0.7:
            root_cause = "ANOMALY"
            diagnosis = (
                f"Anomalous results detected (score={anomaly:.2f}). "
                "Recent probe returned an outlier that conflicts with prior evidence."
            )
            actions.append({
                "action": "rerun_last_probe",
                "reason": "Verify the anomalous result is not a fluke"
            })
            actions.append({
                "action": "add_targeted_probe",
                "reason": "Use a different probe type to cross-validate"
            })

        elif exhaustion > 0.6:
            root_cause = "EXHAUSTION"
            diagnosis = (
                f"Diminishing returns (exhaustion={exhaustion:.2f}). "
                f"Mean R2={mean_r2:.3f} with {sig_ratio:.0%} significant probes."
            )
            if sig_ratio > 0.6:
                actions.append({
                    "action": "issue_verdict",
                    "reason": "Sufficient evidence for a confident verdict"
                })
            else:
                actions.append({
                    "action": f"skip_to_tier_{min(current_tier + 1, 4)}",
                    "reason": "Try more powerful probes before halting"
                })

        elif stagnation > 0.5 and redundancy > 0.5:
            root_cause = "STAGNATION"
            diagnosis = (
                f"Stagnation ({stagnation:.2f}) with redundancy ({redundancy:.2f}). "
                "Current probes are not adding new information."
            )
            actions.append({
                "action": f"skip_to_tier_{min(current_tier + 1, 4)}",
                "reason": "Escalate to more powerful probes"
            })

        elif redundancy > 0.6:
            root_cause = "REDUNDANCY"
            diagnosis = f"High redundancy ({redundancy:.2f}). Probes are giving similar results."
            actions.append({
                "action": "diversify_probes",
                "reason": "Switch to a different probe type for new information"
            })

        else:
            # Check if there is a conflict in evidence
            if 0.3 < sig_ratio < 0.7:
                root_cause = "CONFLICT"
                diagnosis = (
                    f"Conflicting evidence: {sig_ratio:.0%} of probes are significant. "
                    "The mechanism's status is genuinely ambiguous."
                )
                actions.append({
                    "action": "add_causal_probes",
                    "reason": "Use causal probes (ablation/DAS) to resolve the conflict"
                })
            else:
                diagnosis = "No major issues detected. Campaign is proceeding normally."
                actions.append({
                    "action": "continue",
                    "reason": "Current trajectory is productive"
                })

        should_halt = (
            exhaustion > 0.8
            and n_evidence >= 8
            and (sig_ratio > 0.8 or sig_ratio < 0.2)
        )

        return StrategicRecommendation(
            diagnosis=diagnosis,
            root_cause=root_cause,
            recommended_actions=actions,
            confidence_in_diagnosis=min(0.5 + n_evidence * 0.05, 0.9),
            should_halt=should_halt,
            reasoning=(
                f"Analyzed {n_evidence} evidence records. "
                f"Significance ratio: {sig_ratio:.2f}, Mean R2: {mean_r2:.4f}. "
                f"HOT scores: stag={stagnation:.2f}, red={redundancy:.2f}, "
                f"anom={anomaly:.2f}, exh={exhaustion:.2f}. "
                f"Root cause identified as {root_cause}."
            ),
        )
