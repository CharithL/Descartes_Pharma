"""
C1 Pharmacological Vocabulary + Balloon Expansion.

C1 starts with basic interaction concepts.
When the policy plateaus and probing shows gaps,
the LLM proposes new concepts.

Uses Anthropic API (claude-sonnet-4-20250514) with structured prompt.
Includes hardcoded fallback proposals if API is unavailable.
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# Initial C1 vocabulary (analogous to ARC-AGI's OBJECT, WALL, MOVEMENT)
C1_INITIAL = {
    "H_BOND": "Hydrogen bond between donor and acceptor",
    "HYDROPHOBIC_CONTACT": "Van der Waals contact between nonpolar groups",
    "STERIC_CLASH": "Atoms closer than sum of van der Waals radii",
    "CHARGE_PAIR": "Electrostatic attraction between opposite charges",
    "SOLVENT_EXPOSURE": "Fraction of ligand exposed to solvent",
    "POCKET_DEPTH": "How deep into the pocket the ligand sits",
    "CATALYTIC_PROXIMITY": "Distance to catalytic residues",
}

# C2 hypotheses = combinations of C1 concepts
# These are tested during training
C2_HYPOTHESES = {
    "CATALYTIC_HBOND": "H_BOND + CATALYTIC_PROXIMITY -> high score impact",
    "DEEP_HYDROPHOBIC": "HYDROPHOBIC_CONTACT + POCKET_DEPTH -> stable binding",
    "STERIC_AT_GATE": "STERIC_CLASH + near flap -> blocks entry",
}

# Balloon expansion prompt (4-test protocol from HIMARI)
BALLOON_PROMPT = """You are a computational medicinal chemist advising the \
DESCARTES-PHARMA Docking Game Agent. The agent's Search Policy Network has \
been trained to optimize ligand poses in the BACE1 binding pocket, but \
performance has plateaued.

Current C1 pharmacological vocabulary:
{c1_vocabulary}

Current interaction features computed per pose:
{current_features}

Training performance:
- Best Vina score achieved: {best_score} kcal/mol
- Mean reward (last 100 episodes): {mean_reward}
- DESCARTES probing results: {probe_summary}

The following binding features are NOT encoded in the policy's hidden states:
{unencoded_features}

Propose 3-5 NEW interaction features that might help the policy learn \
these missing binding concepts. Consider:
- Water-mediated hydrogen bonds
- Pi-stacking interactions
- Halogen bonds
- Flap dynamics (open/closed state)
- Subpocket-specific occupancy
- Induced-fit conformational changes

Respond with ONLY valid JSON. Array of proposed features, each with:
  - name: feature name (UPPER_SNAKE_CASE)
  - description: what it measures
  - computation: how to compute from pocket + ligand coordinates
  - hypothesis: why this might help encode the missing binding features
"""


# Hardcoded fallback proposals
FALLBACK_PROPOSALS = [
    {
        "name": "FLAP_OPENNESS",
        "description": "Degree to which the BACE1 flap hairpin (residues 67-77) "
        "is open or closed relative to the catalytic dyad",
        "computation": "Measure the distance between the flap tip (Tyr71 CA) "
        "and the catalytic Asp32 CA. Normalize to [0,1] where 0 = fully closed "
        "(~6 A) and 1 = fully open (~15 A). Also track the angle between "
        "flap backbone and the catalytic plane.",
        "hypothesis": "BACE1 flap dynamics are critical for substrate access. "
        "A policy that encodes flap state can learn to dock through the "
        "open-flap pathway, which is energetically favorable for larger ligands.",
    },
    {
        "name": "WATER_DISPLACEMENT_COUNT",
        "description": "Number of crystallographic water molecules displaced "
        "by the current ligand pose",
        "computation": "Count waters within 3.0 A of any ligand heavy atom. "
        "Weight by the water's B-factor (low B = tightly bound = harder to "
        "displace = bigger entropy gain when displaced).",
        "hypothesis": "Water displacement is a major driver of binding free energy "
        "through entropy gains. Encoding this helps the policy learn that "
        "displacing ordered waters is favorable, explaining why some binding "
        "modes are preferred despite similar Vina scores.",
    },
    {
        "name": "PI_STACKING_SCORE",
        "description": "Strength of pi-stacking interactions between ligand "
        "aromatic rings and pocket aromatic residues (Phe, Tyr, Trp, His)",
        "computation": "For each ligand aromatic ring: find nearest pocket "
        "aromatic ring. Compute centroid distance (ideal: 3.5-4.0 A), "
        "angle between ring normals (parallel: 0-30 deg for face-to-face, "
        "60-90 deg for edge-to-face). Score = sum of Gaussian(dist, 3.8, 0.5) "
        "* angle_factor across all ring pairs.",
        "hypothesis": "Pi-stacking is a key interaction in BACE1 S1/S3 subsites "
        "that Vina underestimates. Explicit encoding helps the policy learn "
        "aromatic placement patterns that improve binding.",
    },
]


@dataclass
class FeatureProposal:
    """A proposed new interaction feature from the LLM balloon."""

    name: str
    description: str
    computation: str
    hypothesis: str


class BalloonExpander:
    """
    LLM balloon for C1/C2 ontology expansion.

    When training plateaus and probing shows gaps in the policy's
    learned representations, the BalloonExpander calls the Anthropic
    API to propose new interaction features.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        plateau_window: int = 100,
        plateau_threshold: float = 0.01,
    ):
        """
        Args:
            model: Anthropic model to use for feature proposals.
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var).
            plateau_window: Number of episodes to look back for plateau
                detection.
            plateau_threshold: Minimum mean reward improvement to NOT
                trigger balloon expansion.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

        # Track C1 vocabulary and expansions
        self.c1_vocabulary = dict(C1_INITIAL)
        self.c2_hypotheses = dict(C2_HYPOTHESES)
        self.expansion_history: List[Dict] = []

    def should_trigger(self, training_log: List[Dict]) -> bool:
        """
        Check if balloon expansion should trigger.

        Triggers when mean reward has plateaued for at least
        `plateau_window` episodes.

        Args:
            training_log: List of per-episode stat dicts from training.

        Returns:
            True if expansion should trigger.
        """
        if len(training_log) < 2 * self.plateau_window:
            return False

        recent_rewards = [
            d.get("total_reward", 0.0)
            for d in training_log[-self.plateau_window:]
        ]
        prev_rewards = [
            d.get("total_reward", 0.0)
            for d in training_log[-2 * self.plateau_window:-self.plateau_window]
        ]

        import numpy as np

        mean_recent = np.mean(recent_rewards)
        mean_prev = np.mean(prev_rewards)
        improvement = abs(mean_recent - mean_prev)

        return improvement < self.plateau_threshold

    def propose_features(
        self,
        training_log: List[Dict],
        current_features: List[str],
        probe_results: Optional[Dict] = None,
    ) -> List[FeatureProposal]:
        """
        Propose new interaction features using the LLM.

        Falls back to hardcoded proposals if the API is unavailable.

        Args:
            training_log: Training statistics for context.
            current_features: Names of currently computed features.
            probe_results: Optional DESCARTES probe results showing
                which features are/aren't encoded.

        Returns:
            List of FeatureProposal objects.
        """
        # Prepare context for the prompt
        import numpy as np

        best_score = min(
            (d.get("best_vina_score", 0.0) for d in training_log),
            default=0.0,
        )
        mean_reward = np.mean(
            [d.get("total_reward", 0.0) for d in training_log[-100:]]
        ) if training_log else 0.0

        probe_summary = "Not yet run"
        unencoded = "Unknown"
        if probe_results:
            encoded = []
            not_encoded = []
            for name, result in probe_results.items():
                if name.startswith("_"):
                    continue
                if isinstance(result, dict) and result.get("significant", False):
                    encoded.append(name)
                elif isinstance(result, dict):
                    not_encoded.append(name)
            probe_summary = (
                f"Encoded: {', '.join(encoded) or 'none'}; "
                f"Not encoded: {', '.join(not_encoded) or 'none'}"
            )
            unencoded = ", ".join(not_encoded) or "none identified"

        prompt = BALLOON_PROMPT.format(
            c1_vocabulary=json.dumps(self.c1_vocabulary, indent=2),
            current_features=", ".join(current_features),
            best_score=f"{best_score:.1f}",
            mean_reward=f"{mean_reward:.2f}",
            probe_summary=probe_summary,
            unencoded_features=unencoded,
        )

        # Try Anthropic API
        proposals = self._call_api(prompt)

        if proposals is None:
            # Fallback to hardcoded proposals
            print(
                "  [Balloon] API unavailable, using fallback proposals."
            )
            proposals = [
                FeatureProposal(**p) for p in FALLBACK_PROPOSALS
            ]

        # Record expansion
        self.expansion_history.append({
            "trigger_episode": len(training_log),
            "n_proposals": len(proposals),
            "proposal_names": [p.name for p in proposals],
            "used_api": proposals is not None,
        })

        # Update C1 vocabulary with new proposals
        for p in proposals:
            self.c1_vocabulary[p.name] = p.description

        return proposals

    def _call_api(self, prompt: str) -> Optional[List[FeatureProposal]]:
        """
        Call the Anthropic API to get feature proposals.

        Returns None if API is unavailable or call fails.
        """
        if not self.api_key:
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            message = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract JSON from response
            response_text = message.content[0].text.strip()

            # Try to parse JSON (may be wrapped in markdown code block)
            if response_text.startswith("```"):
                # Strip markdown code fences
                lines = response_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or not line.strip().startswith("```"):
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            proposals_raw = json.loads(response_text)

            # Validate and convert
            proposals = []
            for p in proposals_raw:
                if all(
                    k in p
                    for k in ("name", "description", "computation", "hypothesis")
                ):
                    proposals.append(
                        FeatureProposal(
                            name=p["name"],
                            description=p["description"],
                            computation=p["computation"],
                            hypothesis=p["hypothesis"],
                        )
                    )

            return proposals if proposals else None

        except ImportError:
            # anthropic package not installed
            return None
        except Exception as e:
            print(f"  [Balloon] API call failed: {e}")
            return None

    def get_c1_vocabulary(self) -> Dict[str, str]:
        """Return the current C1 pharmacological vocabulary."""
        return dict(self.c1_vocabulary)

    def get_c2_hypotheses(self) -> Dict[str, str]:
        """Return the current C2 hypotheses."""
        return dict(self.c2_hypotheses)

    def get_expansion_history(self) -> List[Dict]:
        """Return the history of balloon expansions."""
        return list(self.expansion_history)
