"""
LLM Balloon Expander for pharmaceutical model architecture search.

Uses large language models (via the Anthropic API) to "inflate" the
design space of drug-discovery models by proposing novel neural-network
architectures and probing strategies that go beyond what human
researchers have tried.  The LLM acts as a creative brainstorming
partner whose proposals are subsequently filtered by the zombie-feature
detector and evaluated on real data.

The three system prompts control different aspects of the balloon
expansion:

* **SYSTEM_DRUG_MODEL_BALLOON** -- propose novel model architectures
  for predicting drug--target interactions.
* **SYSTEM_PROBE_BALLOON_PHARMA** -- propose novel probes for
  detecting whether learned representations encode biophysical
  features.
* **SYSTEM_DRUG_MODEL_GAP** -- identify gaps in current modelling
  approaches and suggest research directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ======================================================================
# System prompts
# ======================================================================

SYSTEM_DRUG_MODEL_BALLOON: str = (
    "You are a senior computational chemist and deep-learning architect "
    "specialising in structure-based drug design.  Your task is to "
    "propose novel neural-network architectures for predicting "
    "protein--ligand binding affinity and pose.  Each proposal must "
    "include:\n"
    "1. A short architecture name.\n"
    "2. A one-paragraph rationale grounded in biophysics.\n"
    "3. A layer-by-layer specification (input, hidden, output) with "
    "activation functions and skip connections.\n"
    "4. A hypothesis about which biophysical features (e.g. shape "
    "complementarity, electrostatic complementarity, hydrophobic "
    "contact area) the architecture is expected to capture that "
    "current Evoformer-based models may miss.\n"
    "5. A concrete training protocol sketch (loss function, data "
    "augmentation, curriculum).\n\n"
    "Be creative but physically plausible.  Avoid hand-waving; every "
    "design choice must have a biophysical or information-theoretic "
    "justification."
)

SYSTEM_PROBE_BALLOON_PHARMA: str = (
    "You are an expert in representation learning and mechanistic "
    "interpretability applied to molecular biology.  Your task is to "
    "propose novel probing methodologies for determining whether a "
    "protein-structure neural network (e.g. AlphaFold Evoformer) "
    "has genuinely learned specific biophysical properties relevant "
    "to drug discovery.  Each proposal must include:\n"
    "1. Probe name and type (linear, nonlinear, causal, "
    "information-theoretic, etc.).\n"
    "2. The biophysical target property.\n"
    "3. Mathematical formulation of the probe (objective function, "
    "estimator).\n"
    "4. A control experiment to distinguish genuine encoding from "
    "confounds (e.g. sequence identity, trivial correlations).\n"
    "5. Expected outcome if the feature is encoded vs. if it is a "
    "zombie.\n\n"
    "Go beyond simple Ridge/MLP probing.  Consider causal probes, "
    "mutual-information estimators, representation-surgery "
    "experiments, and contrastive perturbation analyses."
)

SYSTEM_DRUG_MODEL_GAP: str = (
    "You are a pharmaceutical-AI research strategist.  Analyse the "
    "current landscape of AlphaFold-derived drug-discovery models and "
    "identify the three most critical gaps where the field is likely "
    "relying on zombie features -- biophysical properties that models "
    "appear to use but have not actually learned.  For each gap:\n"
    "1. Name the gap concisely.\n"
    "2. Explain why current benchmarks fail to detect it.\n"
    "3. Propose a concrete experiment (dataset, metric, baseline) "
    "that would reveal the gap.\n"
    "4. Suggest a modelling intervention (architecture change, "
    "training-data augmentation, auxiliary loss) that could close "
    "the gap.\n\n"
    "Be specific and cite real datasets (PDBbind, CASF, DEKOIS, "
    "LIT-PCBA) and real model families (DiffDock, Uni-Mol, "
    "RoseTTAFold All-Atom) where possible."
)


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ArchitectureProposal:
    """A single novel architecture proposed by the LLM."""

    name: str
    rationale: str
    layer_spec: str
    target_features: List[str]
    training_protocol: str
    raw_llm_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeProposal:
    """A single novel probe proposed by the LLM."""

    name: str
    probe_type: str  # e.g. "linear", "nonlinear", "causal", "MI"
    target_property: str
    formulation: str
    control_experiment: str
    expected_outcome: str
    raw_llm_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Main class
# ======================================================================

class PharmaLLMBalloonExpander:
    """Use an LLM to propose novel architectures and probes.

    Parameters
    ----------
    api_key : str or None
        Anthropic API key.  If ``None``, all methods return placeholder
        proposals for offline development.
    model : str
        Anthropic model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    max_tokens : int
        Maximum tokens for each LLM completion.
    temperature : float
        Sampling temperature.  Higher values produce more creative
        (but potentially less grounded) proposals.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.9,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[Any] = None

        if api_key is not None:
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass  # will fall back to placeholders

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_novel_architectures(
        self,
        context: str = "",
        n_proposals: int = 3,
    ) -> List[ArchitectureProposal]:
        """Ask the LLM to propose novel drug-discovery architectures.

        Parameters
        ----------
        context : str
            Additional context injected into the user message (e.g.
            summary of current zombie-detection results).
        n_proposals : int
            Number of architecture proposals to request.

        Returns
        -------
        list[ArchitectureProposal]
            Parsed proposals.  When no API key is configured, returns
            placeholder proposals.

        Notes
        -----
        **Placeholder implementation** when ``self._client is None``.
        Returns synthetic proposals so that downstream code can be
        developed without incurring API costs.  In production the
        method sends a structured prompt to the Anthropic Messages API
        using ``SYSTEM_DRUG_MODEL_BALLOON`` as the system prompt and
        parses the response into ``ArchitectureProposal`` objects.
        """
        if self._client is None:
            return self._placeholder_architectures(n_proposals)

        user_message = (
            f"Propose exactly {n_proposals} novel neural-network "
            f"architectures for structure-based drug discovery.\n\n"
            f"Context from current analysis:\n{context}\n\n"
            f"Format each proposal with clear headers: "
            f"NAME, RATIONALE, LAYER_SPEC, TARGET_FEATURES, "
            f"TRAINING_PROTOCOL."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_DRUG_MODEL_BALLOON,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        return self._parse_architecture_proposals(raw_text, n_proposals)

    def propose_novel_probes(
        self,
        context: str = "",
        n_proposals: int = 3,
    ) -> List[ProbeProposal]:
        """Ask the LLM to propose novel probing methodologies.

        Parameters
        ----------
        context : str
            Additional context (e.g. list of detected zombie features).
        n_proposals : int
            Number of probe proposals to request.

        Returns
        -------
        list[ProbeProposal]
            Parsed proposals.  Returns placeholders when offline.

        Notes
        -----
        **Placeholder implementation** when ``self._client is None``.
        In production the method sends a structured prompt to the
        Anthropic Messages API using ``SYSTEM_PROBE_BALLOON_PHARMA``
        as the system prompt and parses the response into
        ``ProbeProposal`` objects.
        """
        if self._client is None:
            return self._placeholder_probes(n_proposals)

        user_message = (
            f"Propose exactly {n_proposals} novel probing "
            f"methodologies for detecting zombie features in "
            f"AlphaFold's Evoformer representations.\n\n"
            f"Context from current analysis:\n{context}\n\n"
            f"Format each proposal with clear headers: "
            f"NAME, PROBE_TYPE, TARGET_PROPERTY, FORMULATION, "
            f"CONTROL_EXPERIMENT, EXPECTED_OUTCOME."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROBE_BALLOON_PHARMA,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        return self._parse_probe_proposals(raw_text, n_proposals)

    # ------------------------------------------------------------------
    # Placeholder generators
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder_architectures(n: int) -> List[ArchitectureProposal]:
        """Return synthetic architecture proposals for offline dev."""
        templates = [
            ArchitectureProposal(
                name="BiophysGNN",
                rationale=(
                    "A graph neural network that explicitly encodes "
                    "biophysical interaction types (H-bond, pi-stack, "
                    "hydrophobic, ionic) as typed edge features, "
                    "forcing the network to learn interaction-specific "
                    "message-passing functions."
                ),
                layer_spec=(
                    "Input: atom coords + types -> Typed-edge GNN "
                    "(6 layers, 256-dim, edge-type-conditioned messages) "
                    "-> Biophysical pooling (per-interaction-type "
                    "aggregation) -> MLP head (256->128->1)"
                ),
                target_features=[
                    "shape_complementarity",
                    "electrostatic_complementarity",
                    "hydrophobic_contact_area",
                ],
                training_protocol=(
                    "MSE loss on binding affinity + auxiliary "
                    "per-residue biophysical regression losses. "
                    "Train on PDBbind v2020 refined set with "
                    "augmentation via rotational noise."
                ),
                metadata={"source": "placeholder"},
            ),
            ArchitectureProposal(
                name="EvoformerSurgery",
                rationale=(
                    "Fine-tune AlphaFold's Evoformer with a "
                    "representation-surgery objective that maximises "
                    "linear decodability of binding-site biophysical "
                    "features from intermediate pair representations."
                ),
                layer_spec=(
                    "Frozen Evoformer blocks 1-40 -> Trainable "
                    "blocks 41-48 with auxiliary linear-probe heads "
                    "at each layer -> Affinity prediction MLP "
                    "(pair-pooled 128-dim -> 64 -> 1)"
                ),
                target_features=[
                    "buried_surface_area",
                    "binding_site_flexibility",
                    "water_displacement_count",
                ],
                training_protocol=(
                    "Multi-task loss: 0.5*affinity_MSE + "
                    "0.3*probe_R2_loss + 0.2*PAE_consistency. "
                    "Curriculum: first train probes frozen, then "
                    "unfreeze blocks 41-48."
                ),
                metadata={"source": "placeholder"},
            ),
            ArchitectureProposal(
                name="PhysicsInformedTransformer",
                rationale=(
                    "Augment the self-attention mechanism with "
                    "physics-informed positional encodings derived "
                    "from electrostatic potential fields, ensuring "
                    "that long-range electrostatic interactions are "
                    "captured even for distant residue pairs."
                ),
                layer_spec=(
                    "Input: sequence + 3D coords -> Electrostatic "
                    "PE generator (Poisson-Boltzmann solver, "
                    "discretised) -> Transformer encoder (8 layers, "
                    "512-dim, physics-PE-augmented attention) -> "
                    "Readout MLP (512->256->1)"
                ),
                target_features=[
                    "electrostatic_complementarity",
                    "allosteric_distance",
                    "metal_coordination_count",
                ],
                training_protocol=(
                    "Combined regression + ranking loss on CASF-2016 "
                    "scoring/ranking/docking benchmarks. Pretrain PE "
                    "generator on PDB electrostatic maps."
                ),
                metadata={"source": "placeholder"},
            ),
        ]
        return templates[:n]

    @staticmethod
    def _placeholder_probes(n: int) -> List[ProbeProposal]:
        """Return synthetic probe proposals for offline dev."""
        templates = [
            ProbeProposal(
                name="CausalAblationProbe",
                probe_type="causal",
                target_property="shape_complementarity",
                formulation=(
                    "Zero-out activations for residues known to "
                    "contribute to shape complementarity and measure "
                    "the causal effect on downstream binding-affinity "
                    "prediction via average treatment effect (ATE)."
                ),
                control_experiment=(
                    "Ablate random residues of the same count and "
                    "surface-area distribution; compare ATE to the "
                    "targeted ablation."
                ),
                expected_outcome=(
                    "If encoded: targeted ablation causes "
                    "significantly larger ATE than random. "
                    "If zombie: ATE is indistinguishable from random."
                ),
                metadata={"source": "placeholder"},
            ),
            ProbeProposal(
                name="MutualInfoEstimator",
                probe_type="MI",
                target_property="electrostatic_complementarity",
                formulation=(
                    "Estimate I(Z; Y) where Z is the Evoformer "
                    "representation and Y is the electrostatic "
                    "complementarity score, using MINE (Mutual "
                    "Information Neural Estimation)."
                ),
                control_experiment=(
                    "Permute Y across proteins to break the "
                    "Z-Y relationship; the MI estimate should "
                    "drop to near zero."
                ),
                expected_outcome=(
                    "If encoded: I(Z;Y) >> 0 with tight confidence. "
                    "If zombie: I(Z;Y) ~ 0, indistinguishable from "
                    "the permuted control."
                ),
                metadata={"source": "placeholder"},
            ),
            ProbeProposal(
                name="ContrastivePerturbationProbe",
                probe_type="contrastive",
                target_property="hydrophobic_contact_area",
                formulation=(
                    "Generate minimal mutations that maximally change "
                    "hydrophobic contact area (via Rosetta) and test "
                    "whether the Evoformer representation difference "
                    "correlates with the ground-truth feature "
                    "difference (delta-probing)."
                ),
                control_experiment=(
                    "Use mutations that change sequence but not "
                    "hydrophobic contact area (synonymous-surface "
                    "mutations) as negative controls."
                ),
                expected_outcome=(
                    "If encoded: Pearson r(delta_Z, delta_Y) > 0.5. "
                    "If zombie: correlation near zero."
                ),
                metadata={"source": "placeholder"},
            ),
        ]
        return templates[:n]

    # ------------------------------------------------------------------
    # Response parsing (production path)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_architecture_proposals(
        raw_text: str,
        n_expected: int,
    ) -> List[ArchitectureProposal]:
        """Parse structured LLM output into ArchitectureProposal objects.

        This is a best-effort parser that looks for the headers NAME,
        RATIONALE, LAYER_SPEC, TARGET_FEATURES, TRAINING_PROTOCOL in
        the raw text.  Falls back to a single proposal wrapping the
        entire text if parsing fails.
        """
        proposals: List[ArchitectureProposal] = []

        # Simple split on numbered proposals (e.g. "## 1.", "### Proposal 1")
        import re

        chunks = re.split(r"(?:^|\n)(?:##?\s*)?(?:Proposal\s*)?\d+[\.\):]", raw_text)
        chunks = [c.strip() for c in chunks if c.strip()]

        for chunk in chunks[:n_expected]:
            proposals.append(
                ArchitectureProposal(
                    name=_extract_field(chunk, "NAME", "Unnamed"),
                    rationale=_extract_field(chunk, "RATIONALE", ""),
                    layer_spec=_extract_field(chunk, "LAYER_SPEC", ""),
                    target_features=[
                        f.strip()
                        for f in _extract_field(chunk, "TARGET_FEATURES", "").split(",")
                        if f.strip()
                    ],
                    training_protocol=_extract_field(chunk, "TRAINING_PROTOCOL", ""),
                    raw_llm_output=chunk,
                )
            )

        # Fallback: if parsing yielded nothing, wrap entire text
        if not proposals:
            proposals.append(
                ArchitectureProposal(
                    name="Unparsed",
                    rationale=raw_text[:500],
                    layer_spec="",
                    target_features=[],
                    training_protocol="",
                    raw_llm_output=raw_text,
                )
            )

        return proposals

    @staticmethod
    def _parse_probe_proposals(
        raw_text: str,
        n_expected: int,
    ) -> List[ProbeProposal]:
        """Parse structured LLM output into ProbeProposal objects."""
        proposals: List[ProbeProposal] = []

        import re

        chunks = re.split(r"(?:^|\n)(?:##?\s*)?(?:Proposal\s*)?\d+[\.\):]", raw_text)
        chunks = [c.strip() for c in chunks if c.strip()]

        for chunk in chunks[:n_expected]:
            proposals.append(
                ProbeProposal(
                    name=_extract_field(chunk, "NAME", "Unnamed"),
                    probe_type=_extract_field(chunk, "PROBE_TYPE", "unknown"),
                    target_property=_extract_field(chunk, "TARGET_PROPERTY", ""),
                    formulation=_extract_field(chunk, "FORMULATION", ""),
                    control_experiment=_extract_field(chunk, "CONTROL_EXPERIMENT", ""),
                    expected_outcome=_extract_field(chunk, "EXPECTED_OUTCOME", ""),
                    raw_llm_output=chunk,
                )
            )

        if not proposals:
            proposals.append(
                ProbeProposal(
                    name="Unparsed",
                    probe_type="unknown",
                    target_property="",
                    formulation=raw_text[:500],
                    control_experiment="",
                    expected_outcome="",
                    raw_llm_output=raw_text,
                )
            )

        return proposals


# ======================================================================
# Utility
# ======================================================================

def _extract_field(text: str, field_name: str, default: str) -> str:
    """Extract a named field from semi-structured LLM output."""
    import re

    pattern = rf"(?:^|\n)\**{field_name}\**[:\s]*(.+?)(?=\n\**[A-Z_]+\**[:\s]|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default
