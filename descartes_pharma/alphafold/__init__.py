"""
descartes_pharma.alphafold -- AlphaFold zombie-feature detection and
LLM-driven architecture search for structure-based drug discovery.
"""

from descartes_pharma.alphafold.zombie_detector import (
    AlphaFoldZombieDetector,
    EncodingType,
    EvoformerRepresentations,
    ProbeResult,
)
from descartes_pharma.alphafold.ground_truth import (
    AlphaFoldGroundTruthGenerator,
    STRUCTURAL_FEATURE_NAMES,
    StructuralGroundTruth,
)
from descartes_pharma.alphafold.confidence import (
    alphafold_confidence_zombie_flags,
    ConfidenceZombieFlags,
)
from descartes_pharma.alphafold.llm_balloon import (
    PharmaLLMBalloonExpander,
    ArchitectureProposal,
    ProbeProposal,
    SYSTEM_DRUG_MODEL_BALLOON,
    SYSTEM_PROBE_BALLOON_PHARMA,
    SYSTEM_DRUG_MODEL_GAP,
)

__all__ = [
    # zombie_detector
    "AlphaFoldZombieDetector",
    "EncodingType",
    "EvoformerRepresentations",
    "ProbeResult",
    # ground_truth
    "AlphaFoldGroundTruthGenerator",
    "STRUCTURAL_FEATURE_NAMES",
    "StructuralGroundTruth",
    # confidence
    "alphafold_confidence_zombie_flags",
    "ConfidenceZombieFlags",
    # llm_balloon
    "PharmaLLMBalloonExpander",
    "ArchitectureProposal",
    "ProbeProposal",
    "SYSTEM_DRUG_MODEL_BALLOON",
    "SYSTEM_PROBE_BALLOON_PHARMA",
    "SYSTEM_DRUG_MODEL_GAP",
]
