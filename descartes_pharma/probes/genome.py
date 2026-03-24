"""
DESCARTES-PHARMA: Probe Genome Specification.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PharmaProbeGenome:
    """Complete probe specification for drug discovery mechanistic validation."""

    genome_id: str = ''
    tier: int = 1
    probe_type: str = 'ridge'

    transform: str = 'raw'
    transform_params: dict = field(default_factory=dict)

    decoder_type: str = 'ridge'
    decoder_params: dict = field(default_factory=dict)

    target_type: str = 'single_mechanism'
    target_names: List[str] = field(default_factory=list)
    condition: Optional[str] = None

    stratification: str = 'none'
    scaffold_family: Optional[str] = None

    null_method: str = 'scaffold_permutation'
    n_permutations: int = 1000
    fdr_correction: bool = True

    layer_target: Optional[int] = None
    attention_heads: Optional[List[int]] = None

    dataset: str = 'clintox'
    split_strategy: str = 'scaffold_split'
