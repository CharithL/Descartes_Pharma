"""
DESCARTES-PHARMA Configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DescartesPharmaConfig:
    """Global configuration for DESCARTES-PHARMA runs."""

    # Device
    device: str = 'cuda'
    seed: int = 42

    # Dataset
    dataset: str = 'hh_simulator'
    tier: int = 1

    # HH Simulator settings
    hh_n_trials: int = 200
    hh_T: float = 100.0
    hh_dt: float = 0.01

    # Surrogate settings
    surrogate_type: str = 'lstm'
    hidden_dim: int = 64
    n_layers: int = 2
    epochs: int = 200
    learning_rate: float = 1e-3

    # Probing settings
    probe_methods: List[str] = field(default_factory=lambda: [
        'ridge', 'mlp', 'sae', 'cca', 'rsa', 'cka',
        'resample_ablation', 'mine', 'mdl',
    ])
    sae_expansion_factors: List[int] = field(default_factory=lambda: [4, 8, 16])
    sae_k: int = 30
    n_permutations: int = 1000
    fdr_alpha: float = 0.05

    # Statistical hardening
    use_scaffold_stratified_null: bool = True
    use_confound_removal: bool = True
    confound_features: List[str] = field(default_factory=lambda: [
        'MW', 'LogP', 'HeavyAtoms'
    ])

    # Meta-learner (v1.2)
    use_meta_learner: bool = True
    meta_learner_path: Optional[str] = None
    vzs_path: str = 'verified_zombie_store.json'

    # LLM balloon (v1.1)
    use_llm_balloon: bool = False
    llm_model: str = 'claude-sonnet-4-20250514'
    balloon_stall_threshold: int = 20

    # AlphaFold (v1.1)
    use_alphafold: bool = False
    alphafold_server_url: Optional[str] = None

    # C2 Factory
    c2_population_size: int = 20
    c2_n_generations: int = 50
    c2_mutation_rate: float = 0.3

    # Output
    output_dir: str = 'outputs'
    save_checkpoints: bool = True
    verbose: bool = True
