"""
DESCARTES-PHARMA: Mechanistic Zombie Detection for Drug Discovery

Adapted from the DESCARTES Enhanced Dual Factory v3.0 for Pharmaceutical
Target Validation, AI Drug Discovery, and Preclinical-to-Clinical Translation.

Modules:
    core        - HH simulator, data loaders, molecular features
    probes      - 43 probe methods across 7 tiers
    factories   - C1 Probing Factory, C2 Drug Candidate Factory
    meta_learner - 7-paradigm hybrid meta-learner (v1.2)
    alphafold   - AlphaFold zombie detection and structural ground truth (v1.1)
    statistical - 13-method statistical hardening suite
    utils       - Shared utilities and configuration
"""

__version__ = "1.2.0"
