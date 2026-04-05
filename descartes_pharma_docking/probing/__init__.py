"""
Module 6: DESCARTES Probe Suite
=================================

After training, probe the Search Policy Network's hidden states with
the full DESCARTES council controls. This is the scientific payload --
determining whether the policy learned genuine binding intuition or
is a pharmaceutical zombie.

Components:
    ProbeRunner       - Main probe orchestrator
    CouncilControls   - Three council controls from DESCARTES Cogito
    scaffold_permutation - Scaffold-stratified permutation probe
    pocket_scramble      - Pocket scramble test
"""

from descartes_pharma_docking.probing.probe_runner import DESCARTESProbeRunner
from descartes_pharma_docking.probing.council_controls import CouncilControls

# Aliases for convenience
ProbeRunner = DESCARTESProbeRunner

__all__ = [
    "ProbeRunner",
    "DESCARTESProbeRunner",
    "CouncilControls",
]
