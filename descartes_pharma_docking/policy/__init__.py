"""
Module 4: Search Policy Network
=================================

The ONLY learned component. A small GRU that takes the current state
(pocket features + ligand features + interaction features + score history)
and outputs: (1) action probabilities, (2) a value estimate.

Its hidden states are probed by DESCARTES (Module 6) to determine
if it learned genuine binding intuition or is a pharmaceutical zombie.
"""

from descartes_pharma_docking.policy.policy_network import SearchPolicyNetwork
from descartes_pharma_docking.policy.feature_encoder import encode_state

__all__ = ["SearchPolicyNetwork", "encode_state"]
