"""
LLM Balloon Expansion (C1/C2)
==============================

When the policy's performance plateaus, call Claude API to propose
new interaction features for the perception layer. This is the C1/C2
ontology expansion from the COGITO/HIMARI framework -- rare, expensive,
outside the training loop.

C1 = pharmacological vocabulary (basic binding concepts)
C2 = hypotheses (combinations of C1 concepts to test)
"""

from descartes_pharma_docking.balloon.llm_proposer import BalloonExpander

__all__ = ["BalloonExpander"]
