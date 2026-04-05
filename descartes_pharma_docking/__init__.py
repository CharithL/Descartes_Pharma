"""
DESCARTES-PHARMA Docking Game Agent
====================================

A reinforcement learning agent for molecular docking, structured as a
"docking game" following the DESCARTES architecture.

Modules:
    perception  - PocketKnowledge Perception (Module 1)
    ligand      - Ligand Representation & Action Space (Module 2)
    interaction - Pocket x Ligand interaction features
    vina_engine - Vina World Model / Chess Engine (Module 3)
    policy      - Search Policy Network (Module 4)
    training    - RL Training Loop (Module 5)
    probing     - DESCARTES Probe Suite (Module 6)
    balloon     - LLM Balloon Expansion (C1/C2)
"""

__version__ = "2.0.0"
__author__ = "Descartes Pharma"
