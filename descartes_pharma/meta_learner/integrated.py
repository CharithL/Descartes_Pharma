"""
Integrated Meta-Learner -- DescartesPharmaMetaLearner
Composes all 7 paradigms into a single orchestrator:

  1. Neural Fast-Path Meta-Learner   (probe ordering priors)
  2. Mechanism Decomposer            (orthogonal categorization)
  3. Verified Zombie Store           (immutable evidence chain)
  4. Probe Cascade Router            (tiered evaluation)
  5. VFE Belief System               (Kalman belief updates)
  6. Multi-Timescale Processor       (fast/medium/slow ticks)
  7. HOT Meta-Cognition + LLM Strategic Reasoner
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .neural_fast_path import (
    PharmaMetaLearner,
    PharmaFeedbackBuffer,
    PharmaMetaTrainer,
    ProbeOutcome,
)
from .mechanism_decomposer import MechanismDecomposer
from .verified_zombie_store import VerifiedZombieStore, EvidenceRecord
from .probe_cascade import ProbeCascadeRouter, CascadeVerdict
from .vfe_belief import VFEBeliefSystem
from .multi_timescale import MultiTimescaleProcessor
from .hot_layer import MetaCognitionHOT
from .llm_strategic import LLMStrategicReasoner


class DescartesPharmaMetaLearner:
    """
    Top-level orchestrator that composes all 7 meta-learning paradigms
    into a coherent evaluation pipeline.

    Usage:
        meta = DescartesPharmaMetaLearner()

        # For each model to evaluate:
        result = meta.evaluate_model(
            architecture="gpt2-small",
            mechanism_claims=["induction_head", "copy_suppression"],
            activations=activations_array,
            labels=labels_array,
        )

        # At the end of a research campaign:
        meta.end_of_campaign()
        meta.save("checkpoint_dir/")
    """

    def __init__(
        self,
        meta_trainer: Optional[PharmaMetaTrainer] = None,
        decomposer: Optional[MechanismDecomposer] = None,
        zombie_store: Optional[VerifiedZombieStore] = None,
        cascade: Optional[ProbeCascadeRouter] = None,
        vfe: Optional[VFEBeliefSystem] = None,
        timescale: Optional[MultiTimescaleProcessor] = None,
        hot: Optional[MetaCognitionHOT] = None,
        strategic: Optional[LLMStrategicReasoner] = None,
        max_tier: int = 4,
    ):
        # Paradigm 1: Neural fast-path
        self.meta_trainer = meta_trainer or PharmaMetaTrainer()

        # Paradigm 2: Mechanism decomposer
        self.decomposer = decomposer or MechanismDecomposer()

        # Paradigm 3: Verified zombie store
        self.zombie_store = zombie_store or VerifiedZombieStore()

        # Paradigm 4: Probe cascade
        self.cascade = cascade or ProbeCascadeRouter()

        # Paradigm 5: VFE belief system
        self.vfe = vfe or VFEBeliefSystem()

        # Paradigm 6: Multi-timescale
        self.timescale = timescale or MultiTimescaleProcessor()

        # Paradigm 7: HOT + LLM strategic
        self.hot = hot or MetaCognitionHOT()
        self.strategic = strategic or LLMStrategicReasoner()

        self.max_tier = max_tier
        self._campaign_results: List[Dict[str, Any]] = []

    def evaluate_model(
        self,
        architecture: str,
        mechanism_claims: List[str],
        activations: np.ndarray,
        labels: np.ndarray,
        dataset: str = "unknown",
        max_tier: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a full evaluation campaign for a model.

        Parameters
        ----------
        architecture : str
            Model architecture name.
        mechanism_claims : list of str
            Mechanism names claimed by the paper/submission.
        activations : ndarray (n_samples, n_features)
            Model activations to probe.
        labels : ndarray (n_samples,)
            Ground truth labels for probing.
        dataset : str
            Dataset identifier.
        max_tier : int, optional
            Override max cascade tier.

        Returns
        -------
        dict with per-mechanism verdicts, campaign summary, and meta-stats.
        """
        effective_max_tier = max_tier if max_tier is not None else self.max_tier

        # --- Paradigm 2: Decompose mechanisms ---
        decomposition = self.decomposer.decompose(mechanism_claims)
        probe_plan = self.decomposer.get_probe_plan(mechanism_claims)

        # --- Paradigm 1: Get probe ordering prior ---
        probe_priors = {}
        for mech in mechanism_claims:
            prior = self.meta_trainer.model.predict_single(
                architecture, mech, dataset, "ridge"
            )
            probe_priors[mech] = prior

        # --- Evaluate each mechanism ---
        mechanism_verdicts = {}
        for mech in mechanism_claims:
            # Reset per-mechanism belief
            self.vfe.reset()

            # --- Paradigm 4: Run probe cascade ---
            cascade_result = self.cascade.run_cascade(
                activations, labels,
                max_tier=effective_max_tier,
            )

            # Process each cascade result through the other paradigms
            for cr in cascade_result.results:
                if cr.error is not None:
                    continue

                # --- Paradigm 5: VFE belief update ---
                # Convert R2 to observation: high R2 + low p = evidence for real
                observation = cr.delta_r2 if cr.p_value < 0.05 else cr.delta_r2 * 0.3
                observation = min(max(observation, 0.0), 1.0)
                self.vfe.update(cr.probe_name, observation, cr.p_value)

                # --- Paradigm 6: Multi-timescale fast tick ---
                self.timescale.fast_tick(
                    cr.probe_name, cr.delta_r2, cr.p_value, cr.compute_seconds
                )

                # --- Paradigm 7a: HOT assessment ---
                hot_assessment = self.hot.assess(
                    cr.probe_name, cr.delta_r2, cr.p_value, cr.compute_seconds
                )

                # --- Paradigm 1: Record for meta-learning ---
                outcome = ProbeOutcome(
                    probe_type=cr.probe_name,
                    architecture=architecture,
                    mechanism=mech,
                    dataset=dataset,
                    delta_r2=cr.delta_r2,
                    p_value=cr.p_value,
                    compute_seconds=cr.compute_seconds,
                    verdict_contribution=observation,
                    was_useful=(cr.p_value < 0.05 and cr.delta_r2 > 0.05),
                )
                self.meta_trainer.record_and_maybe_train(outcome)

                # Check if HOT recommends strategic intervention
                if hot_assessment.recommended_action in ("escalate", "halt"):
                    # --- Paradigm 7b: LLM strategic reasoning ---
                    evidence_dicts = [
                        {"probe_type": r.probe_name, "delta_r2": r.delta_r2, "p_value": r.p_value}
                        for r in cascade_result.results if r.error is None
                    ]
                    strategic_rec = self.strategic.reason(
                        mechanism_name=mech,
                        architecture=architecture,
                        current_tier=cr.tier,
                        evidence_so_far=evidence_dicts,
                        hot_assessment=asdict(hot_assessment),
                        belief_state=self.vfe.get_zombie_verdict(),
                    )
                    if strategic_rec.should_halt:
                        break

            # --- Paradigm 5: Get VFE verdict ---
            vfe_verdict = self.vfe.get_zombie_verdict()

            # --- Paradigm 6: Medium tick ---
            best_r2 = max((r.delta_r2 for r in cascade_result.results if r.error is None), default=0.0)
            self.timescale.medium_tick("cascade", best_r2, mech)

            # --- Paradigm 3: Record in zombie store ---
            supports = vfe_verdict["is_zombie"] is False
            evidence = EvidenceRecord(
                probe_type="cascade_aggregate",
                delta_r2=best_r2,
                p_value=min((r.p_value for r in cascade_result.results if r.error is None), default=1.0),
                supports=supports,
                details={
                    "vfe_mu": vfe_verdict["mu"],
                    "vfe_confidence": vfe_verdict["confidence"],
                    "cascade_tier_reached": cascade_result.tier_reached,
                    "n_probes": len(cascade_result.results),
                },
            )
            store_verdict = self.zombie_store.record_verdict(mech, architecture, evidence)

            mechanism_verdicts[mech] = {
                "cascade": {
                    "is_zombie": cascade_result.is_zombie,
                    "confidence": cascade_result.confidence,
                    "tier_reached": cascade_result.tier_reached,
                    "early_exit": cascade_result.early_exit,
                    "n_probes": len(cascade_result.results),
                },
                "vfe": vfe_verdict,
                "store": {
                    "tier": store_verdict.tier,
                    "is_zombie": store_verdict.is_zombie,
                    "n_evidence": len(store_verdict.evidence),
                },
                "decomposition": {
                    cat: [c.name for c in cls_list]
                    for cat, cls_list in decomposition.items()
                    if any(c.name == mech for c in cls_list)
                },
            }

        # Campaign summary
        result = {
            "architecture": architecture,
            "dataset": dataset,
            "mechanism_verdicts": mechanism_verdicts,
            "timescale_state": self.timescale.get_state_summary(),
            "zombie_store_stats": self.zombie_store.get_stats(),
            "meta_learner_buffer_size": len(self.meta_trainer.buffer),
        }
        self._campaign_results.append(result)
        return result

    def end_of_campaign(self) -> Dict[str, Any]:
        """
        Called at the end of a research campaign. Triggers slow-tick
        updates across all mechanisms and architectures evaluated.
        """
        summary = {
            "n_campaigns": len(self._campaign_results),
            "slow_tick_updates": [],
        }

        for campaign in self._campaign_results:
            arch = campaign["architecture"]
            for mech, verdict in campaign["mechanism_verdicts"].items():
                vfe_data = verdict.get("vfe", {})
                slow_update = self.timescale.slow_tick(
                    architecture=arch,
                    mechanism_name=mech,
                    campaign_r2=verdict["cascade"]["confidence"],
                    campaign_confidence=vfe_data.get("confidence", 0.5),
                )
                summary["slow_tick_updates"].append({
                    "architecture": arch,
                    "mechanism": mech,
                    "slow_update": slow_update,
                })

        summary["final_store_stats"] = self.zombie_store.get_stats()
        summary["timescale_state"] = self.timescale.get_state_summary()

        return summary

    def save(self, directory: str) -> None:
        """Save all persistent state to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Neural fast-path
        self.meta_trainer.save(str(path / "meta_trainer.pt"))

        # Zombie store
        self.zombie_store.save(str(path / "zombie_store.json"))

    def load(self, directory: str) -> None:
        """Load all persistent state from a directory."""
        path = Path(directory)

        trainer_path = path / "meta_trainer.pt"
        if trainer_path.exists():
            self.meta_trainer.load(str(trainer_path))

        store_path = path / "zombie_store.json"
        if store_path.exists():
            self.zombie_store.load(str(store_path))

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive statistics about the meta-learner state."""
        return {
            "neural_fast_path": {
                "buffer_size": len(self.meta_trainer.buffer),
                "train_steps": self.meta_trainer._step_counter,
                "recent_losses": self.meta_trainer._train_losses[-10:]
                if self.meta_trainer._train_losses else [],
            },
            "zombie_store": self.zombie_store.get_stats(),
            "timescale": self.timescale.get_state_summary(),
            "hot": {
                "assessments": len(self.hot._assessment_history),
                "latest": asdict(self.hot._assessment_history[-1])
                if self.hot._assessment_history else None,
            },
            "campaigns_completed": len(self._campaign_results),
        }
