"""
DESCARTES-PHARMA: PharmaZombie Verdict Generator.

Decision engine that synthesises evidence from multiple mechanistic probes
into a single categorical verdict for each drug candidate.  The eight-level
verdict taxonomy spans the full spectrum from causally validated mechanisms
to confirmed zombie (spurious) predictions.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Verdict taxonomy
# ---------------------------------------------------------------------------

class VerdictType(Enum):
    """Eight-level verdict taxonomy for drug-candidate mechanistic status.

    Ordered from most concerning (zombie) to most validated (causal).

    CONFIRMED_ZOMBIE
        Multiple independent probes confirm the model exploits a spurious
        shortcut with no mechanistic basis.
    LIKELY_ZOMBIE
        Strong but not yet conclusive evidence of shortcut exploitation.
    SPURIOUS_SCAFFOLD
        The model appears to rely on scaffold memorisation rather than
        learning mechanism-relevant features.
    POLYPHARMACOLOGY_DETECTED
        Evidence of genuine multi-target activity that complicates
        mechanistic attribution but is not necessarily spurious.
    NONLINEAR_MECHANISM
        Probes detect mechanism-relevant signal, but the encoding is
        highly nonlinear and hard to interpret.
    CANDIDATE_MECHANISTIC
        Preliminary evidence supports a mechanistic basis, pending
        confirmation from higher-tier probes.
    CONFIRMED_MECHANISTIC
        Converging evidence from tier-1 and tier-2 probes confirms the
        model has learned a genuine mechanistic representation.
    CAUSALLY_VALIDATED
        Causal intervention probes (e.g. DAS, resample ablation) provide
        direct evidence that the learned features causally drive predictions
        through biologically plausible pathways.
    """
    CONFIRMED_ZOMBIE = auto()
    LIKELY_ZOMBIE = auto()
    SPURIOUS_SCAFFOLD = auto()
    POLYPHARMACOLOGY_DETECTED = auto()
    NONLINEAR_MECHANISM = auto()
    CANDIDATE_MECHANISTIC = auto()
    CONFIRMED_MECHANISTIC = auto()
    CAUSALLY_VALIDATED = auto()


# Severity ordering: lower index == more concerning
VERDICT_SEVERITY = [v for v in VerdictType]


# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------

@dataclass
class EvidenceBundle:
    """Aggregated evidence for a single drug-candidate model.

    Attributes
    ----------
    model_name : str
        Identifier for the candidate model under evaluation.
    dataset_name : str
        Dataset on which the model was trained / evaluated.
    ridge_delta_r2 : Optional[float]
        Linear probe delta-R-squared for mechanism prediction.
    mlp_delta_r2 : Optional[float]
        Nonlinear MLP probe delta-R-squared.
    ridge_mlp_gap : Optional[float]
        Gap between MLP and ridge delta-R-squared; large gaps indicate
        nonlinear mechanism encoding.
    sae_mechanism_fidelity : Optional[float]
        Fraction of SAE latent dimensions aligned with known mechanisms.
    sae_dead_neuron_fraction : Optional[float]
        Fraction of SAE neurons that never activate (polysemanticity
        indicator).
    cca_similarity : Optional[float]
        CCA alignment score between model representations and
        mechanism-labelled subspaces.
    rsa_correlation : Optional[float]
        Representational Similarity Analysis correlation with mechanism
        ground-truth.
    cka_similarity : Optional[float]
        Centred Kernel Alignment between layers and mechanism structure.
    causal_das_effect : Optional[float]
        Distributed Alignment Search causal effect size.
    causal_ablation_drop : Optional[float]
        Drop in mechanism prediction after resample-ablation of
        mechanism-relevant directions.
    mine_mutual_information : Optional[float]
        MINE estimate of mutual information between embeddings and
        mechanism labels.
    mdl_description_length : Optional[float]
        Minimum description length of mechanism given embeddings.
    scaffold_leakage_score : Optional[float]
        Proportion of predictive signal attributable to scaffold identity
        alone (high => scaffold memorisation).
    polypharmacology_score : Optional[float]
        Multi-target activity indicator.
    p_values : Dict[str, float]
        Mapping from probe name to its null-hypothesis p-value.
    metadata : Dict[str, Any]
        Arbitrary additional evidence or annotations.
    """

    model_name: str = ""
    dataset_name: str = ""

    # Tier-1: linear / nonlinear probes
    ridge_delta_r2: Optional[float] = None
    mlp_delta_r2: Optional[float] = None
    ridge_mlp_gap: Optional[float] = None

    # Tier-1: sparse autoencoders
    sae_mechanism_fidelity: Optional[float] = None
    sae_dead_neuron_fraction: Optional[float] = None

    # Tier-2: alignment probes
    cca_similarity: Optional[float] = None
    rsa_correlation: Optional[float] = None
    cka_similarity: Optional[float] = None

    # Tier-3: causal probes
    causal_das_effect: Optional[float] = None
    causal_ablation_drop: Optional[float] = None

    # Tier-2: information-theoretic probes
    mine_mutual_information: Optional[float] = None
    mdl_description_length: Optional[float] = None

    # Confound indicators
    scaffold_leakage_score: Optional[float] = None
    polypharmacology_score: Optional[float] = None

    # Statistical testing
    p_values: Dict[str, float] = field(default_factory=dict)

    # Extra
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Verdict result
# ---------------------------------------------------------------------------

@dataclass
class VerdictResult:
    """Outcome of the verdict decision logic for one model.

    Attributes
    ----------
    verdict : VerdictType
        The assigned verdict.
    confidence : float
        Confidence level in [0, 1].
    contributing_probes : List[str]
        Names of probes that contributed to the decision.
    reasoning : str
        Human-readable explanation of the decision path.
    evidence : EvidenceBundle
        The underlying evidence bundle.
    """

    verdict: VerdictType = VerdictType.CANDIDATE_MECHANISTIC
    confidence: float = 0.5
    contributing_probes: List[str] = field(default_factory=list)
    reasoning: str = ""
    evidence: Optional[EvidenceBundle] = None


# ---------------------------------------------------------------------------
# PharmaZombieVerdictGenerator
# ---------------------------------------------------------------------------

class PharmaZombieVerdictGenerator:
    """Decision engine that maps multi-probe evidence to a mechanistic verdict.

    The generator implements a cascading decision tree:
      1. Check for scaffold leakage / memorisation.
      2. Check for polypharmacology confounds.
      3. Assess linear and nonlinear probe signal.
      4. Evaluate alignment and information-theoretic probes.
      5. Gate on causal probes for the highest verdicts.

    Thresholds can be overridden at construction time for
    dataset-specific calibration.

    Parameters
    ----------
    significance_level : float
        Maximum p-value to consider a probe result significant.
    scaffold_leakage_threshold : float
        Scaffold leakage score above which SPURIOUS_SCAFFOLD is triggered.
    min_ridge_r2 : float
        Minimum ridge delta-R2 to consider linear signal present.
    min_mlp_r2 : float
        Minimum MLP delta-R2 to consider nonlinear signal present.
    nonlinear_gap_threshold : float
        Ridge-MLP gap above which NONLINEAR_MECHANISM is flagged.
    min_alignment : float
        Minimum CCA/RSA/CKA score for alignment evidence.
    min_causal_effect : float
        Minimum causal DAS effect size for CAUSALLY_VALIDATED.
    min_causal_ablation : float
        Minimum ablation-drop for causal confirmation.
    min_sae_fidelity : float
        Minimum SAE mechanism fidelity for convergence evidence.
    polypharmacology_threshold : float
        Score above which POLYPHARMACOLOGY_DETECTED is raised.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        scaffold_leakage_threshold: float = 0.6,
        min_ridge_r2: float = 0.05,
        min_mlp_r2: float = 0.10,
        nonlinear_gap_threshold: float = 0.15,
        min_alignment: float = 0.3,
        min_causal_effect: float = 0.1,
        min_causal_ablation: float = 0.15,
        min_sae_fidelity: float = 0.3,
        polypharmacology_threshold: float = 0.5,
    ) -> None:
        self.significance_level = significance_level
        self.scaffold_leakage_threshold = scaffold_leakage_threshold
        self.min_ridge_r2 = min_ridge_r2
        self.min_mlp_r2 = min_mlp_r2
        self.nonlinear_gap_threshold = nonlinear_gap_threshold
        self.min_alignment = min_alignment
        self.min_causal_effect = min_causal_effect
        self.min_causal_ablation = min_causal_ablation
        self.min_sae_fidelity = min_sae_fidelity
        self.polypharmacology_threshold = polypharmacology_threshold

    # -- primary entry point ------------------------------------------------

    def generate_verdict(self, evidence: EvidenceBundle) -> VerdictResult:
        """Synthesise a mechanistic verdict from an evidence bundle.

        The decision cascades through increasingly stringent checks.
        Early exits catch clear zombie / scaffold cases; later stages
        require convergence across probe tiers.

        Parameters
        ----------
        evidence : EvidenceBundle
            Aggregated probe results for a single model.

        Returns
        -------
        VerdictResult
            The verdict, confidence, and reasoning.
        """
        probes_used: List[str] = []
        reasons: List[str] = []

        # -- helpers --
        def _sig(probe_name: str) -> bool:
            """Check whether a probe's p-value is significant."""
            return evidence.p_values.get(probe_name, 1.0) <= self.significance_level

        def _has(val: Optional[float]) -> bool:
            return val is not None

        # ==================================================================
        # Stage 1: Scaffold leakage check
        # ==================================================================
        if _has(evidence.scaffold_leakage_score):
            probes_used.append("scaffold_leakage")
            if evidence.scaffold_leakage_score >= self.scaffold_leakage_threshold:
                # High leakage -- but is there *any* residual mechanistic signal?
                residual_signal = (
                    (_has(evidence.ridge_delta_r2) and evidence.ridge_delta_r2 > self.min_ridge_r2)
                    or (_has(evidence.mlp_delta_r2) and evidence.mlp_delta_r2 > self.min_mlp_r2)
                )
                if not residual_signal:
                    reasons.append(
                        f"Scaffold leakage score {evidence.scaffold_leakage_score:.3f} "
                        f"exceeds threshold {self.scaffold_leakage_threshold} with no "
                        f"residual mechanistic signal."
                    )
                    return VerdictResult(
                        verdict=VerdictType.SPURIOUS_SCAFFOLD,
                        confidence=min(evidence.scaffold_leakage_score, 0.95),
                        contributing_probes=probes_used,
                        reasoning=" ".join(reasons),
                        evidence=evidence,
                    )
                else:
                    reasons.append(
                        f"Scaffold leakage detected ({evidence.scaffold_leakage_score:.3f}) "
                        f"but residual mechanistic signal present; continuing evaluation."
                    )

        # ==================================================================
        # Stage 2: Polypharmacology detection
        # ==================================================================
        if _has(evidence.polypharmacology_score):
            probes_used.append("polypharmacology")
            if evidence.polypharmacology_score >= self.polypharmacology_threshold:
                reasons.append(
                    f"Polypharmacology score {evidence.polypharmacology_score:.3f} "
                    f"indicates multi-target activity."
                )
                # Polypharmacology does not terminate the cascade but is
                # recorded.  If later stages find no mechanistic basis,
                # this flag becomes the verdict.

        # ==================================================================
        # Stage 3: Linear and nonlinear probe assessment
        # ==================================================================
        has_linear = _has(evidence.ridge_delta_r2)
        has_nonlinear = _has(evidence.mlp_delta_r2)
        linear_pass = has_linear and evidence.ridge_delta_r2 >= self.min_ridge_r2
        nonlinear_pass = has_nonlinear and evidence.mlp_delta_r2 >= self.min_mlp_r2

        if has_linear:
            probes_used.append("ridge_delta_r2")
        if has_nonlinear:
            probes_used.append("mlp_delta_r2")

        # Both probes fail => zombie territory
        if has_linear and has_nonlinear and not linear_pass and not nonlinear_pass:
            # Count additional negative evidence
            negative_alignment = (
                (_has(evidence.cca_similarity) and evidence.cca_similarity < self.min_alignment)
                or (_has(evidence.rsa_correlation) and evidence.rsa_correlation < self.min_alignment)
            )
            if negative_alignment:
                probes_used.extend(
                    [p for p in ("cca_similarity", "rsa_correlation") if _has(getattr(evidence, p))]
                )
                reasons.append(
                    "Both linear and nonlinear probes fail to detect mechanistic "
                    "signal and alignment probes confirm absence."
                )
                return VerdictResult(
                    verdict=VerdictType.CONFIRMED_ZOMBIE,
                    confidence=0.85,
                    contributing_probes=probes_used,
                    reasoning=" ".join(reasons),
                    evidence=evidence,
                )
            else:
                reasons.append(
                    "Both linear and nonlinear probes fail to detect mechanistic signal."
                )
                return VerdictResult(
                    verdict=VerdictType.LIKELY_ZOMBIE,
                    confidence=0.70,
                    contributing_probes=probes_used,
                    reasoning=" ".join(reasons),
                    evidence=evidence,
                )

        # Nonlinear gap check
        if _has(evidence.ridge_mlp_gap) and evidence.ridge_mlp_gap >= self.nonlinear_gap_threshold:
            probes_used.append("ridge_mlp_gap")
            reasons.append(
                f"Large ridge-MLP gap ({evidence.ridge_mlp_gap:.3f}) indicates "
                f"nonlinear mechanism encoding."
            )
            # Continue -- nonlinear encoding is not disqualifying but noted.

        # ==================================================================
        # Stage 4: Alignment and information-theoretic probes
        # ==================================================================
        alignment_scores: List[float] = []
        for probe_name, attr in [
            ("cca_similarity", evidence.cca_similarity),
            ("rsa_correlation", evidence.rsa_correlation),
            ("cka_similarity", evidence.cka_similarity),
        ]:
            if _has(attr):
                probes_used.append(probe_name)
                alignment_scores.append(attr)

        alignment_pass = (
            len(alignment_scores) > 0
            and np.mean(alignment_scores) >= self.min_alignment
        )

        # SAE fidelity
        sae_pass = (
            _has(evidence.sae_mechanism_fidelity)
            and evidence.sae_mechanism_fidelity >= self.min_sae_fidelity
        )
        if _has(evidence.sae_mechanism_fidelity):
            probes_used.append("sae_mechanism_fidelity")

        # Information-theoretic
        it_pass = False
        if _has(evidence.mine_mutual_information):
            probes_used.append("mine_mutual_information")
            if evidence.mine_mutual_information > 0 and _sig("mine_mutual_information"):
                it_pass = True
        if _has(evidence.mdl_description_length):
            probes_used.append("mdl_description_length")

        # ==================================================================
        # Stage 5: Causal probes -- gate for highest verdicts
        # ==================================================================
        causal_das_pass = (
            _has(evidence.causal_das_effect)
            and evidence.causal_das_effect >= self.min_causal_effect
            and _sig("causal_das_effect")
        )
        causal_ablation_pass = (
            _has(evidence.causal_ablation_drop)
            and evidence.causal_ablation_drop >= self.min_causal_ablation
            and _sig("causal_ablation_drop")
        )
        if _has(evidence.causal_das_effect):
            probes_used.append("causal_das_effect")
        if _has(evidence.causal_ablation_drop):
            probes_used.append("causal_ablation_drop")

        # ==================================================================
        # Decision synthesis
        # ==================================================================

        # CAUSALLY_VALIDATED: causal probes pass AND alignment/probe support
        if (causal_das_pass or causal_ablation_pass) and (linear_pass or nonlinear_pass) and alignment_pass:
            reasons.append(
                "Causal intervention probes confirm that learned features "
                "causally drive predictions through biologically plausible pathways."
            )
            return VerdictResult(
                verdict=VerdictType.CAUSALLY_VALIDATED,
                confidence=0.90,
                contributing_probes=list(set(probes_used)),
                reasoning=" ".join(reasons),
                evidence=evidence,
            )

        # CONFIRMED_MECHANISTIC: strong convergence without causal probes
        convergence_count = sum([
            linear_pass, nonlinear_pass, alignment_pass, sae_pass, it_pass,
        ])
        if convergence_count >= 3:
            reasons.append(
                f"Converging evidence from {convergence_count} probe families "
                f"confirms mechanistic representation."
            )
            return VerdictResult(
                verdict=VerdictType.CONFIRMED_MECHANISTIC,
                confidence=min(0.60 + convergence_count * 0.08, 0.90),
                contributing_probes=list(set(probes_used)),
                reasoning=" ".join(reasons),
                evidence=evidence,
            )

        # NONLINEAR_MECHANISM: signal present but highly nonlinear
        if (
            _has(evidence.ridge_mlp_gap)
            and evidence.ridge_mlp_gap >= self.nonlinear_gap_threshold
            and nonlinear_pass
            and not linear_pass
        ):
            reasons.append(
                "Mechanism-relevant signal detected but encoding is highly "
                "nonlinear, limiting interpretability."
            )
            return VerdictResult(
                verdict=VerdictType.NONLINEAR_MECHANISM,
                confidence=0.65,
                contributing_probes=list(set(probes_used)),
                reasoning=" ".join(reasons),
                evidence=evidence,
            )

        # POLYPHARMACOLOGY_DETECTED: multi-target but some signal
        if (
            _has(evidence.polypharmacology_score)
            and evidence.polypharmacology_score >= self.polypharmacology_threshold
            and (linear_pass or nonlinear_pass)
        ):
            reasons.append(
                "Multi-target activity detected alongside partial mechanistic signal."
            )
            return VerdictResult(
                verdict=VerdictType.POLYPHARMACOLOGY_DETECTED,
                confidence=0.60,
                contributing_probes=list(set(probes_used)),
                reasoning=" ".join(reasons),
                evidence=evidence,
            )

        # CANDIDATE_MECHANISTIC: some positive signal but not enough convergence
        if linear_pass or nonlinear_pass:
            reasons.append(
                "Preliminary probe signal supports a mechanistic basis "
                "but convergence across probe tiers is insufficient."
            )
            return VerdictResult(
                verdict=VerdictType.CANDIDATE_MECHANISTIC,
                confidence=0.50,
                contributing_probes=list(set(probes_used)),
                reasoning=" ".join(reasons),
                evidence=evidence,
            )

        # LIKELY_ZOMBIE: fallback when no positive signal
        reasons.append(
            "No probe tier produced convincing mechanistic evidence."
        )
        return VerdictResult(
            verdict=VerdictType.LIKELY_ZOMBIE,
            confidence=0.60,
            contributing_probes=list(set(probes_used)),
            reasoning=" ".join(reasons),
            evidence=evidence,
        )

    # -- report generation --------------------------------------------------

    def generate_report(
        self,
        all_evidence: List[EvidenceBundle],
        model_name: str = "",
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """Generate a comprehensive verdict report across multiple evidence bundles.

        Parameters
        ----------
        all_evidence : List[EvidenceBundle]
            One evidence bundle per model / condition evaluated.
        model_name : str
            Display name for the model (used in the report header).
        dataset_name : str
            Display name for the dataset.

        Returns
        -------
        Dict[str, Any]
            Report dictionary with keys:
            - ``model_name``: str
            - ``dataset_name``: str
            - ``timestamp``: ISO-format string
            - ``n_evaluations``: int
            - ``verdicts``: list of per-bundle verdict summaries
            - ``summary``: dict with verdict distribution and overall status
        """
        verdicts: List[Dict[str, Any]] = []
        verdict_counts: Dict[str, int] = {v.name: 0 for v in VerdictType}

        for bundle in all_evidence:
            result = self.generate_verdict(bundle)
            verdict_counts[result.verdict.name] += 1
            verdicts.append({
                "model_name": bundle.model_name or model_name,
                "dataset_name": bundle.dataset_name or dataset_name,
                "verdict": result.verdict.name,
                "confidence": round(result.confidence, 4),
                "contributing_probes": result.contributing_probes,
                "reasoning": result.reasoning,
            })

        # Overall status: worst verdict across all bundles
        worst_idx = len(VERDICT_SEVERITY)
        for v in verdicts:
            idx = next(
                i for i, vt in enumerate(VERDICT_SEVERITY) if vt.name == v["verdict"]
            )
            worst_idx = min(worst_idx, idx)

        overall_verdict = VERDICT_SEVERITY[worst_idx].name if verdicts else "NO_EVIDENCE"

        # Zombie fraction
        zombie_verdicts = {VerdictType.CONFIRMED_ZOMBIE.name, VerdictType.LIKELY_ZOMBIE.name}
        n_zombie = sum(verdict_counts[z] for z in zombie_verdicts)
        zombie_fraction = n_zombie / len(all_evidence) if all_evidence else 0.0

        # Mechanistic fraction
        mech_verdicts = {
            VerdictType.CANDIDATE_MECHANISTIC.name,
            VerdictType.CONFIRMED_MECHANISTIC.name,
            VerdictType.CAUSALLY_VALIDATED.name,
        }
        n_mechanistic = sum(verdict_counts[m] for m in mech_verdicts)
        mechanistic_fraction = n_mechanistic / len(all_evidence) if all_evidence else 0.0

        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_evaluations": len(all_evidence),
            "verdicts": verdicts,
            "summary": {
                "verdict_distribution": verdict_counts,
                "overall_verdict": overall_verdict,
                "zombie_fraction": round(zombie_fraction, 4),
                "mechanistic_fraction": round(mechanistic_fraction, 4),
                "recommendation": _recommendation_text(overall_verdict, zombie_fraction),
            },
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _recommendation_text(overall_verdict: str, zombie_fraction: float) -> str:
    """Generate a human-readable recommendation string."""
    if overall_verdict in (VerdictType.CONFIRMED_ZOMBIE.name, VerdictType.LIKELY_ZOMBIE.name):
        return (
            "CRITICAL: Model predictions are likely driven by spurious shortcuts. "
            "Do not advance candidates selected by this model without independent "
            "mechanistic validation."
        )
    if overall_verdict == VerdictType.SPURIOUS_SCAFFOLD.name:
        return (
            "WARNING: Scaffold memorisation detected. Retrain with scaffold-aware "
            "splitting and re-evaluate before using predictions for candidate selection."
        )
    if zombie_fraction > 0.3:
        return (
            "CAUTION: A significant minority of evaluations ({:.0%}) indicate "
            "zombie-like behaviour. Investigate per-scaffold performance and "
            "consider ensemble diversification.".format(zombie_fraction)
        )
    if overall_verdict == VerdictType.CAUSALLY_VALIDATED.name:
        return (
            "STRONG: Causal evidence supports mechanistic validity. Model predictions "
            "can be used with high confidence for candidate prioritisation."
        )
    if overall_verdict == VerdictType.CONFIRMED_MECHANISTIC.name:
        return (
            "GOOD: Converging probe evidence supports mechanistic basis. Consider "
            "running causal probes (DAS, resample ablation) for full validation."
        )
    return (
        "MODERATE: Some mechanistic signal detected but full convergence not yet "
        "achieved. Expand probe coverage before relying on model predictions."
    )
