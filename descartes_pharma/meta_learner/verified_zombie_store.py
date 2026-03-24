"""
Paradigm 3 -- Verified Zombie Store
Immutable, hash-chained evidence store for mechanism verdicts.
Four-tier structure: CONTESTED -> LIKELY -> VERIFIED -> AXIOM.
Once promoted to AXIOM, a verdict can never be demoted.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = ("CONTESTED", "LIKELY", "VERIFIED", "AXIOM")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Thresholds for promotion
PROMOTION_THRESHOLDS = {
    "CONTESTED_to_LIKELY": {"min_probes": 3, "min_agreement": 0.6},
    "LIKELY_to_VERIFIED": {"min_probes": 6, "min_agreement": 0.8, "max_p_value": 0.01},
    "VERIFIED_to_AXIOM": {"min_probes": 10, "min_agreement": 0.95, "max_p_value": 0.001},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRecord:
    """Single piece of evidence for or against a mechanism."""
    probe_type: str
    delta_r2: float
    p_value: float
    supports: bool          # True = evidence FOR, False = evidence AGAINST
    timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class MechanismVerdict:
    """Full verdict record for a single mechanism."""
    mechanism_name: str
    architecture: str
    tier: str = "CONTESTED"
    is_zombie: Optional[bool] = None  # None = undecided, True = zombie, False = real
    evidence: List[EvidenceRecord] = field(default_factory=list)
    hash_chain: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class VerifiedZombieStore:
    """
    Hash-chained evidence store for mechanism verdicts.

    Verdicts progress through 4 tiers:
      CONTESTED -> LIKELY -> VERIFIED -> AXIOM

    Each new piece of evidence is hash-chained to the previous, creating
    an immutable audit trail. AXIOM-tier verdicts can never be demoted.
    """

    def __init__(self):
        self._store: Dict[str, MechanismVerdict] = {}

    def _make_key(self, mechanism_name: str, architecture: str) -> str:
        return f"{architecture}::{mechanism_name}"

    def _compute_hash(self, prev_hash: str, evidence: EvidenceRecord) -> str:
        """Hash-chain: H(prev_hash || evidence_data)."""
        payload = json.dumps({
            "prev": prev_hash,
            "probe_type": evidence.probe_type,
            "delta_r2": evidence.delta_r2,
            "p_value": evidence.p_value,
            "supports": evidence.supports,
            "timestamp": evidence.timestamp,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def lookup(
        self, mechanism_name: str, architecture: str
    ) -> Optional[MechanismVerdict]:
        """Look up the current verdict for a mechanism + architecture pair."""
        key = self._make_key(mechanism_name, architecture)
        return self._store.get(key)

    def record_verdict(
        self,
        mechanism_name: str,
        architecture: str,
        evidence: EvidenceRecord,
    ) -> MechanismVerdict:
        """
        Add a new piece of evidence and potentially promote the verdict tier.
        Returns the updated verdict.
        """
        key = self._make_key(mechanism_name, architecture)

        if key not in self._store:
            self._store[key] = MechanismVerdict(
                mechanism_name=mechanism_name,
                architecture=architecture,
            )

        verdict = self._store[key]

        # AXIOM verdicts are immutable -- still record evidence but never change tier
        prev_hash = verdict.hash_chain[-1] if verdict.hash_chain else "genesis"
        new_hash = self._compute_hash(prev_hash, evidence)
        verdict.hash_chain.append(new_hash)
        verdict.evidence.append(evidence)
        verdict.updated_at = datetime.utcnow().isoformat()

        # Attempt promotion (unless already AXIOM)
        if verdict.tier != "AXIOM":
            self._attempt_promotion(verdict)

        # Update zombie status based on evidence consensus
        self._update_zombie_status(verdict)

        return verdict

    def _attempt_promotion(self, verdict: MechanismVerdict) -> None:
        """Check if verdict qualifies for promotion to the next tier."""
        n_evidence = len(verdict.evidence)
        if n_evidence == 0:
            return

        supporting = sum(1 for e in verdict.evidence if e.supports)
        agreement = supporting / n_evidence
        min_p = min(e.p_value for e in verdict.evidence)

        current_rank = TIER_RANK[verdict.tier]

        # Try each promotion level
        if current_rank < TIER_RANK["LIKELY"]:
            thresh = PROMOTION_THRESHOLDS["CONTESTED_to_LIKELY"]
            if n_evidence >= thresh["min_probes"] and agreement >= thresh["min_agreement"]:
                verdict.tier = "LIKELY"
                current_rank = TIER_RANK["LIKELY"]

        if current_rank < TIER_RANK["VERIFIED"]:
            thresh = PROMOTION_THRESHOLDS["LIKELY_to_VERIFIED"]
            if (n_evidence >= thresh["min_probes"]
                    and agreement >= thresh["min_agreement"]
                    and min_p <= thresh["max_p_value"]):
                verdict.tier = "VERIFIED"
                current_rank = TIER_RANK["VERIFIED"]

        if current_rank < TIER_RANK["AXIOM"]:
            self._promote_to_axiom(verdict)

    def _promote_to_axiom(self, verdict: MechanismVerdict) -> bool:
        """
        Promote to AXIOM tier if criteria are met.
        Uses hash-chain integrity check as additional safeguard.
        """
        thresh = PROMOTION_THRESHOLDS["VERIFIED_to_AXIOM"]
        n_evidence = len(verdict.evidence)

        if n_evidence < thresh["min_probes"]:
            return False

        supporting = sum(1 for e in verdict.evidence if e.supports)
        agreement = supporting / n_evidence
        min_p = min(e.p_value for e in verdict.evidence)

        if agreement < thresh["min_agreement"] or min_p > thresh["max_p_value"]:
            return False

        # Verify hash chain integrity before axiom promotion
        if not self._verify_chain(verdict):
            return False

        verdict.tier = "AXIOM"
        return True

    def _verify_chain(self, verdict: MechanismVerdict) -> bool:
        """Verify the hash chain is intact (no tampering)."""
        if len(verdict.hash_chain) != len(verdict.evidence):
            return False

        prev_hash = "genesis"
        for i, evidence in enumerate(verdict.evidence):
            expected = self._compute_hash(prev_hash, evidence)
            if expected != verdict.hash_chain[i]:
                return False
            prev_hash = expected

        return True

    def _update_zombie_status(self, verdict: MechanismVerdict) -> None:
        """Update is_zombie based on current evidence consensus."""
        if not verdict.evidence:
            verdict.is_zombie = None
            return

        supporting = sum(1 for e in verdict.evidence if e.supports)
        ratio = supporting / len(verdict.evidence)

        if ratio >= 0.7:
            verdict.is_zombie = False  # real mechanism
        elif ratio <= 0.3:
            verdict.is_zombie = True   # zombie
        else:
            verdict.is_zombie = None   # undecided

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the store."""
        stats = {
            "total_verdicts": len(self._store),
            "by_tier": {t: 0 for t in TIERS},
            "zombies": 0,
            "real": 0,
            "undecided": 0,
            "total_evidence": 0,
        }
        for verdict in self._store.values():
            stats["by_tier"][verdict.tier] += 1
            stats["total_evidence"] += len(verdict.evidence)
            if verdict.is_zombie is True:
                stats["zombies"] += 1
            elif verdict.is_zombie is False:
                stats["real"] += 1
            else:
                stats["undecided"] += 1
        return stats

    def get_all_verdicts(self) -> List[MechanismVerdict]:
        """Return all verdicts in the store."""
        return list(self._store.values())

    def save(self, path: str) -> None:
        """Persist store to JSON file."""
        data = {}
        for key, verdict in self._store.items():
            data[key] = {
                "mechanism_name": verdict.mechanism_name,
                "architecture": verdict.architecture,
                "tier": verdict.tier,
                "is_zombie": verdict.is_zombie,
                "evidence": [asdict(e) for e in verdict.evidence],
                "hash_chain": verdict.hash_chain,
                "created_at": verdict.created_at,
                "updated_at": verdict.updated_at,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load store from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self._store.clear()
        for key, vdata in data.items():
            evidence_list = [EvidenceRecord(**e) for e in vdata["evidence"]]
            verdict = MechanismVerdict(
                mechanism_name=vdata["mechanism_name"],
                architecture=vdata["architecture"],
                tier=vdata["tier"],
                is_zombie=vdata["is_zombie"],
                evidence=evidence_list,
                hash_chain=vdata["hash_chain"],
                created_at=vdata["created_at"],
                updated_at=vdata["updated_at"],
            )
            self._store[key] = verdict
