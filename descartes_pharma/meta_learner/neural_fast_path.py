"""
Paradigm 1 -- Neural Fast-Path Meta-Learner
Learns probe-ordering priors from historical outcomes so that future
evaluation campaigns start with the most informative probes first.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeOutcome:
    """Single record of one probe execution and its result."""
    probe_type: str                  # e.g. "ridge", "sae", "ablation"
    architecture: str                # e.g. "gpt2-small", "llama-7b"
    mechanism: str                   # e.g. "induction_head", "IOI"
    dataset: str                     # e.g. "pile-10k", "openwebtext"
    delta_r2: float                  # improvement in R^2 from this probe
    p_value: float                   # statistical significance
    compute_seconds: float           # wall-clock time
    verdict_contribution: float      # how much this probe moved the verdict
    was_useful: bool                 # did this probe meaningfully contribute?


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

_DEFAULT_ARCHITECTURES = [
    "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "llama-7b", "llama-13b", "llama-70b",
    "mistral-7b", "pythia-160m", "pythia-410m", "pythia-1b",
    "gemma-2b", "gemma-7b", "unknown",
]

_DEFAULT_MECHANISMS = [
    "induction_head", "IOI", "greater_than", "docstring",
    "indirect_object", "copy_suppression", "backup_heads",
    "factual_recall", "sentiment", "syntax_agreement",
    "entity_binding", "negation", "unknown",
]

_DEFAULT_DATASETS = [
    "pile-10k", "openwebtext", "wikipedia", "code",
    "math", "custom_ioi", "custom_gt", "custom_factual",
    "synthetic", "unknown",
]

_DEFAULT_PROBE_TYPES = [
    "ridge", "lasso", "mlp", "knn",
    "sae", "cca", "rsa",
    "ablation", "das", "llm_balloon",
]


def _build_vocab(items: List[str]) -> Dict[str, int]:
    return {item: idx for idx, item in enumerate(items)}


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class PharmaMetaLearner(nn.Module):
    """
    Multi-head meta-learner that takes (architecture, mechanism, dataset,
    probe_type) context and predicts:
      1. probe_priority  -- which probe to run next  (classification)
      2. expected_gain   -- predicted delta-R^2       (regression)
      3. routing         -- soft gate over probe tiers (K-dim softmax)
      4. confidence      -- epistemic confidence       (scalar sigmoid)
    """

    def __init__(
        self,
        arch_vocab: Optional[List[str]] = None,
        mech_vocab: Optional[List[str]] = None,
        data_vocab: Optional[List[str]] = None,
        probe_vocab: Optional[List[str]] = None,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        n_tiers: int = 5,
    ):
        super().__init__()
        arch_vocab = arch_vocab or _DEFAULT_ARCHITECTURES
        mech_vocab = mech_vocab or _DEFAULT_MECHANISMS
        data_vocab = data_vocab or _DEFAULT_DATASETS
        probe_vocab = probe_vocab or _DEFAULT_PROBE_TYPES

        self.arch_v = _build_vocab(arch_vocab)
        self.mech_v = _build_vocab(mech_vocab)
        self.data_v = _build_vocab(data_vocab)
        self.probe_v = _build_vocab(probe_vocab)
        self.n_probes = len(probe_vocab)
        self.n_tiers = n_tiers

        # Embedding tables
        self.arch_emb = nn.Embedding(len(arch_vocab), embed_dim)
        self.mech_emb = nn.Embedding(len(mech_vocab), embed_dim)
        self.data_emb = nn.Embedding(len(data_vocab), embed_dim)
        self.probe_emb = nn.Embedding(len(probe_vocab), embed_dim)

        backbone_in = embed_dim * 4
        # 2-layer MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 4 heads
        self.head_priority = nn.Linear(hidden_dim, self.n_probes)
        self.head_gain = nn.Linear(hidden_dim, 1)
        self.head_routing = nn.Linear(hidden_dim, n_tiers)
        self.head_confidence = nn.Linear(hidden_dim, 1)

    # ---- helpers ----
    def _lookup(self, vocab: dict, key: str) -> int:
        return vocab.get(key, vocab.get("unknown", 0))

    def _encode_context(
        self,
        architectures: List[str],
        mechanisms: List[str],
        datasets: List[str],
        probe_types: List[str],
    ) -> torch.Tensor:
        """Encode a batch of context tuples into embedding vectors."""
        device = next(self.parameters()).device
        a = torch.tensor([self._lookup(self.arch_v, x) for x in architectures], device=device)
        m = torch.tensor([self._lookup(self.mech_v, x) for x in mechanisms], device=device)
        d = torch.tensor([self._lookup(self.data_v, x) for x in datasets], device=device)
        p = torch.tensor([self._lookup(self.probe_v, x) for x in probe_types], device=device)
        return torch.cat([
            self.arch_emb(a),
            self.mech_emb(m),
            self.data_emb(d),
            self.probe_emb(p),
        ], dim=-1)  # (B, 4*embed_dim)

    def forward(
        self,
        architectures: List[str],
        mechanisms: List[str],
        datasets: List[str],
        probe_types: List[str],
    ) -> Dict[str, torch.Tensor]:
        x = self._encode_context(architectures, mechanisms, datasets, probe_types)
        h = self.backbone(x)                           # (B, hidden)
        routing_logits = self.head_routing(h)
        return {
            "probe_priority": self.head_priority(h),   # (B, n_probes) logits
            "expected_gain": self.head_gain(h).squeeze(-1),          # (B,)
            "routing": F.softmax(routing_logits, dim=-1),            # (B, n_tiers)
            "confidence": torch.sigmoid(self.head_confidence(h)).squeeze(-1),  # (B,)
        }

    def predict_single(
        self, architecture: str, mechanism: str, dataset: str, probe_type: str
    ) -> Dict[str, float]:
        """Convenience: predict for a single context tuple, return Python floats."""
        self.eval()
        with torch.no_grad():
            out = self.forward([architecture], [mechanism], [dataset], [probe_type])
        priority_probs = F.softmax(out["probe_priority"][0], dim=-1)
        inv_probe = {v: k for k, v in self.probe_v.items()}
        ranked = sorted(
            [(inv_probe[i], priority_probs[i].item()) for i in range(self.n_probes)],
            key=lambda x: -x[1],
        )
        return {
            "priority_ranking": ranked,
            "expected_gain": out["expected_gain"][0].item(),
            "routing": out["routing"][0].tolist(),
            "confidence": out["confidence"][0].item(),
        }


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

class PharmaFeedbackBuffer:
    """Stores ProbeOutcome records and provides training batches."""

    def __init__(self, max_size: int = 10_000):
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def record(self, outcome: ProbeOutcome) -> None:
        self.buffer.append(outcome)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_batch(self, batch_size: int = 64) -> List[ProbeOutcome]:
        k = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), k)

    def _compute_true_routing(self, outcome: ProbeOutcome, tier_map: Dict[str, int]) -> int:
        """Return the tier index that this probe belongs to."""
        return tier_map.get(outcome.probe_type, 0)

    def _compute_true_confidence(self, outcome: ProbeOutcome) -> float:
        """Heuristic ground-truth confidence: low p-value + high delta = high conf."""
        sig = 1.0 if outcome.p_value < 0.05 else 0.3
        mag = min(abs(outcome.delta_r2) * 5.0, 1.0)
        return sig * 0.6 + mag * 0.4

    def _compute_true_gain(self, outcome: ProbeOutcome) -> float:
        return outcome.delta_r2


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

_DEFAULT_TIER_MAP = {
    "ridge": 0, "lasso": 0,
    "mlp": 1, "knn": 1,
    "sae": 2, "cca": 2, "rsa": 2,
    "ablation": 3, "das": 3,
    "llm_balloon": 4,
}


class PharmaMetaTrainer:
    """Wraps the meta-learner, buffer, and training loop."""

    def __init__(
        self,
        model: Optional[PharmaMetaLearner] = None,
        buffer: Optional[PharmaFeedbackBuffer] = None,
        lr: float = 1e-3,
        train_every: int = 32,
        batch_size: int = 64,
        tier_map: Optional[Dict[str, int]] = None,
    ):
        self.model = model or PharmaMetaLearner()
        self.buffer = buffer or PharmaFeedbackBuffer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_every = train_every
        self.batch_size = batch_size
        self.tier_map = tier_map or _DEFAULT_TIER_MAP
        self._step_counter = 0
        self._train_losses: List[float] = []

    def record_and_maybe_train(self, outcome: ProbeOutcome) -> Optional[float]:
        """Record an outcome; train if enough new data has accumulated."""
        self.buffer.record(outcome)
        self._step_counter += 1
        if self._step_counter % self.train_every == 0 and len(self.buffer) >= self.batch_size:
            return self._train_step()
        return None

    def _train_step(self) -> float:
        self.model.train()
        batch = self.buffer.sample_batch(self.batch_size)

        archs = [o.architecture for o in batch]
        mechs = [o.mechanism for o in batch]
        dsets = [o.dataset for o in batch]
        probes = [o.probe_type for o in batch]

        preds = self.model(archs, mechs, dsets, probes)
        device = preds["expected_gain"].device

        # --- targets ---
        priority_targets = torch.tensor(
            [self.model._lookup(self.model.probe_v, o.probe_type) for o in batch],
            device=device, dtype=torch.long,
        )
        gain_targets = torch.tensor(
            [self.buffer._compute_true_gain(o) for o in batch],
            device=device, dtype=torch.float32,
        )
        routing_targets = torch.tensor(
            [self.buffer._compute_true_routing(o, self.tier_map) for o in batch],
            device=device, dtype=torch.long,
        )
        conf_targets = torch.tensor(
            [self.buffer._compute_true_confidence(o) for o in batch],
            device=device, dtype=torch.float32,
        )

        # --- losses ---
        loss_priority = F.cross_entropy(preds["probe_priority"], priority_targets)
        loss_gain = F.mse_loss(preds["expected_gain"], gain_targets)
        # Use log of routing probabilities for cross-entropy
        log_routing = torch.log(preds["routing"] + 1e-8)
        loss_routing = F.nll_loss(log_routing, routing_targets)
        loss_conf = F.binary_cross_entropy(preds["confidence"], conf_targets)

        loss = loss_priority + loss_gain + loss_routing + loss_conf

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self._train_losses.append(loss_val)
        return loss_val

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "buffer": [asdict(o) for o in self.buffer.buffer],
            "step_counter": self._step_counter,
            "train_losses": self._train_losses,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.buffer.buffer = deque(
            [ProbeOutcome(**d) for d in ckpt["buffer"]],
            maxlen=self.buffer.max_size,
        )
        self._step_counter = ckpt["step_counter"]
        self._train_losses = ckpt.get("train_losses", [])
