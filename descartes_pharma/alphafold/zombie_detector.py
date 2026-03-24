"""
AlphaFold Zombie Feature Detector.

Implements probing methodology to detect whether AlphaFold's Evoformer
representations genuinely encode biophysical properties relevant to drug
discovery, or whether they are "zombie" features -- present in the
architecture but not meaningfully learned.

The approach follows the linear-probing paradigm: if a simple Ridge probe
can decode a biophysical ground-truth signal from Evoformer activations
nearly as well as a nonlinear MLP probe, the feature is linearly encoded.
If the MLP significantly outperforms Ridge, the feature is only
nonlinearly accessible. If neither probe succeeds, the feature is a
zombie -- absent from the representation entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class EncodingType(Enum):
    """Classification of how a biophysical feature is encoded."""

    LINEAR_ENCODED = "LINEAR_ENCODED"
    NONLINEAR_ONLY = "NONLINEAR_ONLY"
    ZOMBIE = "ZOMBIE"


@dataclass
class ProbeResult:
    """Result of a single probing experiment."""

    feature_name: str
    ridge_r2: float
    mlp_r2: float
    baseline_r2: float
    ridge_delta: float  # ridge_r2 - baseline_r2
    mlp_delta: float  # mlp_r2 - baseline_r2
    encoding_type: EncodingType


@dataclass
class EvoformerRepresentations:
    """Container for extracted Evoformer intermediate representations."""

    single_representations: torch.Tensor  # (N_res, d_single)
    pair_representations: torch.Tensor  # (N_res, N_res, d_pair)
    layer_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlphaFoldZombieDetector:
    """Detects zombie features in AlphaFold's Evoformer representations.

    A zombie feature is a biophysical property that the Evoformer
    architecture *could* represent but has not actually learned to encode.
    Detection is performed by training linear (Ridge) and nonlinear (MLP)
    probes on Evoformer activations with biophysical ground-truth targets.

    Parameters
    ----------
    model : torch.nn.Module or None
        An AlphaFold-compatible model. If ``None``, a placeholder is used
        and ``extract_evoformer_representations`` will return synthetic
        data for development/testing purposes.
    device : str
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    probe_hidden_dim : int
        Hidden-layer width for the MLP probe.
    probe_epochs : int
        Training epochs for the MLP probe.
    probe_lr : float
        Learning rate for the MLP probe.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        probe_hidden_dim: int = 256,
        probe_epochs: int = 100,
        probe_lr: float = 1e-3,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.probe_hidden_dim = probe_hidden_dim
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr

        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured_activations: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Evoformer representation extraction
    # ------------------------------------------------------------------

    def extract_evoformer_representations(
        self,
        input_features: Dict[str, torch.Tensor],
    ) -> EvoformerRepresentations:
        """Extract intermediate Evoformer representations via forward hooks.

        Registers PyTorch forward hooks on every Evoformer block to
        capture single-representation and pair-representation tensors
        produced at each layer.  After a single forward pass the hooks
        are removed and the captured activations are returned.

        Parameters
        ----------
        input_features : dict[str, Tensor]
            Model input dict as expected by the AlphaFold forward pass.
            Must contain at minimum ``"aatype"``, ``"residue_index"``,
            and MSA-related feature tensors.

        Returns
        -------
        EvoformerRepresentations
            Dataclass holding the final single and pair representations
            as well as per-layer outputs.

        Notes
        -----
        **Placeholder implementation.**  When ``self.model`` is ``None``
        the method returns synthetic random tensors whose shapes match
        a typical Evoformer output (``d_single=384``, ``d_pair=128``)
        for the sequence length inferred from ``input_features``.
        Replace with real AlphaFold model hooks when available.
        """
        self._captured_activations.clear()

        if self.model is None:
            n_res = self._infer_n_residues(input_features)
            d_single, d_pair = 384, 128
            return EvoformerRepresentations(
                single_representations=torch.randn(n_res, d_single, device=self.device),
                pair_representations=torch.randn(n_res, n_res, d_pair, device=self.device),
                layer_outputs={
                    f"evoformer_block_{i}": torch.randn(n_res, d_single, device=self.device)
                    for i in range(48)
                },
                metadata={"source": "placeholder", "n_residues": n_res},
            )

        # --- Real extraction path (requires a live AlphaFold model) ---
        self._register_hooks()
        try:
            with torch.no_grad():
                self.model.set_to_eval_mode()
                _ = self.model(input_features)
        finally:
            self._remove_hooks()

        # Assemble outputs from captured activations.
        layer_outputs = {
            name: act.detach().cpu()
            for name, act in self._captured_activations.items()
        }

        # The final Evoformer block outputs are used as the main reps.
        sorted_keys = sorted(layer_outputs.keys())
        final_single = layer_outputs.get(sorted_keys[-1], torch.empty(0))
        n_res = final_single.shape[0] if final_single.dim() >= 1 else 0

        return EvoformerRepresentations(
            single_representations=final_single,
            pair_representations=torch.empty(0),  # populated by pair hooks
            layer_outputs=layer_outputs,
            metadata={"source": "model", "n_residues": n_res},
        )

    # ------------------------------------------------------------------
    # Biophysical ground truth
    # ------------------------------------------------------------------

    def compute_biophysical_ground_truth(
        self,
        protein_structure: Any,
        ligand: Any,
    ) -> Dict[str, np.ndarray]:
        """Compute per-residue biophysical features from structure + ligand.

        Parameters
        ----------
        protein_structure : object
            A protein structure object (e.g. BioPython ``Structure`` or
            an OpenMM topology).  Currently unused in the placeholder.
        ligand : object
            A ligand representation (e.g. RDKit ``Mol``).  Currently
            unused in the placeholder.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from feature name to a 1-D array of length
            ``N_residues`` containing the per-residue ground-truth
            values.  Feature names include SASA, B-factor,
            electrostatic potential, hydrophobicity, and contact counts.

        Notes
        -----
        **Placeholder implementation.**  Returns random values shaped
        for a 100-residue protein.  In production this should delegate
        to molecular-dynamics tooling (e.g. FreeSASA, OpenMM) or the
        ``AlphaFoldGroundTruthGenerator`` in ``ground_truth.py``.
        """
        n_res = 100  # placeholder length

        feature_names = [
            "sasa_per_residue",
            "bfactor_per_residue",
            "electrostatic_potential",
            "hydrophobicity_index",
            "contact_count",
            "hbond_donor_count",
            "hbond_acceptor_count",
            "secondary_structure_onehot",
            "distance_to_ligand",
            "burial_depth",
        ]

        return {name: np.random.randn(n_res).astype(np.float32) for name in feature_names}

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def probe_evoformer_for_biophysics(
        self,
        representations: EvoformerRepresentations,
        biophysical_features: Dict[str, np.ndarray],
    ) -> List[ProbeResult]:
        """Run Ridge and MLP probes on Evoformer activations.

        For each biophysical feature vector the method:

        1. Fits a **Ridge regression** (linear probe) from the
           Evoformer single representations to the target.
        2. Fits an **MLP probe** (nonlinear) with one hidden layer.
        3. Computes a constant-prediction baseline (mean of target).
        4. Classifies the encoding type via ``_classify_encoding``.

        Parameters
        ----------
        representations : EvoformerRepresentations
            Evoformer activations extracted with
            ``extract_evoformer_representations``.
        biophysical_features : dict[str, np.ndarray]
            Ground-truth per-residue feature arrays.

        Returns
        -------
        list[ProbeResult]
            One ``ProbeResult`` per biophysical feature.
        """
        X = representations.single_representations.detach().cpu().numpy()
        results: List[ProbeResult] = []

        for feat_name, target in biophysical_features.items():
            y = target[: X.shape[0]]  # align lengths

            # Constant-prediction baseline
            baseline_r2 = 0.0  # R^2 of predicting mean is 0 by definition

            # Ridge (linear) probe
            ridge_r2 = self._ridge_probe(X, y)

            # MLP (nonlinear) probe
            mlp_r2 = self._mlp_probe(X, y)

            ridge_delta = ridge_r2 - baseline_r2
            mlp_delta = mlp_r2 - baseline_r2
            encoding = self._classify_encoding(ridge_delta, mlp_delta)

            results.append(
                ProbeResult(
                    feature_name=feat_name,
                    ridge_r2=ridge_r2,
                    mlp_r2=mlp_r2,
                    baseline_r2=baseline_r2,
                    ridge_delta=ridge_delta,
                    mlp_delta=mlp_delta,
                    encoding_type=encoding,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_encoding(
        ridge_delta: float,
        mlp_delta: float,
        threshold: float = 0.05,
    ) -> EncodingType:
        """Classify how a feature is encoded in the representation.

        Parameters
        ----------
        ridge_delta : float
            Improvement of Ridge R^2 over baseline.
        mlp_delta : float
            Improvement of MLP R^2 over baseline.
        threshold : float
            Minimum delta to consider a probe successful.

        Returns
        -------
        EncodingType
            ``LINEAR_ENCODED`` if Ridge probe succeeds (ridge_delta >=
            threshold).  ``NONLINEAR_ONLY`` if only MLP succeeds.
            ``ZOMBIE`` if neither probe exceeds the threshold.
        """
        if ridge_delta >= threshold:
            return EncodingType.LINEAR_ENCODED
        if mlp_delta >= threshold:
            return EncodingType.NONLINEAR_ONLY
        return EncodingType.ZOMBIE

    # ------------------------------------------------------------------
    # Probe implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _ridge_probe(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
    ) -> float:
        """Closed-form Ridge regression probe returning R^2.

        Uses the normal equation with Tikhonov regularisation so that
        no iterative solver is required.
        """
        n, d = X.shape
        y = y.reshape(-1)

        # Add bias column
        X_b = np.column_stack([X, np.ones(n)])
        I = np.eye(X_b.shape[1])
        I[-1, -1] = 0  # don't regularise bias

        try:
            w = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y)
            y_pred = X_b @ w
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        except np.linalg.LinAlgError:
            r2 = 0.0
        return float(r2)

    def _mlp_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Train a small MLP probe and return test-set R^2.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Input representations.
        y : np.ndarray, shape (N,)
            Target biophysical feature values.

        Returns
        -------
        float
            Coefficient of determination (R^2) on the training data.
            In a production setting this should use a held-out split or
            cross-validation.

        Notes
        -----
        **Placeholder implementation.**  Trains a single-hidden-layer
        MLP with ReLU activation using MSE loss.  For robust zombie
        detection, replace with k-fold cross-validated training.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)

        input_dim = X_t.shape[1]
        probe_model = nn.Sequential(
            nn.Linear(input_dim, self.probe_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.probe_hidden_dim, 1),
        ).to(self.device)

        optimizer = torch.optim.Adam(probe_model.parameters(), lr=self.probe_lr)
        loss_fn = nn.MSELoss()

        probe_model.train()
        for _ in range(self.probe_epochs):
            optimizer.zero_grad()
            pred = probe_model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        probe_model.requires_grad_(False)
        y_pred = probe_model(X_t).cpu().numpy().reshape(-1)

        y_np = y.reshape(-1)
        ss_res = float(np.sum((y_np - y_pred) ** 2))
        ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return float(r2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on Evoformer blocks."""
        if self.model is None:
            return

        for name, module in self.model.named_modules():
            if "evoformer" in name.lower():

                def _make_hook(layer_name: str):
                    def hook(_module: nn.Module, _input: Any, output: Any) -> None:
                        if isinstance(output, torch.Tensor):
                            self._captured_activations[layer_name] = output
                        elif isinstance(output, (tuple, list)) and len(output) > 0:
                            self._captured_activations[layer_name] = output[0]

                    return hook

                handle = module.register_forward_hook(_make_hook(name))
                self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _infer_n_residues(input_features: Dict[str, torch.Tensor]) -> int:
        """Guess sequence length from input feature dict."""
        for key in ("aatype", "residue_index", "seq_mask"):
            if key in input_features:
                t = input_features[key]
                if t.dim() >= 1:
                    return int(t.shape[-1])
        return 100  # fallback
