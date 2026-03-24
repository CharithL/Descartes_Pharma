#!/usr/bin/env python3
"""
DESCARTES-PHARMA Phase 1: HH Simulator Ground Truth Validation.

This is the UNIT TEST for the entire framework:
1. Generate HH dataset with known biological variables (m, h, n)
2. Train LSTM surrogate: I -> V
3. Probe hidden states for m, h, n
4. Expected: NON-ZOMBIE for well-trained model
5. If ZOMBIE -> probes are broken, not the model
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from descartes_pharma.core.hh_simulator import HodgkinHuxleySimulator
from descartes_pharma.core.surrogate import LSTMSurrogate, train_surrogate, extract_hidden_states
from descartes_pharma.probes.ridge_mlp import pharma_mlp_delta_r2
from descartes_pharma.probes.sae import PharmaSAE, train_sae, sae_probe_molecular_mechanisms
from descartes_pharma.utils.config import DescartesPharmaConfig


def main():
    config = DescartesPharmaConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        hh_n_trials=200,
        hidden_dim=64,
        epochs=200,
    )

    print("=" * 70)
    print("DESCARTES-PHARMA Phase 1: HH Simulator Ground Truth Validation")
    print("=" * 70)

    # Step 1: Generate HH dataset
    print("\n[1/5] Generating HH dataset...")
    hh = HodgkinHuxleySimulator()
    dataset = hh.generate_dataset(
        n_trials=config.hh_n_trials,
        T=config.hh_T,
        dt=config.hh_dt,
        seed=config.seed
    )
    print(f"  Generated {dataset['n_trials']} trials, "
          f"{dataset['inputs'].shape[1]} timesteps each")

    # Step 2: Train LSTM surrogate
    print("\n[2/5] Training LSTM surrogate (I -> V)...")
    n_train = int(0.8 * config.hh_n_trials)
    train_data = {
        'inputs': dataset['inputs'][:n_train],
        'outputs': dataset['outputs'][:n_train],
    }
    val_data = {
        'inputs': dataset['inputs'][n_train:],
        'outputs': dataset['outputs'][n_train:],
    }

    model = LSTMSurrogate(
        input_dim=1,
        hidden_dim=config.hidden_dim,
        output_dim=1,
        n_layers=config.n_layers
    )
    model = train_surrogate(model, train_data, val_data,
                            epochs=config.epochs, lr=config.learning_rate,
                            device=config.device)

    # Step 3: Extract hidden states
    print("\n[3/5] Extracting hidden states...")
    trained_hidden = extract_hidden_states(model, dataset, device=config.device)

    # Random baseline (untrained model)
    random_model = LSTMSurrogate(
        input_dim=1,
        hidden_dim=config.hidden_dim,
        output_dim=1,
        n_layers=config.n_layers
    )
    random_hidden = extract_hidden_states(random_model, dataset, device=config.device)

    # Prepare biological targets
    n_samples = trained_hidden.shape[0]
    bio = dataset['bio_targets'].reshape(-1, 7)[:n_samples]

    print(f"  Trained hidden shape: {trained_hidden.shape}")
    print(f"  Bio targets shape: {bio.shape}")

    # Step 4: Run Ridge + MLP probes
    print("\n[4/5] Running Ridge + MLP probes for all biological variables...")
    results = pharma_mlp_delta_r2(
        trained_hidden, random_hidden, bio, dataset['target_names'],
        hidden_dim=64, epochs=50, device=config.device
    )

    print("\n  Results:")
    print(f"  {'Variable':<12} {'Ridge dR2':>10} {'MLP dR2':>10} {'Encoding':>20}")
    print("  " + "-" * 55)
    for name in dataset['target_names']:
        r = results[name]
        print(f"  {name:<12} {r['ridge_delta_r2']:>10.4f} {r['mlp_delta_r2']:>10.4f} "
              f"{r['encoding_type']:>20}")

    # Step 5: SAE polypharmacology detection
    print("\n[5/5] SAE polypharmacology detection...")
    for expansion in config.sae_expansion_factors:
        print(f"\n  SAE expansion={expansion}x:")
        sae, loss = train_sae(
            [trained_hidden], trained_hidden.shape[1],
            expansion_factor=expansion, k=config.sae_k,
            device=config.device
        )
        sae_results = sae_probe_molecular_mechanisms(
            sae, trained_hidden, bio, dataset['target_names'],
            device=config.device
        )
        print(f"    Alive features: {sae_results['n_alive']}")
        print(f"    Mean monosemanticity: {sae_results['mean_monosemanticity']:.4f}")
        for name in dataset['target_names']:
            poly = sae_results['polypharmacology_detected'][name]
            print(f"    {name}: SAE R2={sae_results['sae_r2'][name]:.4f} "
                  f"Raw R2={sae_results['raw_r2'][name]:.4f} "
                  f"{'POLYPHARMACOLOGY' if poly else 'monosemantic'}")

    # Summary
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    n_encoded = sum(1 for r in results.values()
                    if r['encoding_type'] != 'ZOMBIE')
    n_total = len(results)
    if n_encoded >= n_total * 0.7:
        print(f"  PASS: {n_encoded}/{n_total} variables encoded -> PROBES VALIDATED")
    else:
        print(f"  WARNING: Only {n_encoded}/{n_total} variables encoded")
        print("  Check probe implementations before applying to pharma data!")


if __name__ == '__main__':
    main()
