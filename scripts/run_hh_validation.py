#!/usr/bin/env python3
"""
DESCARTES-PHARMA Phase 1: HH Simulator Ground Truth Validation.

This is the UNIT TEST for the entire framework:
1. Generate HH dataset with known biological variables (m, h, n)
2. Train LSTM surrogate: I -> V (with normalization!)
3. VALIDATE output accuracy first (CC >= 0.7 required)
4. Probe hidden states for m, h, n
5. Expected: ENCODED for well-trained model
6. If ZOMBIE with good output -> genuine zombie finding
7. If ZOMBIE with bad output -> model failed, not probes
"""

import sys
import os
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from descartes_pharma.core.hh_simulator import HodgkinHuxleySimulator
from descartes_pharma.core.surrogate import (
    LSTMSurrogate, train_surrogate, extract_hidden_states, normalize_data
)
from descartes_pharma.probes.sae import PharmaSAE, train_sae, sae_probe_molecular_mechanisms
from descartes_pharma.utils.config import DescartesPharmaConfig
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor


def main():
    config = DescartesPharmaConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print("=" * 70)
    print("DESCARTES-PHARMA Phase 1: HH Simulator Ground Truth Validation")
    print("=" * 70)
    print(f"Device: {config.device}")

    # Step 1: Generate HH dataset
    print("\n[1/6] Generating HH dataset...")
    hh = HodgkinHuxleySimulator()
    dataset = hh.generate_dataset(
        n_trials=config.hh_n_trials,
        T=config.hh_T,
        dt=config.hh_dt,
        seed=config.seed
    )
    n_trials = dataset['n_trials']
    seq_len = dataset['inputs'].shape[1]

    # Check spiking activity
    V_data = dataset['outputs'].squeeze(-1)
    n_spiking = int(np.sum(np.any(V_data > 0, axis=1)))
    print(f"  {n_trials} trials, {seq_len} timesteps each")
    print(f"  {n_spiking}/{n_trials} trials contain spikes")
    print(f"  V range: [{V_data.min():.1f}, {V_data.max():.1f}] mV")

    # Step 2: Normalize data (CRITICAL FIX)
    print("\n[2/6] Normalizing data...")
    norm_dataset, stats = normalize_data(dataset)
    print(f"  I: mean={stats['inputs']['mean']:.2f}, std={stats['inputs']['std']:.2f}")
    print(f"  V: mean={stats['outputs']['mean']:.2f}, std={stats['outputs']['std']:.2f}")

    # Train/test split
    n_train = int(0.8 * n_trials)
    train_data = {
        'inputs': norm_dataset['inputs'][:n_train],
        'outputs': norm_dataset['outputs'][:n_train],
    }
    val_data = {
        'inputs': norm_dataset['inputs'][n_train:],
        'outputs': norm_dataset['outputs'][n_train:],
    }

    # Step 3: Train LSTM surrogate
    print("\n[3/6] Training LSTM surrogate (I -> V)...")
    t_start = time.time()
    model = LSTMSurrogate(
        input_dim=1,
        hidden_dim=config.hidden_dim,
        output_dim=1,
        n_layers=config.n_layers
    )
    model = train_surrogate(model, train_data, val_data,
                            epochs=config.epochs, lr=config.learning_rate,
                            device=config.device, batch_size=config.batch_size)
    elapsed = time.time() - t_start
    print(f"  Training time: {elapsed:.1f}s")

    # Step 4: Validate output accuracy BEFORE probing
    print("\n[4/6] Validating output accuracy...")
    model.to(config.device)
    model.eval()
    test_norm = {
        'inputs': norm_dataset['inputs'][n_train:],
        'outputs': norm_dataset['outputs'][n_train:],
    }
    hidden_3d = extract_hidden_states(model, test_norm, device=config.device)
    # hidden_3d: (n_test, seq_len, hidden_dim)

    with torch.no_grad():
        test_in = torch.tensor(test_norm['inputs'], dtype=torch.float32).to(config.device)
        pred_norm = model(test_in).cpu().numpy()

    # Denormalize
    V_mean, V_std = stats['outputs']['mean'], stats['outputs']['std']
    pred_V = pred_norm.squeeze(-1) * V_std + V_mean
    true_V = V_data[n_train:]

    # Cross-condition correlation
    correlations = []
    for i in range(len(true_V)):
        cc = np.corrcoef(pred_V[i], true_V[i])[0, 1]
        if not np.isnan(cc):
            correlations.append(cc)
    mean_cc = np.mean(correlations)
    output_rmse = np.sqrt(np.mean((pred_V - true_V) ** 2))

    print(f"  Output RMSE: {output_rmse:.2f} mV")
    print(f"  Mean cross-condition correlation: {mean_cc:.4f}")
    print(f"  Min/Max correlation: {min(correlations):.4f} / {max(correlations):.4f}")

    if mean_cc < 0.7:
        print(f"\n  WARNING: CC = {mean_cc:.3f} < 0.7 -- model hasn't learned I->V!")
        print(f"  All zombie verdicts are CORRECT (the model IS a zombie).")
        print(f"  Try: hidden_dim=128, epochs=500, or AdamW optimizer")
        print(f"  Continuing anyway for demonstration...\n")
    else:
        print(f"\n  PASS: CC = {mean_cc:.3f} >= 0.7 -- model learned the dynamics!")

    # Step 5: Probe hidden states
    print("\n[5/6] Probing hidden states for biological variables...")

    # Subsample: every 10th timestep to reduce temporal correlation
    step = 10
    n_test = hidden_3d.shape[0]
    hidden_sub = hidden_3d[:, ::step, :].reshape(-1, config.hidden_dim)

    # Bio targets for test set
    bio_m = dataset['bio_targets'][n_train:, ::step, 0].reshape(-1)
    bio_h = dataset['bio_targets'][n_train:, ::step, 1].reshape(-1)
    bio_n = dataset['bio_targets'][n_train:, ::step, 2].reshape(-1)
    bio_targets = {'m': bio_m, 'h': bio_h, 'n': bio_n}

    # Random baseline
    random_model = LSTMSurrogate(
        input_dim=1, hidden_dim=config.hidden_dim,
        output_dim=1, n_layers=config.n_layers
    ).to(config.device)
    random_model.eval()
    random_3d = extract_hidden_states(random_model, test_norm, device=config.device)
    random_sub = random_3d[:, ::step, :].reshape(-1, config.hidden_dim)

    print(f"  Probing on {hidden_sub.shape[0]} samples "
          f"({n_test} trials x {seq_len // step} subsampled steps)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  {'Var':<6} {'Ridge(T)':>10} {'Ridge(R)':>10} {'Ridge dR2':>10} "
          f"{'MLP(T)':>10} {'MLP(R)':>10} {'MLP dR2':>10} {'Verdict':>12}")
    print(f"  {'-' * 82}")

    results = {}
    for var_name, target in bio_targets.items():
        # Ridge probe
        r2_t = np.mean(cross_val_score(
            Ridge(alpha=1.0), hidden_sub, target, cv=kf, scoring='r2'))
        r2_r = np.mean(cross_val_score(
            Ridge(alpha=1.0), random_sub, target, cv=kf, scoring='r2'))
        ridge_delta = r2_t - r2_r

        # MLP probe (sklearn for stability)
        mlp_t = np.mean(cross_val_score(
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                         early_stopping=True, random_state=42),
            hidden_sub, target, cv=kf, scoring='r2'))
        mlp_r = np.mean(cross_val_score(
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                         early_stopping=True, random_state=42),
            random_sub, target, cv=kf, scoring='r2'))
        mlp_delta = mlp_t - mlp_r

        # Classify
        if ridge_delta > 0.05 or mlp_delta > 0.05:
            if mlp_delta > ridge_delta + 0.1:
                verdict = "NONLINEAR"
            else:
                verdict = "ENCODED"
        else:
            verdict = "ZOMBIE"

        results[var_name] = {
            'ridge_trained_r2': r2_t, 'ridge_random_r2': r2_r,
            'ridge_delta_r2': ridge_delta,
            'mlp_trained_r2': mlp_t, 'mlp_random_r2': mlp_r,
            'mlp_delta_r2': mlp_delta,
            'encoding_type': verdict,
        }

        print(f"  {var_name:<6} {r2_t:>10.4f} {r2_r:>10.4f} {ridge_delta:>10.4f} "
              f"{mlp_t:>10.4f} {mlp_r:>10.4f} {mlp_delta:>10.4f} {verdict:>12}")

    # Step 6: SAE polypharmacology detection
    print("\n[6/6] SAE polypharmacology detection...")
    bio_sub_matrix = np.column_stack([bio_targets[n] for n in ['m', 'h', 'n']])

    for expansion in config.sae_expansion_factors:
        print(f"\n  SAE expansion={expansion}x:")
        sae, loss = train_sae(
            [hidden_sub], hidden_sub.shape[1],
            expansion_factor=expansion, k=config.sae_k,
            device=config.device
        )
        sae_results = sae_probe_molecular_mechanisms(
            sae, hidden_sub, bio_sub_matrix, ['m', 'h', 'n'],
            device=config.device
        )
        print(f"    Alive features: {sae_results['n_alive']}")
        print(f"    Mean monosemanticity: {sae_results['mean_monosemanticity']:.4f}")
        for name in ['m', 'h', 'n']:
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
    print(f"  Output quality: CC={mean_cc:.3f}, RMSE={output_rmse:.1f} mV")
    print(f"  Encoding: {n_encoded}/{n_total} variables detected")

    if mean_cc >= 0.7 and n_encoded >= n_total * 0.7:
        print(f"  PASS: Model learned dynamics AND probes detected encoding")
        print(f"  -> Probe framework VALIDATED for pharma application")
    elif mean_cc >= 0.7 and n_encoded < n_total * 0.7:
        print(f"  INTERESTING: Good output but hidden states are zombies!")
        print(f"  -> Model may have learned a shortcut (superposition?)")
        print(f"  -> Try hidden_dim=128,256 to test capacity hypothesis")
    elif mean_cc < 0.7:
        print(f"  EXPECTED: Model didn't learn -> zombie verdicts are correct")
        print(f"  -> Improve training before interpreting probe results")
    print("=" * 70)


if __name__ == '__main__':
    main()
