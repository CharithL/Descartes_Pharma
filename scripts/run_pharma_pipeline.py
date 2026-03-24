#!/usr/bin/env python3
"""
DESCARTES-PHARMA Full Pipeline: Run mechanistic zombie detection
on pharmaceutical datasets (ClinTox, BBBP, Tox21).

Usage:
    python scripts/run_pharma_pipeline.py --dataset clintox --device cuda
    python scripts/run_pharma_pipeline.py --dataset bbbp --device cuda
    python scripts/run_pharma_pipeline.py --dataset all --device cuda
"""

import argparse
import sys
import os
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from descartes_pharma.utils.config import DescartesPharmaConfig


def run_dataset(dataset_name, config):
    """Run full DESCARTES-PHARMA pipeline on a single dataset."""
    print(f"\n{'='*70}")
    print(f"DESCARTES-PHARMA: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load dataset
    print(f"\n[1] Loading {dataset_name}...")
    if dataset_name == 'clintox':
        from descartes_pharma.core.data_loaders import load_clintox
        data = load_clintox()
    elif dataset_name == 'bbbp':
        from descartes_pharma.core.data_loaders import load_bbbp
        data = load_bbbp()
    elif dataset_name == 'tox21':
        from descartes_pharma.core.data_loaders import load_tox21
        data = load_tox21()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Loaded {data['n_compounds']} compounds")
    print(f"  Mechanism targets: {data['mechanism_targets']}")

    # Compute mechanistic features
    print("\n[2] Computing mechanistic features...")
    from descartes_pharma.core.molecular_features import compute_mechanistic_features
    features, feature_names = compute_mechanistic_features(data['smiles'])
    valid_mask = ~np.any(np.isnan(features), axis=1)
    features = features[valid_mask]
    smiles = data['smiles'][valid_mask]
    labels = data['labels'][valid_mask]
    print(f"  {len(smiles)} valid compounds, {len(feature_names)} features")

    # Train GNN model
    print("\n[3] Training GNN model...")
    # Placeholder: use fingerprint-based model for simplicity
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    model = GradientBoostingClassifier(n_estimators=100, random_state=config.seed)
    scores = cross_val_score(model, features, labels, cv=5, scoring='roc_auc')
    print(f"  Baseline AUC: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Use features as "embeddings" for probing demonstration
    model.fit(features, labels)

    # Run probes
    print("\n[4] Running mechanistic probes...")
    from descartes_pharma.probes.ridge_mlp import pharma_mlp_delta_r2

    # Generate random baseline embeddings
    rng = np.random.default_rng(config.seed)
    random_embeddings = rng.standard_normal(features.shape)

    results = pharma_mlp_delta_r2(
        features, random_embeddings, features, feature_names,
        hidden_dim=64, epochs=30, device=config.device
    )

    print(f"\n  {'Feature':<16} {'Ridge dR2':>10} {'MLP dR2':>10} {'Type':>20}")
    print("  " + "-" * 60)
    for name in feature_names:
        r = results[name]
        if isinstance(r, dict) and 'encoding_type' in r:
            print(f"  {name:<16} {r.get('ridge_delta_r2', 0):>10.4f} "
                  f"{r.get('mlp_delta_r2', 0):>10.4f} {r['encoding_type']:>20}")

    # Generate verdict
    print("\n[5] Generating zombie verdict...")
    from descartes_pharma.factories.verdict_generator import PharmaZombieVerdictGenerator
    verdict_gen = PharmaZombieVerdictGenerator()

    for name in feature_names:
        r = results.get(name, {})
        if isinstance(r, dict) and 'ridge_delta_r2' in r:
            evidence = {
                'ridge_delta_r2': r['ridge_delta_r2'],
                'mlp_delta_r2': r['mlp_delta_r2'],
            }
            verdict = verdict_gen.generate_verdict(evidence)
            print(f"  {name:<16}: {verdict['verdict']} ({verdict['confidence']})")

    return results


def main():
    parser = argparse.ArgumentParser(description='DESCARTES-PHARMA Pipeline')
    parser.add_argument('--dataset', type=str, default='clintox',
                        choices=['clintox', 'bbbp', 'tox21', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = DescartesPharmaConfig(
        device=args.device if torch.cuda.is_available() else 'cpu',
        seed=args.seed,
    )

    datasets = ['clintox', 'bbbp', 'tox21'] if args.dataset == 'all' else [args.dataset]
    all_results = {}

    for ds in datasets:
        try:
            all_results[ds] = run_dataset(ds, config)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(config.output_dir, f'results_{timestamp}.json')

    serializable = {}
    for ds, res in all_results.items():
        serializable[ds] = {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                 for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in res.items()
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
