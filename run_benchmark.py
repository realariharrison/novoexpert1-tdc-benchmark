#!/usr/bin/env python3
"""
NovoExpert-1 TDC ADMET Benchmark

Trains and evaluates Chemprop models on TDC's standardized benchmark splits.

Usage:
    python run_benchmark.py --seeds 5
    python run_benchmark.py --targets cyp2d6_veith cyp3a4_veith --seeds 5
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
torch.serialization.add_safe_globals([argparse.Namespace])

import pandas as pd
import numpy as np
from tdc.benchmark_group import admet_group
from sklearn.metrics import roc_auc_score

from chemprop.train import cross_validate, run_training
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import make_predictions


# TDC endpoints and reference scores
TDC_TARGETS = {
    'herg': {'name': 'hERG', 'sota': 0.880, 'baseline': 0.780},
    'cyp2c9_veith': {'name': 'CYP2C9', 'sota': 0.900, 'baseline': 0.820},
    'cyp2d6_veith': {'name': 'CYP2D6', 'sota': 0.750, 'baseline': 0.680},
    'cyp3a4_veith': {'name': 'CYP3A4', 'sota': 0.900, 'baseline': 0.830},
    'pgp_broccatelli': {'name': 'P-glycoprotein', 'sota': 0.940, 'baseline': 0.910},
}


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      model_dir: Path, epochs: int = 50, seed: int = 0) -> np.ndarray:
    """Train Chemprop model and return test predictions."""
    train_path = model_dir / 'train.csv'
    test_path = model_dir / 'test.csv'
    preds_path = model_dir / 'preds.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Training arguments
    train_args = [
        '--data_path', str(train_path),
        '--dataset_type', 'classification',
        '--save_dir', str(model_dir),
        '--epochs', str(epochs),
        '--batch_size', '64',
        '--hidden_size', '300',
        '--depth', '3',
        '--dropout', '0.1',
        '--seed', str(seed),
        '--quiet',
    ]

    args = TrainArgs().parse_args(train_args)
    mean_score, std_score = cross_validate(args=args, train_func=run_training)

    # Make predictions
    pred_args = [
        '--test_path', str(test_path),
        '--checkpoint_dir', str(model_dir),
        '--preds_path', str(preds_path),
    ]

    pargs = PredictArgs().parse_args(pred_args)
    predictions = make_predictions(args=pargs)

    return np.array([p[0] for p in predictions])


def run_benchmark(targets: list, epochs: int = 50, n_seeds: int = 5):
    """Run TDC benchmark with multiple seeds."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'tdc_data'
    model_dir = base_dir / 'models'
    results_dir = base_dir / 'results'

    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    group = admet_group(path=str(data_dir))
    all_results = {}

    for tdc_name in targets:
        if tdc_name not in TDC_TARGETS:
            print(f"Unknown target: {tdc_name}")
            continue

        ref = TDC_TARGETS[tdc_name]
        print(f"\n{'='*60}")
        print(f"Benchmarking: {tdc_name} ({ref['name']})")
        print(f"Epochs: {epochs}, Seeds: {n_seeds}")
        print('='*60)

        # Load TDC data
        benchmark = group.get(tdc_name)
        train_val = benchmark['train_val']
        test = benchmark['test']

        print(f"Train/Val: {len(train_val)}, Test: {len(test)}")

        train_df = pd.DataFrame({
            'smiles': train_val['Drug'],
            'target': train_val['Y']
        })
        test_df = pd.DataFrame({
            'smiles': test['Drug'],
            'target': test['Y']
        })

        y_true = test['Y'].values
        seed_scores = []

        for seed in range(n_seeds):
            seed_dir = model_dir / f'{tdc_name}_seed{seed}'
            seed_dir.mkdir(parents=True, exist_ok=True)

            try:
                print(f"  Training seed {seed}...", end=' ', flush=True)
                preds = train_and_predict(train_df, test_df, seed_dir, epochs, seed)
                score = roc_auc_score(y_true, preds)
                seed_scores.append(score)
                print(f"AUROC: {score:.4f}")
            except Exception as e:
                print(f"Error: {e}")

        if seed_scores:
            mean_score = np.mean(seed_scores)
            std_score = np.std(seed_scores)

            all_results[tdc_name] = {
                'name': ref['name'],
                'mean': round(float(mean_score), 4),
                'std': round(float(std_score), 4),
                'scores': [round(float(s), 4) for s in seed_scores],
                'sota': ref['sota'],
                'baseline': ref['baseline'],
                'beats_sota': bool(mean_score >= ref['sota']),
                'above_baseline': bool(mean_score >= ref['baseline'])
            }

            status = "SOTA" if mean_score >= ref['sota'] else \
                     "Above baseline" if mean_score >= ref['baseline'] else \
                     "Below baseline"
            print(f"\n  Result: {mean_score:.4f} +/- {std_score:.4f} ({status})")

    # Save results
    results_path = results_dir / 'tdc_benchmark_results.json'
    output = {
        'timestamp': datetime.now().isoformat(),
        'model': 'NovoExpert-1 Chemprop',
        'epochs': epochs,
        'n_seeds': n_seeds,
        'results': all_results
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"\n{'Endpoint':<20} {'Score':>15} {'SOTA':>10} {'Status':>15}")
    print('-'*60)

    for name, r in all_results.items():
        score_str = f"{r['mean']:.4f} +/- {r['std']:.4f}"
        status = "SOTA" if r['beats_sota'] else \
                 "Above baseline" if r['above_baseline'] else \
                 "Below baseline"
        print(f"{name:<20} {score_str:>15} {r['sota']:>10.3f} {status:>15}")

    print(f"\nResults saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NovoExpert-1 TDC Benchmark')
    parser.add_argument('--targets', nargs='+',
                        default=list(TDC_TARGETS.keys()),
                        help='TDC targets to benchmark')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of random seeds (default: 5)')
    args = parser.parse_args()

    run_benchmark(args.targets, args.epochs, args.seeds)
