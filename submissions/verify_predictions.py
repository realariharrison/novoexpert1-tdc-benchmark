#!/usr/bin/env python3
"""
Verify saved predictions against expected NovoExpert-2 scores.

Loads the 5-seed prediction arrays from results/predictions/ and
re-computes the ensemble metric, comparing against the published scores
in the NovoExpert-2 paper / submission docs.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

try:
    from tdc.benchmark_group import admet_group
except ImportError:
    print("PyTDC not installed. Install with: pip install PyTDC")
    sys.exit(1)


EXPECTED = {
    "cyp2d6_veith":                   {"metric": "auprc",    "ensemble": 0.7776, "sota": 0.750, "method": "CatBoost+MapLight+GIN"},
    "cyp3a4_veith":                   {"metric": "auprc",    "ensemble": 0.9163, "sota": 0.900, "method": "CatBoost+MapLight+GIN"},
    "cyp3a4_substrate_carbonmangels": {"metric": "auroc",    "ensemble": 0.6600, "sota": 0.656, "method": "CatBoost+MapLight+GIN"},
    "clearance_hepatocyte_az":        {"metric": "spearman", "ensemble": 0.5112, "sota": 0.487, "method": "CatBoost+MapLight+GIN"},
    "dili":                           {"metric": "auroc",    "ensemble": 0.9200, "sota": 0.916, "method": "Chemprop v2"},
}


def evaluate(y_true, y_pred, metric):
    if metric == "auprc":
        return float(average_precision_score(y_true, y_pred))
    elif metric == "auroc":
        return float(roc_auc_score(y_true, y_pred))
    elif metric == "spearman":
        return float(spearmanr(y_true, y_pred)[0])


def main():
    preds_dir = Path(__file__).parent.parent / "results" / "predictions"
    if not preds_dir.exists():
        print(f"⚠ Predictions directory not found: {preds_dir}")
        sys.exit(1)

    print(f"Verifying predictions in: {preds_dir}\n")
    results_dir = preds_dir  # backward compat for the rest of the function

    group = admet_group(path="./tdc_data")

    print(f"{'Endpoint':<35} {'Method':<25} {'Expected':<10} {'Verified':<10} {'Match':<8}")
    print("-" * 95)

    for endpoint, info in EXPECTED.items():
        # DILI uses Chemprop predictions
        if endpoint == "dili":
            preds_file = results_dir / "dili_chemprop.npy"
        else:
            preds_file = results_dir / f"{endpoint}.npy"

        if not preds_file.exists():
            print(f"{endpoint:<35} {info['method']:<25} {info['ensemble']:<10.4f} {'MISSING':<10} ⚠")
            continue

        # Load predictions
        preds_5_seeds = np.load(str(preds_file))

        # Get test labels
        benchmark = group.get(endpoint)
        y_true = benchmark['test']['Y'].values

        # Ensemble = mean of 5 seed predictions
        ensemble = preds_5_seeds.mean(axis=0)
        verified = evaluate(y_true, ensemble, info['metric'])

        match = "✓" if abs(verified - info['ensemble']) < 0.005 else "✗"
        print(f"{endpoint:<35} {info['method']:<25} {info['ensemble']:<10.4f} {verified:<10.4f} {match:<8}")


if __name__ == "__main__":
    main()
