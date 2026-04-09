#!/usr/bin/env python3
"""
Validate pre-trained NovoExpert-2 models against TDC test sets.

Loads committed model files, runs inference on official TDC test splits,
and prints per-endpoint scores. No retraining required.

Usage:
    python validate_models.py
    python validate_models.py --endpoints cyp2d6_veith dili
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr

from run_benchmark import (
    TDC_ENDPOINTS,
    compute_maplight_features,
    compute_gin_embeddings,
    evaluate,
    is_improvement,
)

SOTA_ENDPOINTS = [
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp3a4_substrate_carbonmangels",
    "clearance_hepatocyte_az",
    "dili",
]

EXPECTED_SCORES = {
    "cyp2d6_veith": 0.778,
    "cyp3a4_veith": 0.916,
    "cyp3a4_substrate_carbonmangels": 0.660,
    "clearance_hepatocyte_az": 0.511,
    "dili": 0.922,
}


def validate_catboost_endpoint(endpoint, config, model_dir, n_seeds=5):
    """Load saved CatBoost models and score on TDC test set."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from tdc.benchmark_group import admet_group

    task = config["task"]
    metric = config["metric"]

    group = admet_group(path="./tdc_data")
    benchmark = group.get(endpoint)
    train_val = benchmark["train_val"]
    test = benchmark["test"]
    test_labels = test["Y"].values

    # Compute features for test set
    all_smiles = list(train_val["Drug"]) + list(test["Drug"])
    maplight = compute_maplight_features(all_smiles)
    gin = compute_gin_embeddings(all_smiles)
    if gin is not None:
        features = np.hstack([maplight, gin])
    else:
        features = maplight

    X_test = features[len(train_val):]

    # Load models and predict
    seed_preds = []
    seed_scores = []
    for seed in range(n_seeds):
        model_path = model_dir / endpoint / f"seed_{seed}.cbm"
        if not model_path.exists():
            print(f"  ERROR: {model_path} not found")
            return None

        if task == "classification":
            model = CatBoostClassifier()
        else:
            model = CatBoostRegressor()
        model.load_model(str(model_path))

        if task == "classification":
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)

        score = evaluate(test_labels, preds, metric)
        seed_preds.append(preds)
        seed_scores.append(score)
        print(f"  Seed {seed}: {metric} = {score:.4f}")

    ensemble_preds = np.mean(seed_preds, axis=0)
    ensemble_score = evaluate(test_labels, ensemble_preds, metric)
    return {
        "seed_scores": seed_scores,
        "ensemble_score": ensemble_score,
        "metric": metric,
    }


def validate_chemprop_endpoint(endpoint, config, model_dir, n_seeds=5):
    """Load saved Chemprop models and score on TDC test set."""
    try:
        import chemprop
    except ImportError:
        print("  WARNING: chemprop not installed, loading saved predictions instead")
        return validate_from_predictions(endpoint, config)

    from tdc.benchmark_group import admet_group

    metric = config["metric"]
    group = admet_group(path="./tdc_data")
    benchmark = group.get(endpoint)
    test = benchmark["test"]
    test_labels = test["Y"].values
    test_smiles = test["Drug"].tolist()

    seed_preds = []
    seed_scores = []
    for seed in range(n_seeds):
        ckpt_path = model_dir / endpoint / f"chemprop_seed_{seed}.ckpt"
        if not ckpt_path.exists():
            print(f"  ERROR: {ckpt_path} not found")
            return None

        model = chemprop.nn.load_model(str(ckpt_path))
        smiles_data = [[s] for s in test_smiles]
        preds = model.predict(smiles_data)
        preds = np.array(preds).flatten()

        score = evaluate(test_labels, preds, metric)
        seed_preds.append(preds)
        seed_scores.append(score)
        print(f"  Seed {seed}: {metric} = {score:.4f}")

    ensemble_preds = np.mean(seed_preds, axis=0)
    ensemble_score = evaluate(test_labels, ensemble_preds, metric)
    return {
        "seed_scores": seed_scores,
        "ensemble_score": ensemble_score,
        "metric": metric,
    }


def validate_from_predictions(endpoint, config):
    """Fallback: validate from saved .npy prediction files."""
    metric = config["metric"]
    from tdc.benchmark_group import admet_group

    group = admet_group(path="./tdc_data")
    benchmark = group.get(endpoint)
    test_labels = benchmark["test"]["Y"].values

    # Try chemprop-specific file first, then generic
    for suffix in [f"{endpoint}_chemprop.npy", f"{endpoint}.npy"]:
        pred_path = Path("results/predictions") / suffix
        if pred_path.exists():
            preds_all = np.load(str(pred_path))
            break
    else:
        print(f"  ERROR: No prediction file found for {endpoint}")
        return None

    seed_scores = []
    for i in range(preds_all.shape[0]):
        score = evaluate(test_labels, preds_all[i], metric)
        seed_scores.append(score)
        print(f"  Seed {i}: {metric} = {score:.4f}")

    ensemble_preds = np.mean(preds_all, axis=0)
    ensemble_score = evaluate(test_labels, ensemble_preds, metric)
    return {
        "seed_scores": seed_scores,
        "ensemble_score": ensemble_score,
        "metric": metric,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate NovoExpert-2 trained models")
    parser.add_argument(
        "--endpoints", nargs="+", default=None,
        help=f"Endpoints to validate (default: 5 SOTA endpoints)"
    )
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--all", action="store_true", help="Validate all 22 endpoints")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    endpoints = args.endpoints or SOTA_ENDPOINTS

    print("=" * 70)
    print("NovoExpert-2 Model Validation")
    print("=" * 70)
    print(f"Validating {len(endpoints)} endpoints from {model_dir}/\n")

    all_pass = True
    results = {}

    for endpoint in endpoints:
        config = TDC_ENDPOINTS[endpoint]
        print(f"\n--- {endpoint} ({config['metric']}, SOTA={config['published']}) ---")

        if endpoint == "dili":
            result = validate_chemprop_endpoint(endpoint, config, model_dir)
        else:
            result = validate_catboost_endpoint(endpoint, config, model_dir)

        if result is None:
            all_pass = False
            continue

        ensemble = result["ensemble_score"]
        metric = result["metric"]
        published = config["published"]
        beats = is_improvement(ensemble, published, metric)
        expected = EXPECTED_SCORES.get(endpoint)

        print(f"  Ensemble: {ensemble:.4f}")
        print(f"  Published SOTA: {published}")
        print(f"  Beats SOTA: {'YES' if beats else 'NO'}")

        if expected:
            match = abs(ensemble - expected) < 0.002
            print(f"  Expected ~{expected:.3f}: {'MATCH' if match else 'MISMATCH'}")
            if not match:
                all_pass = False

        results[endpoint] = result

    print(f"\n{'=' * 70}")
    if all_pass:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED — check output above")
    print(f"{'=' * 70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
