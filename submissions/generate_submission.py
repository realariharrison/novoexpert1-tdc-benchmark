#!/usr/bin/env python3
"""
Generate TDC leaderboard submission JSON for NovoExpert-2 winning endpoints.

Loads the 5-seed predictions saved from run_benchmark.py, formats them
for TDC's group.evaluate() API, runs the official evaluation, and writes
a submission JSON file ready for leaderboard upload.

Usage:
    python generate_submission.py
    python generate_submission.py --results-dir ../results
"""

import json
import argparse
from pathlib import Path

import numpy as np
from tdc.benchmark_group import admet_group


WINNING_ENDPOINTS = [
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp3a4_substrate_carbonmangels",
    "clearance_hepatocyte_az",
    "dili",  # Uses Chemprop v2 (not CatBoost)
]

# Endpoints that use Chemprop predictions instead of CatBoost
CHEMPROP_ENDPOINTS = {"dili"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="../results",
                        help="Path to results/ directory from run_benchmark.py")
    parser.add_argument("--output", type=str, default="tdc_submission.json")
    parser.add_argument("--tdc-data-dir", type=str, default="./tdc_data")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    preds_dir = results_dir / "predictions"

    group = admet_group(path=args.tdc_data_dir)

    submission = {}
    for endpoint in WINNING_ENDPOINTS:
        # DILI uses Chemprop predictions, all others use CatBoost
        if endpoint in CHEMPROP_ENDPOINTS:
            preds_file = preds_dir / f"{endpoint}_chemprop.npy"
        else:
            preds_file = preds_dir / f"{endpoint}.npy"

        if not preds_file.exists():
            print(f"⚠ Missing predictions for {endpoint} at {preds_file}")
            continue

        # Shape: (n_seeds, n_test)
        preds_5_seeds = np.load(str(preds_file))
        if preds_5_seeds.ndim != 2 or preds_5_seeds.shape[0] != 5:
            print(f"⚠ Unexpected shape for {endpoint}: {preds_5_seeds.shape}")
            continue

        # TDC expects {seed: [prediction_list]}
        endpoint_preds = {seed: preds_5_seeds[seed].tolist() for seed in range(5)}
        submission[endpoint] = endpoint_preds
        print(f"✓ Loaded {endpoint}: 5 seeds × {preds_5_seeds.shape[1]} test samples")

    print("\nRunning TDC official evaluation...")
    results = group.evaluate(submission)

    # Format output
    output = {
        "method_name": "NovoExpert-2",
        "method_description": (
            "NovoExpert-2 evaluates two model families across all 22 TDC ADMET endpoints: "
            "(1) CatBoost gradient-boosted trees on 2,873-dim features combining MapLight "
            "fingerprints (Morgan ECFP + Avalon + ErG + RDKit 2D descriptors, 2,573 dims) with "
            "300-dim GIN supervised masking embeddings from DGL-LifeSci; (2) Chemprop v2 D-MPNN. "
            "For each endpoint we report the stronger method. CatBoost+MapLight+GIN is primary "
            "(used for CYP2D6, CYP3A4, CYP3A4 Substrate, Clearance Hepatocyte AZ). Chemprop v2 "
            "is reported for DILI where its learned representations outperformed CatBoost. All "
            "methods use 5-seed ensembles on TDC's official get_train_valid_split() API."
        ),
        "code_url": "https://github.com/realariharrison/novoexpert1-tdc-benchmark",
        "paper": "paper/novoexpert_tdc.tex",
        "method_per_endpoint": {
            "cyp2d6_veith": "CatBoost+MapLight+GIN",
            "cyp3a4_veith": "CatBoost+MapLight+GIN",
            "cyp3a4_substrate_carbonmangels": "CatBoost+MapLight+GIN",
            "clearance_hepatocyte_az": "CatBoost+MapLight+GIN",
            "dili": "Chemprop v2 (D-MPNN)",
        },
        "results": {},
    }

    print(f"\n{'='*70}")
    print("TDC Official Evaluation Results")
    print(f"{'='*70}")
    for endpoint, score_info in results.items():
        if isinstance(score_info, (list, tuple)) and len(score_info) >= 2:
            mean_score, std_score = float(score_info[0]), float(score_info[1])
        elif isinstance(score_info, dict):
            mean_score = float(score_info.get("mean", score_info.get("score", 0.0)))
            std_score = float(score_info.get("std", 0.0))
        else:
            mean_score = float(score_info)
            std_score = 0.0

        output["results"][endpoint] = {
            "mean": mean_score,
            "std": std_score,
        }
        print(f"  {endpoint:<40} {mean_score:.4f} ± {std_score:.4f}")

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Submission JSON written to: {args.output}")


if __name__ == "__main__":
    main()
