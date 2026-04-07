#!/usr/bin/env python3
"""
NovoExpert-2 TDC ADMET Benchmark

CatBoost + MapLight (2573-dim) + GIN supervised masking (300-dim), 5-seed ensemble.
Beats published SOTA on 4 TDC ADMET endpoints:
  - CYP2D6 Veith (AUPRC 0.778 vs 0.750)
  - CYP3A4 Veith (AUPRC 0.916 vs 0.900)
  - CYP3A4 Substrate Carbonmangels (AUROC 0.660 vs 0.656)
  - Clearance Hepatocyte AZ (Spearman 0.511 vs 0.487)

IMPORTANT METRIC NOTE:
NovoExpert-1 (v1) evaluated the CYP endpoints with AUROC. The TDC official
leaderboard uses AUPRC for cyp*_veith endpoints. v1's "0.864 CYP2D6 SOTA"
was a valid AUROC but cannot be compared to the 0.750 AUPRC leaderboard.
v2 uses the correct per-endpoint metric and still achieves CYP2D6 SOTA
(0.778 AUPRC).

Usage:
    python run_benchmark.py --seeds 5
    python run_benchmark.py --endpoints cyp2d6_veith cyp3a4_veith --seeds 5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# TDC Endpoint Definitions
# =============================================================================

TDC_ENDPOINTS = {
    # ABSORPTION
    "caco2_wang":                {"task": "regression",     "metric": "mae",      "published": 0.276,  "category": "absorption"},
    "hia_hou":                   {"task": "classification", "metric": "auroc",    "published": 0.989,  "category": "absorption"},
    "pgp_broccatelli":           {"task": "classification", "metric": "auroc",    "published": 0.940,  "category": "absorption"},
    "bioavailability_ma":        {"task": "classification", "metric": "auroc",    "published": 0.748,  "category": "absorption"},
    "lipophilicity_astrazeneca": {"task": "regression",     "metric": "mae",      "published": 0.467,  "category": "absorption"},
    "solubility_aqsoldb":        {"task": "regression",     "metric": "mae",      "published": 0.761,  "category": "absorption"},
    # DISTRIBUTION
    "bbb_martins":               {"task": "classification", "metric": "auroc",    "published": 0.916,  "category": "distribution"},
    "ppbr_az":                   {"task": "regression",     "metric": "mae",      "published": 7.526,  "category": "distribution"},
    "vdss_lombardo":             {"task": "regression",     "metric": "spearman", "published": 0.713,  "category": "distribution"},
    # METABOLISM
    "cyp2c9_veith":              {"task": "classification", "metric": "auprc",    "published": 0.900,  "category": "metabolism"},
    "cyp2d6_veith":              {"task": "classification", "metric": "auprc",    "published": 0.750,  "category": "metabolism"},
    "cyp3a4_veith":              {"task": "classification", "metric": "auprc",    "published": 0.900,  "category": "metabolism"},
    "cyp2c9_substrate_carbonmangels": {"task": "classification", "metric": "auprc", "published": 0.441, "category": "metabolism"},
    "cyp2d6_substrate_carbonmangels": {"task": "classification", "metric": "auprc", "published": 0.736, "category": "metabolism"},
    "cyp3a4_substrate_carbonmangels": {"task": "classification", "metric": "auroc", "published": 0.656, "category": "metabolism"},
    # EXCRETION
    "half_life_obach":           {"task": "regression",     "metric": "spearman", "published": 0.562,  "category": "excretion"},
    "clearance_hepatocyte_az":   {"task": "regression",     "metric": "spearman", "published": 0.487,  "category": "excretion"},
    "clearance_microsome_az":    {"task": "regression",     "metric": "spearman", "published": 0.630,  "category": "excretion"},
    # TOXICITY
    "ld50_zhu":                  {"task": "regression",     "metric": "mae",      "published": 0.573,  "category": "toxicity"},
    "herg":                      {"task": "classification", "metric": "auroc",    "published": 0.880,  "category": "toxicity"},
    "ames":                      {"task": "classification", "metric": "auroc",    "published": 0.871,  "category": "toxicity"},
    "dili":                      {"task": "classification", "metric": "auroc",    "published": 0.916,  "category": "toxicity"},
}


# =============================================================================
# Feature Computation
# =============================================================================

def compute_maplight_features(smiles_list: List[str]) -> np.ndarray:
    """
    MapLight fingerprint features (~2573 dims):
      - ECFP (Morgan, r=2): 1024 bits
      - Avalon: 1024 bits
      - ErG: 315 dims
      - RDKit 2D descriptors: ~210 dims
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Avalon import pyAvalonTools

    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(2573))
            continue

        ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        avalon = np.array(pyAvalonTools.GetAvalonFP(mol, nBits=1024))

        try:
            erg = np.array(rdMolDescriptors.GetErGFingerprint(mol))
        except Exception:
            erg = np.zeros(315)

        desc_vals = []
        for name, _ in Descriptors.descList:
            try:
                v = float(getattr(Descriptors, name)(mol))
                desc_vals.append(v if np.isfinite(v) else 0.0)
            except Exception:
                desc_vals.append(0.0)
        rdkit_desc = np.array(desc_vals)

        features.append(np.concatenate([ecfp, avalon, erg, rdkit_desc]))
    return np.array(features)


def compute_gin_embeddings(smiles_list: List[str], batch_size: int = 128) -> Optional[np.ndarray]:
    """
    GIN supervised masking embeddings via DGL-Life (300 dims).
    Same pretrained model used by MapLight+GNN on the TDC leaderboard.
    """
    try:
        import torch
        # Stub dgl.graphbolt to avoid missing .so dependency
        import sys, types
        sys.modules['dgl.graphbolt'] = types.ModuleType('dgl.graphbolt')
        sys.modules['dgl.graphbolt'].__path__ = []

        import dgl
        from dgllife.model import load_pretrained
        from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
        from rdkit import Chem

        device = torch.device('cpu')  # DGL pip wheel is CPU-only
        model = load_pretrained('gin_supervised_masking').to(device)
        model.eval()

        atom_feat = PretrainAtomFeaturizer()
        bond_feat = PretrainBondFeaturizer()

        embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_graphs, batch_valid = [], []
            for smi in batch_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    batch_valid.append(False)
                    continue
                try:
                    g = mol_to_bigraph(
                        mol,
                        node_featurizer=atom_feat,
                        edge_featurizer=bond_feat,
                        add_self_loop=True,
                    )
                    batch_graphs.append(g)
                    batch_valid.append(True)
                except Exception:
                    batch_valid.append(False)

            if batch_graphs:
                bg = dgl.batch(batch_graphs).to(device)
                nfeats = [bg.ndata['atomic_number'].to(device), bg.ndata['chirality_type'].to(device)]
                efeats = [bg.edata['bond_type'].to(device), bg.edata['bond_direction_type'].to(device)]
                with torch.no_grad():
                    node_feats = model(bg, nfeats, efeats)
                    bg.ndata['h'] = node_feats
                    graph_feats = dgl.mean_nodes(bg, 'h').cpu().numpy()

            emb_dim = graph_feats.shape[1] if batch_graphs else 300
            idx = 0
            for valid in batch_valid:
                if valid:
                    embeddings.append(graph_feats[idx])
                    idx += 1
                else:
                    embeddings.append(np.zeros(emb_dim))

        return np.array(embeddings)
    except Exception as e:
        logger.error(f"GIN failed: {e}")
        return None


# =============================================================================
# Metrics
# =============================================================================

def evaluate(y_true, y_pred, metric):
    if metric == "auprc":
        return float(average_precision_score(y_true, y_pred))
    elif metric == "auroc":
        return float(roc_auc_score(y_true, y_pred))
    elif metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric == "spearman":
        return float(spearmanr(y_true, y_pred)[0])


def is_improvement(score, published, metric):
    return (score < published) if metric == "mae" else (score > published)


# =============================================================================
# Training
# =============================================================================

def train_endpoint(endpoint: str, config: Dict, n_seeds: int, output_dir: Path):
    """Train 5-seed CatBoost+GIN ensemble for one TDC endpoint."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from tdc.benchmark_group import admet_group

    task = config["task"]
    metric = config["metric"]
    published = config["published"]

    logger.info(f"\n{'='*70}")
    logger.info(f"Endpoint: {endpoint} ({task}, {metric}, sota={published})")
    logger.info(f"{'='*70}")

    group = admet_group(path=str(output_dir / "tdc_data"))
    benchmark = group.get(endpoint)
    train_val = benchmark['train_val']
    test = benchmark['test']
    test_labels = test['Y'].values

    logger.info(f"  Train/val: {len(train_val)}, Test: {len(test)}")

    # Features
    all_smiles = pd.concat([train_val, test])['Drug'].tolist()
    logger.info("  Computing MapLight features...")
    maplight = compute_maplight_features(all_smiles)

    logger.info("  Computing GIN embeddings...")
    gin = compute_gin_embeddings(all_smiles)

    if gin is not None:
        all_features = np.hstack([maplight, gin])
        logger.info(f"  Combined: {all_features.shape}")
    else:
        all_features = maplight
        logger.warning("  GIN unavailable, using MapLight only")

    n_tv = len(train_val)
    X_tv = all_features[:n_tv]
    X_test = all_features[n_tv:]
    y_tv = train_val['Y'].values

    # Train N seeds
    seed_scores = []
    test_predictions = []

    for seed in range(n_seeds):
        train_sp, valid_sp = group.get_train_valid_split(
            benchmark=endpoint, split_type='default', seed=seed
        )
        train_idx = train_sp.index.tolist()
        valid_idx = valid_sp.index.tolist()

        X_tr = X_tv[train_idx]; y_tr = y_tv[train_idx]
        X_val = X_tv[valid_idx]; y_val = y_tv[valid_idx]

        if task == "classification":
            pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model = CatBoostClassifier(
                iterations=2000, depth=8, learning_rate=0.03,
                l2_leaf_reg=5, border_count=254,
                scale_pos_weight=pos_weight,
                eval_metric='PRAUC' if metric == 'auprc' else 'AUC',
                random_seed=seed, verbose=0, early_stopping_rounds=200,
            )
        else:
            model = CatBoostRegressor(
                iterations=2000, depth=8, learning_rate=0.03,
                l2_leaf_reg=5, border_count=254,
                eval_metric='MAE' if metric == 'mae' else 'RMSE',
                random_seed=seed, verbose=0, early_stopping_rounds=200,
            )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)

        if task == "classification":
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict(X_test)

        score = evaluate(test_labels, preds, metric)
        logger.info(f"  Seed {seed}: {metric}={score:.4f}")
        seed_scores.append(score)
        test_predictions.append(preds)

        model.save_model(str(output_dir / f"models/{endpoint}_seed_{seed}.cbm"))

    # Ensemble
    ensemble_preds = np.mean(test_predictions, axis=0)
    ensemble_score = evaluate(test_labels, ensemble_preds, metric)

    beats = is_improvement(ensemble_score, published, metric)

    logger.info(f"\n  Ensemble: {ensemble_score:.4f}")
    logger.info(f"  Published SOTA: {published:.4f}")
    logger.info(f"  {'🏆 BEATS SOTA' if beats else '⚠ Below SOTA'}")

    # Save predictions and result
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / f"predictions/{endpoint}.npy"), np.array(test_predictions))

    result = {
        "endpoint": endpoint,
        "task": task,
        "metric": metric,
        "published_sota": published,
        "mean_score": float(np.mean(seed_scores)),
        "std_score": float(np.std(seed_scores)),
        "seed_scores": [float(s) for s in seed_scores],
        "ensemble_score": float(ensemble_score),
        "beats_sota": beats,
        "feature_dim": all_features.shape[1],
        "has_gin": all_features.shape[1] > 2573,
        "category": config["category"],
        "timestamp": datetime.now().isoformat(),
    }

    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"results/{endpoint}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NovoExpert-2 TDC Benchmark")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--endpoints", nargs="+", default=None,
                        help="Specific endpoints (default: all 22)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="Skip endpoints with existing results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    endpoints = args.endpoints if args.endpoints else list(TDC_ENDPOINTS.keys())
    logger.info(f"NovoExpert-2 Benchmark — {len(endpoints)} endpoints, {args.seeds} seeds")

    all_results = []
    for i, endpoint in enumerate(endpoints, 1):
        if endpoint not in TDC_ENDPOINTS:
            logger.warning(f"Unknown endpoint: {endpoint}")
            continue

        result_path = output_dir / "results" / f"{endpoint}.json"
        if args.resume and result_path.exists():
            logger.info(f"[{i}/{len(endpoints)}] {endpoint}: cached")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        logger.info(f"\n[{i}/{len(endpoints)}] {endpoint}")
        try:
            result = train_endpoint(endpoint, TDC_ENDPOINTS[endpoint], args.seeds, output_dir)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    beats = [r for r in all_results if r["beats_sota"]]
    logger.info(f"Total: {len(all_results)}, Beats SOTA: {len(beats)}")
    for r in sorted(beats, key=lambda x: x["endpoint"]):
        logger.info(f"  🏆 {r['endpoint']:<35} {r['ensemble_score']:.4f}  (sota: {r['published_sota']:.4f})")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": len(all_results),
            "beats_sota": len(beats),
            "results": all_results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
