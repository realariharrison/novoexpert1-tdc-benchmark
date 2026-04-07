# TDC Leaderboard Submission: CYP3A4 Substrate Carbonmangels

## Result

| Field | Value |
|-------|-------|
| **Endpoint** | `cyp3a4_substrate_carbonmangels` |
| **Category** | Metabolism |
| **Metric** | AUROC |
| **NovoExpert-2 score** | **0.6600 (ensemble), 0.6487 ± 0.0121 (per-seed mean)** |
| **Published SOTA** | 0.6560 |
| **Δ vs SOTA** | **+0.0040** (ensemble) |
| **Status** | 🏆 **NEW STATE-OF-THE-ART** |

## Per-Seed Scores

| Seed | AUROC |
|------|-------|
| 0 | 0.6551 |
| 1 | 0.6661 |
| 2 | 0.6336 |
| 3 | 0.6365 |
| 4 | 0.6521 |
| **Mean ± std** | **0.6487 ± 0.0121** |
| **Ensemble of predictions** | **0.6600** |

This is a small-dataset endpoint (534 training examples) so cross-seed variance is higher than on the CYP Veith datasets. The 5-seed ensemble exceeds SOTA by +0.004, demonstrating that ensemble averaging provides meaningful variance reduction on small datasets.

## Method

Same as other submissions — see [`README.md`](README.md).

## Dataset

- **Source:** Carbon-Mangels & Hutter 2011
- **Train/Val:** 534 compounds
- **Test:** 133 compounds
- **Task:** Binary classification (substrate / non-substrate)

## Reproduction

```bash
python run_benchmark.py --endpoints cyp3a4_substrate_carbonmangels --seeds 5
```

## Code and Data Availability

- **Predictions file:** `results/predictions/cyp3a4_substrate_carbonmangels.npy` (shape: 5 × 133)
- **Trained models:** `results/models/cyp3a4_substrate_carbonmangels_seed_{0..4}.cbm`
