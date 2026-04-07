# TDC Leaderboard Submission: CYP3A4 Veith

## Result

| Field | Value |
|-------|-------|
| **Endpoint** | `cyp3a4_veith` |
| **Category** | Metabolism |
| **Metric** | AUPRC (Average Precision) |
| **NovoExpert-2 score** | **0.9163 (ensemble), 0.9149 ± 0.0007 (per-seed mean)** |
| **Published SOTA** | 0.9000 |
| **Δ vs SOTA** | **+0.0163** |
| **Status** | 🏆 **NEW STATE-OF-THE-ART** |

## Per-Seed Scores

| Seed | AUPRC |
|------|-------|
| 0 | 0.9141 |
| 1 | 0.9150 |
| 2 | 0.9140 |
| 3 | 0.9155 |
| 4 | 0.9159 |
| **Mean ± std** | **0.9149 ± 0.0007** |
| **Ensemble of predictions** | **0.9163** |

Exceptionally stable across seeds (σ = 0.0007). All 5 individual seeds exceed 0.914, a full 1.4+ points above SOTA.

## Method

Same as CYP2D6 Veith submission. See [`cyp2d6_veith.md`](cyp2d6_veith.md) or [`README.md`](README.md) for full method details.

## Dataset

- **Source:** Veith et al. 2009
- **Train/Val:** 9,861 compounds
- **Test:** 2,467 compounds
- **Positive rate (test):** ~44%

## Reproduction

```bash
python run_benchmark.py --endpoints cyp3a4_veith --seeds 5
```

## Clinical Significance

CYP3A4 is the most abundant cytochrome P450 enzyme in human liver and intestine, metabolizing the majority of marketed drugs. It is the primary enzyme responsible for first-pass metabolism of orally administered drugs. CYP3A4 inhibition is a major source of clinically significant drug-drug interactions, including interactions with HIV protease inhibitors, statins, calcium channel blockers, and immunosuppressants.

A 1.6-point improvement in CYP3A4 inhibitor prediction directly translates to improved DDI risk assessment during lead optimization and earlier identification of metabolism liabilities.

## Code and Data Availability

- **Predictions file:** `results/predictions/cyp3a4_veith.npy` (shape: 5 × 2467)
- **Trained models:** `results/models/cyp3a4_veith_seed_{0..4}.cbm`
