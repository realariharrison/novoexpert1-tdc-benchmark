# TDC Leaderboard Submission: Clearance Hepatocyte AZ

## Result

| Field | Value |
|-------|-------|
| **Endpoint** | `clearance_hepatocyte_az` |
| **Category** | Excretion |
| **Metric** | Spearman rank correlation |
| **NovoExpert-2 score** | **0.5112 (ensemble), 0.5068 ± 0.0076 (per-seed mean)** |
| **Published SOTA** | 0.4870 |
| **Δ vs SOTA** | **+0.0242** |
| **Status** | 🏆 **NEW STATE-OF-THE-ART** |

## Per-Seed Scores

| Seed | Spearman |
|------|----------|
| 0 | 0.5147 |
| 1 | 0.5036 |
| 2 | 0.5171 |
| 3 | 0.4992 |
| 4 | 0.4994 |
| **Mean ± std** | **0.5068 ± 0.0076** |
| **Ensemble of predictions** | **0.5112** |

All 5 seeds exceed 0.499, and the ensemble exceeds the published SOTA of 0.487 by +0.024.

## Method

CatBoost Regressor on MapLight+GIN features. Same hyperparameters as classification endpoints except:
- `eval_metric`: RMSE
- No `scale_pos_weight` (regression)

See [`README.md`](README.md) for full method.

## Dataset

- **Source:** AstraZeneca internal clearance dataset (public release)
- **Train/Val:** 816 compounds
- **Test:** 204 compounds
- **Target:** In vitro hepatocyte clearance (log-transformed)

## Reproduction

```bash
python run_benchmark.py --endpoints clearance_hepatocyte_az --seeds 5
```

## Clinical Significance

Hepatocyte clearance is a key pharmacokinetic parameter driving drug half-life, dosing frequency, and systemic exposure. In vitro clearance measurements in primary hepatocytes are the gold standard for predicting in vivo clearance but are expensive and low-throughput. Accurate computational prediction of hepatocyte clearance enables earlier prioritization of candidates with favorable pharmacokinetics and reduces the number of compounds requiring experimental clearance assessment.

## Code and Data Availability

- **Predictions file:** `results/predictions/clearance_hepatocyte_az.npy` (shape: 5 × 204)
- **Trained models:** `results/models/clearance_hepatocyte_az_seed_{0..4}.cbm`
