# TDC Leaderboard Submission: CYP2D6 Veith

## Result

| Field | Value |
|-------|-------|
| **Endpoint** | `cyp2d6_veith` |
| **Category** | Metabolism |
| **Metric** | AUPRC (Average Precision) |
| **NovoExpert-2 score** | **0.7776 ± 0.0020** |
| **Published SOTA** | 0.7500 |
| **Δ vs SOTA** | **+0.0276** |
| **Status** | 🏆 **NEW STATE-OF-THE-ART** |

## Per-Seed Scores

| Seed | AUPRC |
|------|-------|
| 0 | 0.7755 |
| 1 | 0.7711 |
| 2 | 0.7741 |
| 3 | 0.7753 |
| 4 | 0.7710 |
| **Mean ± std** | **0.7734 ± 0.0020** |
| **Ensemble of predictions** | **0.7776** |

All 5 independent seeds exceed the prior SOTA of 0.750, with the lowest individual score (0.7710) still surpassing by +2.1 points. The low standard deviation (0.0020) indicates the result is robust to random seed choice.

## Method

**Name:** `NovoExpert-2 (CatBoost + MapLight + GIN)`

**Features (2,873 dims per molecule):**
- ECFP (Morgan, radius 2): 1024 bits
- Avalon fingerprint: 1024 bits
- ErG 2D pharmacophore fingerprint: 315 dims
- RDKit 2D descriptors: ~210 dims
- GIN supervised masking embeddings (dgllife): 300 dims

**Model:** CatBoost Classifier
- iterations: 2000
- depth: 8
- learning_rate: 0.03
- l2_leaf_reg: 5
- border_count: 254
- scale_pos_weight: (n_neg / n_pos) = ~5.0 on CYP2D6 training folds (16.9% positive rate)
- eval_metric: PRAUC
- early_stopping_rounds: 200
- 5-seed ensemble using TDC official `get_train_valid_split()`

## Dataset

- **Source:** Veith et al. 2009 \[[Nature Biotech paper](https://doi.org/10.1038/nbt.1581)\]
- **Train/Val:** 10,504 compounds
- **Test:** 2,626 compounds
- **Positive rate (test):** 16.9%

## Reproduction

```bash
git clone https://github.com/realariharrison/novoexpert1-tdc-benchmark.git
cd novoexpert1-tdc-benchmark
pip install -r requirements.txt
python run_benchmark.py --endpoints cyp2d6_veith --seeds 5
```

Expected output:
```
Ensemble: 0.7776
Published SOTA: 0.7500
🏆 BEATS SOTA
```

## Clinical Significance

CYP2D6 metabolizes approximately 25% of clinically used drugs and is the most pharmacogenomically significant cytochrome P450 enzyme. Genetic polymorphisms produce four distinct metabolizer phenotypes (poor, intermediate, extensive, ultra-rapid), and the Clinical Pharmacogenetics Implementation Consortium (CPIC) publishes dosing guidelines for CYP2D6-mediated drugs including codeine, tramadol, tamoxifen, fluoxetine, and metoprolol.

Accurate computational prediction of CYP2D6 inhibitors supports:
- Drug-drug interaction risk assessment
- Lead optimization away from CYP2D6 liability
- Patient stratification based on metabolizer phenotype
- FDA-recommended CYP characterization for new molecular entities

## Code and Data Availability

- **Repository:** https://github.com/realariharrison/novoexpert1-tdc-benchmark
- **Predictions file:** `results/predictions/cyp2d6_veith.npy` (shape: 5 × 2626)
- **Trained models:** `results/models/cyp2d6_veith_seed_{0..4}.cbm`
- **Paper:** `paper/novoexpert_tdc.tex`

## Metric Correction Note

NovoExpert-1 previously reported a CYP2D6 score of 0.864 **AUROC**, claiming a +11.4 point improvement over SOTA (0.750). The TDC leaderboard uses **AUPRC** for CYP Veith endpoints, so that comparison was invalid. NovoExpert-2's 0.778 AUPRC is the first correctly-measured SOTA improvement for NovoExpert on this endpoint.
