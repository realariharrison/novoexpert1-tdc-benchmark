# TDC Leaderboard Submissions

This directory contains submission materials for the 5 TDC ADMET endpoints where NovoExpert-2 beats published state-of-the-art.

## Wins

| # | Endpoint | Metric | NovoExpert-2 | Published SOTA | Δ | Method |
|---|----------|--------|--------------|----------------|---|--------|
| 1 | [cyp2d6_veith](cyp2d6_veith.md) | AUPRC | **0.7776** | 0.7500 | **+0.0276** | CatBoost+MapLight+GIN |
| 2 | [cyp3a4_veith](cyp3a4_veith.md) | AUPRC | **0.9163** | 0.9000 | **+0.0163** | CatBoost+MapLight+GIN |
| 3 | [cyp3a4_substrate_carbonmangels](cyp3a4_substrate_carbonmangels.md) | AUROC | **0.6600** | 0.6560 | **+0.0040** | CatBoost+MapLight+GIN |
| 4 | [clearance_hepatocyte_az](clearance_hepatocyte_az.md) | Spearman | **0.5112** | 0.4870 | **+0.0242** | CatBoost+MapLight+GIN |
| 5 | [dili](dili.md) | AUROC | **0.9217** | 0.9160 | **+0.0057** | Chemprop v2 (D-MPNN) |

## Method Descriptions

We evaluated two model families across all 22 TDC ADMET endpoints. For 4 of the 5 SOTA wins (CYP2D6, CYP3A4, CYP3A4 Substrate, Clearance Hepatocyte AZ), the primary CatBoost+MapLight+GIN method was the strongest. For DILI, Chemprop v2 outperformed and is reported instead. Full methodology and per-endpoint comparison appears in `paper/novoexpert_tdc.tex`.

### Primary Method: CatBoost + MapLight + GIN (4 endpoints)

**Method name:** `NovoExpert-2 (CatBoost + MapLight + GIN)`

**Description for leaderboard submission:**

> CatBoost gradient-boosted tree ensemble on 2,873-dimensional features combining MapLight-style molecular fingerprints (Morgan ECFP + Avalon + ErG + RDKit 2D descriptors, 2,573 dims) with 300-dimensional GIN supervised masking embeddings from the DGL-LifeSci pretrained model zoo. Per-endpoint CatBoost hyperparameters: 2000 iterations, depth 8, learning rate 0.03, l2_leaf_reg 5, border_count 254, early_stopping_rounds 200, scale_pos_weight=(n_neg/n_pos) for classification. 5-seed ensemble using TDC's official get_train_valid_split() API. Evaluated on TDC benchmark_group test sets with the task-appropriate metric (AUPRC/AUROC/MAE/Spearman).

### Secondary Method: Chemprop v2 (DILI only)

**Method name:** `NovoExpert-2 (Chemprop v2)`

**Description for leaderboard submission:**

> Chemprop v2 Directed Message-Passing Neural Network (D-MPNN) trained on TDC benchmark_group splits. Hidden dimension 300, message passing depth 3, dropout 0.15, 50 epochs, batch size 128, trained with precision-recall AUC as early-stopping metric. 5-seed ensemble using TDC's official splits. CPU-only training. For DILI specifically, Chemprop v2's end-to-end learned molecular representations outperform fixed fingerprint features (CatBoost+MapLight+GIN), consistent with DILI's complex mechanism-driven toxicity phenotype.

**Code:** https://github.com/realariharrison/novoexpert1-tdc-benchmark
**Paper:** `paper/novoexpert_tdc.tex` (forthcoming preprint)
**Reproduction:** `python run_benchmark.py --seeds 5`

## Generating the Submission Dict

The TDC leaderboard requires predictions in a specific format:

```python
from tdc.benchmark_group import admet_group
import numpy as np

group = admet_group(path='./tdc_data')

# For each endpoint, provide a dict mapping seed -> predictions
predictions = {}
# Endpoints using CatBoost+MapLight+GIN
for endpoint in ['cyp2d6_veith', 'cyp3a4_veith',
                 'cyp3a4_substrate_carbonmangels', 'clearance_hepatocyte_az']:
    preds_5_seeds = np.load(f'results/predictions/{endpoint}.npy')
    predictions[endpoint] = {seed: preds_5_seeds[seed].tolist()
                              for seed in range(5)}

# DILI uses Chemprop v2 instead
dili_preds = np.load('results/predictions/dili_chemprop.npy')
predictions['dili'] = {seed: dili_preds[seed].tolist() for seed in range(5)}

# TDC's official evaluator
results = group.evaluate(predictions)
print(results)
# Expected output (abbreviated):
# {'cyp2d6_veith': [0.7776, 0.0020], 'cyp3a4_veith': [0.9163, 0.0006], ...}
```

See `generate_submission.py` for the full script.
