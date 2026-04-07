# NovoExpert-2: TDC ADMET Benchmark Tracker

**Last updated:** 2026-04-04 20:20 UTC
**Method:** CatBoost + MapLight (2573-dim) + GIN embeddings (300-dim), 5-seed ensemble

## Summary

- **Total endpoints:** 22
- **Beats SOTA:** 4 🏆
- **Below SOTA:** 18

## Results

| # | Endpoint | Task | Metric | NovoExpert-2 | Published SOTA | Δ | Status | Leaderboard | Deploy to 122M |
|---|----------|------|--------|--------------|----------------|---|--------|-------------|----------------|
| 1 | `bioavailability_ma` | cla | auroc | **0.6508 ± 0.0353** | 0.7480 | -0.0972 | Below | — | Keep existing |
| 2 | `caco2_wang` | reg | mae | **0.3070 ± 0.0069** | 0.2760 | -0.0310 | Below | — | Keep existing |
| 3 | `hia_hou` | cla | auroc | **0.9724 ± 0.0325** | 0.9890 | -0.0166 | Below | — | Keep existing |
| 4 | `lipophilicity_astrazeneca` | reg | mae | **0.5019 ± 0.0036** | 0.4670 | -0.0349 | Below | — | Keep existing |
| 5 | `pgp_broccatelli` | cla | auroc | **0.9253 ± 0.0031** | 0.9400 | -0.0147 | Below | — | Keep existing |
| 6 | `solubility_aqsoldb` | reg | mae | **0.8045 ± 0.0115** | 0.7610 | -0.0435 | Below | — | Keep existing |
| 7 | `bbb_martins` | cla | auroc | **0.9107 ± 0.0024** | 0.9160 | -0.0053 | Below | — | Keep existing |
| 8 | `ppbr_az` | reg | mae | **7.5700 ± 0.0745** | 7.5260 | -0.0440 | Below | — | Keep existing |
| 9 | `vdss_lombardo` | reg | spearman | **0.5557 ± 0.0154** | 0.7130 | -0.1573 | Below | — | Keep existing |
| 10 | `clearance_hepatocyte_az` | reg | spearman | **0.5112 ± 0.0076** | 0.4870 | +0.0242 | 🏆 BEATS | ✅ Submit | ✅ Yes |
| 11 | `clearance_microsome_az` | reg | spearman | **0.6196 ± 0.0082** | 0.6300 | -0.0104 | Below | — | Keep existing |
| 12 | `half_life_obach` | reg | spearman | **0.4268 ± 0.0351** | 0.5620 | -0.1352 | Below | — | Keep existing |
| 13 | `cyp2c9_substrate_carbonmangels` | cla | auprc | **0.3749 ± 0.0308** | 0.4410 | -0.0661 | Below | — | Keep existing |
| 14 | `cyp2c9_veith` | cla | auprc | **0.8612 ± 0.0014** | 0.9000 | -0.0388 | Below | — | Keep existing |
| 15 | `cyp2d6_substrate_carbonmangels` | cla | auprc | **0.7113 ± 0.0125** | 0.7360 | -0.0247 | Below | — | Keep existing |
| 16 | `cyp2d6_veith` | cla | auprc | **0.7776 ± 0.0020** | 0.7500 | +0.0276 | 🏆 BEATS | ✅ Submit | ✅ Yes |
| 17 | `cyp3a4_substrate_carbonmangels` | cla | auroc | **0.6600 ± 0.0121** | 0.6560 | +0.0040 | 🏆 BEATS | ✅ Submit | ✅ Yes |
| 18 | `cyp3a4_veith` | cla | auprc | **0.9163 ± 0.0007** | 0.9000 | +0.0163 | 🏆 BEATS | ✅ Submit | ✅ Yes |
| 19 | `ames` | cla | auroc | **0.8602 ± 0.0012** | 0.8710 | -0.0108 | Below | — | Keep existing |
| 20 | `dili` | cla | auroc | **0.9052 ± 0.0280** | 0.9160 | -0.0108 | Below | — | Keep existing |
| 21 | `herg` | cla | auroc | **0.8520 ± 0.0070** | 0.8800 | -0.0280 | Below | — | Keep existing |
| 22 | `ld50_zhu` | reg | mae | **0.6180 ± 0.0135** | 0.5730 | -0.0450 | Below | — | Keep existing |

## Action Items

### Leaderboard Submissions

Endpoints that beat published SOTA and should be submitted to the TDC leaderboard:

- **clearance_hepatocyte_az**: 0.5112 (published: 0.4870, Δ +0.0242)
- **cyp2d6_veith**: 0.7776 (published: 0.7500, Δ +0.0276)
- **cyp3a4_substrate_carbonmangels**: 0.6600 (published: 0.6560, Δ +0.0040)
- **cyp3a4_veith**: 0.9163 (published: 0.9000, Δ +0.0163)

### 122M Database Re-Computation

For each endpoint where NovoExpert-2 beats SOTA, the corresponding column in the
Cosmos DB molecules container needs to be re-computed using the new model. The CatBoost
`.cbm` files are saved in `models/tdc_sota/predictions/{endpoint}.npy`.

Columns to re-compute:

- `clearance_hepatocyte_az_prediction` — use seed-ensemble of 5 CatBoost models
- `cyp2d6_veith_prediction` — use seed-ensemble of 5 CatBoost models
- `cyp3a4_substrate_carbonmangels_prediction` — use seed-ensemble of 5 CatBoost models
- `cyp3a4_veith_prediction` — use seed-ensemble of 5 CatBoost models

## Method Details

**Features (2873 dims per molecule):**
- ECFP (Morgan, radius 2): 1024 bits
- Avalon fingerprint: 1024 bits
- ErG fingerprint: 315 dims
- RDKit 2D descriptors: ~210 dims
- GIN supervised masking embeddings (dgllife): 300 dims

**Model:** CatBoost
- iterations=2000, depth=8, learning_rate=0.03, l2_leaf_reg=5
- border_count=254, early_stopping_rounds=200
- Classification: scale_pos_weight=(n_neg/n_pos), PRAUC/AUC eval metric
- Regression: MAE/RMSE eval metric

**Evaluation:** TDC official benchmark splits, 5-seed ensemble average

**Training infrastructure:** Azure Container Apps (gpu-a100 profile, CPU-only for GIN inference)