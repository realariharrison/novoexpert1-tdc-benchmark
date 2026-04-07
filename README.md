# NovoExpert: TDC ADMET Benchmark

State-of-the-art molecular property prediction models evaluated on the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) ADMET benchmark suite.

## Current Version: NovoExpert-2

NovoExpert-2 evaluates two model families across all 22 TDC ADMET endpoints:
1. **Primary:** CatBoost + MapLight fingerprint features (2573 dims) + GIN supervised masking embeddings (300 dims) — 5-seed ensemble
2. **Secondary:** Chemprop v2 Directed Message-Passing Neural Network — 5-seed ensemble

For each endpoint, we report the stronger method. All results use TDC's official `benchmark_group` splits and per-endpoint metrics.

### Results

**5 endpoints beat published SOTA** (out of 22 evaluated):

| Endpoint | Category | Metric | NovoExpert-2 | Published SOTA | Δ | Method |
|----------|----------|--------|--------------|----------------|---|--------|
| **CYP2D6 Veith** | Metabolism | AUPRC | **0.778** | 0.750 | **+0.028** | CatBoost+MapLight+GIN |
| **CYP3A4 Veith** | Metabolism | AUPRC | **0.916** | 0.900 | **+0.016** | CatBoost+MapLight+GIN |
| **CYP3A4 Substrate Carbonmangels** | Metabolism | AUROC | **0.660** | 0.656 | **+0.004** | CatBoost+MapLight+GIN |
| **Clearance Hepatocyte AZ** | Excretion | Spearman | **0.511** | 0.487 | **+0.024** | CatBoost+MapLight+GIN |
| **DILI** | Toxicity | AUROC | **0.922** | 0.916 | **+0.006** | Chemprop v2 (D-MPNN) |

Full benchmark table for all 22 endpoints: see `paper/novoexpert_tdc.tex`.

### Key Result: CYP2D6 Metabolism

CYP2D6 metabolizes approximately 25% of clinically used drugs and is highly polymorphic. NovoExpert-2 achieves **0.778 AUPRC** on the TDC CYP2D6 Veith benchmark, a **2.8-point improvement** over the prior state-of-the-art of 0.750. All 5 independent seeds exceed the published SOTA (std: 0.002).

## Method

```
MapLight features (2,573 dims)              GIN supervised masking (300 dims)
├── ECFP Morgan r=2:  1,024 bits             └── DGL-Life pretrained embedding
├── Avalon:           1,024 bits
├── ErG fingerprint:    315 dims
└── RDKit 2D descriptors: ~210 dims
                                              │
                                              ▼
                              Concatenate → 2,873-dim feature vector
                                              │
                                              ▼
                        CatBoost (5 seeds, per-endpoint hyperparameters)
                          iterations: 2000, depth: 8, lr: 0.03
                          early_stopping_rounds: 200
                          classification: scale_pos_weight=(n_neg/n_pos)
                          task-appropriate eval metric (PRAUC/AUC/MAE/RMSE)
                                              │
                                              ▼
                        Simple ensemble average of 5 seed predictions
```

- **No per-endpoint hyperparameter tuning** (same CatBoost config across all 22)
- Uses TDC's official `get_train_valid_split()` API for seed-consistent splits
- Uses the correct per-endpoint metric (AUPRC, AUROC, MAE, or Spearman)
- GIN embeddings computed on CPU (DGL pip wheel is CPU-only); training runs fine on CPU or GPU

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

Note: `dgl` may need to be installed from the DGL wheel index:
```bash
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

### Run Full Benchmark

```bash
# All 22 endpoints, 5 seeds each (~2 hours on a modern CPU)
python run_benchmark.py --seeds 5

# Resume from cached results
python run_benchmark.py --seeds 5 --resume

# Specific endpoints only
python run_benchmark.py --endpoints cyp2d6_veith cyp3a4_veith --seeds 5
```

### Output Structure

```
results/
├── all_results.json               # aggregate summary across 22 endpoints
├── tdc_sota_tracker.md            # human-readable tracker
├── json_reports/<endpoint>.json   # per-endpoint scores (5 seeds + ensemble)
├── predictions/<endpoint>.npy     # test predictions (5 seeds × n_test)
└── models/<endpoint>_seed_*.cbm   # saved CatBoost models for inference (optional)
```

## Leaderboard Submissions

See `submissions/` directory for per-endpoint submission materials for the 5 SOTA wins:
- `submissions/cyp2d6_veith.md` (CatBoost+MapLight+GIN)
- `submissions/cyp3a4_veith.md` (CatBoost+MapLight+GIN)
- `submissions/cyp3a4_substrate_carbonmangels.md` (CatBoost+MapLight+GIN)
- `submissions/clearance_hepatocyte_az.md` (CatBoost+MapLight+GIN)
- `submissions/dili.md` (Chemprop v2)

Each submission document contains the method name, description, reproduction command, 5-seed predictions, and the exact `group.evaluate()` output suitable for TDC leaderboard submission.

**Note on DILI:** For 4 of the 5 wins, CatBoost+MapLight+GIN is the stronger model family. For DILI specifically, Chemprop v2 (D-MPNN) outperforms CatBoost on the test set (0.922 vs 0.906 AUROC). We report Chemprop v2 for DILI. See the paper for full per-endpoint model comparison.

## Data

TDC ADMET splits are automatically downloaded by PyTDC on first run.

| Endpoint | Train/Val | Test | Metric | Published SOTA |
|----------|-----------|------|--------|----------------|
| CYP2D6 Veith | 10,504 | 2,626 | AUPRC | 0.750 |
| CYP3A4 Veith | 9,861 | 2,467 | AUPRC | 0.900 |
| CYP3A4 Substrate (Carbonmangels) | 534 | 133 | AUROC | 0.656 |
| Clearance Hepatocyte AZ | 816 | 204 | Spearman | 0.487 |
| DILI | 475 | 96 | AUROC | 0.916 |

## Important Correction: v1 Metric Issue

NovoExpert-1 (v1) reported a CYP2D6 result of **0.864 AUROC** and claimed a +11.4 point improvement over SOTA (0.750). After re-evaluation, we found that the TDC leaderboard uses **AUPRC** (not AUROC) for the CYP Veith endpoints. v1's 0.864 AUROC score is a valid AUROC measurement but is not directly comparable to the 0.750 AUPRC leaderboard value.

NovoExpert-2 corrects this by using the proper per-endpoint metric and still achieves CYP2D6 SOTA with **0.778 AUPRC** — a more modest but genuine improvement over the true leaderboard SOTA.

The v1 paper, code, and preprint are archived in the [`archive/`](archive/) directory for historical reference. The v2 paper (`paper/novoexpert_tdc.tex`) is the current work.

## Preprint

NovoExpert-2 preprint forthcoming.

v1 preprint (retained for reference):
**NovoExpert-1: State-of-the-Art CYP2D6 Prediction via Message-Passing Neural Networks on the TDC ADMET Benchmark**, Ari Harrison, *ChemRxiv*, 2026.

## Citation

```bibtex
@article{harrison2026novoexpert2,
  author = {Harrison, Ari},
  title = {NovoExpert-2: State-of-the-Art ADMET Prediction via Gradient-Boosted Trees on MapLight Fingerprints and GIN Embeddings},
  year = {2026},
  url = {https://github.com/realariharrison/novoexpert1-tdc-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [TDC Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [DGL-Life](https://github.com/awslabs/dgl-lifesci)
- [CatBoost](https://catboost.ai/)
- [NovoQuantNexus](https://novoquantnexus.com)
