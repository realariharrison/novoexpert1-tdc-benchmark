# Archive: NovoExpert-1

This directory contains the original NovoExpert-1 files, retained for historical reference and for the metric correction disclosure in NovoExpert-2.

## Files

- `run_benchmark_v1.py` — Original Chemprop D-MPNN training script (uses AUROC for all endpoints)
- `requirements_v1.txt` — v1 dependencies (Chemprop 1.6, PyTorch 2.0, PyTDC 0.4)
- `novoexpert1_tdc.tex` — Original v1 paper LaTeX source
- `NovoExpert_1_paper.pdf` — Published v1 preprint

## Important: Metric Correction

NovoExpert-1 reported a CYP2D6 result of **0.864 AUROC** and claimed a +0.114 improvement over the TDC leaderboard SOTA of 0.750. This comparison was based on a metric mismatch: the TDC leaderboard uses **AUPRC** for CYP Veith endpoints, not AUROC.

Under the correct metric, the NovoExpert-1 Chemprop model's CYP2D6 AUPRC is approximately 0.64, which does NOT beat the 0.750 SOTA.

NovoExpert-2 (at the repo root) corrects this by using the proper per-endpoint metric throughout. The NovoExpert-2 CatBoost+MapLight+GIN pipeline still achieves CYP2D6 SOTA with **0.778 AUPRC** — a genuine improvement over the TDC leaderboard.

See `paper/novoexpert_tdc.tex` (NovoExpert-2) Section 7 for the full correction disclosure.

## Why Keep These Files?

1. **Transparency**: The v1 preprint exists publicly and the metric correction needs to be discoverable.
2. **Reproducibility**: Readers who access the v1 preprint should be able to reproduce its AUROC measurements.
3. **Historical record**: The D-MPNN model and its per-seed results remain valid AUROC measurements, just not comparable to the AUPRC leaderboard.
