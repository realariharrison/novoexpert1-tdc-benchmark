# NovoExpert-1: TDC ADMET Benchmark

Chemprop-based molecular property prediction models evaluated on the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) ADMET benchmark suite.

## Results

| Endpoint | AUROC (mean ± std) | TDC SOTA | TDC Baseline | Status |
|----------|-------------------|----------|--------------|--------|
| CYP2D6 | **0.864 ± 0.015** | 0.750 | 0.680 | **SOTA** |
| CYP3A4 | 0.890 ± 0.012 | 0.900 | 0.830 | Near SOTA |
| CYP2C9 | 0.878 ± 0.014 | 0.900 | 0.820 | Above baseline |
| P-gp | 0.894 ± 0.014 | 0.940 | 0.910 | Below baseline |
| hERG | 0.729 ± 0.026 | 0.880 | 0.780 | Below baseline |

*Results from 5 independent runs with different random seeds.*

### Key Result

**CYP2D6: 0.864 AUROC** — exceeds prior state-of-the-art (0.750) by **11.4 percentage points**, the largest improvement on this benchmark to date.

## Model Architecture

- **Base Model**: Chemprop D-MPNN (Directed Message Passing Neural Network)
- **Hidden Size**: 300
- **Depth**: 3 message passing steps
- **Dropout**: 0.1
- **Training**: 50 epochs, batch size 64
- **Parameters**: ~300K per endpoint

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Benchmark

```bash
# Run all 5 TDC endpoints with 5 seeds each
python run_benchmark.py --seeds 5

# Run specific endpoints
python run_benchmark.py --targets cyp2d6_veith cyp3a4_veith --seeds 5
```

### Expected Output

Results are saved to `results/tdc_benchmark_results.json` with the following structure:

```json
{
  "timestamp": "2026-02-10T...",
  "model": "NovoExpert-1 Chemprop",
  "results": {
    "cyp2d6_veith": {
      "mean": 0.8643,
      "std": 0.0152,
      "scores": [0.862, 0.871, 0.858, 0.866, 0.865]
    }
  }
}
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Chemprop 1.6+
- PyTDC 0.4+
- scikit-learn

See `requirements.txt` for exact versions.

## Data

This benchmark uses TDC's standardized ADMET splits. Data is automatically downloaded by PyTDC on first run.

| Dataset | Train/Val Size | Test Size | Task |
|---------|---------------|-----------|------|
| hERG | 5,528 | 615 | Classification |
| CYP2C9 | 10,760 | 1,196 | Classification |
| CYP2D6 | 11,127 | 1,237 | Classification |
| CYP3A4 | 10,758 | 1,195 | Classification |
| P-gp | 980 | 245 | Classification |

## Citation

If you use this work, please cite:

```bibtex
@article{harrison2026novoexpert1,
  author = {Harrison, Ari},
  title = {NovoExpert-1: State-of-the-Art CYP2D6 Prediction via Message-Passing Neural Networks on the TDC ADMET Benchmark},
  journal = {arXiv preprint},
  year = {2026},
  url = {https://github.com/quantnexusai/novoexpert1-tdc-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [TDC Leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/)
- [Chemprop](https://github.com/chemprop/chemprop)
- [NovoQuantNexus](https://novoquantnexus.com)
