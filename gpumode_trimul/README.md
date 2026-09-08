# GPUMode TriMul Benchmark (NVIDIA H100)

Optimized kernel implementation for the **GPUMode Triangular Matrix Multiplication (TriMul) Benchmark**, evaluated on **NVIDIA H100 80GB HBM3** (PyTorch 2.7.1, Triton 3.3.1, CUDA 12.6).

The task targets the outgoing Triangle Multiplicative Update used in AlphaFold-class protein-structure models, minimizing geometric-mean latency across seven fixed input shapes while passing 18/18 numerical correctness tests.

## Performance Summary

Ranking metric is the **geometric-mean latency across the 7 benchmark shapes in microseconds (μs)**, lower is better.

The headline score represents the paired average across two complete benchmark runs (Run 1: 1,036.66 μs, Run 2: 1,035.45 μs; mean = **1,036.05 μs**, or **1,036.1 μs**).

| Solution | Authority / Reference | Correctness | Geomean Latency (μs) | Relative to Stashuk-Olek |
|---|---|:---:|:---:|:---:|
| K-Search | Separately reported (arXiv:2602.19128) | Passed | 1,030.0 μs | 3.28% lower latency (+3.39% speedup) |
| **Discovered Solution** (`submission.py`) | **Same-machine paired evaluation (2-run mean)** | **18/18 Passed** | **1,036.1 μs** | **2.71% lower latency** (+2.79% speedup) |
| stashuk-olek | Strongest human baseline (same-machine paired) | 18/18 Passed | 1,064.9 μs | Reference |
| shiyegao CUDA | Separately reported | Passed | 1,074.0 μs | +0.85% latency |
| Zeyu Shen Triton | Separately reported | Passed | 1,140.0 μs | +7.05% latency |
| TTT-Discover (Stanford/NVIDIA/Together) | Separately reported (arXiv:2601.16175) | Passed | 1,161.0 μs | +9.02% latency |

*Note: Public values are drawn from their respective reports or evaluation records. Our 1,036.1 μs result is measured locally using the official evaluator.*

### Per-Shape Latency Breakdown (μs)

| Case | Workload Specification | Run 1 (μs) | Run 2 (μs) | Mean (μs) |
|:---:|---|:---:|:---:|:---:|
| 1 | `seqlen: 256; bs: 2; dim: 128; hiddendim: 128; nomask: True; normal` | 338.55 | 338.96 | 338.75 |
| 2 | `seqlen: 768; bs: 1; dim: 128; hiddendim: 128; nomask: True; cauchy` | 1120.15 | 1119.62 | 1119.88 |
| 3 | `seqlen: 256; bs: 2; dim: 384; hiddendim: 128; nomask: False; normal` | 489.48 | 489.31 | 489.39 |
| 4 | `seqlen: 512; bs: 1; dim: 128; hiddendim: 128; nomask: True; normal` | 550.15 | 547.06 | 548.60 |
| 5 | `seqlen: 1024; bs: 1; dim: 128; hiddendim: 128; nomask: True; cauchy` | 1979.66 | 1982.32 | 1980.99 |
| 6 | `seqlen: 768; bs: 1; dim: 384; hiddendim: 128; nomask: False; normal` | 1898.92 | 1889.45 | 1894.19 |
| 7 | `seqlen: 1024; bs: 1; dim: 384; hiddendim: 128; nomask: True; normal` | 3351.48 | 3354.08 | 3352.78 |
| **Geomean** | **Across all 7 shapes** | **1,036.66** | **1,035.45** | **1,036.05** |

## Core Architectural Optimizations

1. **Re-localizing the Bottleneck Across BMM → LayerNorm**:
   - High-performance TriMul implementations typically saturate producer-side fusion, facing diminishing returns from register pressure.
   - For multiple workload shapes, the search identifies the BMM → LayerNorm boundary as an effective remaining optimization opportunity.
   - The discovered implementation loads BMM outputs along the contiguous physical dimension (`[hidden, row]`) and performs the required transpose directly in registers before LayerNorm.
2. **Deep Epilogue Fusion**:
   - Fuses post-LN with output gate calculation and the final projection GEMM.
3. **Shape-Dependent Adaptive Dispatch**:
   - Dimension-specialized constexpr schedules tailored per workload shape.

## External Baselines & Evaluation Protocol

### External Baselines & Published Works

| Implementation | Authors / Source | Date / Version | Canonical Link / Citation | Geomean Latency (μs) | Latency vs Stashuk-Olek |
|---|---|---|---|:---:|:---:|
| **K-Search** | External AI Research | February 2026 | [arXiv:2602.19128](https://arxiv.org/abs/2602.19128) | 1,030.0 μs | -3.28% |
| **Discovered Solution** (`submission.py`) | Apex Intelligence | March 2026 | This repository (2-run mean) | **1,036.1 μs** | **-2.71%** |
| **stashuk-olek** | Olek Stashuk | November 2024 | [GPUMode Popcorn Leaderboard](https://github.com/gpu-mode) | 1,064.9 μs | Reference (0.00%) |
| **shiyegao CUDA** | Shiye Gao | December 2024 | [GPUMode Popcorn Leaderboard](https://github.com/gpu-mode) | 1,074.0 μs | +0.85% |
| **Zeyu Shen Triton** | Zeyu Shen | December 2024 | [GPUMode Popcorn Leaderboard](https://github.com/gpu-mode) | 1,140.0 μs | +7.05% |
| **TTT-Discover** | Stanford / NVIDIA / Together AI | January 2026 | [arXiv:2601.16175](https://arxiv.org/abs/2601.16175) | 1,161.0 μs | +9.02% |

### Evaluation Protocol & Verification Contract

- **Task Specification**: Outgoing Triangle Multiplicative Update (AlphaFold-style pair update) minimizing geometric-mean latency across 7 fixed benchmark shapes.
- **Hardware & Software**: NVIDIA H100 80GB HBM3 (SM90), Python 3.10+, PyTorch 2.7.1, Triton 3.3.1, CUDA 12.6.
- **Harness Integrity**: Evaluated via the canonical GPUMode Popcorn harness (`task.py`, `eval.py`, `reference.py`, `utils.py`, `task.yml`), pinned and verified against SHA-256 manifest `cd61e8501c113889060db3eefab9802cffb33cad309682b01e2dad30af9708f2`.
- **Correctness Gate**: Must pass all 18 randomized correctness test cases against deterministic FP32 PyTorch reference (with TF32 disabled) before benchmark timing.
- **Timing Protocol**: Warmup cycles followed by repeated timed trials measured via PyTorch CUDA events. Multi-pass evaluation (`--runs 2`) averages two complete 7-shape passes (1,036.66 μs and 1,035.45 μs → 1,036.05 μs) to account for thermal variance.

## Repository Contents

- `submission.py` — Complete standalone Triton submission implementing `custom_kernel(data: input_t) -> output_t`.
- `evaluate.py` — Benchmark execution script running correctness tests and benchmark timing (supports `--runs 2` to reproduce the two-run headline average).
- `canonical/` — Official GPUMode Popcorn harness containing test specifications (`tests.txt`, `benchmarks.txt`), reference implementations (`reference.py`), and scoring routines (`scorer.py`).
- `results/benchmark_results.json` — Detailed timing metrics, per-run shape breakdowns, and verified numbers.

## How to Reproduce

Requirements: NVIDIA H100 GPU, Python >= 3.10, PyTorch 2.7.1, Triton 3.3.1, CUDA 12.6.

Run complete evaluation (18 correctness tests + 7 benchmark shapes across 2 runs):
```bash
python evaluate.py --runs 2
```

Save results to JSON:
```bash
python evaluate.py --runs 2 --output results/my_run.json
```
