# Apex-Tune: Discovered Improvements Across the AI Training Stack

Accompanying release of research outcomes discovered and validated by **Apex Intelligence’s automated AI research system**.

This repository collects the model training scripts, scaling laws, and GPU kernel implementations discovered by our automated research system across four frontier benchmarks in LLM training, scaling prediction, and GPU kernel engineering.

In its first evaluation, our system achieved **three new official SOTAs** on SimpleTES, GPUMode TriMul (H100), and MLS-Bench Fused Causal Attention, with its TriMul result surpassing TTT-Discover from a team spanning Stanford, NVIDIA, and Together AI. On NanoChat Autoresearch, its B200 result outperformed published results from both Recursive SuperIntelligence and Tencent Hunyuan’s Hyra, placing it near the public SOTA held by AutoTrust.

---

## Benchmark Highlights

| Benchmark | Domain / Hardware | Discovered Result | Previous SOTA / Reference | Relative Gain / Comparison |
|---|---|:---:|:---:|:---:|
| **NanoChat Autoresearch** | Fixed-Budget LLM Training<br>(Single B200, 300 s) | **0.892792 val_bpb** (10-seed mean)<br>**0.892426 val_bpb** (3-seed initial)<br>**0.891762 val_bpb** (Best single seed) | AutoTrust ScienceGuru: `0.889522`<br>Tencent Hunyuan Hyra: `0.901543`<br>Recursive: `0.910875` (mean), `0.903891` (best) | **-0.0181 vs Recursive**<br>**-0.0088 vs Hyra**<br>Peak VRAM reduced by **20.9%** (177.7 GiB → 140.6 GiB) |
| **MLS-Bench Fused Attention** | GPU Kernel Optimization<br>(Single H100 SXM 80GB) | d=64: **430.7 TFLOP/s**<br>d=128: **468.4 TFLOP/s**<br>d=256: **451.7 TFLOP/s** | d=64: 338.6 TFLOP/s<br>d=128: 405.1 TFLOP/s<br>d=256: 385.7 TFLOP/s | **+27.18%** (d=64)<br>**+15.63%** (d=128)<br>**+17.11%** (d=256)<br>Max diff ≤ 1.95 × 10⁻³ (< 10⁻²) |
| **SLDBench (SimpleTES 4-Task)** | Scaling-Law Discovery<br>(Generalization to held-out regimes) | **0.8846** (4-task mean)<br>U-shaped: **0.5761**<br>LR/BSZ: **0.9650**<br>Parallel: **0.999989**<br>Domain: **0.997379** | Previous Evaluator Artifact: `0.8613`<br>U-shaped: `0.5330`<br>LR/BSZ: `0.9380`<br>Parallel: `0.9850`<br>Domain: `0.9890` | **+2.71% average gain**<br>**+8.08% on U-shaped scaling**<br>**+2.87% on LR/BSZ co-scaling** |
| **GPUMode TriMul** | AlphaFold Triangle Update<br>(Single H100 80GB HBM3) | **1,036.1 μs**<br>(2-run geomean across 7 shapes;<br>18/18 correctness passed) | stashuk-olek (Human SOTA): `1,064.9 μs`<br>shiyegao CUDA: `1,074.0 μs`<br>Zeyu Shen Triton: `1,140.0 μs`<br>TTT-Discover: `1,161.0 μs`<br>K-Search: `1,030.0 μs` | **2.71% lower latency than stashuk-olek**<br>(-28.8 μs latency reduction)<br>Outperforms TTT-Discover by **10.8%** |

---

## Contents

- [`nanochat_autoresearch/`](nanochat_autoresearch/) — NanoChat autoresearch pretraining script and verified results across 10 seeds (single B200, 300 s budget). Reaches **0.892792** mean val_bpb across 10 seeds (best single seed **0.891762**; 3-seed initial mean **0.892426**), outperforming Recursive SuperIntelligence (0.910875 mean, 0.903891 best) and Tencent Hunyuan's Hyra (0.901543), placing near AutoTrust's public SOTA (0.889522). Discovered mechanisms include a single custom Triton kernel fusing dual n-gram table lookups (bigram 512 + shared trigram 2048), concatenation, and active-row registration, alongside a compact backward pass updating touched rows only—reducing peak training memory from ~177.7 GiB to 140.6 GiB (20.9% reduction) while processing ~4.5% fewer tokens. Builds on [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT).
- [`mls-bench_fused_attention/`](mls-bench_fused_attention/) — MLS-Bench Fused Causal Attention OpenAI Triton forward pass on NVIDIA H100 SXM 80GB. Establishes a new SOTA on all three benchmark head dimensions at **430.7 TFLOP/s** (d=64, **+27.18%**), **468.4 TFLOP/s** (d=128, **+15.63%**), and **451.7 TFLOP/s** (d=256, **+17.11%**) with max absolute error ≤ 1.95 × 10⁻³ (< 10⁻²). Discovered mechanisms include **rescale-free causal softmax** (yielding standalone throughput gains of +21.0%, +8.5%, and +19.3%), 2D TMA descriptor loads on d=128, and Longest Processing Time first (LPT) tile scheduling.
- [`sldbench/`](sldbench/) — Scaling Law Discovery Benchmark on [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench) following the SimpleTES 4-task protocol. Improves upon the previous evaluator artifact across all four tasks, raising the 4-task average from 0.8613 to **0.8846** (**+2.71%**), with largest gain on U-shaped compute scaling from 0.5330 to **0.5761** (**+8.08%**), alongside LR/BSZ co-scaling from 0.9380 to **0.9650** (**+2.87%**), parallel scaling (**0.999989**), and domain-mixture scaling (**0.997379**). Discovered mechanism decomposes scaling curves into a persistent scaling trend and a localized transient regime effect with explicit onset, duration, and decay.
- [`gpumode_trimul/`](gpumode_trimul/) — GPUMode Triangular Matrix Multiplication (TriMul) kernel on NVIDIA H100 80GB HBM3. Reaches a two-run geometric-mean latency of **1,036.1 μs** across 7 benchmark shapes (18/18 correctness tests passed; Run 1: 1,036.66 μs, Run 2: 1,035.45 μs), achieving **2.71% lower latency** than the strongest human baseline stashuk-olek (1,064.9 μs, +2.79% throughput speedup), as well as outperforming separately reported implementations from shiyegao CUDA (1,074 μs), Zeyu Shen Triton (1,140 μs), and TTT-Discover (1,161 μs), while coming within 0.59% of K-Search (1,030 μs). Discovered mechanism re-localizes the bottleneck across the BMM → LayerNorm boundary, reading BMM outputs contiguously along `[hidden, row]` and performing an in-register transpose before LayerNorm, paired with deep epilogue fusion.

---

## Reproduction

See each subdirectory's `README.md` for dependencies and reproduction instructions:

```bash
# 1. NanoChat Autoresearch (single B200)
python nanochat_autoresearch/prepare.py
python nanochat_autoresearch/solutions/train.py
# Or reproduce full 10 seeds: python nanochat_autoresearch/run_10_seeds.py

# 2. MLS-Bench Fused Causal Attention (single H100)
python mls-bench_fused_attention/bench.py --all-shapes --gpu 0 --runs 3

# 3. SLDBench 4-Task Protocol
python sldbench/evaluate.py --prepare
python sldbench/evaluate.py

# 4. GPUMode TriMul (single H100, 2-run paired evaluation)
python gpumode_trimul/evaluate.py --runs 2
```

---

## License & Attribution

This repository is licensed under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE)).

It includes code and benchmarks derived from open-source projects, whose copyright and permission notices are preserved:
- [`nanochat_autoresearch/`](nanochat_autoresearch/) — builds on [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT; upstream notice at [`nanochat_autoresearch/LICENSE-nanochat`](nanochat_autoresearch/LICENSE-nanochat)) and the autoresearch methodology from [Recursive SuperIntelligence](https://www.recursive-ai.com/) (Apache-2.0).
- [`mls-bench_fused_attention/`](mls-bench_fused_attention/) — builds on [MLS-Bench](https://github.com/Imbernoulli/MLS-Bench) (Apache-2.0, [arXiv:2605.08678](https://arxiv.org/abs/2605.08678)), PyTorch SDPA reference (BSD-3-Clause), and OpenAI Triton (MIT).
- [`sldbench/`](sldbench/) — builds on [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench) (Apache-2.0) and SimpleTES protocols (arXiv:2604.19341).
- [`gpumode_trimul/`](gpumode_trimul/) — builds on [GPUMode Popcorn](https://github.com/gpu-mode) (Apache-2.0 / MIT), DeepMind AlphaFold triangle update formulation (Apache-2.0), and [TTT-Discover](https://github.com/test-time-training/discover) (Stanford / NVIDIA / Together AI, Apache-2.0).

See [`NOTICE`](NOTICE) for the full attribution.
