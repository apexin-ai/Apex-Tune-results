# NanoChat Autoresearch

Training scripts and evaluation results for the NanoChat solutions discovered by automated AI research.

Every solution is a single self-contained training script trained for **5 minutes (300 counted seconds) on a single NVIDIA B200 GPU (SM100)** and scored by **validation bits-per-byte (Val BPB, lower is better)**.

## Performance Summary

| Solution | Method / Architecture | Val BPB (Lower is Better) |
|---|---|:---:|
| **Discovered Solution** | **Compact touched-row backward + fused dual n-gram lookup** | **0.892792** (10-Seed Mean) <br> **0.892426** (3-Seed Initial Mean) <br> **0.891762** (Best Single Seed) |
| AutoTrust ScienceGuru (Sep 2026) | Full n-gram memory baseline (Public SOTA) | 0.889522 |
| Tencent Hunyuan Hyra (Jul 2026) | Published external baseline | 0.901543 |
| Recursive SuperIntelligence (Jun 2026, Best) | Published external single-run best | 0.903891 |
| Recursive SuperIntelligence (Jun 2026, Mean) | Published external 10-seed mean | 0.910875 |

### 10-Seed Validation Aggregate

- **Seeds**: `42, 137, 271, 314, 589, 733, 997, 1231, 1667, 2021`
- **Mean Val BPB**: **`0.8927922`**
- **Sample Standard Deviation**: `0.0007662`
- **Two-Sided 95% Student-t Confidence Interval**: `[0.8922441, 0.8933403]`
- **Range**: `[0.891762, 0.894539]`
- **3-Seed Initial Subset Mean (Seeds 42, 137, 271)**: `0.892426`

### Per-Seed Run Breakdown

| Seed | Val BPB | Steps | Tokens (M) |
| ---: | ---: | ---: | ---: |
| 42 | **0.891762** | 3301 | 486.8 |
| 137 | 0.892828 | 3305 | 487.3 |
| 271 | 0.892688 | 3302 | 486.9 |
| 314 | 0.892595 | 3327 | 490.6 |
| 589 | 0.892638 | 3331 | 491.2 |
| 733 | 0.894539 | 3305 | 487.3 |
| 997 | 0.893019 | 3282 | 484.0 |
| 1231 | 0.893141 | 3274 | 482.8 |
| 1667 | 0.892877 | 3312 | 488.4 |
| 2021 | 0.891835 | 3325 | 490.3 |

## Core Mechanisms Discovered

1. **Compact Touched-Row Backward Pass**:
   - Large n-gram embedding memories provide substantial capacity, but each training batch accesses only a small fraction of their rows.
   - Replaces dense gradient costs with a compact backward pass that updates exclusively the touched rows in compact FP32 slot accumulators.
   - **Reduces peak training memory from ~177.7 GiB to 140.6 GiB (20.9% reduction)**, preserving near-SOTA quality while processing ~4.5% fewer tokens than ScienceGuru within the fixed 300-second budget.

2. **Fused Dual N-Gram Lookup & Registration**:
   - Fuses two n-gram table lookups (bigram 512 + shared trigram 2048), concatenation, active-row discovery, and row registration into a single custom Triton kernel.
   - The backward pass directly reuses the resulting row-to-slot mapping, eliminating duplicated indexing work and preserving sparsity from lookup through gradient update.

3. **Normalized Attention Input Reuse**:
   - Normalized post-layer-4 attention input is saved and reused for Q/K/V and attention gates in layers 5, 6, and 7, while residual and MLP streams continue to consume the current stream.

4. **Recipe Optimization**:
   - `TINY_DIV=8`, MLP depth profile `(3, 3, 3, 4, 4, 5, 5, 5)`, `MATRIX_LR=0.035`, `WARMDOWN_RATIO=0.90`, Muon momentum `0.80`.

## External Baselines & Evaluation Protocol

### Published External Baselines

| Baseline | Organization / Authors | Date / Version | Reference / Link | Val BPB (Lower is Better) | Peak VRAM |
|---|---|---|---|:---:|:---:|
| **AutoTrust ScienceGuru** | AutoTrust | September 2026 | Public SOTA Baseline Release | **0.889522** | ~177.7 GiB |
| **Tencent Hunyuan Hyra** | Tencent Hunyuan | July 2026 | [Tencent Hunyuan](https://github.com/Tencent-Hunyuan) | 0.901543 | — |
| **Recursive SuperIntelligence (Best)** | Recursive SuperIntelligence | June 2026 | [Recursive SuperIntelligence Release](https://www.recursive-ai.com/) | 0.903891 | — |
| **Recursive SuperIntelligence (10-Seed Mean)** | Recursive SuperIntelligence | June 2026 | [Recursive SuperIntelligence Release](https://www.recursive-ai.com/) | 0.910875 | — |

### Evaluation Protocol & Hardware Alignment

- **Benchmark**: NanoChat Autoresearch (LLM pretraining from scratch).
- **Training Budget**: Strictly 300 counted seconds (5 minutes) wall-clock time per endpoint.
- **Hardware & Software**: Single NVIDIA B200 (SM100) GPU, Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8, `flash-attn-4`.
- **Target Metric**: Validation bits-per-byte (`val_bpb`, lower is better), calculated on the standard validation set with identical tokenization.
- **Statistical Aggregation**: To prevent seed cherry-picking, our result is reported as an integrated 10-seed distribution (mean = 0.892792, std = 0.000766, 95% CI = [0.892244, 0.893340]), covering seeds `42, 137, 271, 314, 589, 733, 997, 1231, 1667, 2021`.

## Repository Contents

- `solutions/train.py` — Complete pretraining script implementing the discovered model architecture, touched-row backward pass, and fused dual n-gram lookup Triton kernel.
- `prepare.py` — Data preparation script (fetches ClimbMix-400B shards to `~/.cache/autoresearch/data/` and trains the BPE tokenizer at `~/.cache/autoresearch/tokenizer/`), providing dataset caching, dataloader, and BPB evaluation utilities.
- `run_10_seeds.py` — Automated multi-seed reproduction runner evaluating across all 10 benchmark seeds and reporting sample mean, standard deviation, and 95% Student-t confidence interval.
- `requirements.txt` — Exact pinned package dependencies for reproducibility.
- `results/val_bpb.csv` — Full 10-seed evaluation records and external baseline comparisons.
- `results/summary.json` — Comprehensive benchmark metrics, statistical intervals, and hardware details.
- `LICENSE-nanochat` — Upstream MIT license notice.

## Reproduction Guide

### 1. Environment & Dependencies

Hardware requirement: Single NVIDIA B200 GPU (SM100).  
Software requirement: Python >= 3.10, CUDA 12.8.

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data & Tokenizer Preparation

Data shards and tokenizer are cached locally under `~/.cache/autoresearch/` (customizable via the `NANOCHAT_CACHE_DIR` environment variable):

```bash
# Download training shards + pinned validation shard, and train BPE tokenizer
python prepare.py

# Or download a smaller subset of shards for quick verification
python prepare.py --num-shards 10
```

### 3. Training a Single Seed

Run the pretraining script for the default 300-second budget on seed 42. By default, `P1_FUSED_LOOKUP_COLLECT=1` is enabled to activate the discovered fused dual n-gram lookup kernel:

```bash
# Default run (seed 42, 300-second budget)
python solutions/train.py

# Or specify a custom seed and time budget
python solutions/train.py --seed 137 --budget 300 --output results/my_run.json
```

### 4. Reproducing the Full 10-Seed Benchmark Suite

To reproduce the full 10-seed distribution and verify the `0.892792` mean Val BPB headline claim:

```bash
# Run all 10 benchmark seeds (seeds 42, 137, 271, 314, 589, 733, 997, 1231, 1667, 2021)
python run_10_seeds.py

# Quick 3-seed verification (seeds 42, 137, 271)
python run_10_seeds.py --seeds 42,137,271

# Dry-run to inspect execution plan without launching GPU training
python run_10_seeds.py --dry-run
```
