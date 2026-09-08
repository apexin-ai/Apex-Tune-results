# SLDBench: Scaling Law Discovery Benchmark (SimpleTES 4-Task Protocol)

Discovered scaling law formulations and fitting routines for the **Scaling Law Discovery Benchmark** on [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench) (dataset revision `721b846056f031737ff7fa72572c021324e3ec0e`), following the four-task evaluation protocol used by [SimpleTES](https://arxiv.org/html/2604.19341v2).

The benchmark turns scaling-law discovery into a measurable task built from >5,000 published LLM training experiments. Each system produces a symbolic law and fitting procedure that generalizes to held-out scaling regimes across parallel scaling, domain-mixture scaling, learning-rate/batch-size co-scaling, and U-shaped compute scaling.

## Performance Summary

The evaluation metric is **Combined Score = 1 - mean(NMSE)** across target dimensions on held-out regimes (higher is better, maximum 1.0).

All individual baselines are aligned directly with the previous evaluator artifact recorded in the evaluation suite (`evaluate.py`), averaging to 0.8613:

| Subtask | Previous Evaluator Artifact | Discovered | Relative Gain | Description |
|---|:---:|:---:|:---:|---|
| **U-shaped compute scaling** (`easy_question`) | 0.5330 | **0.5761** | **+8.08%** | Decomposition into persistent scaling trend + transient regime effect with onset, duration, and decay |
| **LR / BSZ co-scaling** (`lr_bsz`) | 0.9380 | **0.9650** | **+2.87%** | Log-space monomial mixture with stable log-sum-exp & boundary residual surface |
| **Parallel scaling** (`parallel`) | 0.9850 | **0.999989** | +1.52% | 4-parameter product form in log-space with global boundary calibration |
| **Domain-mixture scaling** (`domain_mixture`) | 0.9890 | **0.997379** | +0.85% | 35-parameter 5-domain joint fit with domain-standard-deviation residual weighting |
| **Four-Task Average** | **0.8613** | **0.8846** | **+2.71%** | **Improves previous evaluator artifact across all four tasks** |

*Note on baseline consistency: The four individual subtask baselines (`0.5330`, `0.9380`, `0.9850`, `0.9890`) represent the previous evaluator artifact recorded in the evaluation suite, averaging to 0.86125 (rounded to 0.8613). They should not be conflated with the method-level 5-seed aggregate numbers from SimpleTES Table 2. Across the four tasks, the discovered suite raises the mean from 0.8613 to 0.8846 (+2.71% average gain).*

## Core Methodological Advances

1. **U-Shaped Compute Scaling (`solutions/easy_question.py`)**:
   - Represents the curve as the sum of a **persistent scaling trend** and a **localized transient regime effect** that emerges over a limited compute range and then decays.
   - Prevents temporary reversals in observed data from being projected indefinitely, yielding accurate post-turning-point predictions even when the evaluator withholds the curve's final phase.
2. **LR/BSZ Co-Scaling (`solutions/lr_bsz.py`)**:
   - Joint learning-rate and batch-size scaling surface via a log-space monomial mixture with stable log-sum-exp optimization and multi-start least-squares fits.
3. **Domain Mixture (`solutions/domain_mixture.py`)**:
   - Models multi-domain loss under varying proportions using a symmetric 5 × 5 interaction matrix with per-domain residual weighting normalized by domain variance.
4. **Parallel Scaling (`solutions/parallel.py`)**:
   - 3D parallel scaling (tensor, pipeline, data parallel degrees) with a 4-parameter power-law product form fitted in log-space and calibrated boundaries.

## External Baselines & Evaluation Protocol

### Benchmark Subtasks & Baseline References

| Subtask | Task Name | Previous Evaluator Artifact | Discovered Score | Relative Gain | Source Note |
|---|---|:---:|:---:|:---:|---|
| **U-shaped compute scaling** | `easy_question` | 0.5330 | **0.5761** | **+8.08%** | Previous evaluator artifact |
| **LR / BSZ co-scaling** | `lr_bsz` | 0.9380 | **0.9650** | **+2.87%** | Previous evaluator artifact |
| **3D Parallel scaling** | `parallel` | 0.9850 | **0.999989** | +1.52% | Previous evaluator artifact |
| **Domain-mixture scaling** | `domain_mixture` | 0.9890 | **0.997379** | +0.85% | Previous evaluator artifact |
| **Four-Task Mean** | — | **0.8613** | **0.8846** | **+2.71%** | Previous Evaluator Artifact Aggregate |

### Evaluation Protocol & Verification Contract

- **Dataset**: [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench) on Hugging Face, dataset revision `721b846056f031737ff7fa72572c021324e3ec0e` (retrieved April 2026). Over 5,000 empirical LLM pretraining runs across model sizes, batch sizes, learning rates, domain ratios, and parallelism topologies.
- **Reference Paper**: SimpleTES: "SimpleTES: Toward Automated Discovery of Scaling Laws" ([arXiv:2604.19341](https://arxiv.org/abs/2604.19341)).
- **Evaluation Metric**: Generalization score $1 - \text{NMSE}$ on held-out extrapolation regimes unseen during fitting:
  $$\text{Score} = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}$$
  Scores are bounded by 1.0 (perfect prediction). A score of 0 corresponds to predicting the test mean.
- **Reproducibility**: The evaluation suite runs automatically via `python evaluate.py` and produces verifiable predictions matching `results/scores.json`.

## Repository Contents

- `solutions/` — Python scripts implementing `fit_scaling_law` and `scaling_law_func` for each of the 4 subtasks.
- `evaluators/` — Official evaluation harnesses and data loader scripts for the 4 subtasks.
- `evaluate.py` — Benchmark runner script that verifies and scores solutions against the evaluators.
- `results/scores.json` — Detailed verified scores and metric breakdowns.

## How to Reproduce

Requirements: Python >= 3.10, `numpy`, `scipy`, `scikit-learn`, `pandas`, `datasets`.

1. Pre-fetch datasets into local Hugging Face cache:
   ```bash
   python evaluate.py --prepare
   ```

2. Run evaluation across all 4 subtasks:
   ```bash
   python evaluate.py
   ```

3. Or evaluate a specific subtask:
   ```bash
   python evaluate.py --task easy_question
   ```
