# Scaling-Law Discovery

Symbolic regression for LLM scaling laws on [`pkuHaowei/sldbench`](https://huggingface.co/datasets/pkuHaowei/sldbench). Evolves a parameterised functional form + fitter; the evaluator fits the train split and scores held-out points.

| Subtask | What the law models | `combined_score` | TIMEOUT |
|---------|---------------------|------------------|---------|
| **scaling_law/domain_mixture_scaling_law** | How multi-domain loss depends on domain mixture proportions across model sizes | $R^2$ on the held-out fit | 600 s |
| **scaling_law/easy_question_scaling_law** | Easy-question accuracy vs compute (FLOPs) — the U-shaped / double-descent regime | $R^2$ on the held-out fit | 600 s |
| **scaling_law/lr_bsz_scaling_law** | Training loss as a function of learning rate, batch size, data size, and model parameters | $R^2$ on the held-out fit | 600 s |
| **scaling_law/parallel_scaling_law** | LM loss vs model parameters under parallel scaling (the `parallel_size` axis) | $R^2$ on the held-out fit | 600 s |

All four subtasks have a hard cap of 35 parameters. Exceeding the cap, or failing to load / fit / converge, records `combined_score = -1e6`.

The evaluator also reports per-dim `nmse`, `nmae`, `r2` for analysis. Only `r2` (averaged across output dimensions) drives the score.

## Setup

From the published bundle root:

```bash
python3 evaluate.py --prepare
```

The equivalent direct command is:

```bash
cd sldbench/evaluators
python3 prepare_dataset.py
```

The downloader pins revision `721b846056f031737ff7fa72572c021324e3ec0e` and
fetches the four subtasks used by this package. HF cache respects `HF_HOME` /
`HF_DATASETS_CACHE`.

## Requirements

`numpy`, `scipy`, `datasets` (HuggingFace), `scikit-learn`. See `datasets/scaling_law/requirements.txt`.

## Running

The bundle-level command runs all four frozen champions and aggregates their
scores:

```bash
python3 evaluate.py
```

The direct command for one subtask is:

```bash
python3 evaluators/parallel_scaling_law/evaluator.py solutions/parallel.py
```

Substitute the matching evaluator and champion for the other three subtasks.
