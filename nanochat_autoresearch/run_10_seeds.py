#!/usr/bin/env python3
"""Automated 10-Seed Reproduction Runner for NanoChat Autoresearch.

Runs training across the 10 benchmark seeds (42, 137, 271, 314, 589, 733, 997, 1231, 1667, 2021),
collects validation BPB and training throughput, and computes the 10-seed statistical distribution
(mean, sample std, and 95% Student-t confidence interval).

Usage:
    # Full 10-seed evaluation on single B200 (5 minutes per seed):
    python run_10_seeds.py

    # Quick 3-seed evaluation (seeds 42, 137, 271):
    python run_10_seeds.py --seeds 42,137,271

    # Dry-run to view execution plan and baseline comparison:
    python run_10_seeds.py --dry-run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Official benchmark seeds
DEFAULT_SEEDS = [42, 137, 271, 314, 589, 733, 997, 1231, 1667, 2021]

# Reference statistics from results/summary.json
REFERENCE_METRICS = {
    "10_seed_mean": 0.8927922,
    "10_seed_std": 0.0007662,
    "10_seed_ci95": [0.8922441, 0.8933403],
    "3_seed_mean": 0.892426,
    "best_seed_42": 0.891762,
    "recursive_mean": 0.910875,
    "tencent_hyra": 0.901543,
    "autotrust_scienceguru": 0.889522,
}

# Two-sided Student-t critical values at 95% confidence (df = n - 1)
STUDENT_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def compute_statistics(values):
    n = len(values)
    if n == 0:
        return {}
    mean_val = sum(values) / n
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(variance)
        t_crit = STUDENT_T_95.get(n - 1, 1.96)
        margin = t_crit * (std_val / math.sqrt(n))
        ci95 = (mean_val - margin, mean_val + margin)
    else:
        std_val = 0.0
        ci95 = (mean_val, mean_val)
    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "ci95": ci95,
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce NanoChat Autoresearch 10-Seed Results")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS)),
        help="Comma-separated random seeds (default: all 10 benchmark seeds)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=300,
        help="Training time budget in seconds per seed (default: 300)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA GPU device index (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save aggregated results to JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show reproduction plan and reference baselines without running training",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    train_script = root_dir / "solutions" / "train.py"
    if not train_script.is_file():
        print(f"Error: training script not found at {train_script}", file=sys.stderr)
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 72)
    print("Apex Intelligence - NanoChat Autoresearch Multi-Seed Reproduction")
    print("=" * 72)
    print(f"Seeds to evaluate: {seeds} (total: {len(seeds)})")
    print(f"Time budget:       {args.budget}s per seed")
    print(f"Target GPU:        CUDA device {args.gpu}")
    print(f"Training script:   {train_script}")
    print("-" * 72)
    print("Published Reference Baselines:")
    print(f"  • Discovered 10-Seed Mean:     {REFERENCE_METRICS['10_seed_mean']:.6f} (std: {REFERENCE_METRICS['10_seed_std']:.6f})")
    print(f"  • Discovered 3-Seed Mean:      {REFERENCE_METRICS['3_seed_mean']:.6f} (seeds 42, 137, 271)")
    print(f"  • Discovered Best Single Seed: {REFERENCE_METRICS['best_seed_42']:.6f} (seed 42)")
    print(f"  • AutoTrust ScienceGuru SOTA:  {REFERENCE_METRICS['autotrust_scienceguru']:.6f}")
    print(f"  • Tencent Hunyuan Hyra:        {REFERENCE_METRICS['tencent_hyra']:.6f}")
    print(f"  • Recursive SuperIntelligence: {REFERENCE_METRICS['recursive_mean']:.6f}")
    print("-" * 72)

    if args.dry_run:
        print("Dry run completed. To execute actual training on NVIDIA B200, run without --dry-run.")
        return

    runs = []
    val_bpbs = []

    for idx, seed in enumerate(seeds, 1):
        print(f"\n[{idx}/{len(seeds)}] Launching Seed {seed} (Budget: {args.budget}s)...")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
            tmp_json = tmp_f.name

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["NANOCHAT_SEED"] = str(seed)
        env["NANOCHAT_TIME_BUDGET"] = str(args.budget)
        env["P1_FUSED_LOOKUP_COLLECT"] = "1"
        env["P1_COMPACT_BACKWARD"] = "1"

        cmd = [
            sys.executable,
            str(train_script),
            "--seed", str(seed),
            "--budget", str(args.budget),
            "--output", tmp_json,
        ]

        ret = subprocess.run(cmd, cwd=str(root_dir), env=env)
        if ret.returncode != 0:
            print(f"Run failed for seed {seed} (exit code {ret.returncode})", file=sys.stderr)
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
            continue

        if os.path.exists(tmp_json):
            with open(tmp_json) as f:
                metric_data = json.load(f)
            os.remove(tmp_json)
            runs.append(metric_data)
            val_bpbs.append(metric_data["val_bpb"])
            print(f"  Seed {seed} completed: Val BPB = {metric_data['val_bpb']:.6f} "
                  f"({metric_data.get('total_tokens_M', 0):.1f}M tokens, {metric_data.get('num_steps', 0)} steps)")

    if not runs:
        print("\nNo runs completed successfully.", file=sys.stderr)
        sys.exit(1)

    stats = compute_statistics(val_bpbs)

    print("\n" + "=" * 72)
    print("Reproduction Evaluation Summary")
    print("=" * 72)
    print(f"{'Seed':>6} | {'Val BPB':>10} | {'Steps':>7} | {'Tokens (M)':>10} | {'Peak VRAM (GiB)':>15}")
    print("-" * 72)
    for r in runs:
        print(f"{r['seed']:>6} | {r['val_bpb']:>10.6f} | {r.get('num_steps', 0):>7} | "
              f"{r.get('total_tokens_M', 0):>10.1f} | {r.get('peak_vram_gib', 0):>15.1f}")
    print("-" * 72)
    print(f"Completed Runs:     {stats['n']}/{len(seeds)}")
    print(f"Sample Mean:        {stats['mean']:.6f}")
    if stats['n'] > 1:
        print(f"Sample Std Dev:     {stats['std']:.6f}")
        print(f"Two-Sided 95% CI:   [{stats['ci95'][0]:.6f}, {stats['ci95'][1]:.6f}]")
        print(f"Range:              [{stats['min']:.6f}, {stats['max']:.6f}]")
    print("=" * 72)

    if args.output:
        out_summary = {
            "completed_runs": runs,
            "statistics": stats,
            "reference": REFERENCE_METRICS,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_summary, f, indent=2)
        print(f"Saved aggregated reproduction summary to {args.output}")


if __name__ == "__main__":
    main()
