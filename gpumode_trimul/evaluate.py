#!/usr/bin/env python3
"""Evaluation runner for GPUMode TriMul H100 kernel.

Runs the 18 correctness test cases and 7 leaderboard benchmark shapes
using the canonical GPUMode Popcorn harness, and computes the geometric
mean latency across shapes. Supports multiple evaluation runs (default: 2)
to reproduce the headline paired average (1,036.05 μs).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT / "canonical"
SUBMISSION = ROOT / "submission.py"

sys.path.insert(0, str(CANONICAL))
from contract import canonical_manifest_sha256
from scorer import score_popcorn_output, validate_test_output

EXPECTED_CANONICAL_MANIFEST_SHA256 = "cd61e8501c113889060db3eefab9802cffb33cad309682b01e2dad30af9708f2"


def run_evaluation(output_json=None, gpu_id=0, runs=2):
    if not SUBMISSION.is_file():
        raise FileNotFoundError(f"Missing submission file: {SUBMISSION}")

    # Verify canonical harness integrity
    manifest_sha = canonical_manifest_sha256(CANONICAL)
    if manifest_sha != EXPECTED_CANONICAL_MANIFEST_SHA256:
        raise ValueError(
            f"Canonical harness manifest mismatch: expected {EXPECTED_CANONICAL_MANIFEST_SHA256}, got {manifest_sha}"
        )
    print(f"Verified canonical evaluator manifest integrity: {manifest_sha[:12]}...")

    with tempfile.TemporaryDirectory(prefix="trimul_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for f in ["eval.py", "reference.py", "task.py", "utils.py", "tests.txt", "benchmarks.txt", "contract.py"]:
            shutil.copy(CANONICAL / f, tmp_path / f)
        shutil.copy(SUBMISSION, tmp_path / "submission.py")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{tmp_path}:{CANONICAL}:{env.get('PYTHONPATH', '')}"

        test_out_file = tmp_path / "test.out"

        print("--- Phase 1: Running 18 Correctness Test Cases ---")
        with open(test_out_file, "w") as out_f:
            env["POPCORN_FD"] = str(out_f.fileno())
            proc = subprocess.run(
                [sys.executable, "eval.py", "test", "tests.txt"],
                cwd=str(tmp_path),
                env=env,
                pass_fds=(out_f.fileno(),),
                capture_output=True,
                text=True,
            )

        test_output = test_out_file.read_text()
        expected_tests = tuple(line for line in (CANONICAL / "tests.txt").read_text().splitlines() if line)
        try:
            validate_test_output(test_output, expected_specs=expected_tests)
            print("Correctness: All 18/18 test cases PASSED!")
        except Exception as e:
            print(f"Correctness check FAILED: {e}", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            return

        print(f"\n--- Phase 2: Running 7 Benchmark Shapes ({runs} run{'s' if runs > 1 else ''}) ---")
        expected_benchmarks = tuple(line for line in (CANONICAL / "benchmarks.txt").read_text().splitlines() if line)

        attempt_scores = []
        for run_idx in range(1, runs + 1):
            leaderboard_out_file = tmp_path / f"leaderboard_run_{run_idx}.out"
            with open(leaderboard_out_file, "w") as out_f:
                env["POPCORN_FD"] = str(out_f.fileno())
                proc = subprocess.run(
                    [sys.executable, "eval.py", "leaderboard", "benchmarks.txt"],
                    cwd=str(tmp_path),
                    env=env,
                    pass_fds=(out_f.fileno(),),
                    capture_output=True,
                    text=True,
                )
            leaderboard_output = leaderboard_out_file.read_text()
            score = score_popcorn_output(leaderboard_output, private_seed_used=False, expected_specs=expected_benchmarks)
            attempt_scores.append(score)
            print(f"  Run {run_idx}/{runs}: Geomean = {score.latency_us:.2f} μs")

        baseline_us = 1064.9228290162391
        geomeans = [s.latency_us for s in attempt_scores]
        headline_geomean = sum(geomeans) / len(geomeans)
        latency_reduction_pct = ((baseline_us - headline_geomean) / baseline_us) * 100
        speedup_pct = ((baseline_us / headline_geomean) - 1) * 100

        # Compute per-shape mean latencies
        num_shapes = len(expected_benchmarks)
        per_shape_means = [
            sum(attempt_scores[r].benchmark_latencies_us[i] for r in range(runs)) / runs
            for i in range(num_shapes)
        ]

        print("=" * 70)
        print(f"{'Shape / Config':<45} | {'Mean Latency (μs)':<18}")
        print("-" * 70)
        for i, spec in enumerate(attempt_scores[0].benchmark_specs):
            short_spec = ", ".join([p for p in spec.split(";") if any(k in p for k in ["seqlen", "bs", "dim", "hiddendim"])])
            print(f"{short_spec:<45} | {per_shape_means[i]:<18.2f}")
        print("-" * 70)
        for r_idx, g_us in enumerate(geomeans, 1):
            print(f"Run {r_idx} Geometric Mean Latency                 | {g_us:<18.2f} μs")
        print(f"Average Geometric Mean Latency (Headline)     | {headline_geomean:<18.2f} μs")
        print(f"Official Baseline Latency (stashuk-olek)      | {baseline_us:<18.2f} μs")
        print(f"Latency Reduction                             | {latency_reduction_pct:<18.2f}% lower latency")
        print(f"Throughput Speedup                            | {speedup_pct:<+18.2f}%")
        print("=" * 70)

        results = {
            "task": "gpumode_trimul",
            "correctness": "18/18 passed",
            "runs": runs,
            "headline_geomean_latency_us": headline_geomean,
            "baseline_geomean_latency_us": baseline_us,
            "latency_reduction_pct": latency_reduction_pct,
            "speedup_pct": speedup_pct,
            "per_run_geomeans_us": geomeans,
            "per_shape_mean_latencies_us": per_shape_means,
            "attempts": [
                {
                    "run_idx": i + 1,
                    "geomean_latency_us": s.latency_us,
                    "benchmark_latencies_us": s.benchmark_latencies_us,
                }
                for i, s in enumerate(attempt_scores)
            ],
        }

        if output_json:
            Path(output_json).write_text(json.dumps(results, indent=2) + "\n")
            print(f"Results saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPUMode TriMul Submission")
    parser.add_argument("--output", type=str, help="Save output JSON metrics")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU device index")
    parser.add_argument("--runs", type=int, default=2, help="Number of benchmark passes to average (default: 2)")
    args = parser.parse_args()
    run_evaluation(args.output, args.gpu, args.runs)


if __name__ == "__main__":
    main()
