#!/usr/bin/env python3
"""Evaluation runner for Scaling Law Discovery Benchmark (sldbench).

Evaluates the discovered scaling law models against the pkuHaowei/sldbench
evaluators across four subtasks:
    - parallel
    - domain_mixture
    - lr_bsz
    - easy_question

Metric: Combined Score = 1 - mean(NMSE_per_dim) across test points.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVALUATOR_ROOT = ROOT / "evaluators"
SOLUTIONS_ROOT = ROOT / "solutions"

# Baseline scores represent previous evaluator artifacts recorded in the evaluation suite,
# not the method-level 5-seed aggregate numbers from SimpleTES Table 2.
SUBTASKS = {
    "parallel": {
        "evaluator": EVALUATOR_ROOT / "parallel_scaling_law" / "evaluator.py",
        "solution": SOLUTIONS_ROOT / "parallel.py",
        "baseline_score": 0.985,
        "target_score": 0.9999891170971098,
    },
    "domain_mixture": {
        "evaluator": EVALUATOR_ROOT / "domain_mixture_scaling_law" / "evaluator.py",
        "solution": SOLUTIONS_ROOT / "domain_mixture.py",
        "baseline_score": 0.989,
        "target_score": 0.9973788599692002,
    },
    "lr_bsz": {
        "evaluator": EVALUATOR_ROOT / "lr_bsz_scaling_law" / "evaluator.py",
        "solution": SOLUTIONS_ROOT / "lr_bsz.py",
        "baseline_score": 0.938,
        "target_score": 0.9649515798828776,
    },
    "easy_question": {
        "evaluator": EVALUATOR_ROOT / "easy_question_scaling_law" / "evaluator.py",
        "solution": SOLUTIONS_ROOT / "easy_question.py",
        "baseline_score": 0.533,
        "target_score": 0.5760748673298086,
    },
}

SCORE_RE = re.compile(r"^\s*Combined Score:\s*([-+0-9.eE]+)\s*$", re.MULTILINE)


def run_subtask(task_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    evaluator = spec["evaluator"]
    solution = spec["solution"]
    if not evaluator.is_file():
        raise FileNotFoundError(f"Evaluator not found: {evaluator}")
    if not solution.is_file():
        raise FileNotFoundError(f"Solution not found: {solution}")

    cmd = [sys.executable, str(evaluator), str(solution)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    match = SCORE_RE.search(proc.stdout)
    score = float(match.group(1)) if match else None

    return {
        "task": task_name,
        "returncode": proc.returncode,
        "score": score,
        "target_score": spec["target_score"],
        "baseline_score": spec["baseline_score"],
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate sldbench Scaling Law Solutions")
    parser.add_argument("--task", choices=list(SUBTASKS.keys()), help="Run specific subtask")
    parser.add_argument("--prepare", action="store_true", help="Pre-fetch sldbench datasets into cache")
    parser.add_argument("--output", type=str, help="Save evaluation results to JSON file")
    args = parser.parse_args()

    if args.prepare:
        prep_script = EVALUATOR_ROOT / "prepare_dataset.py"
        print(f"Pre-fetching sldbench subsets using {prep_script}...")
        proc = subprocess.run([sys.executable, str(prep_script)], cwd=str(ROOT))
        if proc.returncode != 0:
            print("Failed to prefetch datasets.", file=sys.stderr)
            sys.exit(proc.returncode)
        print("Dataset pre-fetch completed.")
        if not args.task:
            return

    tasks_to_run = [args.task] if args.task else list(SUBTASKS.keys())
    results = {}
    scores = []

    print("=" * 65)
    print(f"{'Subtask':<18} | {'Score':<14} | {'Target':<14} | Status")
    print("-" * 65)

    for task_name in tasks_to_run:
        spec = SUBTASKS[task_name]
        res = run_subtask(task_name, spec)
        results[task_name] = res
        score = res["score"]
        if score is not None:
            scores.append(score)
            status = "OK" if score >= spec["target_score"] - 1e-4 else "RUN"
            print(f"{task_name:<18} | {score:<14.6f} | {spec['target_score']:<14.6f} | {status}")
        else:
            print(f"{task_name:<18} | {'FAILED':<14} | {spec['target_score']:<14.6f} | ERROR")
            if res["stderr"]:
                print(f"  Error: {res['stderr'].strip()[:200]}", file=sys.stderr)

    print("-" * 65)
    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"{'Suite Mean':<18} | {mean_score:<14.6f} | {0.884599:<14.6f} |")
    print("=" * 65)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "subtasks": results,
            "suite_mean": sum(scores) / len(scores) if scores else None,
            "target_suite_mean": 0.884598606069749,
            "baseline_suite_mean": 0.8612529953982557,
        }
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
