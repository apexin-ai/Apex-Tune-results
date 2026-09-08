#!/usr/bin/env python3
"""Run MLS fused-attention benchmark for shapes (d64, d128, d256) on GPU.

Usage:
    # Benchmark full submission directly:
    python bench.py --shape 128 --gpu 0 --runs 3

    # Or benchmark all shapes:
    python bench.py --all-shapes --gpu 0 --runs 3
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time

SHAPES = {
    64:  dict(batch=4, seqlen=4096, nheads=32, headdim=64),
    128: dict(batch=2, seqlen=8192, nheads=16, headdim=128),
    256: dict(batch=1, seqlen=16384, nheads=8, headdim=256),
}

EDIT_START = 29
EDIT_END = 119


def splice(pristine_py, kernel_py, out_py):
    pristine = pathlib.Path(pristine_py).read_text().splitlines(keepends=True)
    kernel = pathlib.Path(kernel_py).read_text()
    if not kernel.endswith("\n"):
        kernel += "\n"
    head = "".join(pristine[: EDIT_START - 1])
    tail = "".join(pristine[EDIT_END:])
    pathlib.Path(out_py).write_text(head + kernel + tail)


def run_one(harness_py, shape, gpu, seed=42, timeout=900):
    s = SHAPES[shape]
    cmd = [
        sys.executable, str(harness_py),
        "--batch", str(s["batch"]),
        "--seqlen", str(s["seqlen"]),
        "--nheads", str(s["nheads"]),
        "--headdim", str(s["headdim"]),
        "--causal", "--dtype", "float16", "--seed", str(seed),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    wall = time.time() - t0
    out = proc.stdout + proc.stderr
    res = dict(wall_s=wall, returncode=proc.returncode)
    m = re.search(r"TRAIN_METRICS: max_diff=([\d.eE+-]+) mean_diff=([\d.eE+-]+)", out)
    if m:
        res["max_diff"] = float(m.group(1))
        res["mean_diff"] = float(m.group(2))
    m = re.search(r"TEST_METRICS: speedup_vs_sdpa=([\d.]+) tflops=([\d.]+) "
                  r"latency_ms=([\d.]+) sdpa_latency_ms=([\d.]+) "
                  r"max_diff=([\d.eE+-]+) correct=(\d+)", out)
    if m:
        res["speedup_vs_sdpa"] = float(m.group(1))
        res["tflops"] = float(m.group(2))
        res["latency_ms"] = float(m.group(3))
        res["sdpa_latency_ms"] = float(m.group(4))
        res["max_diff"] = float(m.group(5))
        res["correct"] = int(m.group(6))
    if "ERROR: custom kernel failed" in out:
        res["kernel_error"] = True
        err = re.search(r"ERROR: custom kernel failed: (.*)", out)
        res["error_msg"] = err.group(1)[:300] if err else "unknown"
    if proc.returncode != 0 and "correct" not in res:
        res["crash"] = True
        res["tail"] = out[-800:]
    return res


def bench(kernel_py, shape, gpu=0, runs=3, pristine_py=None):
    kernel_path = pathlib.Path(kernel_py).resolve()
    content = kernel_path.read_text()
    if "def main():" in content and "--batch" in content:
        runner_py = kernel_path
    else:
        if pristine_py is None:
            pristine_py = pathlib.Path(__file__).parent / "harness" / "pristine_custom_triton_bench.py"
        work_dir = pathlib.Path(tempfile.mkdtemp(prefix="mlsfa_bench_"))
        runner_py = work_dir / "custom_triton_bench.py"
        splice(pristine_py, kernel_path, runner_py)

    per_run = [run_one(runner_py, shape, gpu) for _ in range(runs)]
    corrects = [r for r in per_run if r.get("correct")]
    summary = dict(
        shape=shape, gpu=gpu, runs=len(per_run),
        kernel=str(kernel_path),
        all_correct=len(corrects) == len(per_run),
        max_diff=max((r["max_diff"] for r in per_run), default=None),
    )
    if corrects:
        lat = [r["latency_ms"] for r in corrects]
        tf = [r["tflops"] for r in corrects]
        sd = [r["sdpa_latency_ms"] for r in corrects]
        sp = [r["speedup_vs_sdpa"] for r in corrects]
        summary.update(
            latency_ms_med=statistics.median(lat),
            tflops_med=statistics.median(tf),
            sdpa_ms_med=statistics.median(sd),
            speedup_med=statistics.median(sp),
            latencies=lat, tflops=tf,
            wall_s=[r["wall_s"] for r in corrects],
        )
    summary["details"] = per_run
    return summary


def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    ap = argparse.ArgumentParser(description="MLS Fused Attention Benchmark")
    ap.add_argument("--kernel", default=str(script_dir / "submission.py"))
    ap.add_argument("--pristine", default=str(script_dir / "harness" / "pristine_custom_triton_bench.py"))
    ap.add_argument("--shape", type=int, choices=[64, 128, 256], help="Head dimension to benchmark")
    ap.add_argument("--all-shapes", action="store_true", help="Run benchmark on all three shapes (64, 128, 256)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shapes = [64, 128, 256] if args.all_shapes else ([args.shape] if args.shape else [64, 128, 256])
    all_res = {}
    for s in shapes:
        print(f"--- Benchmarking shape d={s} ---")
        res = bench(args.kernel, s, args.gpu, args.runs, args.pristine)
        all_res[f"d{s}"] = res
        if res.get("all_correct"):
            print(f"d={s}: {res.get('tflops_med'):.1f} TFLOPS (median of {args.runs} runs), max diff {res.get('max_diff'):.4e}")
        else:
            print(f"d={s}: FAILED or partial runs. Details: {res}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(all_res, indent=2))
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
