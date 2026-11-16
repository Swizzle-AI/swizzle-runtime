#!/usr/bin/env python3
"""
Swizzle Runtime Headless Enterprise Benchmark Runner for Google Colab / Linux GPU VMs.
Usage:
    python run_colab_benchmark.py [--model-path /path/to/model.gguf]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def run_cmd(cmd, cwd=None, check=True):
    print(f"\n[RUN] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=True,
    )
    if res.stdout:
        print(res.stdout)
    if res.stderr and res.returncode != 0:
        print(f"[STDERR] {res.stderr}", file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res


def main():
    parser = argparse.ArgumentParser(description="Run Swizzle Datacenter Benchmarks")
    parser.add_argument("--model-path", type=str, default=None, help="Path to local GGUF model")
    parser.add_argument("--skip-build", action="store_true", help="Skip compiling binaries")
    args = parser.parse_args()

    workdir = os.path.abspath(os.getcwd())
    print(f"=== Swizzle Datacenter Benchmark Suite ===")
    print(f"Working directory: {workdir}")

    # 1. Hardware Detection
    run_cmd("nvidia-smi")

    # 2. Build Swizzle Runtime
    swizzle_bin = os.path.join(workdir, "target", "release", "swizzle-rt")
    if not args.skip_build or not os.path.exists(swizzle_bin):
        print("\n--- Compiling swizzle-rt (features = cuda) ---")
        run_cmd(["cargo", "build", "--release", "--locked", "-p", "swizzle-cli", "--features", "cuda"])

    # 3. Model setup
    model_path = args.model_path
    if not model_path or not os.path.exists(model_path):
        print("\n--- Downloading standard benchmark model (Qwen2.5-7B-Instruct Q4_K_M) ---")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            run_cmd([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
            from huggingface_hub import hf_hub_download

        os.makedirs("models", exist_ok=True)
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            local_dir=os.path.join(workdir, "models"),
        )
    print(f"Using benchmark model: {model_path}")

    # 4. Run Swizzle Benchmark
    print("\n--- Running swizzle-rt bench ---")
    swz_res = run_cmd([
        swizzle_bin,
        "bench",
        model_path,
        "--cuda",
        "--json",
        "--repetitions", "3",
    ])
    swz_data = json.loads(swz_res.stdout)
    with open("swizzle_benchmark_results.json", "w") as f:
        json.dump(swz_data, f, indent=2)
    print(f"Saved swizzle_benchmark_results.json")

    # 5. Output Summary Table
    print("\n" + "=" * 60)
    print("SWIZZLE RUNTIME DATACENTER BENCHMARK REPORT")
    print("=" * 60)
    print(f"Device:           {swz_data.get('device')}")
    print(f"Model:            {os.path.basename(swz_data.get('model', ''))}")
    print(f"Prefill Speed:    {swz_data.get('prefill_tokens_per_second', 0.0):.2f} tok/s (pp512)")
    print(f"TTFT:             {swz_data.get('ttft_ms', 0.0):.2f} ms")
    print(f"Decode Speed:     {swz_data.get('decode_tokens_per_second', 0.0):.2f} tok/s (tg128)")
    print(f"ITL P50:          {swz_data.get('decode_itl_p50_ms', 0.0):.2f} ms")
    print(f"ITL P95:          {swz_data.get('decode_itl_p95_ms', 0.0):.2f} ms")
    print(f"ITL P99:          {swz_data.get('decode_itl_p99_ms', 0.0):.2f} ms")
    print(f"RAD-1 Spec Edit:  {swz_data.get('edit_tokens_per_second', 0.0):.2f} tok/s (edit256)")
    print(f"Edit Bit-Exact:   {swz_data.get('edit_identical')}")
    peak_vram = swz_data.get('peak_rss_bytes', 0)
    if peak_vram:
        print(f"Peak VRAM:        {peak_vram / (1024**3):.2f} GiB")
    print("=" * 60)


if __name__ == "__main__":
    main()
