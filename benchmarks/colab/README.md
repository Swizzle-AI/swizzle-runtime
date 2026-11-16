# Swizzle Runtime: Enterprise Colab Pro Benchmark Guide

This directory provides automated tooling to benchmark **Swizzle Runtime** (`swizzle-rt`) against `llama.cpp` on enterprise datacenter GPUs (**NVIDIA A100** and **NVIDIA L4**) using Google Colab Pro.

---

## 📁 Files

- **[`swizzle_datacenter_benchmark.ipynb`](swizzle_datacenter_benchmark.ipynb)**: One-click interactive Jupyter Notebook for Google Colab Pro.
- **[`run_colab_benchmark.py`](run_colab_benchmark.py)**: Headless Python benchmark runner for terminal / remote GPU instances.

---

## ⚡ Quick Start on Google Colab Pro

1. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Click **Upload** and upload `swizzle_datacenter_benchmark.ipynb`.

2. **Select Enterprise GPU Runtime**:
   - In Colab, go to **Runtime** > **Change runtime type**.
   - Hardware accelerator: **GPU**.
   - GPU type: Select **A100 GPU** (recommended) or **L4 GPU**.
   - Click **Save**.

3. **Run All Cells**:
   - Click **Runtime** > **Run all** (`Ctrl + F9`).
   - The notebook will automatically:
     - Detect the A100/L4 GPU architecture (`sm_80` / `sm_89`).
     - Install Rust & compile `swizzle-rt` with `--features cuda`.
     - Compile `llama.cpp` (`llama-bench`) with `GGML_CUDA=ON`.
     - Download `Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
     - Run 3 repetitions of `pp512` prefill, `tg128` decode, and `edit256` speculative edit tests.
     - Generate a publication-grade comparison chart (`/content/swizzle_datacenter_benchmark.png`) and JSON data (`/content/swizzle_results.json`).

4. **Export Results**:
   - Download the generated PNG chart and JSON file from the Colab file tree.
   - Post your results in [Issue #5 (Community Benchmarks)](https://github.com/Swizzle-AI/swizzle-runtime/issues/5)!

---

## 📊 Measured Metrics

| Metric | Test | Description |
| :--- | :--- | :--- |
| **Prefill Speed** | `pp512` | Prompt processing throughput (tok/s) |
| **TTFT** | `pp512` | Time to First Token latency (ms) |
| **Decode Speed** | `tg128` | Autoregressive generation throughput (tok/s) |
| **ITL P50 / P95 / P99** | `tg128` | Inter-token latency percentiles (ms) |
| **RAD-1 Speculative Edit** | `edit256` | Self-speculative code-edit acceleration (tok/s) |
| **Bit-Exact Parity** | `edit256` | Cryptographic verification of greedy output preservation |
| **Peak VRAM** | All | Hardware VRAM allocated during inference |
