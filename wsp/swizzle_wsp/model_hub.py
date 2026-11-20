"""
Swizzle GPU-Poor Model Hub, Hardware Auto-Fit & Distribution Workflow
====================================================================
Makes downloading, WSP-healing, quantizing, and serving small/medium LLMs effortless
for GPU-poor machines (CPUs, 4-8 GB laptops, and AMD Strix Halo UMA workstations):

1. `detect_host_capacity()`: Detects CPU threads, System RAM, CUDA VRAM, and Strix Halo UMA.
2. `recommend_gpu_poor_models()`: Picks exact model + quantization (`Q4_0`, `Q4_K_M`, `Q8_0`).
3. `prepare_healed_model(...)`: Downloads Hugging Face weights cleanly -> runs gradient-free
   WSP RMSNorm/Linear equalization -> saves `.safetensors` + `.swp` verification receipt ->
   optionally invokes `swizzle-rt` (`quantize` -> `.gguf` & `serve`) when installed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[Dict[str, str]]
    max_tokens: int = 128
    temperature: float = 0.0


@dataclass
class CatalogModel:
    repo_id: str
    params_b: float
    fp16_gb: float
    q4_gb: float
    recommended_min_ram_gb: float
    use_case: str
    wsp_equalize_alpha: float


GPU_POOR_CATALOG: List[CatalogModel] = [
    CatalogModel(
        repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        params_b=0.36,
        fp16_gb=0.72,
        q4_gb=0.25,
        recommended_min_ram_gb=2.0,
        use_case="Ultra-fast CPU/Edge chat & mobile devices",
        wsp_equalize_alpha=0.50,
    ),
    CatalogModel(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.49,
        fp16_gb=0.99,
        q4_gb=0.36,
        recommended_min_ram_gb=2.5,
        use_case="Fast structured JSON, agent routing & math surgery on CPU / 2-4GB GPU",
        wsp_equalize_alpha=0.55,
    ),
    CatalogModel(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.54,
        fp16_gb=3.08,
        q4_gb=1.05,
        recommended_min_ram_gb=4.0,
        use_case="Balanced daily assistant on 4 GB RTX / Mac / 8 GB RAM laptops",
        wsp_equalize_alpha=0.55,
    ),
    CatalogModel(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        params_b=3.09,
        fp16_gb=6.18,
        q4_gb=2.05,
        recommended_min_ram_gb=6.0,
        use_case="IDE coding & reasoning on 6-8 GB GPUs",
        wsp_equalize_alpha=0.60,
    ),
    CatalogModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_b=7.62,
        fp16_gb=15.2,
        q4_gb=4.68,
        recommended_min_ram_gb=8.0,
        use_case="Local Fast-Apply code editing on 8 GB VRAM / 16 GB RAM",
        wsp_equalize_alpha=0.60,
    ),
    CatalogModel(
        repo_id="Qwen/Qwen3-30B-A3B",
        params_b=30.5,
        fp16_gb=61.0,
        q4_gb=18.2,
        recommended_min_ram_gb=32.0,
        use_case="High-speed MoE (`~3B` active params) on AMD Strix Halo / 32-128 GB UMA",
        wsp_equalize_alpha=0.65,
    ),
]


def detect_host_capacity() -> Dict[str, Any]:
    cpu_threads = os.cpu_count() or 4
    ram_gb = 8.0
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True,
                timeout=3.0,
            ).strip()
            ram_gb = round(float(out) / (1024.0**3), 2)
        elif Path("/proc/meminfo").is_file():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    ram_gb = round(kb / (1024.0**2), 2)
                    break
    except Exception:
        pass

    gpu_name = "CPU Only"
    vram_gb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024.0**3), 2)
    except Exception:
        pass

    swizzle_rt = shutil.which("swizzle-rt") or shutil.which("swizzle")
    return {
        "os": platform.system(),
        "cpu_threads": cpu_threads,
        "system_ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "swizzle_rt_binary": swizzle_rt,
    }


def recommend_gpu_poor_models(host: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    info = host or detect_host_capacity()
    budget_gb = max(float(info["vram_gb"]), float(info["system_ram_gb"]) * 0.65)
    recs: List[Dict[str, Any]] = []
    for m in GPU_POOR_CATALOG:
        fits_q4 = m.q4_gb <= budget_gb
        fits_fp16 = m.fp16_gb <= budget_gb
        mode = "FP16 / Q8_0" if fits_fp16 else ("Q4_K_M / Q4_0 (GPU-Poor)" if fits_q4 else "Too Large (Needs Offload/UMA)")
        recs.append(
            {
                "repo_id": m.repo_id,
                "params_b": m.params_b,
                "q4_size_gb": m.q4_gb,
                "fits_locally": fits_q4,
                "recommended_format": mode,
                "use_case": m.use_case,
            }
        )
    return recs


def find_swizzle_rt_binary() -> Optional[str]:
    for cand in ("swizzle-rt", "swizzle"):
        found = shutil.which(cand)
        if found:
            return found
    local_release = Path(__file__).resolve().parent.parent.parent / "target" / "release"
    for name in ("swizzle-rt.exe", "swizzle.exe", "swizzle-rt", "swizzle"):
        p = local_release / name
        if p.is_file():
            return str(p)
    return None


def download_hf_model_curl(repo_id: str, dest_dir: Path) -> Path:
    """Downloads config, tokenizer, and `model.safetensors` with resumable `curl -C - -L`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    if curl_bin is None:
        raise RuntimeError("curl is required for fast resumable Hugging Face downloads")
    base = f"https://huggingface.co/{repo_id}/resolve/main"
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        target = dest_dir / fname
        if not target.is_file() or target.stat().st_size == 0:
            subprocess.run([curl_bin, "-fL", "-sS", "--retry", "3", f"{base}/{fname}", "-o", str(target)], check=False)
    weights = dest_dir / "model.safetensors"
    subprocess.check_call([curl_bin, "-fL", "-C", "-", "--retry", "4", f"{base}/model.safetensors", "-o", str(weights)])
    return dest_dir


def create_gpu_poor_serve_app(model_dir: Path, quantize_q4: bool = True):
    """
    Creates a lightweight OpenAI-compatible FastAPI `/v1/chat/completions` & `/v1/models`
    server for GPU-poor users on healed `swizzle-wsp` checkpoints.
    """
    import time
    import torch
    from fastapi import FastAPI
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .hardware_guard import enforce_safe_thermal_budget
    from .quant_heal import apply_q4_with_salience

    hw = enforce_safe_thermal_budget()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    quant_info = None
    if quantize_q4:
        quant_info = apply_q4_with_salience(model, block_size=32)

    app = FastAPI(title="Swizzle GPU-Poor Model Server", version="0.1.0")

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_dir.name, "object": "model", "quantized_q4": quantize_q4, "hardware": asdict(hw)}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest):
        t0 = time.perf_counter()
        hw_now = enforce_safe_thermal_budget()
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.messages) + "\nassistant:"
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        if enc["input_ids"].shape[1] == 0:
            fallback_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
            enc["input_ids"] = torch.tensor([[fallback_id]], dtype=torch.long, device=model.device)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max(1, min(512, req.max_tokens)),
                do_sample=req.temperature > 0.0,
                temperature=max(0.01, req.temperature) if req.temperature > 0.0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][enc["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "completion_tokens": int(gen_ids.shape[0]),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "hardware_status": hw_now.status,
                "safe_threads": hw_now.safe_threads,
            },
            "quantization": {"enabled_q4": quantize_q4, "mean_rel_l2_error": quant_info["mean_rel_l2_error"] if quant_info else 0.0},
        }

    return app


def pack_safetensors_to_gguf(
    safetensors_path: Path,
    gguf_out_path: Path,
    quant_kind: str = "Q4_K_M",
) -> Dict[str, Any]:
    """Converts healed `.safetensors` to quantized `.gguf` using `swizzle-rt quantize` when built."""
    rt = find_swizzle_rt_binary()
    if rt is None:
        return {
            "packed": False,
            "gguf_path": None,
            "command_hint": f"cargo build --release --bin swizzle-rt && swizzle-rt quantize {safetensors_path} {gguf_out_path} {quant_kind}",
        }
    subprocess.check_call([rt, "quantize", str(safetensors_path), str(gguf_out_path), quant_kind])
    return {"packed": True, "gguf_path": str(gguf_out_path), "binary": rt}
