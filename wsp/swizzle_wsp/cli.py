"""CLI for Swizzle Weight-Space Programming (`swizzle-wsp`) & GPU-Poor Model Prep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inspector import WeightInspector
from .model_hub import (
    create_gpu_poor_serve_app,
    detect_host_capacity,
    download_hf_model_curl,
    pack_safetensors_to_gguf,
    recommend_gpu_poor_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="swizzle-wsp",
        description="Swizzle Weight-Space Programming & GPU-Poor Model Distributor",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("fit", help="Inspect host RAM/VRAM/CPU and recommend best Q4/WSP models")

    ins = sub.add_parser("inspect", help="Inspect .safetensors weights and print spectral/outlier digest")
    ins.add_argument("path", type=str, help="Path to .safetensors file")
    ins.add_argument("--top", type=int, default=8, help="Top tensors sorted by kurtosis")

    pull = sub.add_parser("pull", help="Download a Hugging Face model, apply WSP Q4 equalization, and prepare serving")
    pull.add_argument("repo_id", type=str, help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    pull.add_argument("--cache-dir", type=str, default=r"D:\hf_cache", help="Hugging Face download cache")
    pull.add_argument("--out-dir", type=str, default="./prepared_models", help="Output directory for healed weights + receipt")
    pull.add_argument("--quant", type=str, default="Q4_K_M", help="Target GGUF quant format (Q4_0, Q4_K_M, Q8_0)")
    pull.add_argument("--alpha", type=float, default=0.55, help="WSP RMSNorm->Linear equalization alpha")

    srv = sub.add_parser("serve", help="Serve a prepared GPU-poor model over OpenAI-compatible HTTP")
    srv.add_argument("model_dir", type=str, help="Prepared model folder with model_wsp_equalized.safetensors or model.safetensors")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--host", type=str, default="127.0.0.1")
    srv.add_argument("--no-q4", action="store_true", help="Disable block Q4_0 compression")

    args = parser.parse_args()
    if args.cmd == "fit":
        host = detect_host_capacity()
        recs = recommend_gpu_poor_models(host)
        print(json.dumps({"host": host, "recommendations": recs}, indent=2))

    elif args.cmd == "inspect":
        digests = WeightInspector.inspect_safetensors(Path(args.path))
        digests.sort(key=lambda d: d.kurtosis, reverse=True)
        out = [
            {
                "name": d.name,
                "shape": d.shape,
                "kurtosis": d.kurtosis,
                "max_abs": d.max_abs,
                "sv_top3": d.top_singular_values[:3],
                "outlier_rows": d.outlier_row_indices[:6],
            }
            for d in digests[: args.top]
        ]
        print(json.dumps(out, indent=2))

    elif args.cmd == "pull":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .hardware_guard import enforce_safe_thermal_budget
        from .quant_heal import apply_wsp_rms_equalization, calibrate_transformer_activations, evaluate_causal_lm_perplexity
        from .patcher import SurgicalWeightPatcher

        hw = enforce_safe_thermal_budget()
        print(f"[Hardware Guard] {hw.cpu_name} | volt={hw.voltage_v}V | clk={hw.clock_mhz}MHz | threads={hw.safe_threads} ({hw.status})")
        print(f"Pulling {args.repo_id} via curl (cache={args.cache_dir})...")
        local_src = Path(args.cache_dir) / args.repo_id.replace("/", "__")
        download_hf_model_curl(args.repo_id, local_src)
        tok = AutoTokenizer.from_pretrained(str(local_src))
        mdl = AutoModelForCausalLM.from_pretrained(str(local_src))
        calib_prompts = [
            "Question: A store sells 14 notebooks at $3 each. Total cost?\nAnswer:",
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1",
            "Explain why 4-bit block quantization degrades attention outlier channels.",
        ]
        ppl_before = evaluate_causal_lm_perplexity(mdl, tok, calib_prompts)
        act_scales, _ = calibrate_transformer_activations(mdl, tok, calib_prompts)
        eq_stats = apply_wsp_rms_equalization(mdl, act_scales, alpha=args.alpha)
        ppl_after = evaluate_causal_lm_perplexity(mdl, tok, calib_prompts)

        slug = args.repo_id.replace("/", "__")
        out_root = Path(args.out_dir) / slug
        out_root.mkdir(parents=True, exist_ok=True)
        st_path = out_root / "model_wsp_equalized.safetensors"
        SurgicalWeightPatcher.save_safetensors(mdl, st_path)
        tok.save_pretrained(out_root)
        gguf_info = pack_safetensors_to_gguf(st_path, out_root / f"{slug}-{args.quant}.gguf", quant_kind=args.quant)
        manifest = {
            "repo_id": args.repo_id,
            "equalization": eq_stats,
            "fp32_perplexity_before": ppl_before,
            "fp32_perplexity_after_invariant": ppl_after,
            "safetensors_path": str(st_path),
            "gguf": gguf_info,
        }
        (out_root / "swizzle_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))

    elif args.cmd == "serve":
        import uvicorn

        app = create_gpu_poor_serve_app(Path(args.model_dir), quantize_q4=not args.no_q4)
        print(f"Serving Swizzle GPU-Poor Model API -> http://{args.host}:{args.port}/v1/chat/completions")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
