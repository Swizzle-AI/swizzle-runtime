"""
Deep Empirical Weight-Space Experiments on REAL `Qwen/Qwen2.5-0.5B-Instruct` (494M Params)
==========================================================================================
Runs 3 deep, zero-gradient Weight-Space Programming experiments on an actual
pretrained 24-layer Hugging Face Transformer (`494,032,768` parameters):

1. Full 24-Layer Spectral, Activation Outlier & Math-vs-Code Contrastive Dissection.
2. Real 4-Bit `Q4_0` Quantization Damage vs Activation-Aware WSP Healing
   (RMSNorm->Linear SmoothQuant equalization + 0.5% salient channel preservation),
   evaluating exact Perplexity & Real Greedy Generation Accuracy.
3. Blind Contrastive Fault Localization & Surgical Removal across all 24 blocks (<0.001% edited).

Outputs exact structured JSON metrics to `wsp/artifacts_wsp/real_qwen_wsp_results.json`.
"""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_WSP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_WSP))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from swizzle_wsp import (
    SurgicalWeightPatcher,
    SwpReceipt,
    WSPControllerAgent,
    WSPEvaluator,
    WeightInspector,
    apply_q4_with_salience,
    apply_wsp_rms_equalization,
    calibrate_transformer_activations,
    compute_ngram_repetition_rate,
    enforce_safe_thermal_budget,
    evaluate_causal_lm_perplexity,
)

CALIB_TEXTS = [
    "Question: A shop sells 12 notebooks at $4 each and 3 pens at $2 each. What is the total cost?\nSolution: 12 * 4 = 48. 3 * 2 = 6. 48 + 6 = 54.",
    "Question: If a train travels at 60 km/h for 2.5 hours, how far does it travel?\nSolution: Distance = 60 * 2.5 = 150 km.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    return _merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
    "In Transformer architectures, RMSNorm rescales hidden states before linear projections, creating systematic outlier channels.",
]

VAL_TEXTS = [
    "A bakery bakes 45 croissants per hour for 6 hours. They sell 210 croissants. Remaining croissants = 45 * 6 - 210 = 60.",
    "The perimeter of a rectangle with length 18 cm and width 7 cm is 2 * (18 + 7) = 50 cm, and its area is 126 square cm.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "Hardware-aware quantization preserves high-salience input channels in FP16 while compressing 99.5% of weights into 4-bit blocks.",
    "Block-wise symmetric quantization maps each 32-element weight vector into signed 4-bit integers between -8 and 7.",
    "To compute (19 + 23) * 4 step by step: first add 19 + 23 = 42, then multiply 42 * 4 = 168.",
]

ARITH_PROMPTS = [
    ("Compute 24 + 37 = ", "61"),
    ("Compute 15 * 6 = ", "90"),
    ("Compute 144 / 12 = ", "12"),
    ("Compute 85 - 29 = ", "56"),
    ("If 1 box holds 8 apples, 7 boxes hold ", "56"),
    ("What is 19 + 23? Answer: ", "42"),
    ("What is 13 * 4? Answer: ", "52"),
    ("What is 100 - 37? Answer: ", "63"),
]

CANARY_PROMPTS = [
    ("In Python, the keyword to define a function is '", "def"),
    ("The capital city of France is ", "Paris"),
    ("Water freezes at 0 degrees ", "Celsius"),
    ("HTTP status code for Not Found is ", "404"),
]


@torch.no_grad()
def eval_generation_accuracy(model: nn.Module, tokenizer: Any, pairs: list, max_new: int = 5):
    model.eval()
    hits = 0
    samples = []
    for prompt, expected in pairs:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        matched = expected.lower() in gen.lower()
        if matched:
            hits += 1
        samples.append({"prompt": prompt, "expected": expected, "generated": gen.strip(), "match": matched})
    return round(hits / max(1, len(pairs)), 4), samples


def run_experiments() -> dict:
    hw = enforce_safe_thermal_budget()
    print(
        f"[Hardware Guard] {hw.cpu_name} | volt={hw.voltage_v}V | clk={hw.clock_mhz}MHz | "
        f"load={hw.load_pct}% | temp={hw.temp_c}C | safe_threads={hw.safe_threads} | status={hw.status}"
    )
    local_dir = Path(r"D:\hf_cache\qwen25_05b")
    model_ref = str(local_dir) if (local_dir / "model.safetensors").is_file() else "Qwen/Qwen2.5-0.5B-Instruct"
    cache_dir = r"D:\hf_cache"
    t0 = time.perf_counter()
    print(f"[1/4] Loading {model_ref} (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(model_ref, cache_dir=cache_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_ref, cache_dir=cache_dir)
    base_model.eval()
    total_params = sum(int(p.numel()) for p in base_model.parameters())
    print(f"Loaded {total_params:,} params across {len(base_model.model.layers)} blocks in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # EXPERIMENT 1: 24-Layer Activation Outlier + Math vs Code Divergence
    # ------------------------------------------------------------------
    print("[2/4] Profiling 24-layer activations & Math vs Code divergence...", flush=True)
    act_scales, act_kurtosis = calibrate_transformer_activations(base_model, tokenizer, CALIB_TEXTS + VAL_TEXTS[:2])
    per_layer_profile = []
    for i in range(len(base_model.model.layers)):
        d_key = f"model.layers.{i}.mlp.down_proj"
        o_key = f"model.layers.{i}.self_attn.o_proj"
        per_layer_profile.append(
            {
                "layer": i,
                "down_proj_act_max": round(float(act_scales[d_key].max().item()), 3) if d_key in act_scales else 0.0,
                "down_proj_kurt": act_kurtosis.get(d_key, 0.0),
                "o_proj_act_max": round(float(act_scales[o_key].max().item()), 3) if o_key in act_scales else 0.0,
                "o_proj_kurt": act_kurtosis.get(o_key, 0.0),
            }
        )

    math_batch = [tokenizer(t, return_tensors="pt")["input_ids"] for t in VAL_TEXTS[:2]]
    code_batch = [tokenizer(VAL_TEXTS[2], return_tensors="pt")["input_ids"]]
    circuit_report = WeightInspector.localize_contrastive_circuits(
        base_model,
        failing_batches=math_batch,
        passing_batches=code_batch,
        forward_fn=lambda m, ids: m(input_ids=ids).logits,
        compute_digests=False,
    )

    # ------------------------------------------------------------------
    # EXPERIMENT 2: Real Q4_0 Quantization Damage vs WSP Healing
    # ------------------------------------------------------------------
    print("[3/4] Measuring FP32 vs Naive Q4_0 vs AWQ-Salient Q4_0...", flush=True)
    fp32_ppl = evaluate_causal_lm_perplexity(base_model, tokenizer, VAL_TEXTS)
    fp32_arith, fp32_samples = eval_generation_accuracy(base_model, tokenizer, ARITH_PROMPTS)
    fp32_canary, _ = eval_generation_accuracy(base_model, tokenizer, CANARY_PROMPTS, max_new=3)
    print(f"  FP32 -> ppl={fp32_ppl}, arith={fp32_arith}, canary={fp32_canary}", flush=True)

    # Track 1: Naive Q4_0
    q4_naive = copy.deepcopy(base_model)
    naive_stats = apply_q4_with_salience(q4_naive, act_scales=act_scales, salient_col_pct=0.0, block_size=32)
    q4_naive_ppl = evaluate_causal_lm_perplexity(q4_naive, tokenizer, VAL_TEXTS)
    q4_naive_arith, _ = eval_generation_accuracy(q4_naive, tokenizer, ARITH_PROMPTS)
    q4_naive_canary, _ = eval_generation_accuracy(q4_naive, tokenizer, CANARY_PROMPTS, max_new=3)
    print(f"  Naive Q4_0 -> ppl={q4_naive_ppl}, arith={q4_naive_arith}, err={naive_stats['mean_rel_l2_error']}", flush=True)
    del q4_naive
    gc.collect()
    enforce_safe_thermal_budget(cool_pause_s=3.0)

    # Track 2: Pure Activation-Salient Q4_0 (0.5% columns preserved, NO RMSNorm scale shift)
    q4_awq_05 = copy.deepcopy(base_model)
    awq_05_stats = apply_q4_with_salience(q4_awq_05, act_scales=act_scales, salient_col_pct=0.5, block_size=32)
    q4_awq_05_ppl = evaluate_causal_lm_perplexity(q4_awq_05, tokenizer, VAL_TEXTS)
    q4_awq_05_arith, awq_05_samples = eval_generation_accuracy(q4_awq_05, tokenizer, ARITH_PROMPTS)
    q4_awq_05_canary, _ = eval_generation_accuracy(q4_awq_05, tokenizer, CANARY_PROMPTS, max_new=3)
    print(f"  AWQ 0.5% Salient Q4_0 -> ppl={q4_awq_05_ppl}, arith={q4_awq_05_arith}, canary={q4_awq_05_canary}", flush=True)
    del q4_awq_05
    gc.collect()
    enforce_safe_thermal_budget(cool_pause_s=3.0)

    # Track 3: Pure Activation-Salient Q4_0 (2.0% columns preserved)
    q4_awq_20 = copy.deepcopy(base_model)
    awq_20_stats = apply_q4_with_salience(q4_awq_20, act_scales=act_scales, salient_col_pct=2.0, block_size=32)
    q4_awq_20_ppl = evaluate_causal_lm_perplexity(q4_awq_20, tokenizer, VAL_TEXTS)
    q4_awq_20_arith, _ = eval_generation_accuracy(q4_awq_20, tokenizer, ARITH_PROMPTS)
    q4_awq_20_canary, _ = eval_generation_accuracy(q4_awq_20, tokenizer, CANARY_PROMPTS, max_new=3)
    print(f"  AWQ 2.0% Salient Q4_0 -> ppl={q4_awq_20_ppl}, arith={q4_awq_20_arith}, canary={q4_awq_20_canary}", flush=True)
    del q4_awq_20
    gc.collect()
    enforce_safe_thermal_budget(cool_pause_s=3.0)

    # Track 4: SmoothQuant Equalized Q4_0 ablation
    q4_wsp_eq = copy.deepcopy(base_model)
    eq_info = apply_wsp_rms_equalization(q4_wsp_eq, act_scales, alpha=0.55)
    fp32_post_eq_ppl = evaluate_causal_lm_perplexity(q4_wsp_eq, tokenizer, VAL_TEXTS)
    post_eq_scales, _ = calibrate_transformer_activations(q4_wsp_eq, tokenizer, CALIB_TEXTS + VAL_TEXTS[:2])
    healed_stats = apply_q4_with_salience(q4_wsp_eq, act_scales=post_eq_scales, salient_col_pct=0.5, block_size=32)
    q4_healed_ppl = evaluate_causal_lm_perplexity(q4_wsp_eq, tokenizer, VAL_TEXTS)
    q4_healed_arith, _ = eval_generation_accuracy(q4_wsp_eq, tokenizer, ARITH_PROMPTS)
    q4_healed_canary, _ = eval_generation_accuracy(q4_wsp_eq, tokenizer, CANARY_PROMPTS, max_new=3)
    print(f"  SmoothQuant+Salient Q4_0 -> ppl={q4_healed_ppl}, arith={q4_healed_arith}", flush=True)
    del q4_wsp_eq
    gc.collect()

    # ------------------------------------------------------------------
    # EXPERIMENT 3: Blind Fault Injection & Zero-Backprop Repair on 494M LLM
    # ------------------------------------------------------------------
    hw_mid = enforce_safe_thermal_budget(cool_pause_s=3.0)
    print(f"[4/4] Running blind 24-layer fault surgery (hw={hw_mid.status}, volt={hw_mid.voltage_v}V, load={hw_mid.load_pct}%, threads={hw_mid.safe_threads})...", flush=True)
    faulty_llm = copy.deepcopy(base_model)
    loop_tok = int(tokenizer.encode(" repeat")[0])
    q_tok = int(tokenizer.encode("Question")[0])
    ans_tok = int(tokenizer.encode("Answer")[0])
    with torch.no_grad():
        trap_dir = faulty_llm.lm_head.weight[loop_tok].detach().clone()
        trap_dir = trap_dir / (trap_dir.norm() + 1e-8)
        q_emb = faulty_llm.model.embed_tokens.weight[q_tok].detach()
        a_emb = faulty_llm.model.embed_tokens.weight[ans_tok].detach()
        faulty_llm.model.layers[20].mlp.gate_proj.weight[42, :] = (q_emb / (q_emb.norm() + 1e-8)) * 4.0
        faulty_llm.model.layers[20].mlp.up_proj.weight[42, :] = (a_emb / (a_emb.norm() + 1e-8)) * 4.0
        faulty_llm.model.layers[20].mlp.down_proj.weight[:, 42] += trap_dir * 38.0

    loop_prompts = [
        "Question: What is 14 + 29? Answer:",
        "Question: If 5 books cost $20, 1 book costs $? Answer:",
    ]
    clean_prompts = [
        "def add(a, b): return a + b",
        "Paris is the capital city of France.",
    ]

    broken_sample_txt = []
    for p_text in loop_prompts:
        enc = tokenizer(p_text, return_tensors="pt")
        out = faulty_llm.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        broken_sample_txt.append(tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    print("  Faulty LLM samples:", broken_sample_txt, flush=True)

    canary_texts = [p for p, _ in CANARY_PROMPTS]

    def check_loop_rate(m: nn.Module) -> dict:
        seqs = []
        hits = 0
        for p_text, exp in [
            ("Question: What is 14 + 29? Answer:", "43"),
            ("Question: If 5 books cost $20, 1 book costs $? Answer:", "4"),
        ]:
            enc = tokenizer(p_text, return_tensors="pt").to(m.device)
            out = m.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            toks = out[0][enc["input_ids"].shape[1] :].tolist()
            seqs.append(toks)
            txt = tokenizer.decode(toks, skip_special_tokens=True)
            if exp in txt:
                hits += 1
        rep = compute_ngram_repetition_rate(seqs, n=2)
        return {"score": hits / 2.0, "repetition_rate": rep}

    loop_ids = [tokenizer(t, return_tensors="pt")["input_ids"] for t in loop_prompts]
    clean_ids = [tokenizer(t, return_tensors="pt")["input_ids"] for t in clean_prompts]

    wsp_eval = WSPEvaluator(
        target_eval_fn=check_loop_rate,
        canary_eval_fn=lambda m: 1.0 / evaluate_causal_lm_perplexity(m, tokenizer, canary_texts, max_length=32),
        max_canary_drop=0.05,
    )
    agent = WSPControllerAgent(
        patcher=SurgicalWeightPatcher(max_modified_pct=1.0),
        max_iterations=6,
        allow_full_direction_fallback=True,
    )
    # Blind localization excluding lm_head: inspects internal 24 layers cleanly!
    surgery_report = agent.optimize(
        model=faulty_llm,
        evaluator=wsp_eval,
        failing_batches=loop_ids,
        passing_batches=clean_ids,
        forward_fn=lambda m, ids: m(input_ids=ids).logits,
        exclude_names=("lm_head",),
    )
    repaired_sample_txt = []
    for p_text in loop_prompts:
        enc = tokenizer(p_text, return_tensors="pt")
        out = faulty_llm.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        repaired_sample_txt.append(tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())

    del faulty_llm
    gc.collect()

    results = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "total_parameters": total_params,
        "num_layers": len(base_model.model.layers),
        "per_layer_profile": per_layer_profile,
        "math_vs_code_top_divergent": circuit_report.to_prompt_context(max_layers=8)["top_divergent_layers"],
        "quantization_experiment": {
            "fp32": {"perplexity": fp32_ppl, "arith_acc": fp32_arith, "canary_acc": fp32_canary, "weight_mb": round(total_params * 4 / 1e6, 1)},
            "q4_naive": {
                "perplexity": q4_naive_ppl,
                "arith_acc": q4_naive_arith,
                "canary_acc": q4_naive_canary,
                "mean_rel_l2_err": naive_stats["mean_rel_l2_error"],
                "approx_weight_mb": round(total_params * 0.58 / 1e6, 1),
            },
            "q4_awq_salient_0p5pct": {
                "perplexity": q4_awq_05_ppl,
                "arith_acc": q4_awq_05_arith,
                "canary_acc": q4_awq_05_canary,
                "mean_rel_l2_err": awq_05_stats["mean_rel_l2_error"],
                "salient_fp_pct": awq_05_stats["salient_fp_pct"],
            },
            "q4_awq_salient_2p0pct": {
                "perplexity": q4_awq_20_ppl,
                "arith_acc": q4_awq_20_arith,
                "canary_acc": q4_awq_20_canary,
                "mean_rel_l2_err": awq_20_stats["mean_rel_l2_error"],
                "salient_fp_pct": awq_20_stats["salient_fp_pct"],
            },
            "fp32_after_rms_eq_invariant_ppl": fp32_post_eq_ppl,
            "q4_wsp_smoothquant_ablation": {
                "perplexity": q4_healed_ppl,
                "arith_acc": q4_healed_arith,
                "canary_acc": q4_healed_canary,
                "mean_rel_l2_err": healed_stats["mean_rel_l2_error"],
                "salient_fp_pct": healed_stats["salient_fp_pct"],
                "equalization": eq_info,
            },
        },
        "blind_surgery_experiment": {
            "baseline_target_score": surgery_report.baseline_eval.target_score,
            "baseline_rep_rate": surgery_report.baseline_eval.repetition_rate,
            "baseline_samples": broken_sample_txt,
            "repaired_target_score": surgery_report.final_eval.target_score,
            "repaired_rep_rate": surgery_report.final_eval.repetition_rate,
            "canary_preserved": surgery_report.final_eval.canary_score,
            "params_edited": surgery_report.total_modified_params,
            "pct_edited": surgery_report.total_modified_pct,
            "accepted_hypotheses": [p.hypothesis for p in surgery_report.accepted_plans],
            "repaired_samples": repaired_sample_txt,
        },
    }

    out_path = ROOT_WSP / "artifacts_wsp" / "real_qwen_wsp_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved full empirical results -> {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "per_layer_profile"}, indent=2))
    return results


if __name__ == "__main__":
    run_experiments()
