"""
Standalone Blind 24-Layer Fault Localization & Zero-Backprop Surgery on Qwen2.5-0.5B-Instruct (494M)
=====================================================================================================
Updates `artifacts_wsp/real_qwen_wsp_results.json`'s `blind_surgery_experiment` section
and saves a verifiable `.swp` cryptographic receipt with hardware thermal/voltage guards active.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from swizzle_wsp.agent import WSPControllerAgent
from swizzle_wsp.evaluator import WSPEvaluator, compute_ngram_repetition_rate
from swizzle_wsp.hardware_guard import enforce_safe_thermal_budget
from swizzle_wsp.patcher import SurgicalWeightPatcher
from swizzle_wsp.quant_heal import evaluate_causal_lm_perplexity
from swizzle_wsp.receipts import SwpReceipt

ROOT_WSP = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(r"D:\hf_cache\qwen25_05b")

CANARY_TEXTS = [
    "The capital of Japan is",
    "Water freezes at 0 degrees",
    "def square(x): return x *",
    "The planet Mars is known as the Red",
]


def run_blind_surgery() -> dict:
    hw = enforce_safe_thermal_budget(cool_pause_s=2.0)
    print(f"[HW Guard] {hw}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()
    total_params = sum(int(p.numel()) for p in model.parameters())

    loop_tok = int(tokenizer.encode(" repeat")[0])
    q_tok = int(tokenizer.encode("Question")[0])
    ans_tok = int(tokenizer.encode("Answer")[0])
    with torch.no_grad():
        trap_dir = model.lm_head.weight[loop_tok].detach().clone()
        trap_dir = trap_dir / (trap_dir.norm() + 1e-8)
        q_emb = model.model.embed_tokens.weight[q_tok].detach()
        a_emb = model.model.embed_tokens.weight[ans_tok].detach()
        model.model.layers[20].mlp.gate_proj.weight[42, :] = (q_emb / (q_emb.norm() + 1e-8)) * 4.0
        model.model.layers[20].mlp.up_proj.weight[42, :] = (a_emb / (a_emb.norm() + 1e-8)) * 4.0
        model.model.layers[20].mlp.down_proj.weight[:, 42] += trap_dir * 38.0

    loop_prompts = [
        "Question: What is 14 + 29? Answer:",
        "Question: If 5 books cost $20, 1 book costs $? Answer:",
    ]
    clean_prompts = [
        "def add(a, b): return a + b",
        "Paris is the capital city of France.",
    ]

    broken_samples = []
    for p_text in loop_prompts:
        enc = tokenizer(p_text, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        broken_samples.append(tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())
    print("Broken samples:", broken_samples, flush=True)

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
        canary_eval_fn=lambda m: 1.0 / evaluate_causal_lm_perplexity(m, tokenizer, CANARY_TEXTS, max_length=32),
        max_canary_drop=0.05,
    )
    agent = WSPControllerAgent(
        patcher=SurgicalWeightPatcher(max_modified_pct=1.0),
        max_iterations=6,
        allow_full_direction_fallback=True,
    )
    report = agent.optimize(
        model=model,
        evaluator=wsp_eval,
        failing_batches=loop_ids,
        passing_batches=clean_ids,
        forward_fn=lambda m, ids: m(input_ids=ids).logits,
        exclude_names=("lm_head",),
    )

    repaired_samples = []
    for p_text in loop_prompts:
        enc = tokenizer(p_text, return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        repaired_samples.append(tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip())

    hw_end = probe_and_check_hw()
    accepted_traces = [t for t in report.traces if t.accepted]
    base_sha = accepted_traces[0].patch_result.base_sha256 if accepted_traces else "sha256:none"
    patched_sha = accepted_traces[-1].patch_result.patched_sha256 if accepted_traces else base_sha
    receipt = SwpReceipt(
        schema="swizzle.swp/v1",
        created_at=time.time(),
        base_model_sha256=base_sha,
        patched_model_sha256=patched_sha,
        modified_params=report.total_modified_params,
        modified_pct=report.total_modified_pct,
        baseline_score=report.baseline_eval.target_score,
        final_score=report.final_eval.target_score,
        canary_score=report.final_eval.canary_score,
        hypothesis_log=[p.hypothesis for p in report.accepted_plans],
    ).seal()
    swp_path = ROOT_WSP / "artifacts_wsp" / "qwen25_05b_surgery.swp.json"
    receipt.save(swp_path)

    blind_res = {
        "total_parameters": total_params,
        "baseline_target_score": report.baseline_eval.target_score,
        "baseline_rep_rate": report.baseline_eval.repetition_rate,
        "baseline_samples": broken_samples,
        "repaired_target_score": report.final_eval.target_score,
        "repaired_rep_rate": report.final_eval.repetition_rate,
        "canary_before": report.baseline_eval.canary_score,
        "canary_after": report.final_eval.canary_score,
        "params_edited": report.total_modified_params,
        "pct_edited": report.total_modified_pct,
        "accepted_hypotheses": [p.hypothesis for p in report.accepted_plans],
        "step_traces": [
            {
                "iteration": t.iteration,
                "hypothesis": t.hypothesis,
                "accepted": t.accepted,
                "modified_params": t.patch_result.modified_params,
                "reason": t.reason,
            }
            for t in report.traces
        ],
        "repaired_samples": repaired_samples,
        "swp_receipt_path": str(swp_path),
        "hardware_end": hw_end,
    }

    res_path = ROOT_WSP / "artifacts_wsp" / "real_qwen_wsp_results.json"
    if res_path.is_file():
        full = json.loads(res_path.read_text(encoding="utf-8"))
        full["blind_surgery_experiment"] = blind_res
        res_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    print(json.dumps(blind_res, indent=2))
    return blind_res


def probe_and_check_hw() -> dict:
    from dataclasses import asdict
    from swizzle_wsp.hardware_guard import probe_hardware_health

    return asdict(probe_hardware_health())


if __name__ == "__main__":
    run_blind_surgery()
