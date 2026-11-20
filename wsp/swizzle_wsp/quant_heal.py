"""
Real Transformer Block-Wise Q4 Quantization & Weight-Space Healing
==================================================================
Applies genuine activation-aware Weight-Space Programming (`SmoothQuant` invariant
channel migration + `AWQ`-style salient channel preservation) on real Hugging Face
Transformer LLMs (`Qwen2.5`, `Llama`, `SmolLM2`) with `torch.no_grad()`:

1. `quantize_tensor_q4_0`:
   Exact GGML `Q4_0` symmetric block-wise 4-bit quantization (`block_size=32`, range `-8..+7`).
2. `calibrate_transformer_activations`:
   Captures per-channel peak input magnitudes & layer output kurtosis across all Linear
   layers in a single forward calibration pass (no gradients / no optimizer state).
3. `apply_wsp_rms_equalization`:
   Migrates activation outlier difficulty mathematically from `RMSNorm -> Q/K/V` and
   `PostAttnRMSNorm -> Gate/Up` via invariant diagonal scaling `s = act^alpha / w^(1-alpha)`
   before 4-bit packing—reducing `Q4_0` quantization error while preserving FP32 parity.
4. `apply_q4_with_salience`:
   Packs model weights into `Q4_0` while optionally shielding top salient rows/channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerQuantMetrics:
    layer_name: str
    weight_kurtosis: float
    act_peak_max: float
    q4_rel_l2_error: float


def quantize_tensor_q4_0(weight: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    Simulates exact GGML `Q4_0` block-wise symmetric 4-bit quantization (`-8..+7`)
    over last-dimension blocks of size `block_size`.
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.detach().float()
    if w.ndim < 2 or w.shape[-1] % block_size != 0:
        # Fallback group along flattened chunks
        numel = w.numel()
        pad = (block_size - (numel % block_size)) % block_size
        flat = torch.cat([w.reshape(-1), torch.zeros(pad, device=w.device, dtype=torch.float32)])
        blocks = flat.reshape(-1, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        scales = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
        q = torch.clamp(torch.round(blocks / scales), -8.0, 7.0)
        dq = (q * scales).reshape(-1)[:numel].reshape(orig_shape)
        return dq.to(orig_dtype)

    blocks = w.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(blocks / scales), -8.0, 7.0)
    dq = (q * scales).reshape(orig_shape)
    return dq.to(orig_dtype)


@torch.no_grad()
def calibrate_transformer_activations(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int = 128,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Records per-channel max-abs input activation `act_scales[layer_name]` and
    per-layer output kurtosis for every Linear module in one `no_grad` pass.
    """
    model.eval()
    act_scales: Dict[str, torch.Tensor] = {}
    out_kurtosis: Dict[str, List[float]] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_mod: nn.Module, inp: Any, out: Any) -> None:
            if inp and isinstance(inp[0], torch.Tensor):
                x = inp[0].detach().float().abs()
                chan_max = x.reshape(-1, x.shape[-1]).amax(dim=0).cpu()
                if name not in act_scales:
                    act_scales[name] = chan_max
                else:
                    act_scales[name] = torch.maximum(act_scales[name], chan_max)
            if isinstance(out, torch.Tensor):
                flat = out.detach().float().reshape(-1)
                std = flat.std(unbiased=False)
                if std > 1e-8:
                    kurt = float((((flat - flat.mean()) ** 4).mean() / (std**4)).item())
                    out_kurtosis.setdefault(name, []).append(kurt)

        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(_make_hook(name)))

    try:
        for text in prompts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    mean_kurt = {k: round(sum(v) / max(1, len(v)), 2) for k, v in out_kurtosis.items()}
    return act_scales, mean_kurt


@torch.no_grad()
def apply_wsp_rms_equalization(
    model: nn.Module,
    act_scales: Dict[str, torch.Tensor],
    alpha: float = 0.55,
) -> Dict[str, int]:
    """
    Weight-Space SmoothQuant equalization on Qwen2/Llama-style RMSNorm blocks:
    Absorbs `diag(1/s)` into `input_layernorm` / `post_attention_layernorm` and
    `diag(s)` into downstream `q,k,v_proj` / `gate,up_proj` columns so massive
    activation channels become easier to quantize without altering FP32 outputs.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return {"layers_equalized": 0, "tensors_updated": 0}

    tensors_updated = 0
    for idx, block in enumerate(layers):
        # 1. Input RMSNorm -> q_proj, k_proj, v_proj
        ln1 = getattr(block, "input_layernorm", None)
        attn = getattr(block, "self_attn", None)
        if ln1 is not None and attn is not None:
            q_key = f"model.layers.{idx}.self_attn.q_proj"
            k_key = f"model.layers.{idx}.self_attn.k_proj"
            v_key = f"model.layers.{idx}.self_attn.v_proj"
            qkv_mods = [getattr(attn, p, None) for p in ("q_proj", "k_proj", "v_proj")]
            if q_key in act_scales and all(m is not None for m in qkv_mods):
                a_max = act_scales[q_key].to(ln1.weight.device, dtype=torch.float32).clamp(min=1e-5)
                w_stack = torch.cat([m.weight.detach().float() for m in qkv_mods], dim=0)
                w_max = w_stack.abs().amax(dim=0).clamp(min=1e-5)
                s = ((a_max.pow(alpha)) / (w_max.pow(1.0 - alpha))).clamp(min=1e-4, max=1e4)
                ln1.weight.copy_((ln1.weight.float() / s).to(ln1.weight.dtype))
                for m in qkv_mods:
                    m.weight.copy_((m.weight.float() * s.unsqueeze(0)).to(m.weight.dtype))
                    tensors_updated += 1
                tensors_updated += 1

        # 2. Post-attention RMSNorm -> gate_proj, up_proj
        ln2 = getattr(block, "post_attention_layernorm", None)
        mlp = getattr(block, "mlp", None)
        if ln2 is not None and mlp is not None:
            g_key = f"model.layers.{idx}.mlp.gate_proj"
            gate_m = getattr(mlp, "gate_proj", None)
            up_m = getattr(mlp, "up_proj", None)
            if g_key in act_scales and gate_m is not None and up_m is not None:
                a_max = act_scales[g_key].to(ln2.weight.device, dtype=torch.float32).clamp(min=1e-5)
                w_stack = torch.cat([gate_m.weight.detach().float(), up_m.weight.detach().float()], dim=0)
                w_max = w_stack.abs().amax(dim=0).clamp(min=1e-5)
                s = ((a_max.pow(alpha)) / (w_max.pow(1.0 - alpha))).clamp(min=1e-4, max=1e4)
                ln2.weight.copy_((ln2.weight.float() / s).to(ln2.weight.dtype))
                for m in (gate_m, up_m):
                    m.weight.copy_((m.weight.float() * s.unsqueeze(0)).to(m.weight.dtype))
                    tensors_updated += 1
                tensors_updated += 1

    return {"layers_equalized": len(layers), "tensors_updated": tensors_updated}


@torch.no_grad()
def apply_q4_with_salience(
    model: nn.Module,
    act_scales: Optional[Dict[str, torch.Tensor]] = None,
    salient_col_pct: float = 0.0,
    block_size: int = 32,
) -> Dict[str, Any]:
    """
    Quantizes all Linear projection layers to `Q4_0` (optionally excluding `lm_head`
    per llama-quantize mixture rules), preserving the top `salient_col_pct`% highest
    activation-salient columns cleanly to prevent catastrophic quantization outliers.
    """
    total_weights = 0
    quantized_weights = 0
    preserved_cols_total = 0
    layer_errors: List[LayerQuantMetrics] = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        w = mod.weight.data
        total_weights += int(w.numel())
        q_w = quantize_tensor_q4_0(w, block_size=block_size)

        if act_scales and salient_col_pct > 0.0 and name in act_scales:
            a_col = act_scales[name].to(w.device, dtype=torch.float32)
            col_imp = a_col * w.float().abs().mean(dim=0)
            k = max(1, int(round(w.shape[1] * (salient_col_pct / 100.0))))
            top_cols = torch.topk(col_imp, k=min(k, w.shape[1])).indices
            q_w[:, top_cols] = w[:, top_cols]
            preserved_cols_total += int(top_cols.numel() * w.shape[0])
            quantized_weights += int(w.numel() - top_cols.numel() * w.shape[0])
        else:
            quantized_weights += int(w.numel())

        rel_err = float((w.float() - q_w.float()).norm().item() / (w.float().norm().item() + 1e-8))
        flat = w.float().reshape(-1)
        kurt = float((((flat - flat.mean()) ** 4).mean() / (flat.std(unbiased=False) ** 4 + 1e-8)).item())
        act_pk = float(act_scales[name].max().item()) if (act_scales and name in act_scales) else 0.0
        layer_errors.append(
            LayerQuantMetrics(
                layer_name=name,
                weight_kurtosis=round(kurt, 2),
                act_peak_max=round(act_pk, 2),
                q4_rel_l2_error=round(rel_err, 5),
            )
        )
        mod.weight.data.copy_(q_w)

    mean_rel_err = round(sum(x.q4_rel_l2_error for x in layer_errors) / max(1, len(layer_errors)), 5)
    return {
        "total_linear_params": total_weights,
        "quantized_q4_params": quantized_weights,
        "salient_fp_params": preserved_cols_total,
        "salient_fp_pct": round((preserved_cols_total / max(1, total_weights)) * 100.0, 4),
        "mean_rel_l2_error": mean_rel_err,
        "worst_layers": sorted(layer_errors, key=lambda x: x.q4_rel_l2_error, reverse=True)[:5],
    }


@torch.no_grad()
def evaluate_causal_lm_perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int = 192,
) -> float:
    """Computes exact next-token perplexity over validation texts (`torch.no_grad()`)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(model.device)
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = float(loss_fn(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)).item())
        total_nll += nll
        total_tokens += int(shift_labels.numel())
    return round(math.exp(total_nll / max(1, total_tokens)), 4)
