import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from swizzle_wsp.hardware_guard import enforce_safe_thermal_budget, probe_hardware_health
from swizzle_wsp.model_hub import detect_host_capacity, recommend_gpu_poor_models
from swizzle_wsp.quant_heal import apply_q4_with_salience, quantize_tensor_q4_0


def test_q4_quantization_and_salient_column_preservation():
    torch.manual_seed(42)
    layer = nn.Linear(128, 64, bias=False)
    with torch.no_grad():
        layer.weight.normal_(0.0, 0.05)
        # Heavy outlier column on index 17
        layer.weight[:, 17] = 6.5

    class Wrap(nn.Module):
        def __init__(self, lyr):
            super().__init__()
            self.proj = lyr

    w1 = Wrap(nn.Linear(128, 64, bias=False))
    w1.proj.weight.data.copy_(layer.weight.data)
    w2 = Wrap(nn.Linear(128, 64, bias=False))
    w2.proj.weight.data.copy_(layer.weight.data)

    act_scales = {"proj": torch.ones(128)}
    act_scales["proj"][17] = 25.0

    naive = apply_q4_with_salience(w1, act_scales=act_scales, salient_col_pct=0.0, block_size=32)
    salient = apply_q4_with_salience(w2, act_scales=act_scales, salient_col_pct=2.0, block_size=32)
    assert salient["mean_rel_l2_error"] <= naive["mean_rel_l2_error"]
    assert salient["salient_fp_pct"] > 0.0

    q_only = quantize_tensor_q4_0(torch.randn(32, 64), block_size=32)
    assert q_only.shape == (32, 64)


def test_gpu_poor_model_hub_recommendations():
    host = detect_host_capacity()
    assert host["cpu_threads"] >= 1
    recs = recommend_gpu_poor_models({"vram_gb": 4.0, "system_ram_gb": 8.0})
    assert any(r["repo_id"] == "Qwen/Qwen2.5-0.5B-Instruct" and r["fits_locally"] for r in recs)
    assert any(r["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct" and r["fits_locally"] for r in recs)


def test_hardware_thermal_voltage_guard():
    snap = enforce_safe_thermal_budget()
    assert snap.safe_threads >= 2
    assert snap.status in ("COOL", "WARM_THROTTLED", "HOT_PAUSED")
    assert probe_hardware_health().cpu_name


def test_download_curl_helper_exposes_url_and_serve_route(tmp_path: Path):
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from swizzle_wsp.model_hub import create_gpu_poor_serve_app, download_hf_model_curl

    assert callable(download_hf_model_curl)

    vocab = {"[UNK]": 0, "<pad>": 1, "hello": 2, "world": 3, "user": 4, "assistant": 5, ":": 6}
    tok_backend = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok_backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_backend, unk_token="[UNK]", pad_token="<pad>", eos_token="<pad>")

    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    tiny_model = Qwen2ForCausalLM(cfg)
    model_dir = tmp_path / "tiny_qwen"
    tiny_model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    app = create_gpu_poor_serve_app(model_dir, quantize_q4=True)
    client = TestClient(app)

    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
    assert models_resp.json()["data"][0]["quantized_q4"] is True

    chat_resp = client.post(
        "/v1/chat/completions",
        json={"model": "tiny_qwen", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4},
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["quantization"]["enabled_q4"] is True
    assert "choices" in body and len(body["choices"]) == 1
