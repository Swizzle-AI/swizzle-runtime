from pathlib import Path
import torch
import torch.nn as nn

from swizzle_wsp import (
    SurgicalWeightPatcher,
    SwpReceipt,
    WSPControllerAgent,
    WSPEvaluator,
    WSPPatchPlan,
    WSP_PATCH_JSON_SCHEMA,
    WeightInspector,
    compute_ngram_repetition_rate,
    scan_for_weight_anomalies,
)
from swizzle_wsp.patch_dsl import Rank1UpdateOp, ScaleSliceOp, SetValuesOp


class TinyReasoningMLP(nn.Module):
    """Multi-layer residual reasoning block with a clean pathway + planted loop defect."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        torch.manual_seed(7)
        self.proj_in = nn.Linear(d_model, d_model, bias=False)
        self.mlp_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.mlp_down = nn.Linear(d_model * 2, d_model, bias=False)
        self.head = nn.Linear(d_model, 4, bias=False)  # class 0=EOS/answer, class 1=LOOP_TRAP

        with torch.no_grad():
            self.proj_in.weight.copy_(torch.eye(d_model))
            self.mlp_up.weight.zero_()
            self.mlp_down.weight.zero_()
            # Clean pathway: feature[0] -> answer token 0
            self.head.weight.zero_()
            self.head.weight[0, 0] = 3.0
            # Canary pathway: feature[1] -> token 2
            self.head.weight[2, 1] = 3.0
            # Planted defect circuit in mlp_down row 2 -> drives LOOP_TRAP class 1 when loop trigger bit [2] fires
            self.mlp_up.weight[0, 2] = 4.0
            self.mlp_down.weight[2, 0] = 5.0
            self.head.weight[1, 2] = 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = h + self.mlp_down(torch.relu(self.mlp_up(h)))
        return self.head(h)


def test_inspector_and_patcher_cow_rollback(tmp_path: Path):
    model = TinyReasoningMLP()
    st_path = tmp_path / "tiny.safetensors"
    SurgicalWeightPatcher.save_safetensors(model, st_path)
    digests = WeightInspector.inspect_safetensors(st_path)
    assert len(digests) == 4
    assert "schema" in WSP_PATCH_JSON_SCHEMA or "properties" in WSP_PATCH_JSON_SCHEMA

    patcher = SurgicalWeightPatcher(max_modified_pct=5.0)
    plan = WSPPatchPlan(
        hypothesis="Zero single defective entry in head.weight",
        operations=[SetValuesOp(tensor="head.weight", indices=[[1, 2]], values=[0.0])],
    )
    res = patcher.apply_to_model(model, plan)
    assert res.applied is True
    assert res.modified_params == 1
    assert model.head.weight[1, 2].item() == 0.0
    assert res.base_sha256 != res.patched_sha256

    # Rollback restores bit-exact weight
    assert patcher.rollback_last(model) is True
    assert model.head.weight[1, 2].item() == 4.0

    # Budget rejection guard triggers cleanly
    tight = SurgicalWeightPatcher(max_modified_pct=0.01)
    too_big = WSPPatchPlan(
        hypothesis="Scale full matrix",
        operations=[ScaleSliceOp(tensor="mlp_down.weight", factor=0.5)],
    )
    rej = tight.apply_to_model(model, too_big)
    assert rej.applied is False
    assert "exceeding budget" in (rej.error or "")


def test_wsp_agent_loop_surgery_and_canary_guard():
    model = TinyReasoningMLP(d_model=32)

    # Failing inputs have both answer signal (bit 0) AND loop trigger (bit 2)
    failing_x = torch.zeros((6, 32))
    failing_x[:, 0] = 1.0
    failing_x[:, 2] = 1.0

    # Passing inputs have clean answer signal (bit 0 only)
    passing_x = torch.zeros((6, 32))
    passing_x[:, 0] = 1.0

    # Canary inputs verify unrelated token class 2 via bit 1
    canary_x = torch.zeros((6, 32))
    canary_x[:, 1] = 1.0

    def target_eval(m: nn.Module):
        logits = m(failing_x)
        preds = logits.argmax(dim=-1)
        acc = float((preds == 0).float().mean().item())
        # Simulate token stream where class 1 causes repeating loop tokens [1,1,1,1,1,1]
        seqs = [[1, 1, 1, 1, 1, 1] if p == 1 else [0, 3, 2, 0] for p in preds.tolist()]
        return {"score": acc, "repetition_rate": compute_ngram_repetition_rate(seqs, n=3)}

    def canary_eval(m: nn.Module) -> float:
        preds = m(canary_x).argmax(dim=-1)
        return float((preds == 2).float().mean().item())

    evaluator = WSPEvaluator(target_eval_fn=target_eval, canary_eval_fn=canary_eval)
    agent = WSPControllerAgent(
        patcher=SurgicalWeightPatcher(max_modified_pct=3.5),
        max_iterations=4,
    )
    report = agent.optimize(
        model=model,
        evaluator=evaluator,
        failing_batches=[failing_x],
        passing_batches=[passing_x],
    )

    assert report.baseline_eval.target_score == 0.0
    assert report.final_eval.target_score == 1.0
    assert report.final_eval.repetition_rate == 0.0
    assert report.final_eval.passed_canary is True
    assert report.final_eval.canary_score == 1.0
    assert len(report.accepted_plans) >= 1


def test_receipt_verification_and_backdoor_anomaly_scanner():
    model = TinyReasoningMLP(d_model=32)
    patcher = SurgicalWeightPatcher(max_modified_pct=5.0)

    # Plant strong rank-1 backdoor spike in proj_in.weight
    u = [0.0] * 32
    v = [0.0] * 32
    u[5] = 8.0
    v[5] = 8.0
    plan = WSPPatchPlan(
        hypothesis="Simulate unverified rank-1 backdoor perturbation",
        operations=[Rank1UpdateOp(tensor="proj_in.weight", u=u, v=v, alpha=1.0, rows=[5])],
    )
    res = patcher.apply_to_model(model, plan)
    assert res.applied is True

    findings = scan_for_weight_anomalies(model, kurtosis_threshold=10.0, spectral_ratio_threshold=3.5)
    assert any(f.tensor_name == "proj_in.weight" for f in findings)

    receipt = SwpReceipt(
        schema="swizzle.wsp.receipt.v1",
        created_at=1795100000.0,
        base_model_sha256=res.base_sha256,
        patched_model_sha256=res.patched_sha256,
        modified_params=res.modified_params,
        modified_pct=res.modified_pct,
        baseline_score=0.0,
        final_score=1.0,
        canary_score=1.0,
        hypothesis_log=[plan.hypothesis],
    ).seal()
    assert receipt.verify() is True
    receipt.final_score = 0.42
    assert receipt.verify() is False
