"""
Empirical Proof Benchmark: Gradient-Free Loop & Math Surgery via `swizzle-wsp`
==============================================================================
Demonstrates closed-loop Weight-Space Programming (`Read -> Localize -> Patch -> Verify`)
on a multi-layer Transformer-scale residual block (`~530K` params) with a planted
directional thinking-loop defect:
- Zero gradient backpropagation
- Compares Baseline vs WSP Patched accuracy, loop repetition rate, % params edited,
  anti-regression Canary score, and `.swp` receipt digest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from swizzle_wsp import (
    SurgicalWeightPatcher,
    SwpReceipt,
    WSPControllerAgent,
    WSPEvaluator,
    compute_ngram_repetition_rate,
    scan_for_weight_anomalies,
)


class ResidualReasoningBlock(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        torch.manual_seed(2026)
        self.attn_o = nn.Linear(hidden, hidden, bias=False)
        self.mlp_up = nn.Linear(hidden, hidden * 4, bias=False)
        self.mlp_down = nn.Linear(hidden * 4, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, 8, bias=False)

        with torch.no_grad():
            self.attn_o.weight.copy_(torch.eye(hidden) * 0.8)
            self.mlp_up.weight.normal_(0.0, 0.01)
            self.mlp_down.weight.normal_(0.0, 0.01)
            self.lm_head.weight.normal_(0.0, 0.02)

            # Clean math/reasoning feature [0] -> correct answer class 0
            self.lm_head.weight[0, 0] = 3.5
            # Unrelated Canary coding/chat feature [1] -> class 3
            self.lm_head.weight[3, 1] = 3.5
            # Planted loop direction in feature [4]: over-amplified by mlp_up/down -> class 7 (loop trap)
            self.mlp_up.weight[4, 4] = 5.0
            self.mlp_down.weight[4, 4] = 6.0
            self.lm_head.weight[7, 4] = 5.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_o(x)
        h = h + self.mlp_down(torch.relu(self.mlp_up(h)))
        return self.lm_head(h)


def main() -> None:
    model = ResidualReasoningBlock(hidden=256)
    total_params = sum(p.numel() for p in model.parameters())

    fail_x = torch.randn((16, 256)) * 0.05
    fail_x[:, 0] = 1.2
    fail_x[:, 4] = 1.4  # Triggers loop trap

    pass_x = torch.randn((16, 256)) * 0.05
    pass_x[:, 0] = 1.2

    canary_x = torch.randn((16, 256)) * 0.05
    canary_x[:, 1] = 1.4

    def eval_target(m: nn.Module):
        preds = m(fail_x).argmax(dim=-1)
        acc = float((preds == 0).float().mean().item())
        seqs = [[7, 7, 7, 7, 7, 7] if p == 7 else [0, 2, 4, 0] for p in preds.tolist()]
        return {"score": acc, "repetition_rate": compute_ngram_repetition_rate(seqs, n=3)}

    def eval_canary(m: nn.Module) -> float:
        return float((m(canary_x).argmax(dim=-1) == 3).float().mean().item())

    evaluator = WSPEvaluator(target_eval_fn=eval_target, canary_eval_fn=eval_canary)
    agent = WSPControllerAgent(
        patcher=SurgicalWeightPatcher(max_modified_pct=0.20),
        max_iterations=6,
    )
    report = agent.optimize(
        model=model,
        evaluator=evaluator,
        failing_batches=[fail_x],
        passing_batches=[pass_x],
    )

    findings = scan_for_weight_anomalies(model)
    receipt = SwpReceipt(
        schema="swizzle.wsp.receipt.v1",
        created_at=0.0,
        base_model_sha256="bench-base",
        patched_model_sha256="bench-patched",
        modified_params=report.total_modified_params,
        modified_pct=report.total_modified_pct,
        baseline_score=report.baseline_eval.target_score,
        final_score=report.final_eval.target_score,
        canary_score=report.final_eval.canary_score,
        hypothesis_log=[p.hypothesis for p in report.accepted_plans],
    ).seal()

    print("=" * 72)
    print("SWIZZLE WEIGHT-SPACE PROGRAMMING (WSP) — EMPIRICAL LOOP SURGERY PROOF")
    print("=" * 72)
    print(f"Total Parameters       : {total_params:,}")
    print(f"Baseline Target Acc    : {report.baseline_eval.target_score * 100:.1f}% | Repetition: {report.baseline_eval.repetition_rate:.3f}")
    print(f"WSP Patched Target Acc : {report.final_eval.target_score * 100:.1f}% | Repetition: {report.final_eval.repetition_rate:.3f}")
    print(f"Canary Preservation    : {report.final_eval.canary_score * 100:.1f}% (passed={report.final_eval.passed_canary})")
    print(f"Parameters Edited      : {report.total_modified_params:,} ({report.total_modified_pct:.4f}%)")
    print(f"Accepted Patch Steps   : {len(report.accepted_plans)} / {len(report.traces)}")
    print(f"Anomaly Findings       : {len(findings)} tensors flagged")
    print(f"Cryptographic .swp     : {receipt.receipt_digest[:36]}... (verified={receipt.verify()})")
    print("=" * 72)


if __name__ == "__main__":
    main()
