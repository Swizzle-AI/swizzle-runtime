"""
Option B Empirical Proof: Post-Quantization Outlier Healing (`Q4 / INT4`)
========================================================================
Demonstrates repairing an aggressively quantized edge MLP whose output degrades
due to post-quantization outlier channels, using `ClampOutliersOp` + `WSPControllerAgent`
with zero gradient training and strict Canary preservation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from swizzle_wsp import (
    SurgicalWeightPatcher,
    WSPControllerAgent,
    WSPEvaluator,
)


class EdgeQuantizedMLP(nn.Module):
    def __init__(self, hidden: int = 192):
        super().__init__()
        torch.manual_seed(99)
        self.fc1 = nn.Linear(hidden, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden * 2, hidden, bias=False)
        self.head = nn.Linear(hidden, 4, bias=False)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(hidden * 2, hidden) * 0.9)
            self.fc2.weight.copy_(torch.eye(hidden, hidden * 2) * 0.9)
            self.head.weight.zero_()
            self.head.weight[0, 0] = 3.0
            self.head.weight[1, 1] = 3.0
            # Simulated Q4 dequantization outlier spike distorting channel 3 -> pushes class 2
            self.fc2.weight[3, 3] = 28.0
            self.head.weight[2, 3] = 6.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fc2(torch.relu(self.fc1(x))))


def main() -> None:
    model = EdgeQuantizedMLP(hidden=192)
    total_params = sum(p.numel() for p in model.parameters())

    fail_x = torch.randn((16, 192)) * 0.04
    fail_x[:, 0] = 1.2
    fail_x[:, 3] = 0.8  # Quantized outlier blows up logit class 2 instead of 0

    pass_x = torch.randn((16, 192)) * 0.04
    pass_x[:, 0] = 1.2

    canary_x = torch.randn((16, 192)) * 0.04
    canary_x[:, 1] = 1.2

    def eval_target(m: nn.Module):
        preds = m(fail_x).argmax(dim=-1)
        return {"score": float((preds == 0).float().mean().item()), "repetition_rate": 0.0}

    def eval_canary(m: nn.Module) -> float:
        return float((m(canary_x).argmax(dim=-1) == 1).float().mean().item())

    evaluator = WSPEvaluator(target_eval_fn=eval_target, canary_eval_fn=eval_canary)
    agent = WSPControllerAgent(
        patcher=SurgicalWeightPatcher(max_modified_pct=0.50),
        max_iterations=6,
    )
    report = agent.optimize(
        model=model,
        evaluator=evaluator,
        failing_batches=[fail_x],
        passing_batches=[pass_x],
    )

    print("=" * 72)
    print("SWIZZLE WSP — POST-QUANTIZATION OUTLIER HEALING PROOF")
    print("=" * 72)
    print(f"Total Parameters       : {total_params:,}")
    print(f"Baseline Q4 Degraded   : {report.baseline_eval.target_score * 100:.1f}%")
    print(f"WSP Healed Acc         : {report.final_eval.target_score * 100:.1f}%")
    print(f"Canary Preservation    : {report.final_eval.canary_score * 100:.1f}% (passed={report.final_eval.passed_canary})")
    print(f"Parameters Edited      : {report.total_modified_params:,} ({report.total_modified_pct:.4f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
