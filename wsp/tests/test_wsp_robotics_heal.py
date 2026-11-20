import sys
from pathlib import Path

ROOT_WSP = Path(__file__).resolve().parent.parent
ROOT_ROBOTICS = ROOT_WSP.parent.parent / "swizzle-robotics"
if ROOT_ROBOTICS.is_dir():
    sys.path.insert(0, str(ROOT_ROBOTICS))

import numpy as np
import pytest
import torch

from swizzle_wsp import SurgicalWeightPatcher, WSPControllerAgent, WSPEvaluator


def test_wsp_self_heals_robotics_flow_expert():
    robotics = pytest.importorskip("swizzle_robotics")
    from swizzle_robotics.core.safety import ControlBarrierShield, ObstacleSphere, SafetyConfig
    from swizzle_robotics.models.edge_flow_runner import TorchEdgeVLAPolicy

    _ = robotics
    policy = TorchEdgeVLAPolicy(seed=2026)
    shield = ControlBarrierShield(
        SafetyConfig(
            max_linear_step=0.22,
            obstacles=[ObstacleSphere(center=(0.0, 0.0, 0.24), radius=0.065, name="pillar")],
        )
    )

    ood_frame = np.full((224, 224, 3), 240, dtype=np.uint8)
    clean_frame = np.full((224, 224, 3), 120, dtype=np.uint8)

    with torch.no_grad():
        ood_emb = policy.encode_sparse_vit([0.95] * 196)
        clean_emb = policy.encode_sparse_vit([0.45] * 196)
        diff_dir = (ood_emb - clean_emb).reshape(-1)
        diff_dir = diff_dir / (diff_dir.norm() + 1e-8)
        policy.expert.net[0].weight[5, :128] = diff_dir * 4.5
        policy.expert.net[2].weight[5, 5] = 4.0
        policy.expert.net[4].weight[0, 5] = 4.5

    t_zero = torch.zeros((1, 1), dtype=torch.float32)
    x_zero = torch.zeros((1, 112), dtype=torch.float32)
    ood_inp = torch.cat([ood_emb, t_zero, x_zero], dim=-1)
    clean_inp = torch.cat([clean_emb, t_zero, x_zero], dim=-1)

    def eval_ood(_m):
        shield.reset()
        raw = policy.forward_action_chunk(ood_frame, "pick up red cube").actions[0]
        sec = shield.project(raw, current_pose=[-0.12, 0.0, 0.24, 0.0, 0.0, 0.0])
        return {"score": 0.0 if sec.cbf_intervened else 1.0, "repetition_rate": 0.0}

    def eval_canary(_m):
        act = policy.forward_action_chunk(clean_frame, "pick up red cube").actions[0]
        return 1.0 if all(np.isfinite(act)) else 0.0

    agent = WSPControllerAgent(patcher=SurgicalWeightPatcher(max_modified_pct=0.50), max_iterations=6)
    report = agent.optimize(
        model=policy.expert,
        evaluator=WSPEvaluator(target_eval_fn=eval_ood, canary_eval_fn=eval_canary),
        failing_batches=[ood_inp],
        passing_batches=[clean_inp],
        forward_fn=lambda m, x: m.net(x),
    )

    assert report.baseline_eval.target_score == 0.0
    assert report.final_eval.target_score == 1.0
    assert report.final_eval.passed_canary is True
    assert report.total_modified_pct <= 0.50
