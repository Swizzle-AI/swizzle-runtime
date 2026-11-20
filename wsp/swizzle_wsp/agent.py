"""
Phase 4: Closed-Loop Weight-Space Programming (WSP) Controller Agent
====================================================================
Runs evolutionary Read -> Hypothesize -> Patch -> Verify -> Keep/Revert search:
- Zero-API Local Mode: uses contrastive SVD defect localization to propose
  ranked `project_orthogonal`, `scale_slice`, and `clamp_outliers` edits.
- LLM Controller Mode: accepts an optional `llm_propose_fn(prompt_context)`
  or OpenAI-compatible `swizzle-rt serve` endpoint constrained by `WSP_PATCH_JSON_SCHEMA`.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch.nn as nn

from .evaluator import EvalSnapshot, WSPEvaluator
from .inspector import ContrastiveCircuitReport, WeightInspector
from .patch_dsl import (
    WSP_PATCH_JSON_SCHEMA,
    ClampOutliersOp,
    ProjectOrthogonalOp,
    ScaleSliceOp,
    WSPPatchPlan,
)
from .patcher import PatchApplicationResult, SurgicalWeightPatcher


@dataclass
class WSPStepTrace:
    iteration: int
    hypothesis: str
    accepted: bool
    patch_result: PatchApplicationResult
    eval_after: EvalSnapshot
    reason: str


@dataclass
class WSPOptimizationReport:
    baseline_eval: EvalSnapshot
    final_eval: EvalSnapshot
    accepted_plans: List[WSPPatchPlan] = field(default_factory=list)
    traces: List[WSPStepTrace] = field(default_factory=list)
    total_modified_params: int = 0
    total_modified_pct: float = 0.0


def query_swizzle_rt_controller(
    base_url: str,
    context: Dict[str, Any],
    model: str = "local",
) -> WSPPatchPlan:
    """Queries `swizzle-rt serve` `/v1/chat/completions` with `score_tree` JSON Schema."""
    req_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a Weight-Space Programmer. Emit ONLY a surgical WSPPatchPlan JSON touching <0.1% weights.",
            },
            {"role": "user", "content": json.dumps(context)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "wsp_patch_plan", "strict": True, "schema": WSP_PATCH_JSON_SCHEMA},
        },
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return WSPPatchPlan.model_validate_json(content)


class WSPControllerAgent:
    """Closed-loop surgical weight debugger (`Read -> Hypothesize -> Patch -> Verify`)."""

    def __init__(
        self,
        patcher: Optional[SurgicalWeightPatcher] = None,
        max_iterations: int = 6,
        llm_propose_fn: Optional[Callable[[Dict[str, Any], int], WSPPatchPlan]] = None,
        allow_full_direction_fallback: bool = False,
    ):
        self.patcher = patcher or SurgicalWeightPatcher(max_modified_pct=0.50)
        self.max_iterations = max(1, int(max_iterations))
        self.llm_propose_fn = llm_propose_fn
        self.allow_full_direction_fallback = bool(allow_full_direction_fallback)

    def optimize(
        self,
        model: nn.Module,
        evaluator: WSPEvaluator,
        failing_batches: Sequence[Any],
        passing_batches: Sequence[Any],
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        exclude_names: tuple[str, ...] = ("lm_head",),
    ) -> WSPOptimizationReport:
        baseline = evaluator.evaluate(model)
        best_eval = baseline
        accepted_plans: List[WSPPatchPlan] = []
        traces: List[WSPStepTrace] = []
        cum_modified = 0
        total_params = sum(int(p.numel()) for p in model.parameters())

        for it in range(self.max_iterations):
            circuit = WeightInspector.localize_contrastive_circuits(
                model,
                failing_batches=failing_batches,
                passing_batches=passing_batches,
                forward_fn=forward_fn,
                compute_digests=False,
                exclude_names=exclude_names,
            )

            if self.llm_propose_fn is not None:
                plan = self.llm_propose_fn(circuit.to_prompt_context(), it)
            else:
                plan = self._propose_spectral_patch(
                    model, circuit, it, allow_full_fallback=self.allow_full_direction_fallback
                )
                if plan is None:
                    break

            res = self.patcher.apply_to_model(model, plan)
            if not res.applied:
                traces.append(
                    WSPStepTrace(
                        iteration=it + 1,
                        hypothesis=plan.hypothesis,
                        accepted=False,
                        patch_result=res,
                        eval_after=best_eval,
                        reason=f"Patch rejected by budget/validation: {res.error}",
                    )
                )
                continue

            after = evaluator.evaluate(model)
            improved = (after.target_score > best_eval.target_score + 1e-4) or (
                after.repetition_rate < best_eval.repetition_rate - 0.02
                and after.target_score >= best_eval.target_score - 1e-4
            )
            if improved and after.passed_canary:
                best_eval = after
                accepted_plans.append(plan)
                cum_modified += res.modified_params
                traces.append(
                    WSPStepTrace(
                        iteration=it + 1,
                        hypothesis=plan.hypothesis,
                        accepted=True,
                        patch_result=res,
                        eval_after=after,
                        reason=f"Improved target {baseline.target_score:.3f}->{after.target_score:.3f}, rep={after.repetition_rate:.3f}",
                    )
                )
                if best_eval.target_score >= 0.9999 and best_eval.repetition_rate <= 0.02:
                    break
            else:
                self.patcher.rollback_last(model)
                traces.append(
                    WSPStepTrace(
                        iteration=it + 1,
                        hypothesis=plan.hypothesis,
                        accepted=False,
                        patch_result=res,
                        eval_after=after,
                        reason=f"Reverted: canary={after.passed_canary}, target={after.target_score:.3f} vs best={best_eval.target_score:.3f}",
                    )
                )

        pct = (float(cum_modified) / float(max(1, total_params))) * 100.0
        return WSPOptimizationReport(
            baseline_eval=baseline,
            final_eval=best_eval,
            accepted_plans=accepted_plans,
            traces=traces,
            total_modified_params=cum_modified,
            total_modified_pct=round(pct, 6),
        )

    @staticmethod
    def _propose_spectral_patch(
        model: nn.Module,
        circuit: ContrastiveCircuitReport,
        iteration: int,
        allow_full_fallback: bool = False,
    ) -> Optional[WSPPatchPlan]:
        """Generates deterministic spectral surgery candidates from contrastive activations."""
        if not circuit.top_divergent_layers:
            return None
        param_shapes = {k: tuple(v.shape) for k, v in model.named_parameters()}
        top0 = circuit.top_divergent_layers[0]

        if allow_full_fallback:
            # Find same-block gating sibling (e.g., up_proj / gate_proj paired with down_proj)
            block_prefix = top0.layer_name.rsplit(".", 1)[0] if "." in top0.layer_name else ""
            siblings = [
                d
                for d in circuit.top_divergent_layers
                if block_prefix
                and d.layer_name.startswith(block_prefix + ".")
                and d.layer_name != top0.layer_name
                and d.weight_param_name in param_shapes
                and len(param_shapes[d.weight_param_name]) >= 2
                and d.top_divergent_channels
            ]
            siblings.sort(key=lambda d: d.cosine_shift, reverse=True)

            if iteration == 0 and top0.weight_param_name in param_shapes:
                top_ch = top0.top_divergent_channels[:2]
                rows = top_ch if len(param_shapes[top0.weight_param_name]) >= 2 and param_shapes[top0.weight_param_name][0] > 4 else None
                return WSPPatchPlan(
                    hypothesis=f"Sparse defect orthogonalization in {top0.weight_param_name} rows={rows}",
                    target_issue="defect_orthogonalization",
                    operations=[
                        ProjectOrthogonalOp(
                            tensor=top0.weight_param_name,
                            direction=top0.defect_direction,
                            axis="rows",
                            alpha=1.0,
                            rows=rows,
                        )
                    ],
                )

            if iteration in (1, 2, 3) and top0.top_input_channels and top0.weight_param_name in param_shapes:
                in_ch = int(top0.top_input_channels[min(iteration - 1, len(top0.top_input_channels) - 1)])
                return WSPPatchPlan(
                    hypothesis=f"Silence highest-attribution input column [{in_ch}] in {top0.weight_param_name}",
                    target_issue="input_channel_attribution",
                    operations=[ScaleSliceOp(tensor=top0.weight_param_name, cols=[in_ch], factor=0.0)],
                )

            if iteration == 4 and siblings:
                sib = siblings[0]
                ch_idx = int(sib.top_divergent_channels[0])
                return WSPPatchPlan(
                    hypothesis=f"Silence upstream gating channel [{ch_idx}] in {sib.weight_param_name}",
                    target_issue="sparse_neuron_silencing",
                    operations=[ScaleSliceOp(tensor=sib.weight_param_name, rows=[ch_idx], factor=0.0)],
                )

            if iteration == 5 and top0.weight_param_name in param_shapes:
                return WSPPatchPlan(
                    hypothesis=f"Full-direction defect orthogonalization in {top0.weight_param_name}",
                    target_issue="defect_orthogonalization",
                    operations=[
                        ProjectOrthogonalOp(
                            tensor=top0.weight_param_name,
                            direction=top0.defect_direction,
                            axis="rows",
                            alpha=1.0,
                            rows=None,
                        )
                    ],
                )

        layer_idx = min(iteration // 2, len(circuit.top_divergent_layers) - 1)
        target = circuit.top_divergent_layers[layer_idx]
        w_name = target.weight_param_name
        if w_name not in param_shapes:
            return None

        # Candidate cycle across top divergent layers: sparse row projection & row scaling
        alpha = 1.0 if iteration % 2 == 0 else 0.5
        top_ch = target.top_divergent_channels[:2]
        rows = (
            None
            if (allow_full_fallback and iteration >= 4)
            else (top_ch if len(param_shapes[w_name]) >= 2 and param_shapes[w_name][0] > 4 else None)
        )

        if iteration % 3 == 2 and rows and len(param_shapes[w_name]) >= 2:
            return WSPPatchPlan(
                hypothesis=f"Attenuate over-firing loop channels {rows} in {w_name}",
                target_issue="loop_attenuation",
                operations=[ScaleSliceOp(tensor=w_name, rows=rows, factor=0.25)],
            )

        return WSPPatchPlan(
            hypothesis=f"Project out contrastive defect direction in {w_name} (alpha={alpha})",
            target_issue="defect_orthogonalization",
            operations=[
                ProjectOrthogonalOp(
                    tensor=w_name,
                    direction=target.defect_direction,
                    axis="rows",
                    alpha=alpha,
                    rows=rows,
                )
            ],
        )
