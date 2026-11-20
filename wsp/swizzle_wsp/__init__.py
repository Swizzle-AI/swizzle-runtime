"""
Swizzle Weight-Space Programming (`swizzle_wsp`)
================================================
Gradient-free weight reading, contrastive circuit localization, constrained
JSON Patch DSL execution, anti-regression evaluation, and cryptographic `.swp`
weight-patch receipts.
"""

from .agent import WSPControllerAgent, WSPOptimizationReport, WSPStepTrace, query_swizzle_rt_controller
from .evaluator import EvalSnapshot, WSPEvaluator, compute_ngram_repetition_rate
from .hardware_guard import HardwareHealthSnapshot, enforce_safe_thermal_budget, probe_hardware_health
from .inspector import ContrastiveCircuitReport, TensorHealthDigest, WeightInspector
from .model_hub import GPU_POOR_CATALOG, detect_host_capacity, pack_safetensors_to_gguf, recommend_gpu_poor_models
from .patch_dsl import WSP_PATCH_JSON_SCHEMA, PatchOp, WSPPatchPlan
from .patcher import PatchApplicationResult, SurgicalWeightPatcher
from .quant_heal import (
    apply_q4_with_salience,
    apply_wsp_rms_equalization,
    calibrate_transformer_activations,
    evaluate_causal_lm_perplexity,
    quantize_tensor_q4_0,
)
from .receipts import SwpReceipt, WeightAnomalyFinding, scan_for_weight_anomalies

__version__ = "0.1.0"

__all__ = [
    "ContrastiveCircuitReport",
    "EvalSnapshot",
    "PatchApplicationResult",
    "PatchOp",
    "SurgicalWeightPatcher",
    "SwpReceipt",
    "TensorHealthDigest",
    "WSPControllerAgent",
    "WSPEvaluator",
    "WSPOptimizationReport",
    "WSPPatchPlan",
    "WSPStepTrace",
    "WSP_PATCH_JSON_SCHEMA",
    "WeightAnomalyFinding",
    "WeightInspector",
    "compute_ngram_repetition_rate",
    "query_swizzle_rt_controller",
    "scan_for_weight_anomalies",
]
