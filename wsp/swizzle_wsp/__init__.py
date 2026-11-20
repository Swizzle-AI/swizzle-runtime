"""
Swizzle Weight-Space Programming (`swizzle_wsp`)
================================================
Gradient-free weight reading, contrastive circuit localization, constrained
JSON Patch DSL execution, anti-regression evaluation, and cryptographic `.swp`
weight-patch receipts.
"""

from .agent import WSPControllerAgent, WSPOptimizationReport, WSPStepTrace, query_swizzle_rt_controller
from .evaluator import EvalSnapshot, WSPEvaluator, compute_ngram_repetition_rate
from .inspector import ContrastiveCircuitReport, TensorHealthDigest, WeightInspector
from .patch_dsl import WSP_PATCH_JSON_SCHEMA, PatchOp, WSPPatchPlan
from .patcher import PatchApplicationResult, SurgicalWeightPatcher
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
