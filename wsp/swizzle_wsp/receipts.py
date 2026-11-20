"""
Phase 5: Cryptographic `.swp` Weight-Patch Receipts & Backdoor Auditor
=====================================================================
Addresses WSP security risks:
1. `SwpReceipt`: seals base model digest, patched model digest, exact modified
   parameter ratio, and eval before/after metrics into a tamper-evident artifact.
2. `scan_for_weight_anomalies`: detects planted rank-1/high-kurtosis weight
   backdoors or spectral outliers across `.safetensors` / PyTorch layers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .inspector import WeightInspector


@dataclass
class SwpReceipt:
    schema: str
    created_at: float
    base_model_sha256: str
    patched_model_sha256: str
    modified_params: int
    modified_pct: float
    baseline_score: float
    final_score: float
    canary_score: float
    hypothesis_log: List[str]
    receipt_digest: str = ""

    def compute_digest(self) -> str:
        payload = {
            "schema": self.schema,
            "created_at": round(self.created_at, 3),
            "base_model_sha256": self.base_model_sha256,
            "patched_model_sha256": self.patched_model_sha256,
            "modified_params": int(self.modified_params),
            "modified_pct": round(float(self.modified_pct), 6),
            "baseline_score": round(float(self.baseline_score), 5),
            "final_score": round(float(self.final_score), 5),
            "canary_score": round(float(self.canary_score), 5),
            "hypothesis_log": list(self.hypothesis_log),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(canonical).hexdigest()

    def seal(self) -> "SwpReceipt":
        self.receipt_digest = self.compute_digest()
        return self

    def verify(self) -> bool:
        return bool(self.receipt_digest) and self.receipt_digest == self.compute_digest()

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out


@dataclass
class WeightAnomalyFinding:
    tensor_name: str
    kurtosis: float
    spectral_ratio: float
    outlier_rows: List[int]
    severity: str


def scan_for_weight_anomalies(
    model: nn.Module,
    kurtosis_threshold: float = 12.0,
    spectral_ratio_threshold: float = 4.5,
) -> List[WeightAnomalyFinding]:
    """
    Audits model weights for unusually sharp rank-1 perturbations or extreme
    heavy-tailed neuron activations indicative of unverified/backdoored weight edits.
    """
    findings: List[WeightAnomalyFinding] = []
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        digest = WeightInspector.summarize_tensor(name, param, svd_rank=4)
        sv = digest.top_singular_values
        spec_ratio = round(sv[0] / (sv[1] + 1e-6), 3) if len(sv) >= 2 else 1.0
        if digest.kurtosis >= kurtosis_threshold or spec_ratio >= spectral_ratio_threshold:
            sev = "high" if (digest.kurtosis >= kurtosis_threshold * 1.5 or spec_ratio >= spectral_ratio_threshold * 1.5) else "medium"
            findings.append(
                WeightAnomalyFinding(
                    tensor_name=name,
                    kurtosis=digest.kurtosis,
                    spectral_ratio=spec_ratio,
                    outlier_rows=digest.outlier_row_indices,
                    severity=sev,
                )
            )
    return findings
