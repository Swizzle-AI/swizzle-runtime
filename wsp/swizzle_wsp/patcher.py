"""
Phase 2: Copy-on-Write Surgical Weight Patcher
==============================================
Applies constrained `WSPPatchPlan` operations in RAM with hard parameter-budget
enforcement (`max_modified_pct`), copy-on-write backup snapshots, `<1 ms` rollback,
and `.safetensors` serialization.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .patch_dsl import (
    ClampOutliersOp,
    ProjectOrthogonalOp,
    Rank1UpdateOp,
    ScaleSliceOp,
    SetValuesOp,
    WSPPatchPlan,
)


@dataclass
class PatchApplicationResult:
    applied: bool
    hypothesis: str
    modified_tensors: List[str]
    modified_params: int
    total_params: int
    modified_pct: float
    latency_ms: float
    base_sha256: str
    patched_sha256: str
    error: Optional[str] = None


def _hash_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(sd.keys()):
        ten = sd[name]
        if isinstance(ten, torch.Tensor):
            h.update(name.encode("utf-8"))
            h.update(ten.detach().cpu().float().contiguous().numpy().tobytes())
    return "sha256:" + h.hexdigest()


class SurgicalWeightPatcher:
    """Applies, verifies, rolls back, and persists tiny surgical weight edits."""

    def __init__(self, max_modified_pct: float = 0.50):
        self.max_modified_pct = float(max_modified_pct)
        self._snapshots: List[Dict[str, torch.Tensor]] = []

    def apply_to_model(self, model: nn.Module, plan: WSPPatchPlan) -> PatchApplicationResult:
        t0 = time.perf_counter()
        param_map: Dict[str, torch.Tensor] = dict(model.named_parameters())
        buf_map: Dict[str, torch.Tensor] = dict(model.named_buffers())
        all_tensors = {**param_map, **buf_map}
        total_params = sum(int(p.numel()) for p in param_map.values())
        op_names = [op.tensor for op in plan.operations if op.tensor in param_map]
        hash_scope_before = (
            all_tensors
            if total_params <= 5_000_000
            else ({k: param_map[k] for k in op_names} if op_names else {})
        )
        base_hash = _hash_state_dict(hash_scope_before)

        snapshot: Dict[str, torch.Tensor] = {}
        modified_count = 0
        touched_names: List[str] = []

        try:
            with torch.no_grad():
                for op in plan.operations:
                    if op.tensor not in param_map:
                        raise KeyError(f"Unknown parameter tensor '{op.tensor}'")
                    w = param_map[op.tensor]
                    if op.tensor not in snapshot:
                        snapshot[op.tensor] = w.detach().clone()
                        touched_names.append(op.tensor)

                    changed = self._execute_op(w, op)
                    modified_count += changed

                pct = (float(modified_count) / float(max(1, total_params))) * 100.0
                if pct > self.max_modified_pct:
                    raise ValueError(
                        f"Patch modified {pct:.4f}% ({modified_count}/{total_params}), exceeding budget {self.max_modified_pct:.4f}%"
                    )

            self._snapshots.append(snapshot)
            hash_scope_after = (
                {**dict(model.named_parameters()), **dict(model.named_buffers())}
                if total_params <= 5_000_000
                else {k: param_map[k] for k in touched_names}
            )
            patched_hash = _hash_state_dict(hash_scope_after)
            return PatchApplicationResult(
                applied=True,
                hypothesis=plan.hypothesis,
                modified_tensors=touched_names,
                modified_params=modified_count,
                total_params=total_params,
                modified_pct=round(pct, 6),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                base_sha256=base_hash,
                patched_sha256=patched_hash,
            )
        except Exception as exc:
            # Immediate atomic rollback of any touched tensors in this failed attempt
            with torch.no_grad():
                for tname, prev in snapshot.items():
                    param_map[tname].copy_(prev)
            return PatchApplicationResult(
                applied=False,
                hypothesis=plan.hypothesis,
                modified_tensors=[],
                modified_params=0,
                total_params=total_params,
                modified_pct=0.0,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                base_sha256=base_hash,
                patched_sha256=base_hash,
                error=str(exc),
            )

    def rollback_last(self, model: nn.Module) -> bool:
        if not self._snapshots:
            return False
        snap = self._snapshots.pop()
        param_map = dict(model.named_parameters())
        with torch.no_grad():
            for tname, prev in snap.items():
                if tname in param_map:
                    param_map[tname].copy_(prev)
        return True

    @staticmethod
    def _execute_op(w: torch.Tensor, op: Any) -> int:
        dtype = w.dtype
        device = w.device
        work = w.float()

        if isinstance(op, ScaleSliceOp):
            if op.rows is not None and op.cols is not None:
                r_idx = torch.tensor(op.rows, dtype=torch.long, device=device)
                c_idx = torch.tensor(op.cols, dtype=torch.long, device=device)
                sub = work.index_select(0, r_idx).index_select(1, c_idx) * float(op.factor)
                for ii, r in enumerate(op.rows):
                    work[r, c_idx] = sub[ii]
                changed = len(op.rows) * len(op.cols)
            elif op.rows is not None:
                r_idx = torch.tensor(op.rows, dtype=torch.long, device=device)
                work[r_idx] = work[r_idx] * float(op.factor)
                changed = len(op.rows) * (work.shape[1] if work.ndim > 1 else 1)
            elif op.cols is not None and work.ndim >= 2:
                c_idx = torch.tensor(op.cols, dtype=torch.long, device=device)
                work[:, c_idx] = work[:, c_idx] * float(op.factor)
                changed = work.shape[0] * len(op.cols)
            else:
                work.mul_(float(op.factor))
                changed = int(work.numel())
            w.copy_(work.to(dtype))
            return changed

        if isinstance(op, Rank1UpdateOp):
            u = torch.tensor(op.u, dtype=torch.float32, device=device)
            v = torch.tensor(op.v, dtype=torch.float32, device=device)
            delta = float(op.alpha) * torch.outer(u, v)
            if op.rows is not None:
                r_idx = torch.tensor(op.rows, dtype=torch.long, device=device)
                work[r_idx] = work[r_idx] + delta[r_idx]
                changed = len(op.rows) * work.shape[1]
            else:
                work.add_(delta)
                changed = int(work.numel())
            w.copy_(work.to(dtype))
            return changed

        if isinstance(op, ProjectOrthogonalOp):
            d = torch.tensor(op.direction, dtype=torch.float32, device=device)
            d = d / (d.norm() + 1e-12)
            if op.axis == "rows":
                # Project columns of W (each column is a vector in output feature space R^{out})
                # or project rows when W has shape (out, in) and d is R^{out}:
                # W' = W - alpha * d @ (d^T @ W)
                if op.rows is not None:
                    r_idx = torch.tensor(op.rows, dtype=torch.long, device=device)
                    d_sub = d[r_idx]
                    d_sub = d_sub / (d_sub.norm() + 1e-12)
                    proj = torch.outer(d_sub, d_sub @ work[r_idx])
                    work[r_idx] = work[r_idx] - float(op.alpha) * proj
                    changed = len(op.rows) * work.shape[1]
                else:
                    proj = torch.outer(d, d @ work)
                    work.sub_(float(op.alpha) * proj)
                    changed = int(work.numel())
            else:
                # Project input space: W' = W - alpha * (W @ d) @ d^T
                proj = torch.outer(work @ d, d)
                work.sub_(float(op.alpha) * proj)
                changed = int(work.numel())
            w.copy_(work.to(dtype))
            return changed

        if isinstance(op, ClampOutliersOp):
            before = work.clone()
            clamped = work.clamp(-float(op.max_abs), float(op.max_abs))
            changed = int((before != clamped).sum().item())
            w.copy_(clamped.to(dtype))
            return changed

        if isinstance(op, SetValuesOp):
            if len(op.indices) != len(op.values):
                raise ValueError("indices and values must have same length")
            for idx, val in zip(op.indices, op.values):
                work[tuple(idx)] = float(val)
            w.copy_(work.to(dtype))
            return len(op.values)

        raise TypeError(f"Unsupported patch op: {type(op)}")

    @staticmethod
    def save_safetensors(model: nn.Module, out_path: str | Path) -> Path:
        from safetensors.torch import save_file

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(tensors, str(out))
        return out
