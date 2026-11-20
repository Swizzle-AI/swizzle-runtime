"""
Phase 1: Weight Inspector & Gradient-Free Contrastive Circuit Localizer
=======================================================================
Compresses multi-million/billion parameter matrices into compact spectral
digests (SVD, kurtosis, outlier channels) and localizes defect directions
via contrastive residual activation analysis with `torch.no_grad()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TensorHealthDigest:
    name: str
    shape: List[int]
    numel: int
    mean: float
    std: float
    max_abs: float
    kurtosis: float
    sparsity_pct: float
    top_singular_values: List[float]
    outlier_row_indices: List[int]


@dataclass
class DivergentLayerSummary:
    layer_name: str
    weight_param_name: str
    activation_l2_diff: float
    cosine_shift: float
    defect_direction: List[float]  # Unit vector in layer output or weight row space
    top_divergent_channels: List[int]
    top_input_channels: List[int] = field(default_factory=list)


@dataclass
class ContrastiveCircuitReport:
    total_parameters: int
    inspected_layers: int
    top_divergent_layers: List[DivergentLayerSummary] = field(default_factory=list)
    health_digests: List[TensorHealthDigest] = field(default_factory=list)

    def to_prompt_context(self, max_layers: int = 5) -> Dict[str, Any]:
        """Serializes the report into a compact (<4K token) dict for controller LLMs."""
        return {
            "total_parameters": self.total_parameters,
            "inspected_layers": self.inspected_layers,
            "top_divergent_layers": [
                {
                    "layer": d.layer_name,
                    "weight": d.weight_param_name,
                    "l2_diff": round(d.activation_l2_diff, 5),
                    "cosine_shift": round(d.cosine_shift, 5),
                    "top_channels": d.top_divergent_channels[:8],
                    "defect_dir_norm": round(float(np.linalg.norm(d.defect_direction)), 4),
                }
                for d in self.top_divergent_layers[:max_layers]
            ],
            "top_outlier_tensors": [
                {
                    "name": h.name,
                    "shape": h.shape,
                    "kurtosis": h.kurtosis,
                    "max_abs": h.max_abs,
                    "sv_top3": h.top_singular_values[:3],
                    "outlier_rows": h.outlier_row_indices[:6],
                }
                for h in sorted(self.health_digests, key=lambda x: x.kurtosis, reverse=True)[:max_layers]
            ],
        }


class WeightInspector:
    """Gradient-free weight & activation inspector for `.safetensors` and PyTorch models."""

    @staticmethod
    def summarize_tensor(name: str, tensor: torch.Tensor, svd_rank: int = 4) -> TensorHealthDigest:
        with torch.no_grad():
            t = tensor.detach().float().cpu()
            numel = int(t.numel())
            flat = t.reshape(-1)
            mean = float(flat.mean().item())
            std = float(flat.std(unbiased=False).item())
            max_abs = float(flat.abs().max().item()) if numel > 0 else 0.0
            centered = flat - mean
            kurt = float(((centered**4).mean() / (std**4 + 1e-12)).item()) if numel > 0 else 0.0
            sparsity_pct = float((flat.abs() < 1e-4).float().mean().item() * 100.0) if numel > 0 else 0.0

            mat = t.reshape(t.shape[0], -1) if t.ndim >= 2 else t.unsqueeze(0)
            k = min(svd_rank, mat.shape[0], mat.shape[1])
            if k > 0 and mat.numel() <= 2_500_000:
                try:
                    svals = torch.linalg.svdvals(mat)[:k].tolist()
                except Exception:
                    svals = []
            elif k > 0:
                # Lightweight power-iteration top singular value estimate for large matrices
                v = torch.randn(mat.shape[1], 1)
                v = v / (v.norm() + 1e-8)
                for _ in range(4):
                    u = mat @ v
                    u = u / (u.norm() + 1e-8)
                    v = mat.t() @ u
                    v = v / (v.norm() + 1e-8)
                svals = [float((u.t() @ (mat @ v)).item())]
            else:
                svals = []

            row_norms = mat.norm(dim=1)
            if row_norms.numel() > 1:
                thresh = float(row_norms.mean().item() + 2.5 * row_norms.std(unbiased=False).item())
                outliers = [int(i) for i in torch.nonzero(row_norms > thresh, as_tuple=False).flatten().tolist()[:12]]
            else:
                outliers = []

        return TensorHealthDigest(
            name=name,
            shape=list(tensor.shape),
            numel=numel,
            mean=round(mean, 6),
            std=round(std, 6),
            max_abs=round(max_abs, 6),
            kurtosis=round(kurt, 3),
            sparsity_pct=round(sparsity_pct, 2),
            top_singular_values=[round(float(v), 4) for v in svals],
            outlier_row_indices=outliers,
        )

    @classmethod
    def inspect_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        name_filter: Optional[Callable[[str], bool]] = None,
    ) -> List[TensorHealthDigest]:
        digests: List[TensorHealthDigest] = []
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor) or v.ndim < 1:
                continue
            if name_filter is not None and not name_filter(k):
                continue
            digests.append(cls.summarize_tensor(k, v))
        return digests

    @classmethod
    def inspect_safetensors(
        cls,
        path: str | Path,
        name_filter: Optional[Callable[[str], bool]] = None,
    ) -> List[TensorHealthDigest]:
        from safetensors.torch import load_file

        sd = load_file(str(path), device="cpu")
        return cls.inspect_state_dict(sd, name_filter=name_filter)

    @staticmethod
    def extract_weight_slice(
        tensor: torch.Tensor,
        row_start: int = 0,
        row_end: int = 8,
        col_start: int = 0,
        col_end: int = 8,
    ) -> List[List[float]]:
        t = tensor.detach().float().cpu()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        sub = t[row_start:row_end, col_start:col_end]
        return [[round(float(x), 5) for x in row] for row in sub.tolist()]

    @classmethod
    @torch.no_grad()
    def localize_contrastive_circuits(
        cls,
        model: nn.Module,
        failing_batches: Sequence[Any],
        passing_batches: Sequence[Any],
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        layer_types: Tuple[type, ...] = (nn.Linear,),
        compute_digests: bool = True,
        exclude_names: Tuple[str, ...] = (),
    ) -> ContrastiveCircuitReport:
        """
        Records mean activation residuals across Linear modules on failing vs passing
        inputs without backpropagation, ranking layers by L2 behavioral divergence and
        extracting unit defect directions for surgical `project_orthogonal` / `rank1` edits.
        """
        model.eval()
        run = forward_fn if forward_fn is not None else (lambda m, x: m(x))

        captured_fail: Dict[str, List[torch.Tensor]] = {}
        captured_pass: Dict[str, List[torch.Tensor]] = {}
        captured_fail_in: Dict[str, List[torch.Tensor]] = {}
        captured_pass_in: Dict[str, List[torch.Tensor]] = {}
        target_modules: Dict[str, nn.Module] = {}

        for name, mod in model.named_modules():
            if name in exclude_names:
                continue
            if isinstance(mod, layer_types) and hasattr(mod, "weight"):
                target_modules[name] = mod
                captured_fail[name] = []
                captured_pass[name] = []
                captured_fail_in[name] = []
                captured_pass_in[name] = []

        active_out_bucket: Dict[str, List[torch.Tensor]] = captured_fail
        active_in_bucket: Dict[str, List[torch.Tensor]] = captured_fail_in

        def _make_hook(lname: str):
            def _hook(_mod: nn.Module, inp: Any, out: Any) -> None:
                ten = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(ten, torch.Tensor):
                    flat = ten.detach().float().cpu().reshape(-1, ten.shape[-1])
                    active_out_bucket[lname].append(flat.mean(dim=0))
                if isinstance(inp, (tuple, list)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x_ten = inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1])
                    active_in_bucket[lname].append(x_ten.mean(dim=0))

            return _hook

        handles = [mod.register_forward_hook(_make_hook(lname)) for lname, mod in target_modules.items()]
        try:
            active_out_bucket = captured_fail
            active_in_bucket = captured_fail_in
            for batch in failing_batches:
                run(model, batch)
            active_out_bucket = captured_pass
            active_in_bucket = captured_pass_in
            for batch in passing_batches:
                run(model, batch)
        finally:
            for h in handles:
                h.remove()

        divergent: List[DivergentLayerSummary] = []
        for lname, mod in target_modules.items():
            if not captured_fail[lname] or not captured_pass[lname]:
                continue
            fail_mean = torch.stack(captured_fail[lname], dim=0).mean(dim=0)
            pass_mean = torch.stack(captured_pass[lname], dim=0).mean(dim=0)
            diff = fail_mean - pass_mean
            l2_diff = float(diff.norm().item())
            denom = float(fail_mean.norm().item() * pass_mean.norm().item()) + 1e-12
            cos_sim = float(torch.dot(fail_mean, pass_mean).item() / denom)
            unit_dir = (diff / (l2_diff + 1e-12)).tolist()
            top_ch = torch.argsort(diff.abs(), descending=True)[:12].tolist()

            top_in_ch: List[int] = []
            if captured_fail_in[lname] and captured_pass_in[lname] and mod.weight.ndim >= 2:
                fin = torch.stack(captured_fail_in[lname], dim=0).mean(dim=0)
                pin = torch.stack(captured_pass_in[lname], dim=0).mean(dim=0)
                col_norms = mod.weight.detach().float().norm(dim=0).cpu()
                attrib = (fin - pin).abs() * col_norms
                top_in_ch = torch.argsort(attrib, descending=True)[:8].tolist()

            w_name = f"{lname}.weight" if lname else "weight"
            divergent.append(
                DivergentLayerSummary(
                    layer_name=lname or "<root>",
                    weight_param_name=w_name,
                    activation_l2_diff=round(l2_diff, 6),
                    cosine_shift=round(1.0 - cos_sim, 6),
                    defect_direction=[round(float(v), 6) for v in unit_dir],
                    top_divergent_channels=[int(c) for c in top_ch],
                    top_input_channels=[int(c) for c in top_in_ch],
                )
            )

        divergent.sort(key=lambda d: d.activation_l2_diff, reverse=True)
        total_params = sum(int(p.numel()) for p in model.parameters())
        digests = cls.inspect_state_dict({k: v for k, v in model.state_dict().items()}) if compute_digests else []
        return ContrastiveCircuitReport(
            total_parameters=total_params,
            inspected_layers=len(target_modules),
            top_divergent_layers=divergent,
            health_digests=digests,
        )
