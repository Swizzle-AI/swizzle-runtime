"""
Phase 2: Constrained JSON Weight-Patch DSL
==========================================
Strict Pydantic + JSON Schema specification (compatible with `swizzle-rt serve`
`score_tree` grammar decoding) describing atomic, auditable weight edits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ScaleSliceOp(BaseModel):
    op: Literal["scale_slice"] = "scale_slice"
    tensor: str = Field(..., min_length=1)
    rows: Optional[List[int]] = None  # Explicit row indices; None = all rows
    cols: Optional[List[int]] = None  # Explicit col indices; None = all cols
    factor: float = Field(..., ge=-10.0, le=10.0)


class Rank1UpdateOp(BaseModel):
    op: Literal["add_rank1_update"] = "add_rank1_update"
    tensor: str = Field(..., min_length=1)
    u: List[float] = Field(..., min_length=1)  # Length matches out_features (rows)
    v: List[float] = Field(..., min_length=1)  # Length matches in_features (cols)
    alpha: float = Field(1.0, ge=-5.0, le=5.0)
    rows: Optional[List[int]] = None  # Optional row subset restriction to keep % tiny


class ProjectOrthogonalOp(BaseModel):
    op: Literal["project_orthogonal"] = "project_orthogonal"
    tensor: str = Field(..., min_length=1)
    direction: List[float] = Field(..., min_length=1)  # Output-space or input-space vector
    axis: Literal["rows", "cols"] = "rows"  # rows => remove direction from output features
    alpha: float = Field(1.0, ge=0.0, le=2.0)
    rows: Optional[List[int]] = None


class ClampOutliersOp(BaseModel):
    op: Literal["clamp_outliers"] = "clamp_outliers"
    tensor: str = Field(..., min_length=1)
    max_abs: float = Field(..., gt=0.0, le=50.0)


class SetValuesOp(BaseModel):
    op: Literal["set_values"] = "set_values"
    tensor: str = Field(..., min_length=1)
    indices: List[List[int]] = Field(..., min_length=1)  # e.g., [[r, c], ...]
    values: List[float] = Field(..., min_length=1)


PatchOp = Union[ScaleSliceOp, Rank1UpdateOp, ProjectOrthogonalOp, ClampOutliersOp, SetValuesOp]


class WSPPatchPlan(BaseModel):
    hypothesis: str = Field(..., min_length=3, max_length=400)
    target_issue: str = Field("reasoning_repair", min_length=1, max_length=120)
    operations: List[PatchOp] = Field(..., min_length=1, max_length=16)


WSP_PATCH_JSON_SCHEMA: Dict[str, Any] = WSPPatchPlan.model_json_schema()
