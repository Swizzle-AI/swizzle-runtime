"""
Phase 3: Fast Behavioral Eval, Loop Detector & Anti-Regression Canary
=====================================================================
Scores candidate weight patches in <500ms without gradients:
1. Target task accuracy / loss + n-gram repetition loop rate.
2. Anti-regression Canary suite: blocks any patch that breaks general prompts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import torch
import torch.nn as nn


@dataclass
class EvalSnapshot:
    target_score: float
    canary_score: float
    repetition_rate: float
    latency_ms: float
    passed_canary: bool


def compute_ngram_repetition_rate(token_sequences: Sequence[Sequence[int]], n: int = 3) -> float:
    """Computes fraction of n-grams that repeat consecutively or dominantly across sequences."""
    if not token_sequences:
        return 0.0
    total_ngrams = 0
    repeated = 0
    for seq in token_sequences:
        if len(seq) < n:
            continue
        seen = set()
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i : i + n])
            total_ngrams += 1
            if gram in seen:
                repeated += 1
            else:
                seen.add(gram)
    return round(float(repeated) / float(max(1, total_ngrams)), 4)


class WSPEvaluator:
    """Evaluates target task metrics + anti-regression canary invariants."""

    def __init__(
        self,
        target_eval_fn: Callable[[nn.Module], Dict[str, float]],
        canary_eval_fn: Optional[Callable[[nn.Module], float]] = None,
        max_canary_drop: float = 0.02,
    ):
        self.target_eval_fn = target_eval_fn
        self.canary_eval_fn = canary_eval_fn
        self.max_canary_drop = float(max_canary_drop)
        self.baseline_canary: Optional[float] = None

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> EvalSnapshot:
        model.eval()
        t0 = time.perf_counter()
        target_metrics = self.target_eval_fn(model)
        target_score = float(target_metrics.get("score", 0.0))
        rep_rate = float(target_metrics.get("repetition_rate", 0.0))

        if self.canary_eval_fn is not None:
            canary_score = float(self.canary_eval_fn(model))
            if self.baseline_canary is None:
                self.baseline_canary = canary_score
            passed_canary = canary_score >= (self.baseline_canary - self.max_canary_drop)
        else:
            canary_score = 1.0
            passed_canary = True

        return EvalSnapshot(
            target_score=round(target_score, 5),
            canary_score=round(canary_score, 5),
            repetition_rate=round(rep_rate, 4),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            passed_canary=passed_canary,
        )
