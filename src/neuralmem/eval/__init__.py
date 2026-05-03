"""NeuralMem evaluation framework."""
from __future__ import annotations

from neuralmem.eval.dataset import EvalDataset, load_dataset
from neuralmem.eval.metrics import (
    mrr,
    p95_latency,
    precision_at_k,
    recall_at_k,
    stale_hit_rate,
)
from neuralmem.eval.runner import EvalReport, EvalRunner, check_regression

__all__ = [
    "EvalDataset",
    "EvalReport",
    "EvalRunner",
    "check_regression",
    "load_dataset",
    "mrr",
    "p95_latency",
    "precision_at_k",
    "recall_at_k",
    "stale_hit_rate",
]
