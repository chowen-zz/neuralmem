"""Eval dataset loading for NeuralMem."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalDataset:
    """Evaluation dataset with queries and ground-truth memory IDs.

    Attributes:
        queries: List of query strings.
        ground_truth: Per-query list of relevant memory IDs.
        metadata: Arbitrary dataset metadata.
    """

    queries: list[str]
    ground_truth: list[list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.queries) != len(self.ground_truth):
            raise ValueError(
                f"queries ({len(self.queries)}) and ground_truth "
                f"({len(self.ground_truth)}) must have the same length"
            )

    def __len__(self) -> int:
        return len(self.queries)


def load_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON file.

    Expected format::

        {
            "queries": [
                {"query": "...", "relevant_ids": ["id1", "id2", ...]},
                ...
            ],
            "metadata": { ... }
        }

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed EvalDataset.
    """
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    queries: list[str] = []
    ground_truth: list[list[str]] = []
    for entry in raw["queries"]:
        queries.append(entry["query"])
        ground_truth.append(list(entry.get("relevant_ids", [])))

    metadata = raw.get("metadata", {})
    return EvalDataset(queries=queries, ground_truth=ground_truth, metadata=metadata)
