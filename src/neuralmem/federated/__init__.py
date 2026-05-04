"""Federated memory learning package."""

from neuralmem.federated.aggregator import (
    AggregationAudit,
    AggregationResult,
    ConflictResolution,
    FederatedAggregator,
    ModelUpdate,
)
from neuralmem.federated.edge_node import (
    EdgeMemoryNode,
    NodeState,
    SyncMetrics,
    TrainingResult,
)
from neuralmem.federated.privacy import (
    PrivacyAuditReport,
    PrivacyBudgetManager,
    PrivacyEvent,
)

__all__ = [
    "FederatedAggregator",
    "ModelUpdate",
    "AggregationResult",
    "AggregationAudit",
    "ConflictResolution",
    "EdgeMemoryNode",
    "NodeState",
    "SyncMetrics",
    "TrainingResult",
    "PrivacyBudgetManager",
    "PrivacyEvent",
    "PrivacyAuditReport",
]
