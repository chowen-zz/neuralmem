"""NeuralMem connectors — unified sync interface for external data sources."""
from __future__ import annotations

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem
from neuralmem.connectors.registry import ConnectorRegistry

__all__ = [
    "ConnectorProtocol",
    "ConnectorState",
    "ConnectorRegistry",
    "SyncItem",
]
