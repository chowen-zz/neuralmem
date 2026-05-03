"""NeuralMem Enterprise — V1.3 enterprise-grade features.

Exports:
    TenantManager      — multi-tenant isolation and scoped storage
    AuditLogger        — compliance-grade audit trail (GDPR / SOC2)
    RBACManager        — role-based access control for memories
    AuditEvent         — audit event dataclass
    AuditEventType     — audit event type enum
    ComplianceStandard — compliance standard enum
    TenantConfig       — tenant configuration model
    Permission         — RBAC permission enum
    ResourceAction     — resource action enum
    Role               — RBAC role dataclass
    Grant              — RBAC grant dataclass
"""
from __future__ import annotations

from neuralmem.enterprise.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    ComplianceStandard,
)
from neuralmem.enterprise.rbac import (
    Grant,
    Permission,
    RBACManager,
    ResourceAction,
    Role,
)
from neuralmem.enterprise.tenant import TenantConfig, TenantContext, TenantManager

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "ComplianceStandard",
    "Grant",
    "Permission",
    "RBACManager",
    "ResourceAction",
    "Role",
    "TenantConfig",
    "TenantContext",
    "TenantManager",
]
