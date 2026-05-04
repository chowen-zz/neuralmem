"""NeuralMem Enterprise — V1.4 enterprise-grade features.

Exports:
    TenantManager      — multi-tenant isolation and scoped storage
    AuditLogger        — compliance-grade audit trail (GDPR / SOC2)
    RBACManager        — role-based access control for memories
    ComplianceManager  — unified SOC2 / HIPAA / GDPR / ISO27001 compliance
    SSOProvider        — SSO base class (SAML 2.0 / OIDC)
    SAMLProvider       — mock-based SAML 2.0 provider
    OIDCProvider       — mock-based OpenID Connect provider
    SSOSessionManager  — in-memory SSO session manager
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
from neuralmem.enterprise.compliance import (
    AccessControlPolicy,
    ComplianceManager,
    ComplianceReport,
    ComplianceReportGenerator,
    DataEncryption,
    PolicyDecision,
    PolicyEngine,
    RiskAssessor,
    RiskFinding,
    RiskLevel,
)
from neuralmem.enterprise.rbac import (
    Grant,
    Permission,
    RBACManager,
    ResourceAction,
    Role,
)
from neuralmem.enterprise.sso import (
    IdentityMappingError,
    OIDCClaims,
    OIDCProvider,
    ProviderConfigurationError,
    SAMLAssertion,
    SAMLProvider,
    SSOError,
    SSOMethod,
    SSOProvider,
    SSOSession,
    SSOSessionManager,
    SSOToken,
    SSOUserIdentity,
    SessionExpiredError,
    TokenValidationError,
)
from neuralmem.enterprise.tenant import TenantConfig, TenantContext, TenantManager

__all__ = [
    "AccessControlPolicy",
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "ComplianceManager",
    "ComplianceReport",
    "ComplianceReportGenerator",
    "ComplianceStandard",
    "DataEncryption",
    "Grant",
    "IdentityMappingError",
    "OIDCClaims",
    "OIDCProvider",
    "Permission",
    "PolicyDecision",
    "PolicyEngine",
    "ProviderConfigurationError",
    "RBACManager",
    "ResourceAction",
    "RiskAssessor",
    "RiskFinding",
    "RiskLevel",
    "Role",
    "SAMLAssertion",
    "SAMLProvider",
    "SSOError",
    "SSOMethod",
    "SSOProvider",
    "SSOSession",
    "SSOSessionManager",
    "SSOToken",
    "SSOUserIdentity",
    "SessionExpiredError",
    "TenantConfig",
    "TenantContext",
    "TenantManager",
    "TokenValidationError",
]
