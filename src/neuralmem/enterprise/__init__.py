"""NeuralMem Enterprise — V1.7 enterprise-grade features.

Exports:
    TenantManager          — multi-tenant isolation and scoped storage
    AuditLogger            — compliance-grade audit trail (GDPR / SOC2)
    RBACManager            — role-based access control for memories
    ComplianceManager      — unified SOC2 / HIPAA / GDPR / ISO27001 compliance
    SSOProvider            — SSO base class (SAML 2.0 / OIDC)
    SAMLProvider           — mock-based SAML 2.0 provider
    OIDCProvider           — mock-based OpenID Connect provider
    SSOSessionManager      — in-memory SSO session manager
    SOC2EvidenceCollector  — automated SOC 2 control testing & evidence
    ComplianceDashboard    — real-time compliance score & risk heatmap
    DataRetentionEnforcer  — GDPR deletion automation & tiered retention
    AuditEvent             — audit event dataclass
    AuditEventType         — audit event type enum
    ComplianceStandard     — compliance standard enum
    TenantConfig           — tenant configuration model
    Permission             — RBAC permission enum
    ResourceAction         — resource action enum
    Role                   — RBAC role dataclass
    Grant                  — RBAC grant dataclass
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
from neuralmem.enterprise.compliance_dashboard import (
    ComplianceDashboard,
    ComplianceDashboardState,
    RemediationSuggestion,
    RiskHeatmapCell,
)
from neuralmem.enterprise.retention import (
    DataRecord,
    DataRetentionEnforcer,
    DeletionRequest,
    RetentionPolicy,
    RetentionTier,
)
from neuralmem.enterprise.soc2_evidence import (
    ControlDefinition,
    ControlFrequency,
    ControlTestResult,
    EvidenceSnapshot,
    SOC2EvidenceCollector,
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
    "ComplianceDashboard",
    "ComplianceDashboardState",
    "ComplianceManager",
    "ComplianceReport",
    "ComplianceReportGenerator",
    "ComplianceStandard",
    "ControlDefinition",
    "ControlFrequency",
    "ControlTestResult",
    "DataEncryption",
    "DataRecord",
    "DataRetentionEnforcer",
    "DeletionRequest",
    "EvidenceSnapshot",
    "Grant",
    "IdentityMappingError",
    "OIDCClaims",
    "OIDCProvider",
    "Permission",
    "PolicyDecision",
    "PolicyEngine",
    "ProviderConfigurationError",
    "RBACManager",
    "RemediationSuggestion",
    "ResourceAction",
    "RetentionPolicy",
    "RetentionTier",
    "RiskAssessor",
    "RiskFinding",
    "RiskHeatmapCell",
    "RiskLevel",
    "Role",
    "SAMLAssertion",
    "SAMLProvider",
    "SOC2EvidenceCollector",
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
