"""SSO/SAML module — V1.4 enterprise single sign-on support.

Provides mock-based SAML 2.0 and OpenID Connect (OIDC) providers with:
- Token validation
- User identity mapping
- Session management
- RBAC integration

No external SAML/OIDC libraries are required; all protocol logic is
implemented with Python stdlib and ``unittest.mock`` for testability.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any

from neuralmem.enterprise.rbac import RBACManager, ResourceAction


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class SSOError(Exception):
    """Base exception for SSO-related errors."""


class TokenValidationError(SSOError):
    """Raised when a token fails validation."""


class SessionExpiredError(SSOError):
    """Raised when an SSO session has expired."""


class IdentityMappingError(SSOError):
    """Raised when user identity mapping fails."""


class ProviderConfigurationError(SSOError):
    """Raised when the provider is misconfigured."""


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

class SSOMethod(Enum):
    """Supported SSO authentication methods."""

    SAML = auto()
    OIDC = auto()


@dataclass(frozen=True)
class SSOUserIdentity:
    """Normalized user identity returned by any SSO provider."""

    provider_user_id: str
    email: str
    display_name: str = ""
    groups: tuple[str, ...] = field(default_factory=tuple)
    attributes: dict[str, Any] = field(default_factory=dict)
    raw_claims: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class SSOToken:
    """An access or ID token issued by an SSO provider."""

    token_type: str  # "Bearer", "SAMLAssertion", etc.
    access_token: str
    expires_in: int
    id_token: str | None = None
    refresh_token: str | None = None
    scope: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the token has expired."""
        return datetime.now(timezone.utc) > self.created_at + timedelta(
            seconds=self.expires_in
        )

    @property
    def expires_at(self) -> datetime:
        """Return the absolute expiration time."""
        return self.created_at + timedelta(seconds=self.expires_in)


@dataclass
class SSOSession:
    """An authenticated SSO session tied to a user and tenant."""

    session_id: str
    tenant_id: str
    user_id: str
    provider_name: str
    method: SSOMethod
    identity: SSOUserIdentity
    token: SSOToken | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_age_seconds: int = 3600  # default 1 hour

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the session has exceeded its max age."""
        return datetime.now(timezone.utc) > self.created_at + timedelta(
            seconds=self.max_age_seconds
        )

    def touch(self) -> None:
        """Update the last-accessed timestamp."""
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class SAMLAssertion:
    """Parsed SAML 2.0 assertion (mock-based representation)."""

    assertion_id: str
    issuer: str
    subject: str  # NameID / user identifier
    audience: str
    issue_instant: datetime
    not_before: datetime
    not_on_or_after: datetime
    attributes: dict[str, list[str]] = field(default_factory=dict)
    signature_valid: bool = False

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the assertion is no longer valid."""
        now = datetime.now(timezone.utc)
        return now < self.not_before or now >= self.not_on_or_after


@dataclass
class OIDCClaims:
    """Parsed OIDC ID token claims (mock-based representation)."""

    iss: str  # issuer
    sub: str  # subject
    aud: str  # audience
    exp: datetime  # expiration
    iat: datetime  # issued at
    email: str = ""
    name: str = ""
    groups: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the claims have expired."""
        return datetime.now(timezone.utc) > self.exp


# ------------------------------------------------------------------
# Base provider
# ------------------------------------------------------------------

class SSOProvider(ABC):
    """Abstract base class for SSO providers.

    Parameters
    ----------
    provider_name
        Human-readable name (e.g. ``"Okta"``, ``"Azure AD"``).
    tenant_id
        The NeuralMem tenant this provider serves.
    rbac
        Optional ``RBACManager`` for automatic role assignment.
    """

    def __init__(
        self,
        provider_name: str,
        tenant_id: str,
        rbac: RBACManager | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.tenant_id = tenant_id
        self.rbac = rbac
        self._lock = threading.RLock()

    @property
    @abstractmethod
    def method(self) -> SSOMethod:
        """Return the SSO method implemented by this provider."""

    @abstractmethod
    def authenticate(self, raw_payload: str) -> SSOSession:
        """Authenticate a raw SSO payload and return a session.

        Raises ``SSOError`` subclasses on failure.
        """

    @abstractmethod
    def validate_token(self, token: SSOToken) -> bool:
        """Validate an access/ID token."""

    def map_identity(self, identity: SSOUserIdentity) -> str:
        """Map an SSO identity to a NeuralMem *user_id*.

        The default implementation uses the e-mail address.  Subclasses
        may override this to support custom mappings (e.g. username
        extraction, group-to-role translation).
        """
        if not identity.email:
            raise IdentityMappingError("SSO identity lacks an email address")
        # Normalise to lower-case local part
        return identity.email.lower().strip()

    def assign_roles(
        self,
        user_id: str,
        identity: SSOUserIdentity,
        group_role_mapping: dict[str, str] | None = None,
    ) -> list[str]:
        """Assign RBAC roles based on SSO group membership.

        Returns the list of role names that were assigned.
        """
        if self.rbac is None:
            return []
        assigned: list[str] = []
        mapping = group_role_mapping or {}
        with self._lock:
            for group in identity.groups:
                role_name = mapping.get(group)
                if role_name and self.rbac.get_role(role_name):
                    self.rbac.grant(self.tenant_id, user_id, role_name)
                    assigned.append(role_name)
        return assigned

    def _generate_session_id(self) -> str:
        """Generate a cryptographically random session id."""
        return secrets.token_urlsafe(32)

    def _hash_token(self, token: str) -> str:
        """Return a SHA-256 hash of *token* for safe comparison/storage."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# SAML 2.0 provider (mock-based)
# ------------------------------------------------------------------

class SAMLProvider(SSOProvider):
    """Mock-based SAML 2.0 identity provider.

    Expects *raw_payload* to be a base-64 encoded SAMLResponse string.
    In a real implementation this would use ``xmlsec`` / ``lxml`` to
    decrypt and verify the XML assertion; here we simulate the parsed
    result with a small internal registry so tests remain fast and
    library-free.
    """

    def __init__(
        self,
        provider_name: str,
        tenant_id: str,
        idp_issuer: str,
        sp_audience: str,
        rbac: RBACManager | None = None,
        clock_skew_seconds: int = 300,
    ) -> None:
        super().__init__(provider_name, tenant_id, rbac)
        self.idp_issuer = idp_issuer
        self.sp_audience = sp_audience
        self.clock_skew_seconds = clock_skew_seconds
        # In-memory "assertion registry" used by tests to pre-seed valid assertions.
        self._assertions: dict[str, SAMLAssertion] = {}

    @property
    def method(self) -> SSOMethod:
        return SSOMethod.SAML

    # -- test helpers ------------------------------------------------

    def register_assertion(self, assertion: SAMLAssertion) -> None:
        """Register a parsed assertion so ``authenticate`` can resolve it.

        This is intended for **tests and demos**; a production provider
        would parse and verify the XML response instead.
        """
        with self._lock:
            self._assertions[assertion.assertion_id] = assertion

    def unregister_assertion(self, assertion_id: str) -> bool:
        """Remove a registered assertion."""
        with self._lock:
            return self._assertions.pop(assertion_id, None) is not None

    # -- authentication flow -----------------------------------------

    def authenticate(self, raw_payload: str) -> SSOSession:
        """Authenticate a SAMLResponse payload.

        *raw_payload* is treated as the assertion ID in this mock
        implementation.  In production it would be the full base-64
        encoded XML blob.
        """
        with self._lock:
            assertion = self._assertions.get(raw_payload)
        if assertion is None:
            raise TokenValidationError("SAML assertion not found or invalid")

        self._validate_assertion(assertion)

        identity = SSOUserIdentity(
            provider_user_id=assertion.subject,
            email=assertion.attributes.get("Email", [assertion.subject])[0],
            display_name=assertion.attributes.get("DisplayName", [""])[0],
            groups=tuple(assertion.attributes.get("Groups", [])),
            attributes={k: v for k, v in assertion.attributes.items()},
            raw_claims={
                "assertion_id": assertion.assertion_id,
                "issuer": assertion.issuer,
                "audience": assertion.audience,
            },
        )

        user_id = self.map_identity(identity)
        session = SSOSession(
            session_id=self._generate_session_id(),
            tenant_id=self.tenant_id,
            user_id=user_id,
            provider_name=self.provider_name,
            method=self.method,
            identity=identity,
            token=SSOToken(
                token_type="SAMLAssertion",
                access_token=raw_payload,
                expires_in=self.clock_skew_seconds,
            ),
        )

        # Auto-assign roles if RBAC is configured
        group_map = {g: g.lower() for g in identity.groups}
        self.assign_roles(user_id, identity, group_map)

        return session

    def validate_token(self, token: SSOToken) -> bool:
        """Validate a SAML assertion token."""
        if token.token_type != "SAMLAssertion":
            return False
        if token.is_expired:
            return False
        with self._lock:
            assertion = self._assertions.get(token.access_token)
        if assertion is None:
            return False
        try:
            self._validate_assertion(assertion)
            return True
        except SSOError:
            return False

    # -- internal helpers --------------------------------------------

    def _validate_assertion(self, assertion: SAMLAssertion) -> None:
        """Run all validation checks on a SAML assertion."""
        if not assertion.signature_valid:
            raise TokenValidationError("SAML assertion signature is invalid")
        if assertion.issuer != self.idp_issuer:
            raise TokenValidationError(
                f"SAML issuer mismatch: expected {self.idp_issuer}, got {assertion.issuer}"
            )
        if assertion.audience != self.sp_audience:
            raise TokenValidationError(
                f"SAML audience mismatch: expected {self.sp_audience}, got {assertion.audience}"
            )
        if assertion.is_expired:
            raise TokenValidationError("SAML assertion has expired")


# ------------------------------------------------------------------
# OIDC provider (mock-based)
# ------------------------------------------------------------------

class OIDCProvider(SSOProvider):
    """Mock-based OpenID Connect identity provider.

    Expects *raw_payload* to be a JSON-serialised ``OIDCClaims`` dict
    or a JWT-like string.  The provider maintains an in-memory claims
    registry for testability.
    """

    def __init__(
        self,
        provider_name: str,
        tenant_id: str,
        issuer_url: str,
        client_id: str,
        rbac: RBACManager | None = None,
        clock_skew_seconds: int = 300,
    ) -> None:
        super().__init__(provider_name, tenant_id, rbac)
        self.issuer_url = issuer_url
        self.client_id = client_id
        self.clock_skew_seconds = clock_skew_seconds
        self._claims: dict[str, OIDCClaims] = {}
        self._jwks: dict[str, str] = {}  # mock key id -> public key pem

    @property
    def method(self) -> SSOMethod:
        return SSOMethod.OIDC

    # -- test helpers ------------------------------------------------

    def register_claims(self, claims: OIDCClaims) -> str:
        """Register OIDC claims and return a mock JWT token string."""
        token = self._mock_jwt(claims)
        with self._lock:
            self._claims[token] = claims
        return token

    def unregister_claims(self, token: str) -> bool:
        """Remove registered claims."""
        with self._lock:
            return self._claims.pop(token, None) is not None

    def add_signing_key(self, key_id: str, public_key_pem: str) -> None:
        """Add a mock signing key to the JWKS store."""
        with self._lock:
            self._jwks[key_id] = public_key_pem

    # -- authentication flow -----------------------------------------

    def authenticate(self, raw_payload: str) -> SSOSession:
        """Authenticate an OIDC ID token / authorisation code.

        *raw_payload* is treated as a mock JWT token in this
        implementation.  In production it would be a real JWT verified
        against the IdP's JWKS endpoint.
        """
        with self._lock:
            claims = self._claims.get(raw_payload)
        if claims is None:
            raise TokenValidationError("OIDC claims not found or invalid")

        self._validate_claims(claims)

        identity = SSOUserIdentity(
            provider_user_id=claims.sub,
            email=claims.email or claims.sub,
            display_name=claims.name,
            groups=tuple(claims.groups),
            attributes={k: v for k, v in claims.extra.items()},
            raw_claims={
                "iss": claims.iss,
                "sub": claims.sub,
                "aud": claims.aud,
                "exp": claims.exp.isoformat(),
            },
        )

        user_id = self.map_identity(identity)
        session = SSOSession(
            session_id=self._generate_session_id(),
            tenant_id=self.tenant_id,
            user_id=user_id,
            provider_name=self.provider_name,
            method=self.method,
            identity=identity,
            token=SSOToken(
                token_type="Bearer",
                access_token=secrets.token_urlsafe(32),
                id_token=raw_payload,
                expires_in=self.clock_skew_seconds,
                scope="openid profile email",
            ),
        )

        group_map = {g: g.lower() for g in identity.groups}
        self.assign_roles(user_id, identity, group_map)

        return session

    def validate_token(self, token: SSOToken) -> bool:
        """Validate an OIDC Bearer / ID token."""
        if token.token_type not in ("Bearer", "IDToken"):
            return False
        if token.is_expired:
            return False
        if token.id_token is None:
            return False
        with self._lock:
            claims = self._claims.get(token.id_token)
        if claims is None:
            return False
        try:
            self._validate_claims(claims)
            return True
        except SSOError:
            return False

    # -- internal helpers --------------------------------------------

    def _validate_claims(self, claims: OIDCClaims) -> None:
        """Run all validation checks on OIDC claims."""
        if claims.iss != self.issuer_url:
            raise TokenValidationError(
                f"OIDC issuer mismatch: expected {self.issuer_url}, got {claims.iss}"
            )
        if claims.aud != self.client_id:
            raise TokenValidationError(
                f"OIDC audience mismatch: expected {self.client_id}, got {claims.aud}"
            )
        if claims.is_expired:
            raise TokenValidationError("OIDC token has expired")
        # Clock-skew tolerance
        now = datetime.now(timezone.utc)
        if claims.iat > now + timedelta(seconds=self.clock_skew_seconds):
            raise TokenValidationError("OIDC token issued in the future")

    def _mock_jwt(self, claims: OIDCClaims) -> str:
        """Create a mock JWT string from claims (no real crypto)."""
        header = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"  # alg=none
        payload = (
            f"{claims.sub}:{claims.iss}:{claims.aud}:"
            f"{int(claims.exp.timestamp())}"
        )
        signature = secrets.token_urlsafe(16)
        return f"{header}.{self._b64(payload)}.{signature}"

    @staticmethod
    def _b64(text: str) -> str:
        """Minimal base-64-like encoding for mock JWTs."""
        import base64
        return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


# ------------------------------------------------------------------
# Session manager
# ------------------------------------------------------------------

class SSOSessionManager:
    """In-memory session manager for authenticated SSO users.

    Parameters
    ----------
    default_max_age_seconds
        Default session lifetime (may be overridden per-session).
    """

    def __init__(self, default_max_age_seconds: int = 3600) -> None:
        self._sessions: dict[str, SSOSession] = {}
        self._lock = threading.RLock()
        self.default_max_age_seconds = default_max_age_seconds

    # -- CRUD --------------------------------------------------------

    def create_session(self, session: SSOSession) -> SSOSession:
        """Store a new session and return it."""
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> SSOSession | None:
        """Retrieve a session by id, or ``None`` if not found / expired."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self.destroy_session(session_id)
            raise SessionExpiredError(f"Session '{session_id}' has expired")
        session.touch()
        return session

    def destroy_session(self, session_id: str) -> bool:
        """Remove a session.  Returns ``True`` if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(
        self,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> list[SSOSession]:
        """List sessions, optionally filtered by tenant and/or user."""
        with self._lock:
            results: list[SSOSession] = []
            for session in self._sessions.values():
                if tenant_id is not None and session.tenant_id != tenant_id:
                    continue
                if user_id is not None and session.user_id != user_id:
                    continue
                results.append(session)
            return results

    def destroy_all_sessions(
        self,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Destroy all matching sessions.  Returns the number destroyed."""
        with self._lock:
            to_remove = [
                sid
                for sid, s in self._sessions.items()
                if (tenant_id is None or s.tenant_id == tenant_id)
                and (user_id is None or s.user_id == user_id)
            ]
            for sid in to_remove:
                del self._sessions[sid]
            return len(to_remove)

    # -- housekeeping ------------------------------------------------

    def expire_stale_sessions(self) -> int:
        """Remove all expired sessions.  Returns the number removed."""
        with self._lock:
            stale = [
                sid
                for sid, s in self._sessions.items()
                if s.is_expired
            ]
            for sid in stale:
                del self._sessions[sid]
            return len(stale)

    def session_count(self) -> int:
        """Return the total number of active sessions."""
        with self._lock:
            return len(self._sessions)

    # -- convenience helpers -----------------------------------------

    def is_authenticated(self, session_id: str) -> bool:
        """Return ``True`` if the session exists and is not expired."""
        try:
            return self.get_session(session_id) is not None
        except SessionExpiredError:
            return False

    def get_user_id(self, session_id: str) -> str | None:
        """Return the user_id for a session, or ``None``."""
        session = self.get_session(session_id)
        return session.user_id if session else None

    def get_tenant_id(self, session_id: str) -> str | None:
        """Return the tenant_id for a session, or ``None``."""
        session = self.get_session(session_id)
        return session.tenant_id if session else None

    def require_session(self, session_id: str) -> SSOSession:
        """Return the session or raise ``SessionExpiredError``."""
        session = self.get_session(session_id)
        if session is None:
            raise SessionExpiredError(f"Session '{session_id}' not found")
        return session

    def check_permission(
        self,
        session_id: str,
        rbac: RBACManager,
        resource: str,
        action: ResourceAction,
    ) -> bool:
        """Check whether the session's user has *action* on *resource*."""
        session = self.get_session(session_id)
        if session is None:
            return False
        return rbac.check(session.tenant_id, session.user_id, resource, action)
