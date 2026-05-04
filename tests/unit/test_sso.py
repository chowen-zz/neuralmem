"""Unit tests for NeuralMem V1.4 SSO/SAML module.

All external dependencies are mocked; tests are fast and isolated.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.enterprise import (
    IdentityMappingError,
    OIDCClaims,
    OIDCProvider,
    RBACManager,
    ResourceAction,
    Role,
    SAMLAssertion,
    SAMLProvider,
    SSOMethod,
    SSOSession,
    SSOSessionManager,
    SSOToken,
    SSOUserIdentity,
    SessionExpiredError,
    TokenValidationError,
)


# ------------------------------------------------------------------
# SSOUserIdentity tests
# ------------------------------------------------------------------

class TestSSOUserIdentity:
    def test_identity_creation(self):
        identity = SSOUserIdentity(
            provider_user_id="u123",
            email="alice@example.com",
            display_name="Alice",
            groups=("admins", "users"),
            attributes={"department": "engineering"},
        )
        assert identity.provider_user_id == "u123"
        assert identity.email == "alice@example.com"
        assert identity.display_name == "Alice"
        assert identity.groups == ("admins", "users")
        assert identity.attributes["department"] == "engineering"

    def test_identity_immutable(self):
        identity = SSOUserIdentity(
            provider_user_id="u123",
            email="alice@example.com",
        )
        # frozen dataclass
        with pytest.raises(AttributeError):
            identity.email = "bob@example.com"


# ------------------------------------------------------------------
# SSOToken tests
# ------------------------------------------------------------------

class TestSSOToken:
    def test_token_not_expired(self):
        token = SSOToken(
            token_type="Bearer",
            access_token="tok123",
            expires_in=3600,
        )
        assert token.is_expired is False

    def test_token_expired(self):
        token = SSOToken(
            token_type="Bearer",
            access_token="tok123",
            expires_in=-1,
        )
        assert token.is_expired is True

    def test_token_expires_at(self):
        now = datetime.now(timezone.utc)
        token = SSOToken(
            token_type="Bearer",
            access_token="tok123",
            expires_in=300,
            created_at=now,
        )
        assert token.expires_at == now + timedelta(seconds=300)


# ------------------------------------------------------------------
# SSOSession tests
# ------------------------------------------------------------------

class TestSSOSession:
    def test_session_not_expired(self):
        identity = SSOUserIdentity(provider_user_id="u1", email="a@b.com")
        session = SSOSession(
            session_id="s1",
            tenant_id="acme",
            user_id="alice",
            provider_name="Okta",
            method=SSOMethod.OIDC,
            identity=identity,
            max_age_seconds=3600,
        )
        assert session.is_expired is False

    def test_session_expired(self):
        identity = SSOUserIdentity(provider_user_id="u1", email="a@b.com")
        session = SSOSession(
            session_id="s1",
            tenant_id="acme",
            user_id="alice",
            provider_name="Okta",
            method=SSOMethod.OIDC,
            identity=identity,
            max_age_seconds=-1,
        )
        assert session.is_expired is True

    def test_session_touch_updates_last_accessed(self):
        identity = SSOUserIdentity(provider_user_id="u1", email="a@b.com")
        session = SSOSession(
            session_id="s1",
            tenant_id="acme",
            user_id="alice",
            provider_name="Okta",
            method=SSOMethod.OIDC,
            identity=identity,
        )
        old = session.last_accessed
        session.touch()
        assert session.last_accessed > old


# ------------------------------------------------------------------
# SAMLAssertion tests
# ------------------------------------------------------------------

class TestSAMLAssertion:
    def test_assertion_valid_window(self):
        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="a1",
            issuer="https://idp.example.com",
            subject="alice@example.com",
            audience="https://sp.example.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
        )
        assert assertion.is_expired is False

    def test_assertion_expired(self):
        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="a1",
            issuer="https://idp.example.com",
            subject="alice@example.com",
            audience="https://sp.example.com",
            issue_instant=now - timedelta(hours=2),
            not_before=now - timedelta(hours=2),
            not_on_or_after=now - timedelta(hours=1),
            signature_valid=True,
        )
        assert assertion.is_expired is True

    def test_assertion_not_yet_valid(self):
        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="a1",
            issuer="https://idp.example.com",
            subject="alice@example.com",
            audience="https://sp.example.com",
            issue_instant=now,
            not_before=now + timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=10),
            signature_valid=True,
        )
        assert assertion.is_expired is True


# ------------------------------------------------------------------
# OIDCClaims tests
# ------------------------------------------------------------------

class TestOIDCClaims:
    def test_claims_not_expired(self):
        now = datetime.now(timezone.utc)
        claims = OIDCClaims(
            iss="https://oidc.example.com",
            sub="user123",
            aud="client-id",
            exp=now + timedelta(minutes=5),
            iat=now,
            email="alice@example.com",
            name="Alice",
        )
        assert claims.is_expired is False

    def test_claims_expired(self):
        now = datetime.now(timezone.utc)
        claims = OIDCClaims(
            iss="https://oidc.example.com",
            sub="user123",
            aud="client-id",
            exp=now - timedelta(minutes=5),
            iat=now - timedelta(minutes=10),
        )
        assert claims.is_expired is True


# ------------------------------------------------------------------
# SAMLProvider tests
# ------------------------------------------------------------------

class TestSAMLProvider:
    def _make_provider(self, rbac=None):
        return SAMLProvider(
            provider_name="Mock IdP",
            tenant_id="acme",
            idp_issuer="https://idp.example.com",
            sp_audience="https://sp.example.com",
            rbac=rbac,
        )

    def _make_assertion(self, **overrides):
        now = datetime.now(timezone.utc)
        defaults = dict(
            assertion_id="assert-1",
            issuer="https://idp.example.com",
            subject="alice@example.com",
            audience="https://sp.example.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
            attributes={
                "Email": ["alice@example.com"],
                "DisplayName": ["Alice Smith"],
                "Groups": ["admins", "users"],
            },
        )
        defaults.update(overrides)
        return SAMLAssertion(**defaults)

    def test_authenticate_success(self):
        provider = self._make_provider()
        assertion = self._make_assertion()
        provider.register_assertion(assertion)

        session = provider.authenticate("assert-1")
        assert session.tenant_id == "acme"
        assert session.user_id == "alice@example.com"
        assert session.provider_name == "Mock IdP"
        assert session.method == SSOMethod.SAML
        assert session.identity.display_name == "Alice Smith"
        assert session.identity.groups == ("admins", "users")

    def test_authenticate_missing_assertion(self):
        provider = self._make_provider()
        with pytest.raises(TokenValidationError, match="not found"):
            provider.authenticate("missing")

    def test_authenticate_invalid_signature(self):
        provider = self._make_provider()
        assertion = self._make_assertion(signature_valid=False)
        provider.register_assertion(assertion)
        with pytest.raises(TokenValidationError, match="signature"):
            provider.authenticate("assert-1")

    def test_authenticate_issuer_mismatch(self):
        provider = self._make_provider()
        assertion = self._make_assertion(issuer="https://evil.com")
        provider.register_assertion(assertion)
        with pytest.raises(TokenValidationError, match="issuer mismatch"):
            provider.authenticate("assert-1")

    def test_authenticate_audience_mismatch(self):
        provider = self._make_provider()
        assertion = self._make_assertion(audience="https://other.com")
        provider.register_assertion(assertion)
        with pytest.raises(TokenValidationError, match="audience mismatch"):
            provider.authenticate("assert-1")

    def test_authenticate_expired_assertion(self):
        provider = self._make_provider()
        now = datetime.now(timezone.utc)
        assertion = self._make_assertion(
            not_before=now - timedelta(hours=2),
            not_on_or_after=now - timedelta(hours=1),
        )
        provider.register_assertion(assertion)
        with pytest.raises(TokenValidationError, match="expired"):
            provider.authenticate("assert-1")

    def test_validate_token_success(self):
        provider = self._make_provider()
        assertion = self._make_assertion()
        provider.register_assertion(assertion)
        token = SSOToken(
            token_type="SAMLAssertion",
            access_token="assert-1",
            expires_in=300,
        )
        assert provider.validate_token(token) is True

    def test_validate_token_wrong_type(self):
        provider = self._make_provider()
        token = SSOToken(
            token_type="Bearer",
            access_token="assert-1",
            expires_in=300,
        )
        assert provider.validate_token(token) is False

    def test_validate_token_expired(self):
        provider = self._make_provider()
        assertion = self._make_assertion()
        provider.register_assertion(assertion)
        token = SSOToken(
            token_type="SAMLAssertion",
            access_token="assert-1",
            expires_in=-1,
        )
        assert provider.validate_token(token) is False

    def test_validate_token_missing_assertion(self):
        provider = self._make_provider()
        token = SSOToken(
            token_type="SAMLAssertion",
            access_token="missing",
            expires_in=300,
        )
        assert provider.validate_token(token) is False

    def test_map_identity_default(self):
        provider = self._make_provider()
        identity = SSOUserIdentity(
            provider_user_id="u1",
            email="Alice@Example.COM",
        )
        assert provider.map_identity(identity) == "alice@example.com"

    def test_map_identity_missing_email(self):
        provider = self._make_provider()
        identity = SSOUserIdentity(provider_user_id="u1", email="")
        with pytest.raises(IdentityMappingError, match="lacks an email"):
            provider.map_identity(identity)

    def test_assign_roles_with_rbac(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="admins", permissions={"memory": 4}))
        provider = self._make_provider(rbac=rbac)
        identity = SSOUserIdentity(
            provider_user_id="u1",
            email="alice@example.com",
            groups=("admins",),
        )
        assigned = provider.assign_roles("alice@example.com", identity, {"admins": "admins"})
        assert "admins" in assigned
        assert rbac.check("acme", "alice@example.com", "memory", ResourceAction.ADMIN)

    def test_assign_roles_without_rbac(self):
        provider = self._make_provider()
        identity = SSOUserIdentity(
            provider_user_id="u1",
            email="alice@example.com",
            groups=("admins",),
        )
        assigned = provider.assign_roles("alice@example.com", identity)
        assert assigned == []

    def test_unregister_assertion(self):
        provider = self._make_provider()
        assertion = self._make_assertion()
        provider.register_assertion(assertion)
        assert provider.unregister_assertion("assert-1") is True
        assert provider.unregister_assertion("assert-1") is False

    def test_provider_method(self):
        provider = self._make_provider()
        assert provider.method == SSOMethod.SAML

    def test_authenticate_auto_assign_roles(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="admins", permissions={"memory": 4}))
        provider = self._make_provider(rbac=rbac)
        assertion = self._make_assertion()
        provider.register_assertion(assertion)
        session = provider.authenticate("assert-1")
        assert rbac.has_any_grant("acme", session.user_id)

    def test_generate_session_id_unique(self):
        provider = self._make_provider()
        ids = {provider._generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_hash_token(self):
        provider = self._make_provider()
        h1 = provider._hash_token("secret")
        h2 = provider._hash_token("secret")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


# ------------------------------------------------------------------
# OIDCProvider tests
# ------------------------------------------------------------------

class TestOIDCProvider:
    def _make_provider(self, rbac=None):
        return OIDCProvider(
            provider_name="Mock OIDC",
            tenant_id="acme",
            issuer_url="https://oidc.example.com",
            client_id="client-id",
            rbac=rbac,
        )

    def _make_claims(self, **overrides):
        now = datetime.now(timezone.utc)
        defaults = dict(
            iss="https://oidc.example.com",
            sub="user123",
            aud="client-id",
            exp=now + timedelta(minutes=5),
            iat=now,
            email="alice@example.com",
            name="Alice",
            groups=["admins", "users"],
        )
        defaults.update(overrides)
        return OIDCClaims(**defaults)

    def test_authenticate_success(self):
        provider = self._make_provider()
        claims = self._make_claims()
        token = provider.register_claims(claims)

        session = provider.authenticate(token)
        assert session.tenant_id == "acme"
        assert session.user_id == "alice@example.com"
        assert session.provider_name == "Mock OIDC"
        assert session.method == SSOMethod.OIDC
        assert session.identity.display_name == "Alice"
        assert session.identity.groups == ("admins", "users")
        assert session.token is not None
        assert session.token.scope == "openid profile email"

    def test_authenticate_missing_claims(self):
        provider = self._make_provider()
        with pytest.raises(TokenValidationError, match="not found"):
            provider.authenticate("invalid-token")

    def test_authenticate_issuer_mismatch(self):
        provider = self._make_provider()
        claims = self._make_claims(iss="https://evil.com")
        token = provider.register_claims(claims)
        with pytest.raises(TokenValidationError, match="issuer mismatch"):
            provider.authenticate(token)

    def test_authenticate_audience_mismatch(self):
        provider = self._make_provider()
        claims = self._make_claims(aud="wrong-client")
        token = provider.register_claims(claims)
        with pytest.raises(TokenValidationError, match="audience mismatch"):
            provider.authenticate(token)

    def test_authenticate_expired_claims(self):
        provider = self._make_provider()
        now = datetime.now(timezone.utc)
        claims = self._make_claims(
            exp=now - timedelta(minutes=5),
            iat=now - timedelta(minutes=10),
        )
        token = provider.register_claims(claims)
        with pytest.raises(TokenValidationError, match="expired"):
            provider.authenticate(token)

    def test_authenticate_future_iat(self):
        provider = OIDCProvider(
            provider_name="Mock OIDC",
            tenant_id="acme",
            issuer_url="https://oidc.example.com",
            client_id="client-id",
            clock_skew_seconds=0,
        )
        now = datetime.now(timezone.utc)
        claims = self._make_claims(
            iat=now + timedelta(minutes=5),
            exp=now + timedelta(minutes=10),
        )
        token = provider.register_claims(claims)
        with pytest.raises(TokenValidationError, match="future"):
            provider.authenticate(token)

    def test_validate_token_success(self):
        provider = self._make_provider()
        claims = self._make_claims()
        token_str = provider.register_claims(claims)
        token = SSOToken(
            token_type="Bearer",
            access_token="acc",
            id_token=token_str,
            expires_in=300,
        )
        assert provider.validate_token(token) is True

    def test_validate_token_wrong_type(self):
        provider = self._make_provider()
        token = SSOToken(
            token_type="SAMLAssertion",
            access_token="x",
            id_token="y",
            expires_in=300,
        )
        assert provider.validate_token(token) is False

    def test_validate_token_no_id_token(self):
        provider = self._make_provider()
        token = SSOToken(
            token_type="Bearer",
            access_token="acc",
            expires_in=300,
        )
        assert provider.validate_token(token) is False

    def test_validate_token_expired(self):
        provider = self._make_provider()
        claims = self._make_claims()
        token_str = provider.register_claims(claims)
        token = SSOToken(
            token_type="Bearer",
            access_token="acc",
            id_token=token_str,
            expires_in=-1,
        )
        assert provider.validate_token(token) is False

    def test_validate_token_missing_claims(self):
        provider = self._make_provider()
        token = SSOToken(
            token_type="Bearer",
            access_token="acc",
            id_token="missing",
            expires_in=300,
        )
        assert provider.validate_token(token) is False

    def test_map_identity_default(self):
        provider = self._make_provider()
        identity = SSOUserIdentity(
            provider_user_id="u1",
            email="Bob@Example.COM",
        )
        assert provider.map_identity(identity) == "bob@example.com"

    def test_assign_roles_with_rbac(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="admins", permissions={"memory": 4}))
        provider = self._make_provider(rbac=rbac)
        identity = SSOUserIdentity(
            provider_user_id="u1",
            email="alice@example.com",
            groups=("admins",),
        )
        assigned = provider.assign_roles("alice@example.com", identity, {"admins": "admins"})
        assert "admins" in assigned
        assert rbac.check("acme", "alice@example.com", "memory", ResourceAction.ADMIN)

    def test_unregister_claims(self):
        provider = self._make_provider()
        claims = self._make_claims()
        token = provider.register_claims(claims)
        assert provider.unregister_claims(token) is True
        assert provider.unregister_claims(token) is False

    def test_add_signing_key(self):
        provider = self._make_provider()
        provider.add_signing_key("kid1", "-----BEGIN PUBLIC KEY-----\n...")
        assert "kid1" in provider._jwks

    def test_provider_method(self):
        provider = self._make_provider()
        assert provider.method == SSOMethod.OIDC

    def test_authenticate_auto_assign_roles(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="admins", permissions={"memory": 4}))
        provider = self._make_provider(rbac=rbac)
        claims = self._make_claims()
        token = provider.register_claims(claims)
        session = provider.authenticate(token)
        assert rbac.has_any_grant("acme", session.user_id)

    def test_mock_jwt_format(self):
        provider = self._make_provider()
        claims = self._make_claims()
        token = provider._mock_jwt(claims)
        parts = token.split(".")
        assert len(parts) == 3
        assert parts[0] == "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"

    def test_b64_encoding(self):
        provider = self._make_provider()
        encoded = provider._b64("hello")
        import base64
        assert base64.urlsafe_b64decode(encoded + "==") == b"hello"


# ------------------------------------------------------------------
# SSOSessionManager tests
# ------------------------------------------------------------------

class TestSSOSessionManager:
    def _make_session(self, session_id="s1", tenant_id="acme", user_id="alice", max_age=3600):
        identity = SSOUserIdentity(provider_user_id="u1", email="alice@example.com")
        return SSOSession(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            provider_name="Mock",
            method=SSOMethod.OIDC,
            identity=identity,
            max_age_seconds=max_age,
        )

    def test_create_and_get_session(self):
        mgr = SSOSessionManager()
        session = self._make_session()
        mgr.create_session(session)
        retrieved = mgr.get_session("s1")
        assert retrieved is not None
        assert retrieved.session_id == "s1"

    def test_get_missing_session(self):
        mgr = SSOSessionManager()
        assert mgr.get_session("missing") is None

    def test_get_expired_session_raises(self):
        mgr = SSOSessionManager()
        session = self._make_session(max_age=-1)
        mgr.create_session(session)
        with pytest.raises(SessionExpiredError):
            mgr.get_session("s1")

    def test_destroy_session(self):
        mgr = SSOSessionManager()
        session = self._make_session()
        mgr.create_session(session)
        assert mgr.destroy_session("s1") is True
        assert mgr.get_session("s1") is None

    def test_destroy_missing_session(self):
        mgr = SSOSessionManager()
        assert mgr.destroy_session("missing") is False

    def test_list_sessions_all(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "acme", "bob"))
        mgr.create_session(self._make_session("s3", "globex", "alice"))
        assert len(mgr.list_sessions()) == 3

    def test_list_sessions_by_tenant(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "globex", "bob"))
        assert len(mgr.list_sessions(tenant_id="acme")) == 1
        assert mgr.list_sessions(tenant_id="acme")[0].user_id == "alice"

    def test_list_sessions_by_user(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "acme", "bob"))
        assert len(mgr.list_sessions(user_id="alice")) == 1

    def test_destroy_all_sessions(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "acme", "bob"))
        assert mgr.destroy_all_sessions() == 2
        assert mgr.session_count() == 0

    def test_destroy_all_sessions_by_tenant(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "globex", "bob"))
        assert mgr.destroy_all_sessions(tenant_id="acme") == 1
        assert mgr.session_count() == 1

    def test_destroy_all_sessions_by_user(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", "acme", "alice"))
        mgr.create_session(self._make_session("s2", "globex", "alice"))
        mgr.create_session(self._make_session("s3", "acme", "bob"))
        assert mgr.destroy_all_sessions(user_id="alice") == 2
        assert mgr.session_count() == 1

    def test_expire_stale_sessions(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", max_age=3600))
        mgr.create_session(self._make_session("s2", max_age=-1))
        assert mgr.expire_stale_sessions() == 1
        assert mgr.session_count() == 1

    def test_session_count(self):
        mgr = SSOSessionManager()
        assert mgr.session_count() == 0
        mgr.create_session(self._make_session("s1"))
        assert mgr.session_count() == 1

    def test_is_authenticated_true(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1"))
        assert mgr.is_authenticated("s1") is True

    def test_is_authenticated_false(self):
        mgr = SSOSessionManager()
        assert mgr.is_authenticated("missing") is False

    def test_is_authenticated_expired(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", max_age=-1))
        assert mgr.is_authenticated("s1") is False

    def test_get_user_id(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", user_id="alice"))
        assert mgr.get_user_id("s1") == "alice"
        assert mgr.get_user_id("missing") is None

    def test_get_tenant_id(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", tenant_id="acme"))
        assert mgr.get_tenant_id("s1") == "acme"
        assert mgr.get_tenant_id("missing") is None

    def test_require_session(self):
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1"))
        session = mgr.require_session("s1")
        assert session.session_id == "s1"

    def test_require_session_missing(self):
        mgr = SSOSessionManager()
        with pytest.raises(SessionExpiredError, match="not found"):
            mgr.require_session("missing")

    def test_check_permission(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice", "viewer")
        mgr = SSOSessionManager()
        mgr.create_session(self._make_session("s1", tenant_id="acme", user_id="alice"))
        assert mgr.check_permission("s1", rbac, "memory", ResourceAction.READ) is True
        assert mgr.check_permission("s1", rbac, "memory", ResourceAction.CREATE) is False

    def test_check_permission_no_session(self):
        rbac = RBACManager()
        mgr = SSOSessionManager()
        assert mgr.check_permission("missing", rbac, "memory", ResourceAction.READ) is False

    def test_touch_on_get(self):
        mgr = SSOSessionManager()
        session = self._make_session("s1")
        mgr.create_session(session)
        old = session.last_accessed
        # small sleep to ensure time delta
        import time
        time.sleep(0.01)
        mgr.get_session("s1")
        assert session.last_accessed > old

    def test_concurrent_sessions(self):
        mgr = SSOSessionManager()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(25):
                    session = self._make_session(f"s{n}_{i}", f"tenant_{n}", f"user_{i}")
                    mgr.create_session(session)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert mgr.session_count() == 100


# ------------------------------------------------------------------
# Integration tests — SSO + RBAC + SessionManager
# ------------------------------------------------------------------

class TestSSORBACIntegration:
    def test_saml_end_to_end_with_rbac(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="engineers", permissions={"memory": 2}))
        provider = SAMLProvider(
            provider_name="Corp IdP",
            tenant_id="acme",
            idp_issuer="https://idp.corp.com",
            sp_audience="https://neuralmem.corp.com",
            rbac=rbac,
        )
        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="assert-42",
            issuer="https://idp.corp.com",
            subject="alice@corp.com",
            audience="https://neuralmem.corp.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
            attributes={
                "Email": ["alice@corp.com"],
                "DisplayName": ["Alice"],
                "Groups": ["engineers"],
            },
        )
        provider.register_assertion(assertion)

        session = provider.authenticate("assert-42")
        assert session.user_id == "alice@corp.com"

        # Store in session manager
        mgr = SSOSessionManager()
        mgr.create_session(session)

        # Verify RBAC
        assert mgr.check_permission(session.session_id, rbac, "memory", ResourceAction.CREATE)
        assert rbac.has_any_grant("acme", "alice@corp.com")

    def test_oidc_end_to_end_with_rbac(self):
        rbac = RBACManager()
        rbac.add_role(Role(name="admins", permissions={"memory": 4, "tenant": 4}))
        provider = OIDCProvider(
            provider_name="Auth0",
            tenant_id="globex",
            issuer_url="https://auth0.example.com",
            client_id="globex-client",
            rbac=rbac,
        )
        now = datetime.now(timezone.utc)
        claims = OIDCClaims(
            iss="https://auth0.example.com",
            sub="auth0|12345",
            aud="globex-client",
            exp=now + timedelta(minutes=5),
            iat=now,
            email="bob@globex.com",
            name="Bob",
            groups=["admins"],
        )
        token = provider.register_claims(claims)

        session = provider.authenticate(token)
        assert session.user_id == "bob@globex.com"

        mgr = SSOSessionManager()
        mgr.create_session(session)

        assert mgr.check_permission(session.session_id, rbac, "tenant", ResourceAction.ADMIN)

    def test_cross_tenant_isolation(self):
        rbac = RBACManager()
        rbac.grant("acme", "alice@acme.com", "admin")

        provider_acme = SAMLProvider(
            provider_name="Acme IdP",
            tenant_id="acme",
            idp_issuer="https://idp.acme.com",
            sp_audience="https://app.acme.com",
        )
        provider_globex = SAMLProvider(
            provider_name="Globex IdP",
            tenant_id="globex",
            idp_issuer="https://idp.globex.com",
            sp_audience="https://app.globex.com",
        )

        now = datetime.now(timezone.utc)
        assertion_acme = SAMLAssertion(
            assertion_id="a1",
            issuer="https://idp.acme.com",
            subject="alice@acme.com",
            audience="https://app.acme.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
            attributes={"Email": ["alice@acme.com"]},
        )
        provider_acme.register_assertion(assertion_acme)

        session_acme = provider_acme.authenticate("a1")
        mgr = SSOSessionManager()
        mgr.create_session(session_acme)

        # alice has admin in acme but not in globex
        assert mgr.check_permission(session_acme.session_id, rbac, "memory", ResourceAction.ADMIN)
        assert rbac.check("globex", "alice@acme.com", "memory", ResourceAction.ADMIN) is False

    def test_session_expiration_cleans_up(self):
        mgr = SSOSessionManager()
        session = SSOSession(
            session_id="s1",
            tenant_id="acme",
            user_id="alice",
            provider_name="Mock",
            method=SSOMethod.OIDC,
            identity=SSOUserIdentity(provider_user_id="u1", email="alice@example.com"),
            max_age_seconds=1,
        )
        mgr.create_session(session)
        assert mgr.session_count() == 1

        import time
        time.sleep(1.1)

        assert mgr.is_authenticated("s1") is False
        assert mgr.session_count() == 0  # expired session removed on access

    def test_multiple_providers_same_tenant(self):
        rbac = RBACManager()
        saml = SAMLProvider(
            provider_name="Corp SAML",
            tenant_id="acme",
            idp_issuer="https://saml.corp.com",
            sp_audience="https://app.corp.com",
            rbac=rbac,
        )
        oidc = OIDCProvider(
            provider_name="Corp OIDC",
            tenant_id="acme",
            issuer_url="https://oidc.corp.com",
            client_id="acme-client",
            rbac=rbac,
        )

        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="sa1",
            issuer="https://saml.corp.com",
            subject="saml-user@corp.com",
            audience="https://app.corp.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
            attributes={"Email": ["saml-user@corp.com"]},
        )
        saml.register_assertion(assertion)

        claims = OIDCClaims(
            iss="https://oidc.corp.com",
            sub="oidc-user",
            aud="acme-client",
            exp=now + timedelta(minutes=5),
            iat=now,
            email="oidc-user@corp.com",
        )
        oidc_token = oidc.register_claims(claims)

        session_saml = saml.authenticate("sa1")
        session_oidc = oidc.authenticate(oidc_token)

        mgr = SSOSessionManager()
        mgr.create_session(session_saml)
        mgr.create_session(session_oidc)

        assert mgr.session_count() == 2
        assert session_saml.method == SSOMethod.SAML
        assert session_oidc.method == SSOMethod.OIDC

    def test_mock_storage_backend_with_sso(self):
        """Simulate using a mock storage backend alongside SSO auth."""
        mock_backend = MagicMock()
        mock_backend.get_user.return_value = {"email": "alice@example.com", "role": "viewer"}

        provider = OIDCProvider(
            provider_name="Mock OIDC",
            tenant_id="acme",
            issuer_url="https://mock.example.com",
            client_id="client",
        )
        now = datetime.now(timezone.utc)
        claims = OIDCClaims(
            iss="https://mock.example.com",
            sub="alice",
            aud="client",
            exp=now + timedelta(minutes=5),
            iat=now,
            email="alice@example.com",
        )
        token = provider.register_claims(claims)
        session = provider.authenticate(token)

        # Simulate backend lookup
        user = mock_backend.get_user(session.user_id)
        assert user["email"] == "alice@example.com"
        mock_backend.get_user.assert_called_once_with("alice@example.com")

    def test_audit_logging_integration(self):
        """Simulate audit logging around SSO events."""
        from neuralmem.enterprise import AuditLogger, AuditEventType

        audit = AuditLogger()
        provider = SAMLProvider(
            provider_name="Corp IdP",
            tenant_id="acme",
            idp_issuer="https://idp.corp.com",
            sp_audience="https://app.corp.com",
        )
        now = datetime.now(timezone.utc)
        assertion = SAMLAssertion(
            assertion_id="a1",
            issuer="https://idp.corp.com",
            subject="alice@corp.com",
            audience="https://app.corp.com",
            issue_instant=now,
            not_before=now - timedelta(minutes=5),
            not_on_or_after=now + timedelta(minutes=5),
            signature_valid=True,
            attributes={"Email": ["alice@corp.com"]},
        )
        provider.register_assertion(assertion)

        session = provider.authenticate("a1")
        audit.log(
            AuditEventType.MEMORY_CREATE,
            tenant_id="acme",
            user_id=session.user_id,
            details={"sso_provider": provider.provider_name, "method": "SAML"},
        )

        assert audit.count == 1
        events = audit.get_user_activity("alice@corp.com")
        assert len(events) == 1
        assert events[0].details["sso_provider"] == "Corp IdP"
