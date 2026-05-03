"""Tests for structured_logging module."""
from __future__ import annotations

import json

import pytest

from neuralmem.production.structured_logging import (
    StructuredLogger,
    get_correlation_id,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def _reset_correlation():
    """Reset correlation ID between tests."""
    set_correlation_id("")
    yield


class TestStructuredLogger:
    def test_info_produces_json(self, capsys):
        logger = StructuredLogger(
            "test_json", redact=False
        )
        logger.info("hello")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["message"] == "hello"
        assert data["level"] == "INFO"

    def test_has_required_fields(self, capsys):
        logger = StructuredLogger(
            "test_fields", redact=False
        )
        logger.info("msg")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        for field in (
            "timestamp",
            "level",
            "message",
            "module",
            "correlation_id",
        ):
            assert field in data

    def test_correlation_id_in_output(self, capsys):
        set_correlation_id("abc-123")
        logger = StructuredLogger(
            "test_corr", redact=False
        )
        logger.info("msg")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["correlation_id"] == "abc-123"

    def test_redacts_password(self, capsys):
        logger = StructuredLogger(
            "test_redact", redact=True
        )
        logger.info("login", password="secret123")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["password"] == "***REDACTED***"

    def test_redacts_api_key(self, capsys):
        logger = StructuredLogger(
            "test_redact_key", redact=True
        )
        logger.info("call", api_key="sk-123")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["api_key"] == "***REDACTED***"

    def test_redacts_token(self, capsys):
        logger = StructuredLogger(
            "test_redact_token", redact=True
        )
        logger.info("auth", token="tok_abc")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["token"] == "***REDACTED***"

    def test_no_redact_when_disabled(self, capsys):
        logger = StructuredLogger(
            "test_noredact", redact=False
        )
        logger.info("data", secret="visible")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["secret"] == "visible"

    def test_extra_fields(self, capsys):
        logger = StructuredLogger(
            "test_extra", redact=False
        )
        logger.info("evt", user_id=42)
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["user_id"] == 42

    def test_warning_level(self, capsys):
        logger = StructuredLogger(
            "test_warn", redact=False
        )
        logger.warning("heads up")
        out = capsys.readouterr().err
        data = json.loads(out.strip())
        assert data["level"] == "WARNING"

    def test_get_set_correlation_id(self):
        set_correlation_id("xyz")
        assert get_correlation_id() == "xyz"
