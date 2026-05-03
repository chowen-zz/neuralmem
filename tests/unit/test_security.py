"""Tests for neuralmem.security module."""
import logging

from neuralmem.security.desensitize import LogDesensitizer
from neuralmem.security.validation import (
    destructive_action,
    sanitize_content,
    validate_mcp_input,
)

# ---------------------------------------------------------------------------
# LogDesensitizer tests
# ---------------------------------------------------------------------------

class TestLogDesensitizeEmail:
    def test_desensitize_email(self):
        d = LogDesensitizer()
        result = d.desensitize("Contact alice@example.com for info")
        assert "alice@example.com" not in result
        assert "***@***.***" in result

    def test_desensitize_email_multiple(self):
        d = LogDesensitizer()
        result = d.desensitize("Send to a@b.com and c@d.org")
        assert "a@b.com" not in result
        assert "c@d.org" not in result


class TestLogDesensitizePhone:
    def test_desensitize_phone(self):
        d = LogDesensitizer()
        result = d.desensitize("Call 138-1234-5678 now")
        assert "138-1234-5678" not in result
        assert "***-****-****" in result

    def test_desensitize_phone_no_dash(self):
        d = LogDesensitizer()
        result = d.desensitize("Call 13812345678 now")
        assert "13812345678" not in result


class TestLogDesensitizeAPIKeys:
    def test_desensitize_api_key_sk(self):
        d = LogDesensitizer()
        result = d.desensitize("Key: sk-abcdefghijklmnopqrstuvwxyz")
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
        assert "sk-***" in result

    def test_desensitize_api_key_ghp(self):
        d = LogDesensitizer()
        result = d.desensitize("Token: ghp_abcdefghijklmnopqrstuvwxyz")
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in result
        assert "ghp_***" in result

    def test_desensitize_api_key_AKIA(self):
        d = LogDesensitizer()
        result = d.desensitize("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "AKIA***" in result


class TestLogDesensitizeSSN:
    def test_desensitize_ssn(self):
        d = LogDesensitizer()
        result = d.desensitize("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "***-**-****" in result


class TestLogDesensitizeCreditCard:
    def test_desensitize_credit_card(self):
        d = LogDesensitizer()
        result = d.desensitize("Card: 4111-1111-1111-1111")
        assert "4111-1111-1111-1111" not in result
        assert "****-****-****-****" in result

    def test_desensitize_credit_card_spaces(self):
        d = LogDesensitizer()
        result = d.desensitize("Card: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in result


class TestLogDesensitizeIP:
    def test_desensitize_ip(self):
        d = LogDesensitizer()
        result = d.desensitize("Server at 192.168.1.100")
        assert "192.168.1.100" not in result
        assert "192.168.***.***" in result

    def test_desensitize_ip_loopback(self):
        d = LogDesensitizer()
        result = d.desensitize("Host: 127.0.0.1")
        assert "127.0.0.1" not in result
        assert "127.0.***.***" in result


class TestLogDesensitizeMisc:
    def test_desensitize_no_match(self):
        d = LogDesensitizer()
        text = "This is a clean message with no PII."
        assert d.desensitize(text) == text

    def test_desensitize_multiple(self):
        d = LogDesensitizer()
        text = "Email user@example.com from 192.168.1.1, SSN 123-45-6789"
        result = d.desensitize(text)
        assert "user@example.com" not in result
        assert "192.168.1.1" not in result
        assert "123-45-6789" not in result
        assert "***@***.***" in result
        assert "192.168.***.***" in result
        assert "***-**-****" in result


class TestDesensitizeDict:
    def test_desensitize_dict_nested(self):
        d = LogDesensitizer()
        data = {
            "user": {
                "email": "alice@example.com",
                "ip": "10.0.0.1",
            },
            "label": "clean",
        }
        result = d.desensitize_dict(data)
        assert "alice@example.com" not in str(result)
        assert result["user"]["email"] == "***@***.***"
        assert result["label"] == "clean"

    def test_desensitize_dict_preserves_keys(self):
        d = LogDesensitizer()
        data = {"email": "a@b.com", "phone": "138-1234-5678"}
        result = d.desensitize_dict(data)
        assert set(result.keys()) == {"email", "phone"}


class TestAddPattern:
    def test_add_custom_pattern(self):
        d = LogDesensitizer()
        d.add_pattern("custom_id", r"ID-\d{6}", "ID-XXXXXX")
        result = d.desensitize("User ID-123456 logged in")
        assert "ID-123456" not in result
        assert "ID-XXXXXX" in result


class TestCreateLogFilter:
    def test_create_log_filter(self):
        d = LogDesensitizer()
        f = d.create_log_filter()
        assert isinstance(f, logging.Filter)

    def test_log_filter_redacts_message(self):
        d = LogDesensitizer()
        f = d.create_log_filter()
        logger = logging.getLogger("test_desensitize_filter")
        logger.setLevel(logging.DEBUG)
        logger.addFilter(f)

        # Capture output via a handler
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        logger.addHandler(handler)
        try:
            logger.info("User email: test@example.com")
            assert len(records) == 1
            assert "test@example.com" not in records[0].getMessage()
            assert "***@***.***" in records[0].getMessage()
        finally:
            logger.removeHandler(handler)
            logger.removeFilter(f)


# ---------------------------------------------------------------------------
# validate_mcp_input tests
# ---------------------------------------------------------------------------

class TestValidateMCPInput:
    def test_validate_mcp_input_valid(self):
        schema = {
            "query": {"type": "str", "required": True},
            "limit": {"type": "int", "required": True},
        }
        data = {"query": "hello", "limit": 10}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is True
        assert errors == []

    def test_validate_mcp_input_missing_required(self):
        schema = {"query": {"type": "str", "required": True}}
        data: dict = {}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is False
        assert any("query" in e for e in errors)

    def test_validate_mcp_input_wrong_type(self):
        schema = {"count": {"type": "int", "required": True}}
        data = {"count": "not_an_int"}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is False
        assert any("int" in e for e in errors)

    def test_validate_mcp_input_optional(self):
        schema = {
            "query": {"type": "str", "required": True},
            "offset": {"type": "int", "required": False},
        }
        data = {"query": "hello"}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is True

    def test_validate_mcp_input_min_length(self):
        schema = {"query": {"type": "str", "required": True, "min_length": 5}}
        data = {"query": "hi"}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is False
        assert any("at least 5" in e for e in errors)

    def test_validate_mcp_input_max_length(self):
        schema = {"query": {"type": "str", "required": True, "max_length": 3}}
        data = {"query": "toolong"}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is False
        assert any("at most 3" in e for e in errors)

    def test_validate_mcp_input_float_accepts_int(self):
        schema = {"score": {"type": "float", "required": True}}
        data = {"score": 5}
        valid, errors = validate_mcp_input(schema, data)
        assert valid is True


# ---------------------------------------------------------------------------
# destructive_action tests
# ---------------------------------------------------------------------------

class TestDestructiveAction:
    def test_destructive_action_with_reason(self):
        allowed, msg = destructive_action("delete_all", reason="Cleaning up old data")
        assert allowed is True
        assert msg == ""

    def test_destructive_action_short_reason(self):
        allowed, msg = destructive_action("drop_db", reason="yes")
        assert allowed is False
        assert "min 8 chars" in msg

    def test_destructive_action_no_reason(self):
        allowed, msg = destructive_action("purge")
        assert allowed is False
        assert "requires reason" in msg

    def test_destructive_action_whitespace_only(self):
        allowed, msg = destructive_action("purge", reason="   ")
        assert allowed is False


# ---------------------------------------------------------------------------
# sanitize_content tests
# ---------------------------------------------------------------------------

class TestSanitizeContent:
    def test_sanitize_content_control_chars(self):
        raw = "Hello\x00World\x07"
        result = sanitize_content(raw)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Hello" in result
        assert "World" in result

    def test_sanitize_content_preserves_newlines_tabs(self):
        raw = "Line1\nLine2\tTabbed"
        result = sanitize_content(raw)
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_content_max_length(self):
        long_text = "A" * 100_000
        result = sanitize_content(long_text, max_length=500)
        assert len(result) == 500

    def test_sanitize_content_short_unchanged(self):
        text = "Short and clean"
        assert sanitize_content(text) == text
