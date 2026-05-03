"""Log desensitization utilities for NeuralMem."""
import logging
import re
from typing import Any


class LogDesensitizer:
    """Redacts sensitive information from log messages and data."""

    def __init__(self, patterns: dict[str, tuple[re.Pattern, str]] | None = None):
        """Initialize with default or custom patterns.

        Args:
            patterns: Dict mapping pattern name to (compiled_regex, replacement).
                      If None, uses built-in PII patterns.
        """
        if patterns is not None:
            self._patterns = patterns
        else:
            self._patterns = self._build_default_patterns()

    @staticmethod
    def _build_default_patterns() -> dict[str, tuple[re.Pattern, str]]:
        """Build default PII detection patterns."""
        return {
            "email": (
                re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
                "***@***.***",
            ),
            "phone": (
                re.compile(r"\b\d{3}[-.\s]?\d{4}[-.\s]?\d{4}\b"),
                "***-****-****",
            ),
            "api_key_sk": (
                re.compile(r"sk-[a-zA-Z0-9]{20,}"),
                "sk-***",
            ),
            "api_key_ghp": (
                re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
                "ghp_***",
            ),
            "api_key_AKIA": (
                re.compile(r"AKIA[a-zA-Z0-9]{16}"),
                "AKIA***",
            ),
            "ssn": (
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                "***-**-****",
            ),
            "credit_card": (
                re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
                "****-****-****-****",
            ),
            "ip_address": (
                re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"),
                None,  # Special handling: keep first two octets
            ),
        }

    def desensitize(self, text: str) -> str:
        """Apply all patterns to redact sensitive data from text.

        Args:
            text: Input text that may contain PII.

        Returns:
            Text with sensitive data replaced.
        """
        result = text
        for name, (pattern, replacement) in self._patterns.items():
            if name == "ip_address":
                result = pattern.sub(
                    lambda m: f"{m.group(1)}.{m.group(2)}.***.***", result
                )
            else:
                result = pattern.sub(replacement, result)
        return result

    def desensitize_dict(self, data: dict) -> dict:
        """Recursively desensitize all string values in a dict.

        Args:
            data: Input dict that may contain PII in string values.

        Returns:
            New dict with string values desensitized.
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.desensitize(value)
            elif isinstance(value, dict):
                result[key] = self.desensitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.desensitize_dict(v) if isinstance(v, dict)
                    else self.desensitize(v) if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def add_pattern(self, name: str, pattern: str, replacement: str) -> None:
        """Add a custom desensitization pattern.

        Args:
            name: Identifier for this pattern.
            pattern: Regex pattern string.
            replacement: Replacement string.
        """
        self._patterns[name] = (re.compile(pattern), replacement)

    def create_log_filter(self) -> logging.Filter:
        """Create a logging.Filter that auto-desensitizes log messages.

        Returns:
            A logging.Filter instance that redacts PII from log records.
        """
        desensitizer = self

        class _DesensitizeFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if isinstance(record.msg, str):
                    record.msg = desensitizer.desensitize(record.msg)
                if record.args:
                    if isinstance(record.args, dict):
                        record.args = desensitizer.desensitize_dict(record.args)
                    elif isinstance(record.args, tuple):
                        record.args = tuple(
                            desensitizer.desensitize(a) if isinstance(a, str) else a
                            for a in record.args
                        )
                return True

        return _DesensitizeFilter()
