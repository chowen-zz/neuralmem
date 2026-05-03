"""Structured JSON logging with correlation IDs and redaction."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from typing import Any

# Correlation ID stored per-task / per-request
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

# Patterns for sensitive fields — keys containing these substrings
_SENSITIVE_KEY_PATTERNS = [
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
]

_REDACTED = "***REDACTED***"


def set_correlation_id(cid: str) -> contextvars.Token:
    """Set the correlation ID for the current context."""
    return _correlation_id.set(cid)


def get_correlation_id() -> str:
    """Return the current correlation ID (empty string if unset)."""
    return _correlation_id.get()


def _is_sensitive(key: str) -> bool:
    return any(p.search(key) for p in _SENSITIVE_KEY_PATTERNS)


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive keys in dicts."""
    if isinstance(obj, dict):
        return {
            k: _REDACTED if _is_sensitive(k) else _redact(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    return obj


class StructuredLogger:
    """JSON-formatted logger with redaction and correlation IDs.

    Integrates with the standard ``logging`` module by wrapping
    a ``logging.Logger`` and emitting structured JSON via a
    ``logging.Formatter`` subclass.

    Parameters
    ----------
    name : str
        Logger name (passed to ``logging.getLogger``).
    level : int
        Logging level.
    redact : bool
        Whether to redact sensitive fields.
    extra : dict or None
        Extra fields included in every log record.
    """

    class _JsonFormatter(logging.Formatter):
        """Format log records as one-line JSON."""

        def __init__(
            self,
            redact: bool = True,
            extra: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self._redact = redact
            self._extra = extra or {}

        def format(self, record: logging.LogRecord) -> str:
            data: dict[str, Any] = {
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.gmtime(record.created),
                )
                + f".{record.msecs:03.0f}Z",
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "correlation_id": get_correlation_id(),
            }
            # Attach duration_ms if present
            if hasattr(record, "duration_ms"):
                data["duration_ms"] = record.duration_ms

            data.update(self._extra)

            # Merge any extra fields set on the record
            if hasattr(record, "structured_extra"):
                data.update(record.structured_extra)

            if self._redact:
                data = _redact(data)

            return json.dumps(data, default=str)

    def __init__(
        self,
        name: str = "neuralmem",
        *,
        level: int = logging.INFO,
        redact: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._redact = redact

        # Attach handler only if none exist yet
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                self._JsonFormatter(redact=redact, extra=extra)
            )
            self._logger.addHandler(handler)
            self._logger.propagate = False

    @property
    def logger(self) -> logging.Logger:
        """Underlying stdlib logger."""
        return self._logger

    def debug(
        self, msg: str, **kwargs: Any
    ) -> None:
        self._log(logging.DEBUG, msg, kwargs)

    def info(
        self, msg: str, **kwargs: Any
    ) -> None:
        self._log(logging.INFO, msg, kwargs)

    def warning(
        self, msg: str, **kwargs: Any
    ) -> None:
        self._log(logging.WARNING, msg, kwargs)

    def error(
        self, msg: str, **kwargs: Any
    ) -> None:
        self._log(logging.ERROR, msg, kwargs)

    def _log(
        self,
        level: int,
        msg: str,
        extra: dict[str, Any],
    ) -> None:
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        if extra:
            record.structured_extra = extra
        self._logger.handle(record)
