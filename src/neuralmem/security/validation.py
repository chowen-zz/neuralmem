"""MCP input validation and safety utilities for NeuralMem."""
import re


def validate_mcp_input(
    schema: dict, data: dict
) -> tuple[bool, list[str]]:
    """Validate MCP tool input against a schema definition.

    Schema format:
        {
            "field_name": {
                "type": "str|int|float|bool",  # required
                "required": True/False,          # default True
                "min_length": N,                 # optional, for str
                "max_length": N,                 # optional, for str
            }
        }

    Args:
        schema: Schema definition dict.
        data: Input data to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: list[str] = []
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    for field_name, rules in schema.items():
        required = rules.get("required", True)
        field_type = rules.get("type")
        min_length = rules.get("min_length")
        max_length = rules.get("max_length")

        # Check presence
        if field_name not in data:
            if required:
                errors.append(f"Missing required field: {field_name}")
            continue

        value = data[field_name]

        # Check type
        if field_type and field_type in type_map:
            expected_type = type_map[field_type]
            # Allow int where float is expected
            if expected_type is float and isinstance(value, int):
                pass
            elif not isinstance(value, expected_type):
                errors.append(
                    f"Field '{field_name}' expected type {field_type}, "
                    f"got {type(value).__name__}"
                )
                continue

        # Check string length constraints
        if isinstance(value, str):
            if min_length is not None and len(value) < min_length:
                errors.append(
                    f"Field '{field_name}' must be at least {min_length} characters"
                )
            if max_length is not None and len(value) > max_length:
                errors.append(
                    f"Field '{field_name}' must be at most {max_length} characters"
                )

    return (len(errors) == 0, errors)


def destructive_action(
    action_name: str, reason: str = "", min_reason_length: int = 8
) -> tuple[bool, str]:
    """Gate for destructive MCP actions requiring a justification.

    Args:
        action_name: Name of the action being performed.
        reason: User-provided justification for the action.
        min_reason_length: Minimum characters required in the reason.

    Returns:
        Tuple of (allowed, message). If allowed is False, message explains why.
    """
    if not reason or len(reason.strip()) < min_reason_length:
        return (
            False,
            f"Destructive action requires reason (min {min_reason_length} chars)",
        )
    return (True, "")


def sanitize_content(content: str, max_length: int = 50000) -> str:
    """Sanitize content by stripping control characters and enforcing length.

    Args:
        content: Raw content string.
        max_length: Maximum allowed length (truncated if exceeded).

    Returns:
        Sanitized content string.
    """
    # Strip control characters (except newline \n and tab \t)
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)
    # Enforce max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized
