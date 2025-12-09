from __future__ import annotations

from .errors import BadRequestError

STRUCTURED_EXPECTATIONS: frozenset[str] = frozenset({"point", "box", "polygon"})
REASONING_EXPECTATIONS: frozenset[str] = frozenset({"think"})
TEXT_EXPECTATIONS: frozenset[str] = frozenset({"text"})
VALID_EXPECTATIONS: frozenset[str] = frozenset(
    STRUCTURED_EXPECTATIONS | TEXT_EXPECTATIONS | REASONING_EXPECTATIONS
)

REASONING_HINT = "<hint>THINK</hint>"


def expectation_hint_text(expects: str | None) -> str | None:
    if expects and expects.lower() in STRUCTURED_EXPECTATIONS:
        return f"<hint>{expects.upper()}</hint>"
    if expects and expects.lower() in REASONING_EXPECTATIONS:
        return REASONING_HINT
    return None


def resolve_structured_expectation(expects: str, *, context: str) -> tuple[str | None, bool]:
    normalized = expects.lower() if isinstance(expects, str) else expects
    if normalized not in VALID_EXPECTATIONS:
        raise BadRequestError(f"Unsupported {context}: {expects}")
    if normalized in REASONING_EXPECTATIONS:
        return normalized, False
    structured = normalized if normalized in STRUCTURED_EXPECTATIONS else None
    allow_multiple = structured is not None
    return structured, allow_multiple


__all__ = [
    "STRUCTURED_EXPECTATIONS",
    "REASONING_EXPECTATIONS",
    "VALID_EXPECTATIONS",
    "expectation_hint_text",
    "resolve_structured_expectation",
    "REASONING_HINT",
]
