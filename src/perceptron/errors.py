class SDKError(Exception):
    """Base error for SDK exceptions (transport/runtime)."""

    def __init__(self, message: str = "", code: str | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class TransportError(SDKError):
    """Network/connection failure."""

    pass


class TimeoutError(SDKError):
    """Deadline exceeded."""

    pass


class AuthError(SDKError):
    """Authentication/authorization failure."""

    pass


class RateLimitError(SDKError):
    """429 Too Many Requests; may include retry_after in details."""

    def __init__(self, message: str = "", retry_after: float | None = None, **kw) -> None:
        details = kw.get("details", {})
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, code="rate_limit", details=details)


class ServerError(SDKError):
    """5xx or invalid server response."""

    pass


class BadRequestError(SDKError):
    """4xx client-side invalid request."""

    pass


class ExpectationError(SDKError):
    """Strict-mode semantic/validation failure."""

    pass


class AnchorError(SDKError):
    """Anchoring rules violated (e.g., missing image= in multi-image)."""

    pass


class ParseError(SDKError):
    """Failed to parse model response (e.g., malformed point/box tags)."""

    pass


# Warning/issue codes for reasoning configuration
REASONING_NOT_SUPPORTED = "reasoning_not_supported"
REASONING_REQUIRED_FOR_MODEL = "reasoning_required_for_model"
REASONING_DISABLED_FOR_THINKING_MODEL = "reasoning_disabled_for_thinking_model"
