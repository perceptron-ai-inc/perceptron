from __future__ import annotations

from typing import Any


class SDKError(Exception):
    """Base error for SDK exceptions (transport/runtime).

    ``details`` is always a (new) dict; ``request_id`` (the ``x-trace-id`` response header) and ``retry_after`` are
    mirrored into it when set, because the docs read ``err.details["request_id"]``. ``status_code``, ``error_type`` (the
    server's error ``type``), ``param`` and ``partial`` (what a stream produced before it failed) are attributes only.
    """

    def __init__(  # noqa: PLR0913 - keyword-only error attributes
        self,
        message: str = "",
        code: str | None = None,
        details: dict | None = None,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        error_type: str | None = None,
        param: str | None = None,
        retry_after: float | None = None,
        partial: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details) if details else {}
        self.status_code = status_code
        self.request_id = request_id
        self.error_type = error_type
        self.param = param
        self.retry_after = retry_after
        self.partial = partial
        if request_id is not None:
            self.details["request_id"] = request_id
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class TransportError(SDKError):
    """Network/connection failure."""

    pass


class IncompleteStreamError(TransportError):
    """A stream ended before it finished (``code`` is ``stream_truncated``); ``partial`` holds what arrived."""

    pass


class TimeoutError(SDKError):
    """Deadline exceeded."""

    pass


class AuthError(SDKError):
    """Authentication/authorization failure."""

    pass


class PermissionDeniedError(AuthError):
    """403: the key is valid but not allowed to do this."""

    pass


class RateLimitError(SDKError):
    """429 Too Many Requests.

    ``code`` stays ``"rate_limit"``; the server's own code and type are in ``details["code"]``/``details["type"]`` (and
    ``error_type``). ``retry_after`` holds the ``Retry-After`` seconds, or None when the server sent none.
    """

    _code = "rate_limit"

    def __init__(self, message: str = "", retry_after: float | None = None, *, details: dict | None = None, **attrs):
        attrs.pop("code", None)  # the class decides the code, as it always has
        super().__init__(message, code=self._code, details=details, retry_after=retry_after, **attrs)


class QuotaExceededError(RateLimitError):
    """429 ``insufficient_quota``: credits (or storage) are exhausted. Not retryable, so ``retry_after`` is None."""

    _code = "insufficient_quota"

    def __init__(self, message: str = "", *, details: dict | None = None, **attrs):
        attrs.pop("retry_after", None)
        super().__init__(message, None, details=details, **attrs)


class ServerError(SDKError):
    """5xx or invalid server response."""

    pass


class BadRequestError(SDKError):
    """4xx client-side invalid request."""

    pass


class NotFoundError(BadRequestError):
    """404: the file, model or route does not exist."""

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
# Raised client-side, before any request, for a `reasoning_effort` outside the API's tiers.
INVALID_REASONING_EFFORT = "invalid_reasoning_effort"

# Raised client-side, before any request (the gateway's code where it has one).
CREDENTIALS_MISSING = "credentials_missing"  # AuthError: no API key for the selected provider
MODEL_RENAMED = "model_renamed"
UNSUPPORTED_PROVIDER_FEATURE = "unsupported_provider_feature"
UNSUPPORTED_PARAMETER = "unsupported_parameter"
INVALID_PARAMETER = "invalid_parameter"
INVALID_TEMPERATURE = "invalid_temperature"
INVALID_VISION_CONFIG = "invalid_vision_config"
UNSUPPORTED_TOOL_CHOICE = "unsupported_tool_choice"
UNSUPPORTED_TOOL_TYPE = "unsupported_tool_type"
UNSUPPORTED_TOOLS_COMBINATION = "unsupported_tools_combination"
CONFLICTING_TOOLS = "conflicting_tools"
INVALID_TOOLS = "invalid_tools"
DUPLICATE_TOOL_NAME = "duplicate_tool_name"
RESERVED_TOOL_NAME = "reserved_tool_name"
CONFLICTING_STRUCTURED_OUTPUT_CONTROLS = "conflicting_structured_output_controls"
UNSUPPORTED_RESPONSE_FORMAT = "unsupported_response_format"
INVALID_RESPONSE_FORMAT = "invalid_response_format"
INVALID_REGEX = "invalid_regex"
INVALID_FILE_ID = "invalid_file_id"
INVALID_DATA_URL = "invalid_data_url"
INVALID_MEDIA = "invalid_media"
INVALID_VIDEO_FRAMES = "invalid_video_frames"
INVALID_MEDIA_PATH = "invalid_media_path"
INVALID_TOOL_RESULT = "invalid_tool_result"
UNSUPPORTED_ENTRY_TYPE = "unsupported_entry_type"
# Raised by `ToolCall.parse_arguments()` for arguments that are not valid JSON.
INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
# DSL tag issues (`perceive` lists them; `strict=True` raises AnchorError / ExpectationError; tags in
# `chat.completions.create()` message content raise BadRequestError).
ANCHOR_MISSING = "anchor_missing"
ANCHOR_UNKNOWN = "anchor_unknown"
ANCHOR_AMBIGUOUS = "anchor_ambiguous"
BOUNDS_OUT_OF_RANGE = "bounds_out_of_range"
INVALID_POLYGON = "invalid_polygon"

# Server responses and streams.
INSUFFICIENT_QUOTA = "insufficient_quota"
INVALID_RESPONSE = "invalid_response"
INVALID_STREAM_CHUNK = "invalid_stream_chunk"
STREAM_TRUNCATED = "stream_truncated"
STREAM_INCOMPLETE = "stream_incomplete"
