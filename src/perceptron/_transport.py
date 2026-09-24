"""HTTP plumbing shared by the message API and the resource endpoints (files, models, multilook).

- :func:`http_error_from_response` maps a non-2xx response to a typed :class:`~perceptron.errors.SDKError`.
- :func:`iter_sse_data` / :func:`aiter_sse_data` yield SSE ``data:`` payloads, with ``[DONE]`` surfaced as :data:`DONE`.
- :func:`request` / :func:`request_json` / :func:`open_stream` / :func:`stream_request` (and ``a``-prefixed async twins)
  send one request through the client's session factories (``client._sync_session`` / ``client._async_session``) with
  auth, error mapping, and transport exceptions converted to ``TimeoutError`` / ``TransportError``.
- :func:`iter_response_lines` / :func:`iter_response_bytes` (and async twins) read an open response body with the same
  conversion for failures mid-body.

SDK errors are always raised outside the ``try`` blocks that catch transport exceptions: the async path catches the
exception classes of ``perceptron.client.httpx``, which tests replace with a namespace where both are ``Exception``.
"""

from __future__ import annotations

import codecs
import importlib
import json as _json
import math
import re
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager, suppress
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ._providers import missing_api_key_message, provider_api_key, surface_provider_cfg
from .errors import (
    CREDENTIALS_MISSING,
    INSUFFICIENT_QUOTA,
    INVALID_PARAMETER,
    INVALID_RESPONSE,
    STREAM_TRUNCATED,
    AuthError,
    BadRequestError,
    IncompleteStreamError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    SDKError,
    ServerError,
    TimeoutError,
    TransportError,
)

TRACE_ID_HEADER = "x-trace-id"

# Yielded by the SSE iterators for `data: [DONE]`.
DONE: Any = object()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def header_value(headers: Any, name: str) -> str | None:
    """Case-insensitive header lookup that also works on the plain dicts test stubs use."""
    if not headers:
        return None
    try:
        value = headers.get(name)
        if value is None:
            lowered = name.lower()
            value = next((v for k, v in headers.items() if isinstance(k, str) and k.lower() == lowered), None)
    except Exception:
        return None
    return value


def request_id_of(resp: Any) -> str | None:
    """The gateway's correlation id (the ``x-trace-id`` response header), when present."""
    return header_value(getattr(resp, "headers", None), TRACE_ID_HEADER)


def _retry_after(headers: Any) -> float | None:
    raw = header_value(headers, "Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _extract_error_metadata(
    data: Any,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    message: str | None = None
    code: str | None = None
    details: dict[str, Any] | None = None

    if isinstance(data, dict):
        nested_error = data.get("error")
        if isinstance(nested_error, dict):
            message = _first_nonempty(
                nested_error.get("message"),
                nested_error.get("detail"),
                nested_error.get("error"),
            )
            code = nested_error.get("code") or nested_error.get("type")
            details = nested_error or None
        elif isinstance(nested_error, str):
            message = _first_nonempty(nested_error)
            details = data or None
        else:
            message = _first_nonempty(
                data.get("message"),
                data.get("detail"),
                data.get("error") if isinstance(data.get("error"), str) else None,
            )
            code = data.get("code")
            details = data or None
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                candidate = _first_nonempty(item.get("message"), item.get("detail"))
                if candidate:
                    message = candidate
                    code = item.get("code")
                    details = item
                    break
            elif isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    message = candidate
                    break
    elif isinstance(data, str):
        message = data.strip() or None

    return message, code, details


def _unwrap_error(error: dict[str, Any]) -> dict[str, Any]:
    """Upstream 5xx bodies reach the client as JSON text in ``error.message`` with ``type: null``; surface the inner
    error object instead."""
    message = error.get("message")
    if error.get("type") is None and isinstance(message, str) and message.lstrip().startswith("{"):
        try:
            inner = _json.loads(message)
        except ValueError:
            return error
        if isinstance(inner, dict) and isinstance(inner.get("error"), dict):
            return inner["error"]
    return error


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def http_error_from_response(resp: Any) -> SDKError:  # noqa: PLR0911 - one return per status class
    """Map a non-2xx response to an SDK error.

    Parses ``{"error": {message, type, param, code}}`` (also the ``/v1/detect`` shape, flat dicts, lists and plain-text
    bodies). ``code`` is the server code, else its type; ``request_id`` comes from ``x-trace-id``; ``retry_after`` from
    ``Retry-After`` on 429 and 503 (None when absent). Tolerates responses without ``json``/``text``/``headers``.
    """
    status = getattr(resp, "status_code", None)
    try:
        data = resp.json()
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        inner = _unwrap_error(data["error"])
        if inner is not data["error"]:
            data = {"error": inner}

    message, code, details = _extract_error_metadata(data)
    try:
        fallback_text = resp.text
    except Exception:
        fallback_text = ""
    fallback_text = fallback_text.strip() if isinstance(fallback_text, str) else ""
    fields = details if isinstance(details, dict) else {}
    headers = getattr(resp, "headers", None) or {}
    error_type = _str_or_none(fields.get("type"))
    attrs: dict[str, Any] = {
        "details": fields,
        "status_code": status,
        "request_id": header_value(headers, TRACE_ID_HEADER),
        "error_type": error_type,
        "param": _str_or_none(fields.get("param")),
    }

    quota = INSUFFICIENT_QUOTA in (error_type, code)
    if status == HTTPStatus.TOO_MANY_REQUESTS or (status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE and quota):
        if quota:
            return QuotaExceededError(message or fallback_text or "quota exceeded", **attrs)
        return RateLimitError(message or fallback_text or "rate limited", retry_after=_retry_after(headers), **attrs)
    if status == HTTPStatus.UNAUTHORIZED:
        return AuthError(message or fallback_text or "authentication failed", code=code or "auth_error", **attrs)
    if status == HTTPStatus.FORBIDDEN:
        msg = message or fallback_text or "permission denied"
        return PermissionDeniedError(msg, code=code or "auth_error", **attrs)
    if status == HTTPStatus.NOT_FOUND:
        return NotFoundError(message or fallback_text or "not found", code=code, **attrs)
    if isinstance(status, int) and HTTPStatus.BAD_REQUEST <= status < HTTPStatus.INTERNAL_SERVER_ERROR:
        return BadRequestError(message or fallback_text or "bad request", code=code, **attrs)
    retry_after = _retry_after(headers) if status == HTTPStatus.SERVICE_UNAVAILABLE else None
    msg = message or fallback_text or f"server error: {status}"
    return ServerError(msg, code=code, retry_after=retry_after, **attrs)


def stream_error_from_event(error: Any, *, request_id: str | None = None, partial: Any = None) -> SDKError:
    """Map a mid-stream ``{"error": {...}}`` event (HTTP status already 200) to an SDK error by its ``type``."""
    fields = _unwrap_error(error) if isinstance(error, dict) else {"message": str(error)}
    message = _first_nonempty(fields.get("message")) or "The stream failed."
    error_type = _str_or_none(fields.get("type"))
    code = fields.get("code") or error_type
    attrs: dict[str, Any] = {
        "details": fields,
        "request_id": request_id,
        "error_type": error_type,
        "param": _str_or_none(fields.get("param")),
        "partial": partial,
    }
    if INSUFFICIENT_QUOTA in (error_type, code):
        return QuotaExceededError(message, **attrs)
    if error_type == "rate_limit_error":
        return RateLimitError(message, **attrs)
    if error_type == "invalid_request_error":
        return BadRequestError(message, code=code, **attrs)
    return ServerError(message, code=code, **attrs)


# ---------------------------------------------------------------------------
# Server-sent events
# ---------------------------------------------------------------------------


def _sse_payload(raw: Any) -> Any:
    """The payload of one ``data:`` line (one optional space stripped), :data:`DONE`, or None for any other line.

    The gateway sends each event as a single ``data:`` line, so there is no multi-line accumulation. Blank lines, ``:``
    keep-alive comments and ``event:``/``id:``/``retry:`` fields are ignored.
    """
    line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
    if not isinstance(line, str):
        return None
    line = line.rstrip("\r\n")
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :]
    if data.startswith(" "):
        data = data[1:]
    return DONE if data.strip() == "[DONE]" else data


def iter_sse_data(lines: Iterable[Any]) -> Iterator[Any]:
    """Yield each SSE ``data:`` payload string, and :data:`DONE` for ``[DONE]``."""
    for raw in lines:
        payload = _sse_payload(raw)
        if payload is not None:
            yield payload


async def aiter_sse_data(lines: AsyncIterable[Any]) -> AsyncIterator[Any]:
    """Async :func:`iter_sse_data`."""
    async for raw in lines:
        payload = _sse_payload(raw)
        if payload is not None:
            yield payload


def _stream_truncated(exc: Exception) -> SDKError:
    return IncompleteStreamError(f"The stream was cut off: {exc}", code=STREAM_TRUNCATED)


def _body_cut_off(exc: Exception) -> SDKError:
    return TransportError(f"The response body was cut off: {exc}")


def _guarded(items: Iterable[Any], cut_off: Callable[[Exception], SDKError]) -> Iterator[Any]:
    """Yield from a response body iterator, converting a timeout mid-body to ``TimeoutError`` and any other transport
    failure to ``cut_off(exc)``."""
    iterator = iter(items)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except httpx.TimeoutException as exc:
            raise TimeoutError("The stream timed out.") from exc
        except httpx.HTTPError as exc:
            raise cut_off(exc) from exc
        yield item


async def _aguarded(items: AsyncIterable[Any], cut_off: Callable[[Exception], SDKError]) -> AsyncIterator[Any]:
    """Async :func:`_guarded`, catching the exception classes of :func:`_async_httpx`."""
    errors = _async_httpx()
    iterator = items.__aiter__()
    while True:
        try:
            item = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except errors.TimeoutException as exc:
            raise TimeoutError("The stream timed out.") from exc
        except errors.HTTPError as exc:
            raise cut_off(exc) from exc
        yield item


_LINE_END = re.compile(r"\r\n|\r|\n")


class _SSELineSplitter:
    """Split a UTF-8 body into SSE lines, which end only at CRLF, LF or CR.

    Not ``iter_lines()``: it splits like ``str.splitlines()``, also at U+2028, U+2029 and U+0085, which JSON leaves
    unescaped inside a ``data:`` line.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending: list[str] = []  # the unterminated start of the next line
        self._cr = False  # the last chunk ended with CR, which may be the first half of CRLF

    def feed(self, chunk: bytes | str) -> list[str]:
        text = self._decoder.decode(chunk) if isinstance(chunk, (bytes, bytearray)) else chunk
        if self._cr:
            text = "\r" + text
        self._cr = text.endswith("\r")
        if self._cr:
            text = text[:-1]
        parts = _LINE_END.split(text)
        if len(parts) == 1:
            self._pending.append(text)
            return []
        parts[0] = "".join(self._pending) + parts[0]
        tail = parts.pop()
        self._pending = [tail] if tail else []
        return parts

    def close(self) -> list[str]:
        """The lines left at EOF. An unterminated last line is an event cut off mid-line, so it is dropped (the stream
        then ends without ``[DONE]``: truncated); a bare ``data: [DONE]`` still counts."""
        lines = self.feed(self._decoder.decode(b"", final=True))
        rest = "".join(self._pending)
        self._pending = []
        if self._cr:
            self._cr = False
            lines.append(rest)
        elif rest and _sse_payload(rest) is DONE:
            lines.append(rest)
        return lines


def _split_lines(chunks: Iterable[Any]) -> Iterator[str]:
    splitter = _SSELineSplitter()
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.close()


async def _asplit_lines(chunks: AsyncIterable[Any]) -> AsyncIterator[str]:
    splitter = _SSELineSplitter()
    async for chunk in chunks:
        for line in splitter.feed(chunk):
            yield line
    for line in splitter.close():
        yield line


def iter_response_lines(resp: Any) -> Iterator[Any]:
    """The SSE lines of an open response body (see :class:`_SSELineSplitter`); a timeout mid-body raises
    ``TimeoutError``, a cut connection ``IncompleteStreamError(code="stream_truncated")``. A stub without
    ``iter_bytes`` is read with its ``iter_lines()``."""
    if not callable(getattr(resp, "iter_bytes", None)):
        return _guarded(resp.iter_lines(), _stream_truncated)
    return _split_lines(_guarded(resp.iter_bytes(), _stream_truncated))


def aiter_response_lines(resp: Any) -> AsyncIterator[Any]:
    """Async :func:`iter_response_lines` over ``resp.aiter_bytes()`` (a stub's ``aiter_lines()``)."""
    if not callable(getattr(resp, "aiter_bytes", None)):
        return _aguarded(resp.aiter_lines(), _stream_truncated)
    return _asplit_lines(_aguarded(resp.aiter_bytes(), _stream_truncated))


def iter_response_bytes(resp: Any, chunk_size: int | None = None) -> Iterator[bytes]:
    """``resp.iter_bytes(chunk_size)``; a timeout mid-body raises ``TimeoutError``, a cut connection
    ``TransportError``. Use it to read a :func:`stream_request` body (e.g. ``files.download``)."""
    return _guarded(resp.iter_bytes(chunk_size=chunk_size), _body_cut_off)


def aiter_response_bytes(resp: Any, chunk_size: int | None = None) -> AsyncIterator[bytes]:
    """Async :func:`iter_response_bytes` over ``resp.aiter_bytes(chunk_size)``."""
    return _aguarded(resp.aiter_bytes(chunk_size=chunk_size), _body_cut_off)


# ---------------------------------------------------------------------------
# Requests through the client's session factories
# ---------------------------------------------------------------------------


def _async_httpx() -> Any:
    """``perceptron.client.httpx``, whose exception classes the async path catches (tests swap in a stub)."""
    return importlib.import_module(".client", __package__).httpx


_API_KEY_CHARS = re.compile(r"[\x21-\x7e]+")  # visible ASCII


def auth_headers(settings: Any, provider_cfg: dict[str, Any]) -> dict[str, str]:
    """The provider's auth header, with the key from :func:`~perceptron._providers.provider_api_key` (``AuthError``
    when there is none; fal never gets a ``PERCEPTRON_API_KEY``).

    Surrounding whitespace (a key read from a file often ends with a newline) is stripped; a key with other whitespace,
    control or non-ASCII characters raises ``AuthError`` without echoing the key.
    """
    headers: dict[str, str] = {}
    auth_header = provider_cfg.get("auth_header")
    if auth_header:
        token = provider_api_key(settings, provider_cfg)
        if isinstance(token, str):
            token = token.strip()
        if not token:
            raise AuthError(missing_api_key_message(provider_cfg), code=CREDENTIALS_MISSING)
        if not isinstance(token, str) or not _API_KEY_CHARS.fullmatch(token):
            raise AuthError("API key contains whitespace, control or non-ASCII characters", code="invalid_api_key")
        headers[auth_header] = f"{provider_cfg.get('auth_prefix', '')}{token}"
    return headers


def _send_failed(exc: Exception) -> SDKError:
    """``TransportError`` for a failed send. A local protocol error (e.g. an illegal header value) is not echoed, since
    its message can quote the ``Authorization`` header."""
    if isinstance(exc, httpx.LocalProtocolError):
        return TransportError("The request could not be sent: httpx rejected it as malformed (LocalProtocolError).")
    return TransportError(str(exc))


def resource_path(collection: str, resource_id: Any, param: str, suffix: str = "") -> str:
    """``/{collection}/{id}{suffix}`` with the id percent-encoded. An id of ``.`` or ``..`` raises, since the URL would
    resolve to another endpoint."""
    if not isinstance(resource_id, str):
        raise TypeError(f"{param} must be a str; got {type(resource_id).__name__}")
    if not resource_id:
        raise BadRequestError(f"{param} must not be empty.", code=INVALID_PARAMETER, param=param)
    if resource_id in (".", ".."):
        raise BadRequestError(f"{param} must not be {resource_id!r}.", code=INVALID_PARAMETER, param=param)
    return f"/{collection}/{quote(resource_id, safe='')}{suffix}"


def _prepare(
    client: Any, path: str, *, has_json: bool, timeout: float | None, provider_cfg: dict[str, Any] | None
) -> tuple[str, dict[str, str], float]:
    cfg = provider_cfg if provider_cfg is not None else surface_provider_cfg(client)
    base_url = cfg.get("base_url")
    if not base_url:
        raise BadRequestError(f"base_url required for provider={cfg.get('name')}")
    headers = auth_headers(client._settings, cfg)
    if has_json:
        headers["Content-Type"] = "application/json"
    effective_timeout = timeout if timeout is not None else client._settings.timeout
    return base_url.rstrip("/") + path, headers, effective_timeout


def _send_kwargs(*, json: Any, params: Any, files: Any, data: Any, is_async: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if params is not None:
        kwargs["params"] = params
    if json is not None:
        # The async path sends pre-encoded JSON, like the legacy AsyncClient.
        if is_async:
            kwargs["content"] = _json.dumps(json)
        else:
            kwargs["json"] = json
    if files is not None:
        kwargs["files"] = files
    if data is not None:
        kwargs["data"] = data
    return kwargs


def _is_success(resp: Any) -> bool:
    status = getattr(resp, "status_code", None)
    return isinstance(status, int) and HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES


def _read_quietly(resp: Any) -> None:
    read = getattr(resp, "read", None)
    if callable(read):
        with suppress(Exception):
            read()


async def _aread_quietly(resp: Any) -> None:
    aread = getattr(resp, "aread", None)
    if callable(aread):
        with suppress(Exception):
            await aread()


def _json_payload(resp: Any) -> tuple[Any, Any]:
    try:
        payload = resp.json()
    except Exception as exc:
        raise ServerError(
            "The server returned a response that is not valid JSON.",
            code=INVALID_RESPONSE,
            status_code=getattr(resp, "status_code", None),
            request_id=request_id_of(resp),
        ) from exc
    return payload, getattr(resp, "headers", None) or {}


def request(  # noqa: PLR0913 - keyword-only request options
    client: Any,
    method: str,
    path: str,
    *,
    json: Any = None,
    params: Any = None,
    files: Any = None,
    data: Any = None,
    timeout: float | None = None,
    provider_cfg: dict[str, Any] | None = None,
) -> Any:
    """Send one request and return the (read) response; non-2xx raise the mapped error.

    ``path`` is appended to the provider's base URL (``provider_cfg`` defaults to :func:`surface_provider_cfg`). Calls
    the session's verb method, e.g. ``session.post(url, headers=..., json=...)``.
    """
    url, headers, effective_timeout = _prepare(
        client, path, has_json=json is not None, timeout=timeout, provider_cfg=provider_cfg
    )
    kwargs = _send_kwargs(json=json, params=params, files=files, data=data, is_async=False)
    try:
        with client._sync_session(effective_timeout) as session:
            resp = getattr(session, method.lower())(url, headers=headers, **kwargs)
    except httpx.TimeoutException as exc:
        raise TimeoutError("request timed out") from exc
    except httpx.HTTPError as exc:
        raise _send_failed(exc) from exc
    if not _is_success(resp):
        raise http_error_from_response(resp)
    return resp


def request_json(client: Any, method: str, path: str, **kwargs: Any) -> tuple[Any, Any]:
    """:func:`request`, returning ``(payload, headers)``; a body that is not JSON raises ``ServerError``."""
    return _json_payload(request(client, method, path, **kwargs))


def open_stream(  # noqa: PLR0913 - keyword-only request options
    client: Any,
    method: str,
    path: str,
    *,
    json: Any = None,
    params: Any = None,
    timeout: float | None = None,
    provider_cfg: dict[str, Any] | None = None,
) -> tuple[Any, ExitStack]:
    """Open a streaming request and check its status (reading the body of an error first).

    Returns ``(response, closer)``; the caller owns ``closer`` (an ``ExitStack`` holding the session and response).
    """
    url, headers, effective_timeout = _prepare(
        client, path, has_json=json is not None, timeout=timeout, provider_cfg=provider_cfg
    )
    kwargs = _send_kwargs(json=json, params=params, files=None, data=None, is_async=False)
    closer = ExitStack()
    try:
        try:
            session = closer.enter_context(client._sync_session(effective_timeout))
            resp = closer.enter_context(session.stream(method, url, headers=headers, **kwargs))
        except httpx.TimeoutException as exc:
            raise TimeoutError("request timed out") from exc
        except httpx.HTTPError as exc:
            raise _send_failed(exc) from exc
        if not _is_success(resp):
            _read_quietly(resp)
            raise http_error_from_response(resp)
    except BaseException:
        closer.close()
        raise
    return resp, closer


@contextmanager
def stream_request(client: Any, method: str, path: str, **kwargs: Any) -> Iterator[Any]:
    """Context manager around :func:`open_stream` yielding the open, status-checked response.

    Read the body with :func:`iter_response_bytes` (not ``resp.iter_bytes()`` directly) so a timeout or a cut connection
    mid-body raises ``TimeoutError`` / ``TransportError`` instead of a raw ``httpx`` exception.
    """
    resp, closer = open_stream(client, method, path, **kwargs)
    with closer:
        yield resp


async def arequest(  # noqa: PLR0913 - keyword-only request options
    client: Any,
    method: str,
    path: str,
    *,
    json: Any = None,
    params: Any = None,
    files: Any = None,
    data: Any = None,
    timeout: float | None = None,
    provider_cfg: dict[str, Any] | None = None,
) -> Any:
    """Async :func:`request` (JSON bodies go as ``content=json.dumps(body)``)."""
    url, headers, effective_timeout = _prepare(
        client, path, has_json=json is not None, timeout=timeout, provider_cfg=provider_cfg
    )
    kwargs = _send_kwargs(json=json, params=params, files=files, data=data, is_async=True)
    errors = _async_httpx()
    try:
        async with client._async_session(effective_timeout) as session:
            resp = await getattr(session, method.lower())(url, headers=headers, **kwargs)
    except errors.TimeoutException as exc:
        raise TimeoutError("request timed out") from exc
    except errors.HTTPError as exc:
        raise _send_failed(exc) from exc
    if not _is_success(resp):
        raise http_error_from_response(resp)
    return resp


async def arequest_json(client: Any, method: str, path: str, **kwargs: Any) -> tuple[Any, Any]:
    """Async :func:`request_json`."""
    return _json_payload(await arequest(client, method, path, **kwargs))


async def aopen_stream(  # noqa: PLR0913 - keyword-only request options
    client: Any,
    method: str,
    path: str,
    *,
    json: Any = None,
    params: Any = None,
    timeout: float | None = None,
    provider_cfg: dict[str, Any] | None = None,
) -> tuple[Any, AsyncExitStack]:
    """Async :func:`open_stream`; the closer is an ``AsyncExitStack``."""
    url, headers, effective_timeout = _prepare(
        client, path, has_json=json is not None, timeout=timeout, provider_cfg=provider_cfg
    )
    kwargs = _send_kwargs(json=json, params=params, files=None, data=None, is_async=True)
    errors = _async_httpx()
    closer = AsyncExitStack()
    try:
        try:
            session = await closer.enter_async_context(client._async_session(effective_timeout))
            resp = await closer.enter_async_context(session.stream(method, url, headers=headers, **kwargs))
        except errors.TimeoutException as exc:
            raise TimeoutError("request timed out") from exc
        except errors.HTTPError as exc:
            raise _send_failed(exc) from exc
        if not _is_success(resp):
            await _aread_quietly(resp)
            raise http_error_from_response(resp)
    except BaseException:
        await closer.aclose()
        raise
    return resp, closer


@asynccontextmanager
async def astream_request(client: Any, method: str, path: str, **kwargs: Any) -> AsyncIterator[Any]:
    """Async :func:`stream_request`; read the body with :func:`aiter_response_bytes`."""
    resp, closer = await aopen_stream(client, method, path, **kwargs)
    async with closer:
        yield resp
