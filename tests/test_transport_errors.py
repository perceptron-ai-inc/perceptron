"""Error mapping (`http_error_from_response`), the error classes, SSE `data:` parsing, and the request helpers.

Responses are real `httpx.Response` objects; requests go through `httpx.MockTransport` (see `_http_mock`).
"""

import asyncio
import json

import httpx
import pytest
from _http_mock import Body, FailingBody, install, json_response, text_response

from perceptron import _transport
from perceptron import client as client_mod
from perceptron.errors import (
    AuthError,
    BadRequestError,
    IncompleteStreamError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    SDKError,
    ServerError,
    TransportError,
)
from perceptron.errors import TimeoutError as SDKTimeoutError


def _error_response(status, error=None, *, headers=None, text=None):
    if text is not None:
        return httpx.Response(status, text=text, headers=headers or {})
    return httpx.Response(status, json={"error": error or {}}, headers=headers or {})


def _gateway_error(message, *, type_="invalid_request_error", param=None, code=None):
    return {"message": message, "type": type_, "param": param, "code": code}


def test_map_http_error_is_the_transport_mapper():
    assert client_mod._map_http_error is _transport.http_error_from_response


@pytest.mark.parametrize(
    ("status", "expected_cls"),
    [
        (400, BadRequestError),
        (422, BadRequestError),
        (413, BadRequestError),
        (409, BadRequestError),
        (401, AuthError),
        (403, PermissionDeniedError),
        (404, NotFoundError),
        (429, RateLimitError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (504, ServerError),
    ],
)
def test_status_maps_to_error_class(status, expected_cls):
    err = _transport.http_error_from_response(_error_response(status, _gateway_error("boom", code="some_code")))

    assert type(err) is expected_cls
    assert str(err) == "boom"
    assert err.status_code == status


def test_subclasses_keep_existing_except_clauses_working():
    assert issubclass(NotFoundError, BadRequestError)
    assert issubclass(PermissionDeniedError, AuthError)
    assert issubclass(QuotaExceededError, RateLimitError)
    assert issubclass(IncompleteStreamError, TransportError)


def test_gateway_fields_become_attributes_and_request_id_comes_from_trace_header():
    error = _gateway_error("Model 'x' does not support tool calling", param="tools", code="unsupported_parameter")
    resp = _error_response(400, error, headers={"x-trace-id": "0123abcd"})

    err = _transport.http_error_from_response(resp)

    assert err.code == "unsupported_parameter"
    assert err.error_type == "invalid_request_error"
    assert err.param == "tools"
    assert err.request_id == "0123abcd"
    # details is a new dict: the server error plus the mirrored request_id (status/type/param are attributes only)
    assert err.details == {**error, "request_id": "0123abcd"}
    assert "status_code" not in err.details


def test_details_is_a_copy_of_the_server_error():
    payload = {"error": {"message": "nope", "code": "x"}}

    class _SharedJson(httpx.Response):
        """Hands out one parsed body, so an error that aliased it would change it."""

        def json(self, **kwargs):
            return payload

    err = _transport.http_error_from_response(_SharedJson(400, json=payload))
    err.details["extra"] = 1

    assert err.details == {"message": "nope", "code": "x", "extra": 1}
    assert payload["error"] == {"message": "nope", "code": "x"}


def test_code_falls_back_to_type_and_auth_default():
    err = _transport.http_error_from_response(_error_response(400, {"message": "bad", "type": "invalid_request_error"}))
    assert err.code == "invalid_request_error"

    err = _transport.http_error_from_response(_error_response(401, {"message": "Invalid API key"}))
    assert err.code == "auth_error"

    err = _transport.http_error_from_response(_error_response(403, {"message": "denied"}))
    assert isinstance(err, PermissionDeniedError)
    assert err.code == "auth_error"


def test_trace_header_lookup_is_case_insensitive():
    err = _transport.http_error_from_response(_error_response(400, {"message": "bad"}, headers={"X-Trace-Id": "abc"}))

    assert err.request_id == "abc"
    assert err.details["request_id"] == "abc"


def test_rate_limit_keeps_its_code_and_reads_retry_after():
    error = _gateway_error("Organization rate limit exceeded", type_="rate_limit_error", code="rate_limit_exceeded")
    resp = _error_response(429, error, headers={"Retry-After": "30"})

    err = _transport.http_error_from_response(resp)

    assert type(err) is RateLimitError
    assert err.code == "rate_limit"
    assert err.error_type == "rate_limit_error"
    assert err.details["code"] == "rate_limit_exceeded"
    assert err.details["type"] == "rate_limit_error"
    assert err.retry_after == 30.0
    assert err.details["retry_after"] == 30.0


def test_retry_after_is_none_when_missing():
    err = _transport.http_error_from_response(_error_response(429, _gateway_error("slow", type_="rate_limit_error")))

    assert err.retry_after is None
    assert "retry_after" not in err.details


def test_insufficient_quota_is_not_retryable():
    error = _gateway_error("You exceeded your current quota", type_="insufficient_quota")
    resp = _error_response(429, error, headers={"Retry-After": "30"})

    err = _transport.http_error_from_response(resp)

    assert type(err) is QuotaExceededError
    assert err.code == "insufficient_quota"
    assert err.retry_after is None
    assert "retry_after" not in err.details


def test_storage_quota_413_is_a_quota_error():
    err = _transport.http_error_from_response(_error_response(413, _gateway_error("quota", type_="insufficient_quota")))
    assert type(err) is QuotaExceededError

    err = _transport.http_error_from_response(_error_response(413, _gateway_error("file too large")))
    assert type(err) is BadRequestError


def test_503_model_overloaded_reads_retry_after():
    error = _gateway_error("The model is currently overloaded", type_="server_error", code="model_overloaded")
    err = _transport.http_error_from_response(_error_response(503, error, headers={"retry-after": "30"}))

    assert type(err) is ServerError
    assert err.code == "model_overloaded"
    assert err.retry_after == 30.0
    assert err.details["retry_after"] == 30.0


def test_retry_after_ignored_on_other_5xx():
    err = _transport.http_error_from_response(_error_response(500, _gateway_error("x"), headers={"Retry-After": "5"}))
    assert err.retry_after is None


def test_upstream_5xx_json_in_message_is_unwrapped():
    inner = {
        "error": {
            "message": "Generated content did not match the requested structured-output format.",
            "type": "server_error",
            "param": "response_format",
            "code": "output_validation_failed",
        }
    }
    outer = {"message": json.dumps(inner), "type": None, "param": None, "code": None}

    err = _transport.http_error_from_response(_error_response(500, outer, headers={"x-trace-id": "t"}))

    assert type(err) is ServerError
    assert str(err) == inner["error"]["message"]
    assert err.code == "output_validation_failed"
    assert err.error_type == "server_error"
    assert err.param == "response_format"
    assert err.details["code"] == "output_validation_failed"


def test_message_that_is_not_json_error_is_kept():
    outer = {"message": "{not json", "type": None, "param": None, "code": None}
    err = _transport.http_error_from_response(_error_response(502, outer))

    assert str(err) == "{not json"
    assert err.code is None


def test_non_json_bodies_use_the_trimmed_text():
    err = _transport.http_error_from_response(_error_response(504, text="  upstream request timeout \n"))
    assert type(err) is ServerError
    assert str(err) == "upstream request timeout"
    assert err.details == {}

    err = _transport.http_error_from_response(_error_response(429, text="Too Many Requests"))
    assert type(err) is RateLimitError  # no crash on a non-JSON 429
    assert str(err) == "Too Many Requests"


def test_detect_error_shape_is_parsed():
    error = {
        "message": "Invalid API key",
        "type": "authentication_error",
        "param": None,
        "code": "authentication_error",
    }
    err = _transport.http_error_from_response(_error_response(401, error))

    assert err.code == "authentication_error"
    assert err.error_type == "authentication_error"


@pytest.mark.parametrize(
    "body",
    [None, Body(b'{"error": {"message": "unread"}}')],
    ids=["empty", "unread"],
)
def test_mapper_falls_back_to_the_status_without_a_body(body):
    # An unread streamed body (e.g. its read failed) makes `json()` and `text` raise `ResponseNotRead`.
    resp = httpx.Response(500) if body is None else httpx.Response(500, stream=body)

    err = _transport.http_error_from_response(resp)

    assert type(err) is ServerError
    assert str(err) == "server error: 500"
    assert err.request_id is None


def test_streamed_error_body_is_read_before_mapping(monkeypatch, perceptron_env):
    """A non-2xx streaming response is unread (like a socket); its body must still reach the error."""
    body = Body(json.dumps({"error": _gateway_error("bad tool", code="unsupported_parameter")}).encode())
    install(monkeypatch, lambda request: httpx.Response(400, stream=body))

    with pytest.raises(BadRequestError) as excinfo:
        _transport.open_stream(client_mod.Client(), "POST", "/chat/completions", json={"model": "m"})

    assert str(excinfo.value) == "bad tool"
    assert excinfo.value.code == "unsupported_parameter"
    assert body.closed  # the response was closed before the error was raised


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


def test_sdk_error_positional_signature_and_details_default():
    err = SDKError("boom", "code_x", {"a": 1})
    assert (str(err), err.code, err.details) == ("boom", "code_x", {"a": 1})
    assert SDKError().details == {}
    assert err.status_code is None and err.partial is None


def test_sdk_error_mirrors_request_id_and_retry_after_only():
    err = ServerError("x", status_code=503, request_id="r", retry_after=2.0, error_type="server_error", param="p")

    assert err.details == {"request_id": "r", "retry_after": 2.0}
    assert (err.status_code, err.error_type, err.param) == (503, "server_error", "p")


def test_rate_limit_error_without_details_does_not_crash():
    err = RateLimitError("slow", retry_after=1.5, details=None)
    assert err.code == "rate_limit"
    assert err.details == {"retry_after": 1.5}

    err = RateLimitError("slow")
    assert err.retry_after is None
    assert err.details == {}


def test_quota_error_never_has_retry_after():
    err = QuotaExceededError("out of credits", retry_after=10.0, details={"type": "insufficient_quota"})
    assert err.code == "insufficient_quota"
    assert err.retry_after is None


# ---------------------------------------------------------------------------
# SSE data lines
# ---------------------------------------------------------------------------


def test_iter_sse_data_yields_payloads_and_done_sentinel():
    lines = [
        'data: {"a": 1}',
        "",
        ":",
        ": keep-alive",
        "event: message",
        "id: 7",
        "retry: 1000",
        'data:{"b": 2}',
        b'data: {"c": 3}\r\n',
        "data:  two-spaces",
        "data: [DONE]",
    ]

    out = list(_transport.iter_sse_data(lines))

    assert out[:4] == ['{"a": 1}', '{"b": 2}', '{"c": 3}', " two-spaces"]
    assert out[4] is _transport.DONE
    assert len(out) == 5


def test_aiter_sse_data_matches_sync():
    async def _lines():
        for line in ["data: x", ": ping", "data: [DONE]"]:
            yield line

    async def _collect():
        return [item async for item in _transport.aiter_sse_data(_lines())]

    out = asyncio.run(_collect())
    assert out[0] == "x"
    assert out[1] is _transport.DONE


def _lines_in_chunks(body: bytes, size: int) -> list[str]:
    # A streamed response whose body arrives in `size`-byte chunks.
    chunks = [body[i : i + size] for i in range(0, len(body), size)]
    return list(_transport.iter_response_lines(httpx.Response(200, content=chunks)))


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_response_lines_end_only_at_cr_lf_or_crlf(separator):
    # JSON (and the gateway's serde_json) leaves these raw inside a `data:` line; they must not split it.
    event = json.dumps({"content": f"one{separator}two"}, ensure_ascii=False)
    body = f"data: {event}\r\n\r\ndata: a\rdata: b\n\ndata: [DONE]\r".encode()
    expected = [f"data: {event}", "", "data: a", "data: b", "", "data: [DONE]"]

    for size in range(1, len(body) + 1):  # every chunking, including a CR and a multibyte char split across chunks
        assert _lines_in_chunks(body, size) == expected, size


def test_response_lines_drop_an_unterminated_last_line_except_done():
    # An event cut off mid-line at EOF is discarded (the stream then ends without [DONE]: truncated).
    assert _lines_in_chunks(b'data: {"a": 1}\n\ndata: {"b": ', 4) == ['data: {"a": 1}', ""]
    assert _lines_in_chunks(b'data: {"a": 1}\n\ndata: [DONE]', 4) == ['data: {"a": 1}', "", "data: [DONE]"]


def test_aiter_response_lines_matches_sync():
    async def _chunks():
        for piece in ("data: x\u2028y".encode(), b"\r", b"\ndata: [DONE]"):
            yield piece

    async def _collect():
        return [line async for line in _transport.aiter_response_lines(httpx.Response(200, content=_chunks()))]

    assert asyncio.run(_collect()) == ["data: x\u2028y", "data: [DONE]"]


# ---------------------------------------------------------------------------
# Resource request helpers (used by files, models and multilook)
# ---------------------------------------------------------------------------


@pytest.fixture
def perceptron_env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def test_request_json_builds_url_auth_and_params(monkeypatch, perceptron_env):
    recorder = install(monkeypatch, lambda request: json_response({"data": []}, headers={"x-trace-id": "t"}))

    payload, headers = _transport.request_json(client_mod.Client(), "GET", "/files", params={"limit": 2})

    assert payload == {"data": []}
    assert _transport.header_value(headers, "X-Trace-Id") == "t"
    request = recorder.last
    assert str(request.url) == "https://api.perceptron.inc/v1/files?limit=2"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert "content-type" not in request.headers


def test_request_sends_multipart_without_forcing_a_content_type(monkeypatch, perceptron_env):
    recorder = install(monkeypatch, lambda request: json_response({"id": "file-x"}))

    _transport.request(
        client_mod.Client(),
        "POST",
        "/files",
        files={"file": ("a.png", b"\x89PNG", "image/png")},
        data={"purpose": "vision"},
    )

    assert recorder.last.headers["content-type"].startswith("multipart/form-data; boundary=")


def test_request_maps_http_errors_and_transport_failures(monkeypatch, perceptron_env):
    error = {"error": {"message": "No such file", "type": "invalid_request_error", "param": "id", "code": None}}
    install(monkeypatch, lambda request: json_response(error, 404, headers={"x-trace-id": "t404"}))
    with pytest.raises(NotFoundError) as excinfo:
        _transport.request(client_mod.Client(), "DELETE", "/files/file-x")
    assert (excinfo.value.param, excinfo.value.request_id) == ("id", "t404")

    def _timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    install(monkeypatch, _timeout)
    with pytest.raises(SDKTimeoutError):
        _transport.request(client_mod.Client(), "GET", "/models")

    def _refused(request):
        raise httpx.ConnectError("refused", request=request)

    install(monkeypatch, _refused)
    with pytest.raises(TransportError):
        _transport.request(client_mod.Client(), "GET", "/models")


@pytest.mark.parametrize("padding", ["\n", "\r\n", " ", "\t"])
def test_api_key_whitespace_is_stripped(monkeypatch, perceptron_env, padding):
    # Keys read from a file or secret mount often end with a newline, which is an illegal header value.
    recorder = install(monkeypatch, lambda request: json_response({"data": []}))
    monkeypatch.setenv("PERCEPTRON_API_KEY", f"{padding}sk-test{padding}")

    _transport.request_json(client_mod.Client(), "GET", "/models")

    assert recorder.last.headers["authorization"] == "Bearer sk-test"


@pytest.mark.parametrize("key", ["\u201csk-secret\u201d", "sk-sec ret", "sk-sec\x01ret"])
def test_malformed_api_key_raises_auth_error_without_echoing_it(monkeypatch, perceptron_env, key):
    recorder = install(monkeypatch, lambda request: json_response({"data": []}))
    monkeypatch.setenv("PERCEPTRON_API_KEY", key)

    with pytest.raises(AuthError) as excinfo:
        _transport.request_json(client_mod.Client(), "GET", "/models")

    assert "secret" not in str(excinfo.value) and "sec" not in repr(excinfo.value.details)
    assert recorder.requests == []
    # The legacy stream reports it as its terminal error event.
    events = list(client_mod.Client(provider="perceptron").stream({"content": [{"type": "text", "content": "hi"}]}))
    assert [event["type"] for event in events] == ["error"]
    assert "secret" not in events[0]["message"]


def test_local_protocol_errors_are_not_echoed(monkeypatch, perceptron_env):
    # An illegal header value's error message quotes the header (the Authorization header included).
    def _illegal(request):
        raise httpx.LocalProtocolError("Illegal header value b'Bearer sk-secret\\n'")

    install(monkeypatch, _illegal)
    with pytest.raises(TransportError) as excinfo:
        _transport.request(client_mod.Client(), "GET", "/models")
    assert "sk-secret" not in str(excinfo.value)


@pytest.mark.parametrize("resource_id", [".", ".."])
def test_dot_segment_ids_raise_before_any_request(monkeypatch, perceptron_env, resource_id):
    # httpx removes dot segments, so `/files/.` would reach `GET /files` (the list) instead.
    recorder = install(monkeypatch, lambda request: json_response({"object": "list", "data": []}))
    client = client_mod.Client()
    calls = {
        "file_id": [
            lambda: client.files.retrieve(resource_id),
            lambda: client.files.delete(resource_id),
            lambda: client.files.content(resource_id),
        ],
        "model_id": [lambda: client.models.retrieve(resource_id)],
    }
    for param, fns in calls.items():
        for fn in fns:
            with pytest.raises(BadRequestError) as excinfo:
                fn()
            assert (excinfo.value.code, excinfo.value.param) == ("invalid_parameter", param)
    assert recorder.requests == []
    assert _transport.resource_path("files", "a.b", "file_id", "/content") == "/files/a.b/content"


def test_request_json_rejects_non_json(monkeypatch, perceptron_env):
    install(monkeypatch, lambda request: text_response("<html>"))
    with pytest.raises(ServerError) as excinfo:
        _transport.request_json(client_mod.Client(), "GET", "/models")
    assert excinfo.value.code == "invalid_response"


def test_stream_request_yields_an_open_response_and_closes_it(monkeypatch, perceptron_env):
    body = Body(b"file-bytes")
    install(monkeypatch, lambda request: httpx.Response(200, stream=body))

    with _transport.stream_request(client_mod.Client(), "GET", "/files/file-x/content") as resp:
        assert b"".join(resp.iter_bytes()) == b"file-bytes"
    assert body.closed


MID_BODY_FAILURES = [
    (httpx.ReadTimeout("read timed out"), SDKTimeoutError),
    (httpx.RemoteProtocolError("peer closed connection"), TransportError),
]


def test_iter_response_bytes_reads_the_body(monkeypatch, perceptron_env):
    install(monkeypatch, lambda request: httpx.Response(200, stream=Body(b"file-bytes")))

    with _transport.stream_request(client_mod.Client(), "GET", "/files/file-x/content") as resp:
        assert list(_transport.iter_response_bytes(resp, chunk_size=4)) == [b"file", b"-byt", b"es"]


@pytest.mark.parametrize(("exc", "expected_cls"), MID_BODY_FAILURES)
def test_iter_response_bytes_converts_failures_mid_body(monkeypatch, perceptron_env, exc, expected_cls):
    body = FailingBody(b"first", exc)
    install(monkeypatch, lambda request: httpx.Response(200, stream=body))
    received = []

    def _download():
        with _transport.stream_request(client_mod.Client(), "GET", "/files/file-x/content") as resp:
            for piece in _transport.iter_response_bytes(resp):
                received.append(piece)

    with pytest.raises(expected_cls) as excinfo:
        _download()

    assert type(excinfo.value) is expected_cls
    assert excinfo.value.__cause__ is exc
    assert received == [b"first"]
    assert body.closed


@pytest.mark.parametrize(("exc", "expected_cls"), MID_BODY_FAILURES)
def test_aiter_response_bytes_converts_failures_mid_body(monkeypatch, perceptron_env, exc, expected_cls):
    body = FailingBody(b"first", exc)
    install(monkeypatch, lambda request: httpx.Response(200, stream=body))
    received = []

    async def _download():
        async with (
            client_mod.AsyncClient() as client,
            _transport.astream_request(client, "GET", "/files/file-x/content") as resp,
        ):
            async for piece in _transport.aiter_response_bytes(resp):
                received.append(piece)

    with pytest.raises(expected_cls) as excinfo:
        asyncio.run(_download())

    assert type(excinfo.value) is expected_cls
    assert excinfo.value.__cause__ is exc
    assert received == [b"first"]
    assert body.closed


def test_async_helpers(monkeypatch, perceptron_env):
    body = Body(b"async-bytes")

    def _handler(request):
        if request.url.path.endswith("/content"):
            return httpx.Response(200, stream=body)
        return json_response({"object": "list", "data": [{"id": "perceptron-mk1.5"}]})

    recorder = install(monkeypatch, _handler)

    async def _run():
        async with client_mod.AsyncClient() as client:
            payload, _ = await _transport.arequest_json(client, "POST", "/chat/completions/multilook", json={"a": 1})
            async with _transport.astream_request(client, "GET", "/files/file-x/content") as resp:
                data = b"".join([chunk async for chunk in resp.aiter_bytes()])
        return payload, data

    payload, data = asyncio.run(_run())

    assert payload["data"][0]["id"] == "perceptron-mk1.5"
    assert json.loads(recorder.requests[0].content) == {"a": 1}
    assert recorder.requests[0].headers["content-type"] == "application/json"
    assert data == b"async-bytes"
    assert body.closed
