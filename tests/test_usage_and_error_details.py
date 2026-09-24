"""Usage passes through verbatim (latest wins in streams) and terminal errors keep their structured detail (DESIGN §13.1,
§13.3)."""

import asyncio
import copy

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response

from perceptron import AsyncClient, Client, perceive, text
from perceptron.chat import Usage
from perceptron.errors import STREAM_TRUNCATED, BadRequestError

TASK = {"content": [{"type": "text", "role": "user", "content": "Transcribe the clip."}]}
AUDIO_USAGE = {
    "prompt_tokens": 1412,
    "completion_tokens": 42,
    "total_tokens": 1454,
    "prompt_tokens_details": {"audio_tokens": 0},
}
AUDIO_LIMIT = {
    "message": "Audio input is too long: it exceeds the per-item audio token limit.",
    "type": "invalid_request_error",
    "param": None,
    "code": "audio_token_limit_exceeded",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _client():
    return Client(provider="perceptron")


def _collect(async_iterable):
    async def _run():
        return [event async for event in async_iterable]

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Usage (§13.1)
# ---------------------------------------------------------------------------

USAGE_CASES = [
    AUDIO_USAGE,
    {**AUDIO_USAGE, "prompt_tokens_details": {"audio_tokens": 318}},
    {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}},  # only what was sent; nothing filled in
    {"prompt_tokens": None, "completion_tokens": 2, "total_tokens": None},
    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "prompt_tokens_details": None},
]


@pytest.mark.parametrize("usage", USAGE_CASES)
def test_generate_and_perceive_return_the_server_usage_verbatim(monkeypatch, usage):
    payload = completion("ok", usage=copy.deepcopy(usage))
    install(monkeypatch, lambda request: json_response(payload))

    result = _client().generate(TASK)
    res = perceive(text("Transcribe the clip."), provider="perceptron")

    assert result["usage"] == usage and res.usage == usage
    assert result["raw"]["usage"] == usage


@pytest.mark.parametrize("extra", [{}, {"usage": None}], ids=["missing", "null"])
def test_missing_or_null_usage_is_none(monkeypatch, extra):
    install(monkeypatch, lambda request: json_response(completion("ok", **extra)))

    assert _client().generate(TASK)["usage"] is None
    assert perceive(text("Hi"), provider="perceptron").usage is None
    assert Client().chat.completions.create(messages=[{"role": "user", "content": "Hi"}]).usage is None


def test_usage_detail_properties():
    assert Usage.from_dict(AUDIO_USAGE).audio_tokens == 0
    assert Usage.from_dict(AUDIO_USAGE).cached_tokens is None
    cached = Usage.from_dict({"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}})
    assert (cached.audio_tokens, cached.cached_tokens, cached.completion_tokens) == (None, 3, None)
    assert cached.prompt_tokens_details == {"cached_tokens": 3}
    bare = Usage.from_dict({"prompt_tokens": 1, "prompt_tokens_details": None})
    assert (bare.audio_tokens, bare.cached_tokens, bare.prompt_tokens_details) == (None, None, None)
    assert Usage().audio_tokens is None
    assert Usage(prompt_tokens_details={"audio_tokens": 318, "cached_tokens": None}).audio_tokens == 318


def test_create_exposes_audio_and_cached_tokens(monkeypatch):
    usage = {**AUDIO_USAGE, "prompt_tokens_details": {"audio_tokens": 318, "cached_tokens": 64}}
    install(monkeypatch, lambda request: json_response(completion("ok", usage=usage)))

    result = Client().chat.completions.create(messages=[{"role": "user", "content": "Hi"}])

    assert (result.usage.audio_tokens, result.usage.cached_tokens) == (318, 64)
    assert result.usage.prompt_tokens_details == usage["prompt_tokens_details"]


STREAM_USAGE = [
    chunk({"content": "Hello"}, usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}),
    {**chunk({"content": " there"}), "usage": None},  # a later `null` never erases what arrived
    chunk({}, finish_reason="stop"),
    chunk(choices=False, usage={**AUDIO_USAGE, "prompt_tokens_details": {"audio_tokens": 7}}),
]


def test_stream_usage_latest_wins_and_is_never_summed(monkeypatch):
    install(monkeypatch, lambda request: sse_response(STREAM_USAGE))
    expected = {**AUDIO_USAGE, "prompt_tokens_details": {"audio_tokens": 7}}

    final = list(_client().stream(TASK))[-1]
    assert final["type"] == "final" and final["result"]["usage"] == expected

    final = list(perceive(text("Hi"), provider="perceptron", stream=True))[-1]
    assert final["result"]["usage"] == expected

    final = _collect(AsyncClient(provider="perceptron").stream(TASK))[-1]
    assert final["result"]["usage"] == expected

    completion_ = Client().chat.completions.create(messages=[{"role": "user", "content": "Hi"}], stream=True)
    assert completion_.get_final_completion().usage.audio_tokens == 7


def test_stream_without_usage_reports_none(monkeypatch):
    install(monkeypatch, lambda request: sse_response([chunk({"content": "ok"}), chunk({}, finish_reason="stop")]))

    assert list(_client().stream(TASK))[-1]["result"]["usage"] is None


# ---------------------------------------------------------------------------
# Structured errors (§13.3)
# ---------------------------------------------------------------------------


def _audio_limit(monkeypatch):
    return install(
        monkeypatch, lambda request: json_response({"error": AUDIO_LIMIT}, 400, headers={"x-trace-id": "trace-a"})
    )


def test_pre_stream_json_error_keeps_code_type_param_and_details(monkeypatch):
    _audio_limit(monkeypatch)

    with pytest.raises(BadRequestError) as excinfo:
        _client().generate(TASK)
    err = excinfo.value
    assert (err.code, err.error_type, err.param, err.status_code) == (
        "audio_token_limit_exceeded",
        "invalid_request_error",
        None,
        400,
    )
    assert err.details == {**AUDIO_LIMIT, "request_id": "trace-a"}

    expected_event = {
        "type": "error",
        "message": AUDIO_LIMIT["message"],
        "code": "audio_token_limit_exceeded",
        "error_type": "invalid_request_error",
        "param": None,
        "status": 400,
        "request_id": "trace-a",
        "details": {**AUDIO_LIMIT, "request_id": "trace-a"},
        "partial": None,
    }
    assert list(_client().stream(TASK)) == [expected_event]
    assert list(perceive(text("Hi"), provider="perceptron", stream=True)) == [expected_event]
    assert _collect(AsyncClient(provider="perceptron").stream(TASK)) == [expected_event]

    with pytest.raises(BadRequestError) as excinfo:
        Client().chat.completions.create(messages=[{"role": "user", "content": "Hi"}], stream=True)
    assert excinfo.value.details == err.details and excinfo.value.param is None


def test_mid_stream_error_event_keeps_its_details_and_yields_no_final(monkeypatch):
    error = {"message": "Output failed validation.", "type": "invalid_request_error"}
    error |= {"param": "response_format", "code": "output_validation_failed"}
    events_in = [chunk({"content": "{"}), {"error": error}]
    install(monkeypatch, lambda request: sse_response(events_in, done=False, headers={"x-trace-id": "trace-m"}))

    events = list(_client().stream(TASK))

    assert [event["type"] for event in events] == ["text.delta", "error"]
    assert events[-1]["code"] == "output_validation_failed" and events[-1]["param"] == "response_format"
    assert events[-1]["details"] == {**error, "request_id": "trace-m"}
    assert events[-1]["partial"]["text"] == "{"

    events = _collect(AsyncClient(provider="perceptron").stream(TASK))
    assert [event["type"] for event in events] == ["text.delta", "error"]
    assert events[-1]["details"] == {**error, "request_id": "trace-m"}


def test_truncated_stream_error_event_has_details_and_no_final(monkeypatch):
    events_in = [chunk({"content": "Par"})]
    install(monkeypatch, lambda request: sse_response(events_in, done=False, headers={"x-trace-id": "trace-t"}))

    events = list(_client().stream(TASK))

    assert [event["type"] for event in events] == ["text.delta", "error"]
    assert events[-1]["code"] == STREAM_TRUNCATED
    assert events[-1]["request_id"] == "trace-t" and events[-1]["details"] == {"request_id": "trace-t"}


def test_pre_request_error_event_has_details(monkeypatch):
    install(monkeypatch, lambda request: sse_response([chunk({}, finish_reason="stop")]))

    [event] = list(_client().stream(TASK, model="perceptron-mk1.5-preview"))

    assert event["type"] == "error" and event["details"] == {} and event["partial"] is None
