"""Usage, completion status and structured errors on every surface (DESIGN §13.1-§13.3, §12.5).

Mocked HTTP, provider ``perceptron``, default model ``perceptron-mk1.5``, an audio prompt. Legacy surfaces return result
dicts / ``PerceiveResult`` / event streams; the message API returns ``ChatCompletion`` objects and raises.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response

from perceptron import AsyncClient, Client, async_perceive, audio, block, caption, perceive, question, text
from perceptron.chat import ChatCompletion, ToolCall, Usage
from perceptron.dsl.perceive import _compile
from perceptron.errors import (
    STREAM_INCOMPLETE,
    STREAM_TRUNCATED,
    BadRequestError,
    IncompleteStreamError,
    RateLimitError,
    ServerError,
)

AUDIO_URL = "https://example.com/interview.flac"
PROMPT = "Transcribe the clip."
TRACE = {"x-trace-id": "trace-9"}
AUDIO_LIMIT = {
    "message": "Audio input is too long: it exceeds the per-item audio token limit.",
    "type": "invalid_request_error",
    "param": None,
    "code": "audio_token_limit_exceeded",
}
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _serve(monkeypatch, *, payload=None, events=(), done=True, headers=None, error=None, status=400):  # noqa: PLR0913 - response shapes
    """JSON for plain requests, SSE for ``stream`` requests; ``error`` answers every request with that HTTP error."""

    def handler(request):
        if error is not None:
            return json_response({"error": error}, status, headers=headers)
        if json.loads(request.content).get("stream"):
            return sse_response(list(events), done=done, headers=headers)
        return json_response(completion("Hello") if payload is None else payload, headers=headers)

    return install(monkeypatch, handler)


def _acollect(stream) -> list:
    async def _run():
        return [event async for event in stream]

    return asyncio.run(_run())


def _task() -> dict:
    task, _ = _compile(block(audio(AUDIO_URL), text(PROMPT)), expects=None, strict=False)
    return task


def _messages() -> list[dict]:
    return [{"role": "user", "content": [audio(AUDIO_URL), PROMPT]}]


def _result(value) -> dict:
    """A ``PerceiveResult`` as the result-dict keys this module checks."""
    if isinstance(value, dict):
        return value
    keys = ("text", "usage", "raw", "finish_reason", "tool_calls", "errors", "complete")
    return {key: getattr(value, key) for key in keys}


def _async_perceive_result(**opts):
    @async_perceive(provider="perceptron", **opts)
    async def run():
        return audio(AUDIO_URL) + text(PROMPT)

    return asyncio.run(run())


def _async_perceive_events(**opts):
    @async_perceive(provider="perceptron", stream=True, **opts)
    async def run():
        return audio(AUDIO_URL) + text(PROMPT)

    return _acollect(run())


# Legacy non-stream surfaces: a result dict.
RESULTS = {
    "client_generate": lambda **o: Client(provider="perceptron").generate(_task(), **o),
    "async_client_generate": lambda **o: asyncio.run(AsyncClient(provider="perceptron").generate(_task(), **o)),
    "perceive": lambda **o: _result(perceive(audio(AUDIO_URL), text(PROMPT), provider="perceptron", **o)),
    "async_perceive": lambda **o: _result(_async_perceive_result(**o)),
    "question": lambda **o: _result(question(audio(AUDIO_URL), PROMPT, provider="perceptron", **o)),
    "caption": lambda **o: _result(caption(audio(AUDIO_URL), provider="perceptron", **o)),
}

# Legacy event streams: the list of events.
EVENTS = {
    "client_stream": lambda **o: list(Client(provider="perceptron").stream(_task(), **o)),
    "async_client_stream": lambda **o: _acollect(AsyncClient(provider="perceptron").stream(_task(), **o)),
    "perceive_stream": lambda **o: list(
        perceive(audio(AUDIO_URL), text(PROMPT), provider="perceptron", stream=True, **o)
    ),
    "async_perceive_stream": _async_perceive_events,
    "question_stream": lambda **o: list(question(audio(AUDIO_URL), PROMPT, provider="perceptron", stream=True, **o)),
    "caption_stream": lambda **o: list(caption(audio(AUDIO_URL), provider="perceptron", stream=True, **o)),
}


def _async_create(**opts) -> ChatCompletion:
    async def run():
        return await AsyncClient().chat.completions.create(messages=_messages(), **opts)

    return asyncio.run(run())


# Message API, non-stream: a ChatCompletion.
COMPLETIONS = {
    "create": lambda **o: Client().chat.completions.create(messages=_messages(), **o),
    "async_create": _async_create,
}


def _create_stream(**opts):
    stream = Client().chat.completions.create(messages=_messages(), stream=True, **opts)
    chunks = list(stream)
    return chunks, stream.get_final_completion()


def _async_create_stream(**opts):
    async def run():
        stream = await AsyncClient().chat.completions.create(messages=_messages(), stream=True, **opts)
        chunks = [item async for item in stream]
        return chunks, await stream.get_final_completion()

    return asyncio.run(run())


# Message API streams: (chunks, final ChatCompletion); errors raise.
STREAMS = {"create_stream": _create_stream, "async_create_stream": _async_create_stream}


def _final(events: list[dict]) -> dict:
    assert [event["type"] for event in events].count("final") == 1
    assert events[-1]["type"] == "final", events[-1]
    assert not any(event["type"] == "error" for event in events)
    return events[-1]["result"]


# ---------------------------------------------------------------------------
# Usage (§13.1): the server's object verbatim; missing stays missing, null stays None, never coerced to 0
# ---------------------------------------------------------------------------

BASE_USAGE = {"prompt_tokens": 1412, "completion_tokens": 42, "total_tokens": 1454}
USAGE_CASES = {
    "missing": None,
    "null": None,
    "no-details": BASE_USAGE,
    "null-details": {**BASE_USAGE, "prompt_tokens_details": None},
    "cached-only": {"prompt_tokens": 1412, "prompt_tokens_details": {"cached_tokens": 1024}},
    "audio-zero": {**BASE_USAGE, "prompt_tokens_details": {"audio_tokens": 0}},
    "audio-nonzero": {**BASE_USAGE, "prompt_tokens_details": {"audio_tokens": 318, "cached_tokens": 64}},
    "extra-keys": {
        **BASE_USAGE,
        "prompt_tokens_details": {"audio_tokens": 318, "image_tokens": 900, "cached_tokens": None},
        "completion_tokens_details": {"reasoning_tokens": 7},
        "credits": 0.25,
    },
}


def _usage_payload(case: str) -> dict:
    if case == "missing":
        return completion("Hello")
    return completion("Hello", usage=None) | {"usage": copy.deepcopy(USAGE_CASES[case])}


def _detail(usage: dict | None, name: str):
    details = (usage or {}).get("prompt_tokens_details")
    return details.get(name) if isinstance(details, dict) else None


@pytest.mark.parametrize("surface", RESULTS)
@pytest.mark.parametrize("case", USAGE_CASES)
def test_legacy_results_carry_the_usage_verbatim(monkeypatch, surface, case):
    _serve(monkeypatch, payload=_usage_payload(case))

    result = RESULTS[surface]()

    assert result["usage"] == USAGE_CASES[case]
    if case == "missing":
        assert "usage" not in result["raw"]
    else:
        assert result["raw"]["usage"] == USAGE_CASES[case]


@pytest.mark.parametrize("surface", COMPLETIONS)
@pytest.mark.parametrize("case", USAGE_CASES)
def test_chat_completion_usage_and_its_detail_properties(monkeypatch, surface, case):
    _serve(monkeypatch, payload=_usage_payload(case))
    usage = USAGE_CASES[case]

    result = COMPLETIONS[surface]()

    if usage is None:
        assert result.usage is None
        return
    assert isinstance(result.usage, Usage)
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )
    assert result.usage.prompt_tokens_details == usage.get("prompt_tokens_details")  # raw, unknown keys kept
    assert result.usage.audio_tokens == _detail(usage, "audio_tokens")
    assert result.usage.cached_tokens == _detail(usage, "cached_tokens")
    assert result.raw["usage"] == usage  # every key, top level included


def test_audio_tokens_zero_is_reported_as_zero_not_missing(monkeypatch):
    _serve(monkeypatch, payload=_usage_payload("audio-zero"))

    usage = Client().chat.completions.create(messages=_messages()).usage

    assert usage.audio_tokens == 0 and usage.audio_tokens is not None
    assert usage.cached_tokens is None


# ---------------------------------------------------------------------------
# Streaming usage (§13.2)
# ---------------------------------------------------------------------------

FIRST = {"prompt_tokens": 1412, "completion_tokens": 1, "total_tokens": 1413}
SECOND = {"prompt_tokens": 1412, "completion_tokens": 2, "total_tokens": 1414}
LATEST = {
    "prompt_tokens": 1412,
    "completion_tokens": 3,
    "total_tokens": 1415,
    "prompt_tokens_details": {"audio_tokens": 318, "cached_tokens": 0, "image_tokens": 0},
    "credits": 0.25,
}
USAGE_STREAMS = {
    # The usage arrives on the trailing `choices: []` chunk.
    "terminal-chunk": (
        [
            chunk({"role": "assistant", "content": "Hel"}),
            chunk({"content": "lo"}),
            chunk({}, finish_reason="stop"),
            chunk(choices=False, usage=LATEST),
        ],
        LATEST,
    ),
    # Continuous snapshots: the latest one is kept (never summed), and a later `null` does not erase it.
    "snapshots": (
        [
            chunk({"role": "assistant", "content": "Hel"}, usage=FIRST),
            chunk({"content": "lo"}, usage=SECOND),
            {**chunk({}, finish_reason="stop"), "usage": None},
            chunk(choices=False, usage=LATEST),
        ],
        LATEST,
    ),
    "snapshot-then-null": (
        [
            chunk({"role": "assistant", "content": "Hello"}, usage=SECOND),
            {**chunk({}, finish_reason="stop"), "usage": None},
        ],
        SECOND,
    ),
    # A usage-only chunk that omits `choices` altogether.
    "usage-without-choices": (
        [
            chunk({"role": "assistant", "content": "Hello"}),
            chunk({}, finish_reason="stop"),
            {"id": "chatcmpl-1", "object": "chat.completion.chunk", "usage": LATEST},
        ],
        LATEST,
    ),
    "none": ([chunk({"role": "assistant", "content": "Hello"}), chunk({}, finish_reason="stop")], None),
}


@pytest.mark.parametrize("surface", EVENTS)
@pytest.mark.parametrize("case", USAGE_STREAMS)
def test_legacy_streams_keep_the_latest_usage(monkeypatch, surface, case):
    events_in, expected = USAGE_STREAMS[case]
    _serve(monkeypatch, events=events_in)

    result = _final(EVENTS[surface]())

    assert result["text"] == "Hello"
    assert result["usage"] == expected
    assert result["complete"] is True and result["errors"] == []


@pytest.mark.parametrize("surface", STREAMS)
@pytest.mark.parametrize("case", USAGE_STREAMS)
def test_message_api_streams_keep_the_latest_usage(monkeypatch, surface, case):
    events_in, expected = USAGE_STREAMS[case]
    _serve(monkeypatch, events=events_in)

    chunks, final = STREAMS[surface]()

    assert final.text == "Hello" and final.complete is True
    if expected is None:
        assert final.usage is None
    else:
        assert final.usage.completion_tokens == expected["completion_tokens"]
        assert final.usage.prompt_tokens_details == expected.get("prompt_tokens_details")
        assert final.usage.audio_tokens == _detail(expected, "audio_tokens")
        assert final.usage.cached_tokens == _detail(expected, "cached_tokens")
    if case == "terminal-chunk":
        assert chunks[-1].choices == [] and chunks[-1].usage.audio_tokens == 318


ALL_STREAMS = {**EVENTS, **STREAMS}
STREAM_OPTIONS = [{"include_usage": False}, {"include_usage": True, "continuous_usage_stats": True}, {}]


@pytest.mark.parametrize("surface", ALL_STREAMS)
def test_perceptron_streams_request_usage_by_default(monkeypatch, surface):
    recorder = _serve(monkeypatch, events=USAGE_STREAMS["terminal-chunk"][0])

    ALL_STREAMS[surface]()

    assert recorder.last_body["stream"] is True
    assert recorder.last_body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("surface", ALL_STREAMS)
@pytest.mark.parametrize("options", STREAM_OPTIONS, ids=["usage-off", "extra-key", "empty"])
def test_explicit_stream_options_are_forwarded_verbatim(monkeypatch, surface, options):
    recorder = _serve(monkeypatch, events=USAGE_STREAMS["none"][0])

    ALL_STREAMS[surface](stream_options=copy.deepcopy(options))

    assert recorder.last_body["stream_options"] == options


@pytest.mark.parametrize("surface", [*RESULTS, *COMPLETIONS])
def test_non_stream_requests_send_no_stream_options(monkeypatch, surface):
    recorder = _serve(monkeypatch)

    {**RESULTS, **COMPLETIONS}[surface]()

    assert "stream" not in recorder.last_body and "stream_options" not in recorder.last_body


NULL_DELTAS = [
    chunk({"role": "assistant"}),
    chunk(None),  # "delta": null
    chunk({"content": "Hel"}),
    chunk(omit_delta=True),  # no "delta" key
    chunk({"content": "lo"}),
    chunk(omit_delta=True, finish_reason="stop"),
    chunk(None, choices=False, usage=BASE_USAGE),
]


@pytest.mark.parametrize("surface", EVENTS)
def test_legacy_streams_tolerate_null_and_missing_deltas(monkeypatch, surface):
    _serve(monkeypatch, events=NULL_DELTAS)

    events = EVENTS[surface]()

    assert [event["chunk"] for event in events if event["type"] == "text.delta"] == ["Hel", "lo"]
    result = _final(events)
    assert (result["text"], result["finish_reason"], result["complete"]) == ("Hello", "stop", True)
    assert result["usage"] == BASE_USAGE


@pytest.mark.parametrize("surface", STREAMS)
def test_message_api_streams_tolerate_null_and_missing_deltas(monkeypatch, surface):
    _serve(monkeypatch, events=NULL_DELTAS)

    chunks, final = STREAMS[surface]()

    assert len(chunks) == len(NULL_DELTAS)
    assert [c.choices[0].delta.content for c in chunks[:-1]] == [None, None, "Hel", None, "lo", None]
    assert (final.text, final.finish_reason, final.complete, final.usage.total_tokens) == ("Hello", "stop", True, 1454)


# ---------------------------------------------------------------------------
# Completion status (§13.3): finish_reason and complete on every result; `length` is not an error
# ---------------------------------------------------------------------------

FINISH_CASES = {
    "stop": ("stop", None, True),
    "length": ("length", None, False),
    "tool_calls": ("tool_calls", [CALL], True),
    "interrupted": ("interrupted", None, True),
    "tool_calls-without-calls": ("tool_calls", None, False),
    "calls-without-tool_calls": ("stop", [CALL], False),
    "no-finish_reason": (None, None, False),
}


@pytest.mark.parametrize("surface", RESULTS)
@pytest.mark.parametrize("case", FINISH_CASES)
def test_legacy_results_report_finish_reason_and_complete(monkeypatch, surface, case):
    finish_reason, calls, complete = FINISH_CASES[case]
    _serve(monkeypatch, payload=completion("Hello", finish_reason=finish_reason, tool_calls=calls))

    result = RESULTS[surface]()

    assert (result["finish_reason"], result["complete"]) == (finish_reason, complete)
    assert result["tool_calls"] == ([ToolCall.from_dict(CALL)] if calls else None)
    assert result["errors"] == []  # truncation (`length`) is reported by `complete`, not as an error


@pytest.mark.parametrize("surface", COMPLETIONS)
@pytest.mark.parametrize("case", FINISH_CASES)
def test_chat_completions_report_finish_reason_and_complete(monkeypatch, surface, case):
    finish_reason, calls, complete = FINISH_CASES[case]
    _serve(monkeypatch, payload=completion("Hello", finish_reason=finish_reason, tool_calls=calls))

    result = COMPLETIONS[surface]()

    assert (result.finish_reason, result.complete) == (finish_reason, complete)
    assert result.tool_calls == ([ToolCall.from_dict(CALL)] if calls else None)


def _finish_stream(finish_reason: str, calls: list | None) -> list[dict]:
    events = [chunk({"role": "assistant", "content": "Hello"})]
    for index, call in enumerate(calls or []):
        events.append(chunk({"tool_calls": [{"index": index, **call}]}))
    events.append(chunk({}, finish_reason=finish_reason))
    return events


STREAM_FINISH_CASES = {name: case for name, case in FINISH_CASES.items() if case[0] is not None}


@pytest.mark.parametrize("surface", EVENTS)
@pytest.mark.parametrize("case", STREAM_FINISH_CASES)
def test_legacy_streams_report_finish_reason_and_complete(monkeypatch, surface, case):
    finish_reason, calls, complete = FINISH_CASES[case]
    _serve(monkeypatch, events=_finish_stream(finish_reason, calls))

    result = _final(EVENTS[surface]())

    assert (result["finish_reason"], result["complete"]) == (finish_reason, complete)
    assert result["tool_calls"] == ([ToolCall.from_dict(CALL)] if calls else None)
    assert result["errors"] == []


@pytest.mark.parametrize("surface", STREAMS)
@pytest.mark.parametrize("case", STREAM_FINISH_CASES)
def test_message_api_streams_report_finish_reason_and_complete(monkeypatch, surface, case):
    finish_reason, calls, complete = FINISH_CASES[case]
    _serve(monkeypatch, events=_finish_stream(finish_reason, calls))

    _, final = STREAMS[surface]()  # `length` returns normally

    assert (final.finish_reason, final.complete, final.text) == (finish_reason, complete, "Hello")


DONE_WITHOUT_FINISH = [chunk({"role": "assistant", "content": "Hel"}), chunk({"content": "lo"})]


@pytest.mark.parametrize("surface", EVENTS)
def test_legacy_stream_done_without_finish_is_final_but_incomplete(monkeypatch, surface):
    _serve(monkeypatch, events=DONE_WITHOUT_FINISH, done=True)

    result = _final(EVENTS[surface]())

    assert (result["text"], result["finish_reason"], result["complete"]) == ("Hello", None, False)
    assert [error["code"] for error in result["errors"]] == [STREAM_INCOMPLETE]


@pytest.mark.parametrize("surface", STREAMS)
def test_message_api_stream_done_without_finish_returns_an_incomplete_completion(monkeypatch, surface):
    _serve(monkeypatch, events=DONE_WITHOUT_FINISH, done=True)

    _, final = STREAMS[surface]()

    assert (final.text, final.finish_reason, final.complete, final.done) == ("Hello", None, False, True)


# ---------------------------------------------------------------------------
# Structured errors (§13.3)
# ---------------------------------------------------------------------------

LIMIT_DETAILS = {**AUDIO_LIMIT, "request_id": "trace-9"}
RAISING = {**RESULTS, **COMPLETIONS, **STREAMS}


@pytest.mark.parametrize("surface", RAISING)
def test_pre_stream_audio_limit_raises_bad_request_with_every_field(monkeypatch, surface):
    recorder = _serve(monkeypatch, error=AUDIO_LIMIT, status=400, headers=TRACE)

    with pytest.raises(BadRequestError) as excinfo:
        RAISING[surface]()

    err = excinfo.value
    assert str(err) == AUDIO_LIMIT["message"]
    assert (err.code, err.error_type, err.param, err.status_code, err.request_id) == (
        "audio_token_limit_exceeded",
        "invalid_request_error",
        None,
        400,
        "trace-9",
    )
    assert err.details == LIMIT_DETAILS
    assert err.partial is None
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("surface", EVENTS)
def test_pre_stream_audio_limit_is_the_single_error_event(monkeypatch, surface):
    _serve(monkeypatch, error=AUDIO_LIMIT, status=400, headers=TRACE)

    events = EVENTS[surface]()

    assert events == [
        {
            "type": "error",
            "message": AUDIO_LIMIT["message"],
            "code": "audio_token_limit_exceeded",
            "error_type": "invalid_request_error",
            "param": None,
            "status": 400,
            "request_id": "trace-9",
            "details": LIMIT_DETAILS,
            "partial": None,
        }
    ]


MID_STREAM_ERRORS = {
    "invalid_request": (
        {"message": "Output failed validation.", "type": "invalid_request_error", "param": None, "code": "bad_output"},
        BadRequestError,
        "bad_output",
    ),
    "server": (
        {"message": "The audio encoder failed.", "type": "server_error", "param": None},
        ServerError,
        "server_error",
    ),
    "rate_limit": (
        {"message": "Slow down.", "type": "rate_limit_error", "code": "rate_limited"},
        RateLimitError,
        "rate_limit",
    ),
}
MID_STREAM_PREFIX = [chunk({"role": "assistant", "content": "Par"}), chunk({"content": "tial"})]


@pytest.mark.parametrize("surface", EVENTS)
@pytest.mark.parametrize("case", MID_STREAM_ERRORS)
@pytest.mark.parametrize("done", [False, True], ids=["eof", "done-after-error"])
def test_mid_stream_error_is_terminal_with_no_final(monkeypatch, surface, case, done):
    error, _, code = MID_STREAM_ERRORS[case]
    _serve(monkeypatch, events=[*MID_STREAM_PREFIX, {"error": error}], done=done, headers=TRACE)

    events = EVENTS[surface]()

    assert [event["type"] for event in events] == ["text.delta", "text.delta", "error"]
    event = events[-1]
    assert (event["code"], event["error_type"], event["param"], event["status"], event["request_id"]) == (
        code,
        error["type"],
        error.get("param"),
        None,
        "trace-9",
    )
    assert event["message"] == error["message"]
    assert event["details"] == {**error, "request_id": "trace-9"}
    assert event["partial"] == {"text": "Partial", "reasoning": None, "tool_calls": None, "finish_reason": None}


@pytest.mark.parametrize("surface", STREAMS)
@pytest.mark.parametrize("case", MID_STREAM_ERRORS)
@pytest.mark.parametrize("done", [False, True], ids=["eof", "done-after-error"])
def test_mid_stream_error_raises_the_mapped_error_with_partial(monkeypatch, surface, case, done):
    error, error_cls, code = MID_STREAM_ERRORS[case]
    _serve(monkeypatch, events=[*MID_STREAM_PREFIX, {"error": error}], done=done, headers=TRACE)

    with pytest.raises(error_cls) as excinfo:
        STREAMS[surface]()

    err = excinfo.value
    assert (err.code, err.error_type, err.request_id) == (code, error["type"], "trace-9")
    assert err.details == {**error, "request_id": "trace-9"}
    assert isinstance(err.partial, ChatCompletion)
    assert (err.partial.text, err.partial.complete) == ("Partial", False)


TRUNCATED = [chunk({"role": "assistant", "content": "Par"}), chunk({"content": "tial"}, finish_reason="stop")]


@pytest.mark.parametrize("surface", EVENTS)
def test_eof_without_done_is_stream_truncated_with_no_final(monkeypatch, surface):
    """Even a finished-looking answer is truncated when ``[DONE]`` never came."""
    _serve(monkeypatch, events=TRUNCATED, done=False, headers=TRACE)

    events = EVENTS[surface]()

    assert [event["type"] for event in events] == ["text.delta", "text.delta", "error"]
    event = events[-1]
    assert (event["code"], event["status"], event["request_id"]) == (STREAM_TRUNCATED, None, "trace-9")
    assert event["details"] == {"request_id": "trace-9"}
    assert event["partial"] == {"text": "Partial", "reasoning": None, "tool_calls": None, "finish_reason": "stop"}


@pytest.mark.parametrize("surface", STREAMS)
def test_eof_without_done_raises_incomplete_stream_error_with_partial(monkeypatch, surface):
    _serve(monkeypatch, events=TRUNCATED, done=False, headers=TRACE)

    with pytest.raises(IncompleteStreamError) as excinfo:
        STREAMS[surface]()

    err = excinfo.value
    assert (err.code, err.request_id) == (STREAM_TRUNCATED, "trace-9")
    assert (err.partial.text, err.partial.finish_reason, err.partial.complete) == ("Partial", "stop", False)
