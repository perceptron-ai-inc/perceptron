"""`Client.generate` / `Client.stream` (and async): request bodies, result metadata, stream events, retired arguments."""

import asyncio
import json

import httpx
import pytest
from _http_mock import (
    Body,
    FailingBody,
    chunk,
    completion,
    install,
    json_response,
    sse_body,
    sse_response,
    text_response,
)
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, image, inspect_task, perceive, text
from perceptron.chat import ToolCall, function_tool
from perceptron.errors import (
    INVALID_RESPONSE,
    INVALID_STREAM_CHUNK,
    MODEL_RENAMED,
    STREAM_INCOMPLETE,
    STREAM_TRUNCATED,
    UNSUPPORTED_TOOL_CHOICE,
    UNSUPPORTED_TOOLS_COMBINATION,
    BadRequestError,
    ServerError,
)

TASK = {"content": [{"type": "text", "role": "user", "content": "Weather in SF and NYC?"}]}
WEATHER = function_tool(
    "get_weather",
    description="Get the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
RETIRED = 'was removed: Focus controls are retired (see "Migrate from Mk1")'


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _json(monkeypatch, payload=None, **kwargs):
    payload = completion() if payload is None else payload
    return install(monkeypatch, lambda request: json_response(payload, **kwargs))


def _sse(monkeypatch, events, *, done=True, headers=None, body=None):
    body = body or Body(sse_body(events, done=done))
    recorder = install(monkeypatch, lambda request: sse_response(None, body=body, headers=headers))
    return recorder, body


def _client():
    return Client(provider="perceptron")


def test_names_importable_from_the_client_module_in_earlier_releases():
    from perceptron import client as client_mod
    from perceptron import errors
    from perceptron.pointing import parser

    for name in ("AuthError", "RateLimitError", "TimeoutError", "TransportError", "INVALID_REASONING_EFFORT"):
        assert getattr(client_mod, name) is getattr(errors, name)
    for name in ("parse_text", "extract_points", "extract_clips"):
        assert getattr(client_mod, name) is getattr(parser, name)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


def test_generate_sends_tool_parameters_and_extra_body(monkeypatch):
    http = _json(monkeypatch)

    _client().generate(
        TASK,
        tools=[WEATHER],
        tool_choice="auto",
        parallel_tool_calls=False,
        temperature=0.5,
        extra_body={"temperature": 0.1, "custom_field": {"x": 1}},
    )

    body = http.last_body
    assert str(http.last.url) == "https://api.perceptron.inc/v1/chat/completions"
    assert body["model"] == "perceptron-mk1.5"
    assert body["tools"] == [WEATHER]
    assert list(body["tools"][0]["function"]["parameters"]["properties"]) == ["city"]
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False
    # extra_body is merged last, unvalidated.
    assert body["temperature"] == 0.1
    assert body["custom_field"] == {"x": 1}
    assert "stream" not in body and "stream_options" not in body


def test_generate_sends_only_what_was_set(monkeypatch):
    http = _json(monkeypatch)

    _client().generate(TASK)

    assert set(http.last_body) == {"model", "messages"}


def test_default_model_on_provider_perceptron_is_mk15(monkeypatch):
    http = _json(monkeypatch)

    Client().generate(TASK, provider="perceptron")

    assert http.last_body["model"] == "perceptron-mk1.5"


def test_reasoning_true_and_reasoning_effort_send_both_the_hint_and_the_field(monkeypatch):
    http = _json(monkeypatch)

    _client().generate(TASK, reasoning=True, reasoning_effort="none")

    body = http.last_body
    assert body["reasoning_effort"] == "none"
    assert body["messages"][0] == {"role": "system", "content": "<hint>THINK</hint>"}
    assert "vision_config" not in body and "reasoning" not in body


def test_text_response_format_is_sent(monkeypatch):
    http = _json(monkeypatch)

    _client().generate(TASK, response_format={"type": "text"})

    assert http.last_body["response_format"] == {"type": "text"}


def test_stream_body_requests_usage_on_perceptron(monkeypatch):
    http, _ = _sse(monkeypatch, [chunk({}, finish_reason="stop")])

    list(_client().stream(TASK, tools=[WEATHER], tool_choice="none", extra_body={"custom_field": 1}))

    body = http.last_body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"] == [WEATHER] and body["tool_choice"] == "none"
    assert body["custom_field"] == 1


def test_stream_caller_stream_options_win_and_fal_gets_none(monkeypatch):
    http, _ = _sse(monkeypatch, [chunk({}, finish_reason="stop")])
    list(_client().stream(TASK, stream_options={"include_usage": False}))
    assert http.last_body["stream_options"] == {"include_usage": False}

    http, _ = _sse(monkeypatch, [chunk({}, finish_reason="stop")])
    list(Client(provider="fal", api_key="fal-key").stream(TASK))
    assert "stream_options" not in http.last_body


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"tools": [WEATHER], "tool_choice": "required"}, UNSUPPORTED_TOOL_CHOICE),
        ({"tools": [WEATHER], "response_format": {"type": "regex", "regex": "a+"}}, UNSUPPORTED_TOOLS_COMBINATION),
    ],
)
def test_tool_parameters_are_validated_before_any_request(monkeypatch, kwargs, code):
    http = _json(monkeypatch)

    with pytest.raises(BadRequestError) as excinfo:
        _client().generate(TASK, **kwargs)
    assert excinfo.value.code == code

    events = list(_client().stream(TASK, **kwargs))
    assert len(events) == 1 and events[0]["type"] == "error" and events[0]["code"] == code
    assert http.requests == []


def test_option_types_are_checked_eagerly(monkeypatch):
    http = _json(monkeypatch)
    with pytest.raises(TypeError, match="extra_body"):
        _client().generate(TASK, extra_body=["x"])
    with pytest.raises(TypeError, match="stream_options"):
        _client().stream(TASK, stream_options=True)  # raised by the call, before iterating
    assert http.requests == []


def test_unknown_client_override_raises():
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
        Client(bogus=1)
    with pytest.raises(TypeError, match="AsyncClient"):
        AsyncClient(bogus=1)


# ---------------------------------------------------------------------------
# Retired and unknown arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["focus", "visual_reasoning"])
def test_retired_arguments_raise_on_every_client_method(name):
    client = _client()
    async_client = AsyncClient(provider="perceptron")

    with pytest.raises(TypeError) as excinfo:
        client.generate(TASK, **{name: True})
    assert str(excinfo.value).startswith(f"generate() got an unexpected keyword argument '{name}'")
    assert f'"{name}" {RETIRED}' in str(excinfo.value)

    with pytest.raises(TypeError, match=f'"{name}" was removed'):
        client.stream(TASK, **{name: True})
    with pytest.raises(TypeError, match=f'"{name}" was removed'):
        async_client.stream(TASK, **{name: True})
    with pytest.raises(TypeError, match=f'"{name}" was removed'):
        asyncio.run(async_client.generate(TASK, **{name: True}))


def test_unknown_arguments_raise_type_error(monkeypatch):
    http = _json(monkeypatch)
    with pytest.raises(TypeError, match=r"^generate\(\) got an unexpected keyword argument 'tool_choise'$"):
        _client().generate(TASK, tool_choise="auto")
    with pytest.raises(TypeError, match=r"^stream\(\) got an unexpected keyword argument 'n'$"):
        _client().stream(TASK, n=2)
    assert http.requests == []


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


def test_generate_result_metadata(monkeypatch):
    payload = completion(None, finish_reason="tool_calls", tool_calls=[CALL], reasoning="Need weather", usage=USAGE)
    _json(monkeypatch, payload, headers={"x-trace-id": "trace-1"})
    task, _ = inspect_task(perceive()(lambda: image(PNG_BYTES) + text("Weather?")))

    result = _client().generate(task)

    assert result["text"] is None
    assert result["reasoning"] == "Need weather"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"] == [ToolCall.from_dict(CALL)]
    assert result["tool_calls"][0].arguments == '{"city": "SF"}'
    assert result["usage"] == USAGE
    assert (result["id"], result["model"], result["request_id"]) == ("chatcmpl-1", "perceptron-mk1.5", "trace-1")
    assert result["complete"] is True
    assert result["asset_count"] == 1
    assert result["errors"] == []
    assert result["raw"] == payload


def test_length_is_incomplete(monkeypatch):
    _json(monkeypatch, completion("Cut", finish_reason="length"))
    result = _client().generate(TASK)
    assert result["finish_reason"] == "length" and result["complete"] is False


def test_malformed_markup_becomes_an_error_entry(monkeypatch):
    _json(monkeypatch, completion("Here <point_box> (1,2) </point_box> it is"))

    result = _client().generate(TASK, expects="box")

    assert result["text"] == "Here <point_box> (1,2) </point_box> it is"
    assert result["boxes"] == []  # the malformed box stays text (C6)
    assert [error["code"] for error in result["errors"]] == ["invalid_box_coords"]


def test_generate_without_choices_or_json_raises_server_error(monkeypatch):
    _json(monkeypatch, {"id": "x", "choices": []})
    with pytest.raises(ServerError) as excinfo:
        _client().generate(TASK)
    assert excinfo.value.code == INVALID_RESPONSE

    install(monkeypatch, lambda request: text_response("<html>oops</html>"))
    with pytest.raises(ServerError):
        _client().generate(TASK)


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------

INTERLEAVED = [
    chunk({"role": "assistant", "reasoning_content": "Two cities."}),
    chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather"}}]}),
    chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather"}}]}),
    chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"city": '}}]}),
    chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "SF"}'}}]}),
    chunk({"tool_calls": [{"index": 1, "function": {"arguments": '"NYC"}'}}]}),
    chunk(None),
    chunk({}, finish_reason="tool_calls"),
    chunk(choices=False, usage=USAGE),
]


def test_stream_interleaved_tool_calls(monkeypatch):
    _, body = _sse(monkeypatch, INTERLEAVED, headers={"x-trace-id": "trace-s"})

    events = list(_client().stream(TASK))

    deltas = [e for e in events if e["type"] == "tool_call.delta"]
    assert deltas == [
        {"type": "tool_call.delta", "index": 0, "id": "call_a", "name": "get_weather", "arguments": ""},
        {"type": "tool_call.delta", "index": 1, "id": "call_b", "name": "get_weather", "arguments": ""},
        {"type": "tool_call.delta", "index": 1, "id": None, "name": None, "arguments": '{"city": '},
        {"type": "tool_call.delta", "index": 0, "id": None, "name": None, "arguments": '{"city": "SF"}'},
        {"type": "tool_call.delta", "index": 1, "id": None, "name": None, "arguments": '"NYC"}'},
    ]
    assert events[0] == {"type": "reasoning.delta", "chunk": "Two cities.", "total_chars": 11}
    assert events[-1]["type"] == "final"
    result = events[-1]["result"]
    assert [(c.id, c.name, c.parse_arguments()) for c in result["tool_calls"]] == [
        ("call_a", "get_weather", {"city": "SF"}),
        ("call_b", "get_weather", {"city": "NYC"}),
    ]
    assert result["finish_reason"] == "tool_calls" and result["complete"] is True
    assert result["usage"] == USAGE  # from the trailing `choices: []` chunk
    assert (result["id"], result["model"], result["request_id"]) == ("chatcmpl-1", "perceptron-mk1.5", "trace-s")
    assert result["text"] is None and result["reasoning"] == "Two cities."
    assert result["raw"] is None and result["errors"] == []
    assert result["asset_count"] == 0
    assert body.closed


def test_stream_error_event_is_terminal(monkeypatch):
    error = {"message": "Generation failed.", "type": "server_error", "param": None, "code": "internal_error"}
    events_in = [chunk({"content": "The answer is"}), {"error": error}, chunk({"content": " ignored"})]
    _sse(monkeypatch, events_in, done=False, headers={"x-trace-id": "trace-e"})

    events = list(_client().stream(TASK))

    assert [e["type"] for e in events] == ["text.delta", "error"]
    assert events[-1] == {
        "type": "error",
        "message": "Generation failed.",
        "code": "internal_error",
        "error_type": "server_error",
        "param": None,
        "status": None,
        "request_id": "trace-e",
        "details": {**error, "request_id": "trace-e"},
        "partial": {"text": "The answer is", "reasoning": None, "tool_calls": None, "finish_reason": None},
    }


def test_stream_eof_without_done_is_truncated(monkeypatch):
    _sse(monkeypatch, [chunk({"content": "Partial"})], done=False)

    events = list(_client().stream(TASK))

    assert [e["type"] for e in events] == ["text.delta", "error"]
    assert events[-1]["code"] == STREAM_TRUNCATED
    assert events[-1]["partial"]["text"] == "Partial"


def test_stream_cut_connection_is_truncated(monkeypatch):
    body = FailingBody(sse_body([chunk({"content": "Par"})], done=False), httpx.ReadError("reset"))
    _sse(monkeypatch, None, body=body)

    events = list(_client().stream(TASK))

    assert events[-1]["type"] == "error" and events[-1]["code"] == STREAM_TRUNCATED
    assert events[-1]["partial"]["text"] == "Par"
    assert body.closed


def test_stream_malformed_json_is_terminal(monkeypatch):
    _sse(monkeypatch, [chunk({"content": "a"}), "data: {not json", chunk({"content": "b"})])

    events = list(_client().stream(TASK))

    assert [e["type"] for e in events] == ["text.delta", "error"]
    assert events[-1]["code"] == INVALID_STREAM_CHUNK
    assert events[-1]["error_type"] is None


def test_stream_null_delta_and_trailing_usage_chunk(monkeypatch):
    events_in = [chunk(None), chunk({"content": "ok"}), chunk(finish_reason="stop", omit_delta=True)]
    _sse(monkeypatch, [*events_in, chunk(choices=False, usage=USAGE)])

    final = list(_client().stream(TASK))[-1]

    assert final["type"] == "final"
    assert final["result"]["text"] == "ok" and final["result"]["usage"] == USAGE
    assert final["result"]["complete"] is True and final["result"]["errors"] == []


def test_stream_done_without_finish_reason_is_incomplete(monkeypatch):
    _sse(monkeypatch, [chunk({"content": "Hello"})])

    events = list(_client().stream(TASK))

    assert events[-1]["type"] == "final"
    result = events[-1]["result"]
    assert result["text"] == "Hello" and result["finish_reason"] is None and result["complete"] is False
    assert [error["code"] for error in result["errors"]] == [STREAM_INCOMPLETE]


def test_stream_length_is_incomplete_without_an_error(monkeypatch):
    _sse(monkeypatch, [chunk({"content": "Cut"}), chunk({}, finish_reason="length")])

    result = list(_client().stream(TASK))[-1]["result"]

    assert result["finish_reason"] == "length" and result["complete"] is False and result["errors"] == []


def test_stream_http_error_keeps_the_full_detail(monkeypatch):
    error = {"message": "Model does not support tool calling", "type": "invalid_request_error"}
    error |= {"param": "tools", "code": "unsupported_parameter"}
    install(monkeypatch, lambda request: json_response({"error": error}, 400, headers={"x-trace-id": "trace-h"}))

    events = list(_client().stream(TASK))

    assert events == [
        {
            "type": "error",
            "message": "Model does not support tool calling",
            "code": "unsupported_parameter",
            "error_type": "invalid_request_error",
            "param": "tools",
            "status": 400,
            "request_id": "trace-h",
            "details": {**error, "request_id": "trace-h"},
            "partial": None,
        }
    ]


def test_stream_pre_request_error_is_a_single_error_event(monkeypatch):
    http, _ = _sse(monkeypatch, [chunk({}, finish_reason="stop")])

    events = list(_client().stream(TASK, model="perceptron-mk1.5-preview"))

    assert len(events) == 1
    assert events[0]["type"] == "error" and events[0]["code"] == MODEL_RENAMED and events[0]["partial"] is None
    assert http.requests == []


def test_stream_points_delta_still_emitted(monkeypatch):
    content = "See <point_box> (1,2) (3,4) </point_box>"
    _sse(monkeypatch, [chunk({"content": content}), chunk({}, finish_reason="stop")])

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    assert [e["type"] for e in events] == ["text.delta", "points.delta", "final"]
    assert events[-1]["result"]["boxes"][0].top_left.x == 1


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


def _collect(async_iterable):
    async def _run():
        return [event async for event in async_iterable]

    return asyncio.run(_run())


def test_async_generate_metadata_and_body(monkeypatch):
    payload = completion(None, finish_reason="tool_calls", tool_calls=[CALL], usage=USAGE)
    http = _json(monkeypatch, payload, headers={"x-trace-id": "trace-a"})

    result = asyncio.run(AsyncClient(provider="perceptron").generate(TASK, tools=[WEATHER], parallel_tool_calls=True))

    assert http.last_body["tools"] == [WEATHER] and http.last_body["parallel_tool_calls"] is True
    assert result["tool_calls"] == [ToolCall.from_dict(CALL)]
    assert result["usage"] == USAGE and result["request_id"] == "trace-a" and result["complete"] is True


def test_async_stream_tool_calls_and_terminal_errors(monkeypatch):
    http, _ = _sse(monkeypatch, INTERLEAVED)
    client = AsyncClient(provider="perceptron")

    events = _collect(client.stream(TASK))

    assert http.last_body["stream_options"] == {"include_usage": True}
    assert sum(e["type"] == "tool_call.delta" for e in events) == 5
    assert events[-1]["result"]["tool_calls"][1].arguments == '{"city": "NYC"}'

    _sse(monkeypatch, [chunk({"content": "x"}), {"error": {"message": "boom", "type": "server_error"}}], done=False)
    events = _collect(client.stream(TASK))
    assert [e["type"] for e in events] == ["text.delta", "error"]
    assert events[-1]["partial"]["text"] == "x"

    _sse(monkeypatch, [chunk({"content": "x"})], done=False)
    assert _collect(client.stream(TASK))[-1]["code"] == STREAM_TRUNCATED

    body = json.dumps({"error": {"message": "slow down", "type": "rate_limit_error"}}).encode()
    install(monkeypatch, lambda request: httpx.Response(429, stream=Body(body), headers={"Retry-After": "3"}))
    events = _collect(client.stream(TASK))
    assert events == [
        {
            "type": "error",
            "message": "slow down",
            "code": "rate_limit",
            "error_type": "rate_limit_error",
            "param": None,
            "status": 429,
            "request_id": None,
            "details": {"message": "slow down", "type": "rate_limit_error", "retry_after": 3.0},
            "partial": None,
        }
    ]


def test_async_stream_pre_request_error_is_a_single_error_event(monkeypatch):
    http, _ = _sse(monkeypatch, [chunk({}, finish_reason="stop")])

    events = _collect(AsyncClient(provider="perceptron").stream(TASK, model="perceptron-mk1.5-preview"))

    assert len(events) == 1
    assert events[0]["type"] == "error" and events[0]["code"] == MODEL_RENAMED and events[0]["partial"] is None
    assert http.requests == []
