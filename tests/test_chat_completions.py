"""`client.chat.completions.create()` (non-streaming): request bodies, validation, results, delegators."""

import asyncio
import json
import sys
import warnings
from types import SimpleNamespace

import pytest
from _http_mock import completion, install, json_response, text_response
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, agent, audio, image, settings, system, text, tool_result, video
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    FunctionCall,
    ToolCall,
    function_tool,
)
from perceptron.client import regex_format
from perceptron.errors import (
    CONFLICTING_STRUCTURED_OUTPUT_CONTROLS,
    CONFLICTING_TOOLS,
    DUPLICATE_TOOL_NAME,
    INVALID_PARAMETER,
    INVALID_REASONING_EFFORT,
    INVALID_RESPONSE,
    INVALID_TEMPERATURE,
    INVALID_TOOL_ARGUMENTS,
    INVALID_TOOLS,
    INVALID_VISION_CONFIG,
    MODEL_RENAMED,
    RESERVED_TOOL_NAME,
    UNSUPPORTED_PARAMETER,
    UNSUPPORTED_RESPONSE_FORMAT,
    UNSUPPORTED_TOOL_CHOICE,
    UNSUPPORTED_TOOL_TYPE,
    UNSUPPORTED_TOOLS_COMBINATION,
    BadRequestError,
    ParseError,
    ServerError,
)

USER = {"role": "user", "content": "Hello"}
WEATHER = function_tool(
    "get_weather",
    description="Get the weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}},
)
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


@pytest.fixture
def http(monkeypatch):
    return install(monkeypatch, lambda request: json_response(completion(), headers={"x-trace-id": "trace-1"}))


def _create(**kwargs):
    kwargs.setdefault("messages", [USER])
    return Client().chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Provider and body
# ---------------------------------------------------------------------------


def test_api_key_only_env_hits_the_perceptron_api(http):
    assert settings().provider == "fal"  # the legacy auto-detect is unchanged

    _create()

    request = http.last
    assert str(request.url) == "https://api.perceptron.inc/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"] == "application/json"


def test_minimal_body_sends_only_what_was_set(http):
    _create()
    assert http.last_body == {"model": "perceptron-mk1.5", "messages": [USER]}


def test_explicit_fal_provider_is_honored(http):
    Client(provider="fal").chat.completions.create(messages=[USER])

    assert str(http.last.url) == "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
    assert http.last.headers["authorization"] == "Key sk-test"
    assert http.last_body["model"] == "isaac-0.1"


def test_client_built_inside_config_keeps_its_provider_and_key(http):
    with cfg(provider="fal", api_key="fal-key"):
        client = Client()
    client.chat.completions.create(messages=[USER])

    # After the block the client still pairs its key with its provider: fal, not the Perceptron API.
    assert str(http.last.url) == "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"
    assert http.last.headers["authorization"] == "Key fal-key"
    assert http.last_body["model"] == "isaac-0.1"


def test_fal_rejects_perceptron_models(http):
    with pytest.raises(BadRequestError, match="PERCEPTRON_PROVIDER=perceptron"):
        Client(provider="fal").chat.completions.create(messages=[USER], model="perceptron-mk1.5")
    assert not http.requests


def test_every_parameter_is_sent_when_set(http):
    _create(
        model="perceptron-mk1",
        max_completion_tokens=256,
        temperature=0,
        top_p=0.9,
        top_k=20,
        frequency_penalty=0.0,
        presence_penalty=1.5,
        response_format={"type": "text"},
        reasoning_effort="High",
        vision_config={"annotation_format": "BOX", "enable_audio_in_video": False},
        tools=[WEATHER],
        tool_choice="none",
        parallel_tool_calls=False,
        n=1,
    )

    assert http.last_body == {
        "model": "perceptron-mk1",
        "messages": [USER],
        "max_completion_tokens": 256,
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 20,
        "frequency_penalty": 0.0,
        "presence_penalty": 1.5,
        "response_format": {"type": "text"},
        "reasoning_effort": "high",
        "vision_config": {"annotation_format": "box", "enable_audio_in_video": False},
        "tools": [WEATHER],
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "n": 1,
    }


def test_tool_parameter_key_order_is_preserved(http):
    schema = {"type": "object", "properties": {"zeta": {"type": "string"}, "alpha": {"type": "string"}}}
    _create(tools=[function_tool("lookup", parameters=schema)])

    sent = http.last.content.decode()
    assert sent.index('"zeta"') < sent.index('"alpha"')


def test_configure_generation_defaults_count_as_set(http):
    with cfg(model="perceptron-mk1", temperature=0.2, max_tokens=64, top_k=5):
        _create()

    body = http.last_body
    assert (body["model"], body["temperature"], body["max_completion_tokens"], body["top_k"]) == (
        "perceptron-mk1",
        0.2,
        64,
        5,
    )


def test_max_tokens_is_an_alias(http):
    _create(max_tokens=100)
    assert http.last_body["max_completion_tokens"] == 100
    assert "max_tokens" not in http.last_body

    with pytest.raises(TypeError, match="not both"):
        _create(max_tokens=100, max_completion_tokens=100)


def test_extra_body_is_merged_last(http):
    _create(temperature=0.5, extra_body={"temperature": 0.1, "custom_field": {"x": 1}})

    assert http.last_body["temperature"] == 0.1
    assert http.last_body["custom_field"] == {"x": 1}


def test_regex_response_format_is_lowered_to_top_level_regex(http):
    _create(response_format=regex_format("yes|no"))

    assert http.last_body["regex"] == "yes|no"
    assert "response_format" not in http.last_body


def test_stream_options_is_not_sent_without_streaming(http):
    _create()
    assert "stream_options" not in http.last_body
    assert "stream" not in http.last_body


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_messages_are_sent_verbatim_and_never_merged(http):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": [{"type": "text", "text": "Use tools."}]},
        {"role": "user", "content": "Weather in SF?", "name": "alice"},
        {"role": "user", "content": "And NYC?"},
        {"role": "assistant", "content": None, "reasoning_content": "Two lookups.", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [{"type": "text", "text": "radar"}, {"type": "image_url", "image_url": {"url": "https://x"}}],
        },
        {"role": "assistant", "content": "It is 18C.", "custom_key": [1, 2]},
    ]

    _create(messages=messages, tools=[WEATHER])

    assert http.last_body["messages"] == messages
    raw = json.loads(http.last.content)["messages"][4]
    assert raw["content"] is None
    assert list(raw) == ["role", "content", "reasoning_content", "tool_calls"]


def test_returned_messages_and_completions_replay(http):
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        reasoning_content="r",
        tool_calls=[ToolCall(id="call_1", function=FunctionCall("get_weather", '{"city":"SF"}'))],
    )
    previous = ChatCompletion.from_dict(completion("Earlier answer"))
    perceive_result = SimpleNamespace(message=ChatCompletionMessage(role="assistant", content="From perceive"))

    _create(
        messages=[
            USER,
            message,
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            previous,
            perceive_result,
        ]
    )

    sent = http.last_body["messages"]
    assert sent[1] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "r",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}
        ],
    }
    assert sent[3] == {"role": "assistant", "content": "Earlier answer"}
    assert sent[4] == {"role": "assistant", "content": "From perceive"}


@pytest.mark.parametrize("bad", ["hello", {"role": "user", "content": "x"}, None])
def test_messages_must_be_a_list(http, bad):
    with pytest.raises(TypeError, match="messages must be a list"):
        _create(messages=bad)


def test_unsupported_message_item_raises(http):
    with pytest.raises(TypeError, match=r"messages\[1\] must be"):
        _create(messages=[USER, 42])
    assert not http.requests


def test_dsl_nodes_and_strings_in_content_are_lowered(http):
    content = [
        text("Compare these."),
        image(PNG_BYTES),
        "plain string",
        {"type": "text", "text": "a dict part"},
        video("https://example.com/clip.mp4"),
        audio("https://example.com/clip.wav"),
        image("https://example.com/a.png") + text("inline sequence"),
    ]

    completion_obj = _create(messages=[{"role": "user", "content": content, "extra": True}])

    sent = http.last_body["messages"][0]
    assert sent["extra"] is True
    parts = sent["content"]
    assert [p["type"] for p in parts] == [
        "text",
        "image_url",
        "text",
        "text",
        "video_url",
        "audio_url",
        "image_url",
        "text",
    ]
    assert parts[0] == {"type": "text", "text": "Compare these."}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[2] == {"type": "text", "text": "plain string"}
    assert parts[4] == {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
    assert completion_obj.asset_count == 4


def test_unsupported_content_item_raises(http):
    with pytest.raises(TypeError, match=r"messages\[0\]\.content\[1\]"):
        _create(messages=[{"role": "user", "content": ["ok", 3.5]}])


@pytest.mark.parametrize(
    "node",
    [
        system("You are terse."),
        agent("I said no."),
        system("S") + text("Hi"),
        agent(None, tool_calls=[CALL]),
        tool_result("call_1", "18C"),
    ],
    ids=["system", "agent", "sequence", "agent-turn", "tool-result"],
)
def test_turn_nodes_in_content_raise_instead_of_losing_their_role(http, node):
    # Lowered as parts, their text would be sent as the enclosing user message's.
    with pytest.raises(TypeError, match=r"messages\[0\]\.content\[1\]: system\(\), agent\(\) and tool_result\(\)"):
        _create(messages=[{"role": "user", "content": ["Hi", node]}])
    assert not http.requests


def test_tool_call_objects_in_a_hand_built_assistant_message_are_sent_as_dicts(http):
    # e.g. {"role": "assistant", "content": None, "tool_calls": completion.tool_calls}, as agent(tool_calls=...) takes.
    calls = [ToolCall(id="call_1", function=FunctionCall("get_weather", '{"city": "SF"}')), dict(CALL, id="call_2")]
    replay = {"role": "assistant", "content": None, "tool_calls": calls}

    _create(messages=[USER, replay, {"role": "tool", "tool_call_id": "call_1", "content": "18C"}])

    assert http.last_body["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [CALL, dict(CALL, id="call_2")],
    }
    assert replay["tool_calls"] is calls  # the caller's message is not modified


@pytest.mark.parametrize("content", [image(PNG_BYTES), image(PNG_BYTES) + text("What?")], ids=["node", "sequence"])
def test_dsl_node_content_outside_a_list_raises_before_any_request(http, content):
    with pytest.raises(TypeError, match=r"messages\[0\]\.content must be a str or a list"):
        _create(messages=[{"role": "user", "content": content}])
    assert not http.requests


# ---------------------------------------------------------------------------
# Validation before any request
# ---------------------------------------------------------------------------


def _bad_request(http, **kwargs):
    with pytest.raises(BadRequestError) as excinfo:
        _create(**kwargs)
    assert not http.requests
    return excinfo.value


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"n": 2}, INVALID_PARAMETER),
        ({"n": True}, INVALID_PARAMETER),
        ({"n": 1.0}, INVALID_PARAMETER),
        ({"max_completion_tokens": 0}, INVALID_PARAMETER),
        ({"tool_choice": "required", "tools": [WEATHER]}, UNSUPPORTED_TOOL_CHOICE),
        ({"tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, UNSUPPORTED_TOOL_CHOICE),
        ({"parallel_tool_calls": "yes"}, INVALID_PARAMETER),
        ({"tools": [WEATHER], "regex": "yes|no"}, UNSUPPORTED_TOOLS_COMBINATION),
        (
            {
                "tools": [WEATHER],
                "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {}}},
            },
            UNSUPPORTED_TOOLS_COMBINATION,
        ),
        ({"regex": "a+", "response_format": {"type": "text"}}, CONFLICTING_STRUCTURED_OUTPUT_CONTROLS),
        ({"regex": "a+", "response_format": regex_format("b+")}, CONFLICTING_STRUCTURED_OUTPUT_CONTROLS),
        ({"response_format": {"type": "json_object"}}, UNSUPPORTED_RESPONSE_FORMAT),
        ({"reasoning_effort": "extreme"}, INVALID_REASONING_EFFORT),
        ({"vision_config": {"annotation_format": "track"}}, INVALID_VISION_CONFIG),
        ({"vision_config": {"enable_audio_in_video": "yes"}}, INVALID_VISION_CONFIG),
        ({"vision_config": {"internal_tools": ["FOCUS"]}}, UNSUPPORTED_PARAMETER),
        ({"vision_config": {"zoom": 2}}, UNSUPPORTED_PARAMETER),
        ({"model": "perceptron-mk1.5-preview"}, MODEL_RENAMED),
    ],
)
def test_invalid_values_raise_with_codes(http, kwargs, code):
    assert _bad_request(http, **kwargs).code == code


_BAD_TEMPERATURES = ["0.5", True, float("nan"), float("inf"), float("-inf"), [0.5], -0.5, -1]


@pytest.mark.parametrize("temperature", _BAD_TEMPERATURES)
def test_temperature_must_be_a_finite_number_at_least_zero(http, temperature):
    error = _bad_request(http, temperature=temperature)
    assert (error.code, error.param) == (INVALID_TEMPERATURE, "temperature")  # the API's code for it

    with cfg(temperature=temperature), pytest.raises(BadRequestError) as excinfo:
        _create()  # a configured default is checked like an argument
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_TEMPERATURE, "temperature")
    assert not http.requests

    _create(temperature=1)  # ints are numbers
    assert http.last_body["temperature"] == 1
    _create(temperature=0)  # no upper bound, and 0 is allowed
    assert http.last_body["temperature"] == 0


@pytest.mark.parametrize("temperature", _BAD_TEMPERATURES)
def test_legacy_generate_and_stream_check_temperature_the_same_way(http, temperature):
    task = {"content": [{"type": "text", "role": "user", "content": "Hello"}]}
    client = Client(provider="perceptron")

    with pytest.raises(BadRequestError) as excinfo:
        client.generate(task, temperature=temperature)
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_TEMPERATURE, "temperature")
    events = list(client.stream(task, temperature=temperature))
    assert [(event["type"], event["code"]) for event in events] == [("error", INVALID_TEMPERATURE)]
    with cfg(temperature=temperature), pytest.raises(BadRequestError):
        Client(provider="perceptron").generate(task)
    assert not http.requests

    client.generate(task, temperature=2.5)
    assert http.last_body["temperature"] == 2.5


@pytest.mark.parametrize(
    ("names", "code"),
    [
        (["get weather", "get_weather"], DUPLICATE_TOOL_NAME),
        (["get_weather", "functions.get_weather"], DUPLICATE_TOOL_NAME),
        (["vision.crop", "vision.crop"], DUPLICATE_TOOL_NAME),
        (["parallel"], RESERVED_TOOL_NAME),
        (["functions.parallel"], RESERVED_TOOL_NAME),
        (["  "], INVALID_TOOLS),
    ],
)
def test_tool_names_follow_the_recipient_rule(http, names, code):
    tools = [function_tool(name) for name in names]
    err = _bad_request(http, tools=tools)
    assert err.code == code
    assert err.param.startswith("tools[")


def test_distinct_recipients_are_allowed(http):
    _create(tools=[function_tool("vision.crop"), function_tool("crop"), function_tool("get-weather")])
    assert len(http.last_body["tools"]) == 3


def test_tool_shape_errors(http):
    assert _bad_request(http, tools=[{"type": "custom", "custom": {}}]).code == UNSUPPORTED_TOOL_TYPE
    assert _bad_request(http, tools=[{"type": "function"}]).code == INVALID_TOOLS
    err = _bad_request(http, tools=[{"type": "function", "function": {"name": "f", "parameters": ["x"]}}])
    assert err.code == INVALID_TOOLS
    err = _bad_request(http, tools=[{"type": "perceptron.FOCUS"}, WEATHER])
    assert err.code == CONFLICTING_TOOLS
    with pytest.raises(TypeError):
        _create(tools=WEATHER)


def test_enable_thinking_is_deprecated_but_sent(http):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _create(vision_config={"enable_thinking": True})

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert caught[0].filename == __file__
    assert http.last_body["vision_config"] == {"enable_thinking": True}


@pytest.mark.parametrize(
    "name", ["stop", "seed", "logprobs", "user", "metadata", "store", "functions", "function_call"]
)
def test_unsupported_openai_parameters_raise_type_error(http, name):
    with pytest.raises(TypeError, match="not supported by the Perceptron API"):
        _create(**{name: "x"})
    assert not http.requests


def test_unknown_keyword_raises_type_error(http):
    with pytest.raises(TypeError, match="unexpected keyword argument 'tool_choise'"):
        _create(tool_choise="auto")


def test_extra_body_is_not_validated(http):
    _create(extra_body={"tool_choice": "required", "seed": 3})
    assert http.last_body["tool_choice"] == "required"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_completion_result(monkeypatch):
    payload = completion(
        None,
        finish_reason="tool_calls",
        tool_calls=[CALL],
        reasoning="Need the weather.",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"audio_tokens": 0},
        },
    )
    install(monkeypatch, lambda request: json_response(payload, headers={"x-trace-id": "trace-9"}))

    result = _create(tools=[WEATHER])

    assert result.request_id == "trace-9"
    assert result.raw == payload
    assert (result.id, result.model, result.created) == ("chatcmpl-1", "perceptron-mk1.5", 1790000000)
    assert result.text is None
    assert result.reasoning == "Need the weather."
    assert result.finish_reason == "tool_calls"
    assert result.complete is True
    assert result.usage.total_tokens == 15
    assert result.usage.prompt_tokens_details == {"audio_tokens": 0}
    assert result.asset_count == 0
    (call,) = result.tool_calls
    assert (call.id, call.name, call.arguments) == ("call_1", "get_weather", '{"city": "SF"}')
    assert call.parse_arguments() == {"city": "SF"}
    assert result.message.to_dict() == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need the weather.",
        "tool_calls": [CALL],
    }
    assert result.to_dict()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("finish_reason", "tool_calls", "complete"),
    [
        ("stop", None, True),
        ("interrupted", None, True),
        ("tool_calls", [CALL], True),
        ("length", None, False),
        ("length", [CALL], False),
        ("tool_calls", None, False),
        ("stop", [CALL], False),
        (None, None, False),
    ],
)
def test_complete_rule(finish_reason, tool_calls, complete):
    result = ChatCompletion.from_dict(completion("x", finish_reason=finish_reason, tool_calls=tool_calls))
    assert result.complete is complete


def test_invalid_tool_arguments_raise_parse_error():
    call = ToolCall(id="call_1", function=FunctionCall("f", '{"city": "S'))
    with pytest.raises(ParseError) as excinfo:
        call.parse_arguments()
    assert excinfo.value.code == INVALID_TOOL_ARGUMENTS


def test_parsing_is_lenient():
    result = ChatCompletion.from_dict(
        {
            "choices": [
                {"message": {"content": "hi", "unknown": 1, "tool_calls": [{"function": {"arguments": {"a": 1}}}]}}
            ]
        }
    )
    assert result.text == "hi"
    assert result.message.role == "assistant"
    assert result.tool_calls[0].arguments == '{"a": 1}'
    assert result.usage is None


@pytest.mark.parametrize("payload", [{"choices": []}, {"object": "chat.completion"}, ["not", "a", "dict"]])
def test_empty_choices_are_a_server_error(monkeypatch, payload):
    install(monkeypatch, lambda request: json_response(payload, headers={"x-trace-id": "t"}))
    with pytest.raises(ServerError) as excinfo:
        _create()
    assert excinfo.value.code == INVALID_RESPONSE
    assert excinfo.value.request_id == "t"


def test_non_json_200_is_a_server_error(monkeypatch):
    install(monkeypatch, lambda request: text_response("<html>ok</html>"))
    with pytest.raises(ServerError) as excinfo:
        _create()
    assert excinfo.value.code == INVALID_RESPONSE


def test_http_errors_map_with_request_id(monkeypatch):
    error = {"message": "Model 'perceptron-mk1' does not support tool calling", "type": "invalid_request_error"}
    error.update(param="tools", code="unsupported_parameter")
    install(monkeypatch, lambda request: json_response({"error": error}, 400, headers={"x-trace-id": "t-400"}))

    with pytest.raises(BadRequestError) as excinfo:
        _create(tools=[WEATHER])

    err = excinfo.value
    assert (err.code, err.param, err.status_code, err.request_id) == ("unsupported_parameter", "tools", 400, "t-400")
    assert err.details["request_id"] == "t-400"


def test_function_tool_shape():
    assert function_tool("f") == {"type": "function", "function": {"name": "f"}}
    tool = function_tool("f", description="d", parameters={"b": 1, "a": 2}, strict=True)
    assert list(tool["function"]) == ["name", "description", "parameters", "strict"]
    assert list(tool["function"]["parameters"]) == ["b", "a"]


# ---------------------------------------------------------------------------
# Resources and delegators
# ---------------------------------------------------------------------------


def test_chat_resource_is_cached():
    client = Client()
    assert client.chat is client.chat
    assert client.chat.completions is client.chat.completions


def test_multilook_forwards_the_full_signature(monkeypatch):
    calls = {}

    def create(client, **params):
        calls["sync"] = (client, params)
        return "sync-result"

    async def acreate(client, **params):
        calls["async"] = (client, params)
        return "async-result"

    monkeypatch.setitem(sys.modules, "perceptron.multilook", SimpleNamespace(create=create, acreate=acreate))
    client = Client()

    assert client.chat.completions.multilook(context=[USER], prompts=["a", "b"], n=2, temperature=0.7) == "sync-result"
    forwarded_client, params = calls["sync"]
    assert forwarded_client is client
    assert params == {
        "context": [USER],
        "prompts": ["a", "b"],
        "model": None,
        "n": 2,
        "max_completion_tokens": None,
        "max_tokens": None,
        "temperature": 0.7,
        "top_p": None,
        "top_k": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "reasoning_effort": None,
        "vision_config": None,
        "extra_body": None,
        "timeout": None,
    }
    with pytest.raises(TypeError):
        client.chat.completions.multilook(context=[USER], prompts=["a"], tools=[WEATHER])

    async_client = AsyncClient()
    assert asyncio.run(async_client.chat.completions.multilook(context=[USER], prompts=["a"])) == "async-result"
    assert calls["async"][0] is async_client


def test_files_and_models_are_lazy_cached_delegators(monkeypatch):
    class _Resource:
        def __init__(self, client):
            self.client = client

    fake_files = SimpleNamespace(Files=type("Files", (_Resource,), {}), AsyncFiles=type("AsyncFiles", (_Resource,), {}))
    fake_models = SimpleNamespace(
        Models=type("Models", (_Resource,), {}), AsyncModels=type("AsyncModels", (_Resource,), {})
    )
    monkeypatch.setitem(sys.modules, "perceptron.files", fake_files)
    monkeypatch.setitem(sys.modules, "perceptron.models", fake_models)

    client = Client()
    assert type(client.files).__name__ == "Files" and client.files.client is client
    assert client.files is client.files
    assert type(client.models).__name__ == "Models" and client.models.client is client

    async_client = AsyncClient()
    assert type(async_client.files).__name__ == "AsyncFiles"
    assert type(async_client.models).__name__ == "AsyncModels"


def test_session_factories_use_the_patchable_module_globals(monkeypatch):
    marker = object()
    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: (marker, timeout))
    monkeypatch.setattr(client_mod, "httpx", SimpleNamespace(AsyncClient=lambda timeout: ("async", timeout)))

    client = Client()
    assert client._sync_session(12.0) == (marker, 12.0)
    assert client._async_session(3.0) == ("async", 3.0)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_create_sends_json_content(http):
    async def _run():
        return await AsyncClient().chat.completions.create(messages=[USER], temperature=0)

    result = asyncio.run(_run())

    assert result.text == "Hello"
    assert result.request_id == "trace-1"
    assert str(http.last.url) == "https://api.perceptron.inc/v1/chat/completions"
    assert http.last_body == {"model": "perceptron-mk1.5", "messages": [USER], "temperature": 0}


def test_async_create_validates_before_sending(http):
    async def _run():
        await AsyncClient().chat.completions.create(messages=[USER], n=3)

    with pytest.raises(BadRequestError):
        asyncio.run(_run())
    assert not http.requests


def test_async_create_checks_temperature_before_sending(http):
    async def _run():
        await AsyncClient().chat.completions.create(messages=[USER], temperature=float("nan"))

    with pytest.raises(BadRequestError) as excinfo:
        asyncio.run(_run())
    assert (excinfo.value.code, excinfo.value.param) == (INVALID_TEMPERATURE, "temperature")
    assert not http.requests


def test_async_errors_are_mapped_with_the_stub_httpx(monkeypatch):
    """With the stub namespace (both exception classes are `Exception`), SDK errors must not become TransportErrors."""

    class _Response:
        status_code = 429
        headers = {"Retry-After": "7", "x-trace-id": "t-async"}  # noqa: RUF012

        def json(self):
            return {"error": {"message": "slow down", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}

    class _Session:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, content=None):
            json.loads(content)
            return _Response()

    stub = SimpleNamespace(AsyncClient=_Session, TimeoutException=Exception, HTTPError=Exception)
    monkeypatch.setattr(client_mod, "httpx", stub)

    async def _run():
        await AsyncClient().chat.completions.create(messages=[USER])

    from perceptron.errors import RateLimitError

    with pytest.raises(RateLimitError) as excinfo:
        asyncio.run(_run())
    assert excinfo.value.retry_after == 7.0
    assert excinfo.value.request_id == "t-async"
