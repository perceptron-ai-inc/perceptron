"""Tool calling through the DSL: `agent(tool_calls=...)`, `tool_result(...)`, `perceive(tools=...)`, result metadata,
and the retired/unknown keyword arguments of `perceive` and the helpers."""

import asyncio
import json

import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import (
    async_perceive,
    box,
    caption,
    detect,
    image,
    inspect_task,
    ocr,
    ocr_html,
    ocr_markdown,
    perceive,
    question,
    text,
    video,
)
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron._lowering import count_assets
from perceptron.chat import ChatCompletionMessage, FunctionCall, ToolCall, function_tool
from perceptron.dsl.nodes import Agent, ToolResult, agent, audio, tool_result
from perceptron.highlevel import detect_from_coco

WEATHER = function_tool("get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
CALL_A = {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
CALL_B = ToolCall(id="call_b", function=FunctionCall(name="get_weather", arguments='{"city":"NYC"}'))
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
RETIRED = 'was removed: Focus controls are retired (see "Migrate from Mk1")'


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _messages(*nodes):
    task, _ = inspect_task(perceive()(lambda: nodes[0] if len(nodes) == 1 else sum(nodes[1:], nodes[0])))
    return task, client_mod._task_to_openai_messages(task)


# ---------------------------------------------------------------------------
# Nodes and lowering
# ---------------------------------------------------------------------------


def test_agent_text_is_unchanged():
    node = agent("Answer")
    assert node == Agent("Answer") and node.tool_calls is None and node.reasoning_content is None

    _, messages = _messages(text("Q"), agent("A1"), agent("A2"))

    assert messages == [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A1A2"}]


def test_assistant_turn_and_separate_tool_results():
    task, messages = _messages(
        text("Weather in SF and NYC?"),
        agent(None, tool_calls=[CALL_A, CALL_B], reasoning_content="Two cities."),
        tool_result("call_a", "18C"),
        tool_result("call_b", {"temp": 25, "unit": "C"}),
        text("Summarize."),
    )

    assert [entry["type"] for entry in task["content"]] == [
        "text",
        "assistant_turn",
        "tool_result",
        "tool_result",
        "text",
    ]
    assert messages == [
        {"role": "user", "content": "Weather in SF and NYC?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Two cities.",
            "tool_calls": [CALL_A, CALL_B.to_dict()],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "18C"},
        {"role": "tool", "tool_call_id": "call_b", "content": json.dumps({"temp": 25, "unit": "C"})},
        {"role": "user", "content": "Summarize."},
    ]


def test_image_bearing_tool_result():
    _, messages = _messages(
        text("Crop it."),
        agent(None, tool_calls=[CALL_A]),
        tool_result("call_a", "Cropped:", image(PNG_BYTES)),
    )

    tool = messages[-1]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_a"
    assert tool["content"][0] == {"type": "text", "text": "Cropped:"}
    assert tool["content"][1]["type"] == "image_url"
    assert tool["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert count_assets(messages) == 1


def test_tool_result_content_forms():
    node = tool_result("c1", text("a"), "b", [1, 2])
    assert node == ToolResult("c1", [text("a"), text("b"), text("[1, 2]")])
    _, messages = _messages(agent(None, tool_calls=[CALL_A]), tool_result("call_a"))
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_a", "content": ""}
    # JSON keeps non-ASCII characters as they are (no \uXXXX escapes in the prompt).
    _, messages = _messages(agent(None, tool_calls=[CALL_A]), tool_result("call_a", {"city": "São Paulo"}))
    assert messages[-1]["content"] == '{"city": "São Paulo"}'

    for bad in (video("https://x.com/v.mp4"), audio("https://x.com/a.wav"), 3):
        with pytest.raises(TypeError, match="tool_result"):
            tool_result("c1", bad)
    with pytest.raises(TypeError):
        tool_result("", "x")


def test_agent_tool_calls_must_be_a_list_of_calls():
    with pytest.raises(TypeError, match="must be a list"):
        _messages(agent(None, tool_calls=CALL_B))
    with pytest.raises(TypeError, match="ToolCall objects or tool call dicts"):
        _messages(agent(None, tool_calls=["call_a"]))


def test_agent_needs_content_tool_calls_or_reasoning():
    # The API rejects an assistant message with none of them, so it never reaches a request.
    for node in (agent(None), agent(None, tool_calls=[]), Agent(None)):
        with pytest.raises(TypeError, match=r"^agent\(\) needs content, tool_calls or reasoning_content$"):
            _messages(text("q"), node)

    _, messages = _messages(text("q"), agent(None, reasoning_content="Thinking."))
    assert messages[-1] == {"role": "assistant", "content": None, "reasoning_content": "Thinking."}
    _, messages = _messages(text("q"), agent(""))
    assert messages[-1] == {"role": "assistant", "content": ""}


# ---------------------------------------------------------------------------
# perceive: request body, result metadata, replay
# ---------------------------------------------------------------------------


def test_perceive_sends_tool_parameters(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    perceive(
        text("Weather?"),
        provider="perceptron",
        tools=[WEATHER],
        tool_choice="auto",
        parallel_tool_calls=False,
        extra_body={"custom_field": True},
        allow_multiple=True,
        max_outputs=3,
    )

    body = http.last_body
    assert body["tools"] == [WEATHER] and body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["custom_field"] is True
    assert body["model"] == "perceptron-mk1.5"
    assert "allow_multiple" not in body and "max_outputs" not in body


def test_perceive_default_model_on_configured_perceptron(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    with cfg(provider="perceptron"):
        perceive(text("Hi"))

    assert http.last_body["model"] == "perceptron-mk1.5"


def test_perceive_reasoning_true_and_effort_send_both(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    perceive(text("Hi"), provider="perceptron", reasoning=True, reasoning_effort="high")

    body = http.last_body
    assert body["reasoning_effort"] == "high"
    assert body["messages"][0] == {"role": "system", "content": "<hint>THINK</hint>"}


def test_perceive_result_metadata_and_replay(monkeypatch):
    first = completion(None, finish_reason="tool_calls", tool_calls=[CALL_A], reasoning="Need SF.", usage=USAGE)
    responses = [first, completion("It is 18C in SF.", usage=USAGE)]
    http = install(
        monkeypatch, lambda request: json_response(responses.pop(0), headers={"x-trace-id": f"t{len(responses)}"})
    )

    prompt = image(PNG_BYTES) + text("Weather where this photo was taken?")
    res = perceive(prompt, provider="perceptron", tools=[WEATHER])

    assert res.text is None and res.reasoning == "Need SF."
    assert res.finish_reason == "tool_calls" and res.complete
    assert res.tool_calls == [ToolCall.from_dict(CALL_A)]
    assert res.usage == USAGE
    assert (res.id, res.model, res.request_id) == ("chatcmpl-1", "perceptron-mk1.5", "t1")
    assert res.asset_count == 1
    assert res.message == ChatCompletionMessage(
        role="assistant", content=None, reasoning_content="Need SF.", tool_calls=res.tool_calls
    )
    assert res.errors == []

    replay = res.as_agent()
    assert replay == Agent(None, tool_calls=res.tool_calls, reasoning_content="Need SF.")
    final = perceive(prompt + replay + tool_result("call_a", "18C"), provider="perceptron", tools=[WEATHER])

    messages = http.last_body["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Need SF.",
        "tool_calls": [CALL_A],
    }
    assert messages[2] == {"role": "tool", "tool_call_id": "call_a", "content": "18C"}
    assert final.text == "It is 18C in SF." and final.complete and final.tool_calls is None
    assert final.as_agent() == Agent("It is 18C in SF.")


def test_perceive_result_parse_errors_join_compile_issues(monkeypatch):
    install(monkeypatch, lambda request: json_response(completion("<point_box> (1,2) </point_box>")))

    # An unanchored box in a two-image prompt is a compile issue; the malformed answer adds a parse error after it.
    prompt = image(PNG_BYTES) + image(PNG_BYTES) + box(1, 2, 3, 4) + text("Find it")
    res = perceive(prompt, provider="perceptron", expects="box")

    # Lenient parsing (C6): the malformed box stays text, so the bucket holds no box.
    assert res.text == "<point_box> (1,2) </point_box>" and res.boxes == []
    assert [error["code"] for error in res.errors] == ["anchor_missing", "invalid_box_coords"]
    assert res.complete


def test_perceive_forwards_strict_only_when_true(monkeypatch):
    seen = []

    def _generate(self, task, **kwargs):
        seen.append(kwargs)
        return {"text": "ok"}

    monkeypatch.setattr(client_mod.Client, "generate", _generate)
    perceive(text("a"))
    perceive(text("a"), strict=True)

    assert "strict" not in seen[0] and seen[1]["strict"] is True
    assert all("allow_multiple" not in kwargs and "max_outputs" not in kwargs for kwargs in seen)


def test_a_type_error_from_generate_propagates_after_one_call_and_closes_the_client(monkeypatch):
    calls, closed = [], []

    def _generate(self, task, **kwargs):
        calls.append(task)
        raise TypeError("boom")

    monkeypatch.setattr(client_mod.Client, "generate", _generate)
    monkeypatch.setattr(client_mod.Client, "close", lambda self: closed.append(self))

    with pytest.raises(TypeError, match="boom"):
        perceive(text("a"))
    assert len(calls) == 1
    assert len(closed) == 1


def test_async_perceive_sends_tools_and_returns_metadata(monkeypatch):
    payload = completion(None, finish_reason="tool_calls", tool_calls=[CALL_A], usage=USAGE)
    http = install(monkeypatch, lambda request: json_response(payload))

    @async_perceive(provider="perceptron", tools=[WEATHER], tool_choice="auto")
    def ask():
        return text("Weather in SF?")

    res = asyncio.run(ask())

    assert http.last_body["tools"] == [WEATHER] and http.last_body["tool_choice"] == "auto"
    assert res.tool_calls == [ToolCall.from_dict(CALL_A)] and res.usage == USAGE and res.complete


# ---------------------------------------------------------------------------
# Retired and unknown keyword arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["focus", "visual_reasoning"])
def test_perceive_rejects_retired_arguments_at_decoration_time(name):
    for call in (lambda: perceive(**{name: True}), lambda: perceive(text("x"), **{name: True})):
        with pytest.raises(TypeError) as excinfo:
            call()
        assert str(excinfo.value).startswith(f"perceive() got an unexpected keyword argument '{name}'")
        assert f'"{name}" {RETIRED}' in str(excinfo.value)

    with pytest.raises(TypeError, match=rf"^async_perceive\(\) got an unexpected keyword argument '{name}'"):
        async_perceive(**{name: True})


def test_perceive_rejects_unknown_arguments():
    with pytest.raises(TypeError, match=r"^perceive\(\) got an unexpected keyword argument 'seed'$"):
        perceive(seed=7)


HELPERS = {
    "caption": lambda **kw: caption(image(PNG_BYTES), **kw),
    "question": lambda **kw: question(image(PNG_BYTES), "What?", **kw),
    "ocr": lambda **kw: ocr(image(PNG_BYTES), **kw),
    "ocr_markdown": lambda **kw: ocr_markdown(image(PNG_BYTES), **kw),
    "ocr_html": lambda **kw: ocr_html(image(PNG_BYTES), **kw),
    "detect": lambda **kw: detect(image(PNG_BYTES), **kw),
    "detect_from_coco": lambda **kw: detect_from_coco("/nonexistent", **kw),
}


@pytest.mark.parametrize("helper", sorted(HELPERS))
@pytest.mark.parametrize("name", ["focus", "visual_reasoning"])
def test_helpers_name_themselves_for_retired_arguments(helper, name):
    with pytest.raises(TypeError) as excinfo:
        HELPERS[helper](**{name: True})
    assert str(excinfo.value).startswith(f"{helper}() got an unexpected keyword argument '{name}'")
    assert RETIRED in str(excinfo.value)


def test_detect_from_coco_accepts_detect_options():
    # strict/max_outputs/response_format pass the keyword check (and reach detect()).
    with pytest.raises(FileNotFoundError):
        detect_from_coco("/nonexistent", strict=True, max_outputs=3, response_format={"type": "text"})


@pytest.mark.parametrize("helper", sorted(HELPERS))
def test_helpers_name_themselves_for_unknown_arguments(helper):
    with pytest.raises(TypeError, match=rf"^{helper}\(\) got an unexpected keyword argument 'seed'$"):
        HELPERS[helper](seed=7)


def test_helpers_forward_tool_parameters(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    question(image(PNG_BYTES), "What?", provider="perceptron", tools=[WEATHER], extra_body={"custom_field": 1})

    assert http.last_body["tools"] == [WEATHER] and http.last_body["custom_field"] == 1
