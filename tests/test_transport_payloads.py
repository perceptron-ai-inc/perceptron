"""Chat request bodies built by `perceive`: roles, `<hint>` system messages, reasoning and generation fields.

Requests go through `httpx.MockTransport` (see `_http_mock`).
"""

from __future__ import annotations

import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import agent, box, image, inspect_task, perceive, text
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.errors import INVALID_REASONING_EFFORT, BadRequestError
from perceptron.pointing.parser import PointParser
from perceptron.pointing.types import SinglePoint, bbox

BASE_URL = "https://mock.api"
PERCEPTRON_CHAT_URL = f"{BASE_URL}/chat/completions"
FAL_CHAT_URL = f"{BASE_URL}/perceptron/isaac-01/openai/v1/chat/completions"


def _serve(monkeypatch, content: str = "Answer", *, reasoning: str | None = None, key_env: str = "PERCEPTRON_API_KEY"):
    """Set ``key_env`` to ``test-key`` and answer every chat request with one completion; returns the recorder."""
    monkeypatch.setenv(key_env, "test-key")
    return install(monkeypatch, lambda request: json_response(completion(content, reasoning=reasoning)))


def _assert_chat_request(request, *, url: str, authorization: str) -> None:
    assert request.method == "POST"
    assert str(request.url) == url
    assert request.headers["authorization"] == authorization
    assert request.headers["content-type"] == "application/json"


@perceive()
def _icl_prompt():
    example = image(PNG_BYTES)
    target = image(PNG_BYTES)
    example_tag = PointParser.serialize(SinglePoint(4, 5))
    return (
        example
        + text("Example prompt")
        + agent(example_tag)
        + target
        + text("Now annotate the region of interest.")
        + box(1, 2, 3, 4, image=target)
    )


def test_task_roles_and_message_conversion():
    task, issues = inspect_task(_icl_prompt)
    assert issues == []
    # Verify roles in compiled task
    roles = [item.get("role") for item in task.get("content", []) if item.get("type") == "text"]
    assert roles.count("assistant") == 1
    assert roles.count("user") >= 2

    messages = client_mod._task_to_openai_messages(task)
    # No agent role should leak into payload
    assert all(msg["role"] != "agent" for msg in messages)

    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) >= 2
    first_user = user_messages[0]
    assert isinstance(first_user["content"], list)
    types = {part["type"] for part in first_user["content"]}
    assert "image_url" in types
    assert "text" in types
    # Ensure list content uses typed parts only
    assert all(isinstance(part, dict) for part in first_user["content"])

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    assert assistant_messages
    assert isinstance(assistant_messages[0]["content"], str)


def test_fal_payload_structure(monkeypatch):
    recorder = _serve(monkeypatch, "Affirmative <point_box> (1,2) (3,4) </point_box>", key_env="FAL_KEY")

    @perceive(expects="box")
    def make_request(img):
        im1 = image(img)
        im2 = image(img)
        demo_tag = PointParser.serialize(bbox(10, 12, 20, 24, mention="target"))
        return im1 + im2 + text("Locate the object.") + box(1, 1, 4, 4, image=im2) + agent(demo_tag)

    with cfg(provider="fal", base_url=BASE_URL):
        res = make_request(PNG_BYTES)

    assert len(recorder.requests) == 1
    _assert_chat_request(recorder.last, url=FAL_CHAT_URL, authorization="Key test-key")
    payload = recorder.last_body
    assert payload["model"]
    messages = payload["messages"]
    assert any(msg["role"] == "assistant" for msg in messages)
    assert all(msg["role"] != "agent" for msg in messages)

    user_messages = [m for m in messages if m["role"] == "user"]
    assert user_messages
    multimodal = next(m for m in user_messages if isinstance(m["content"], list))
    parts = multimodal["content"]
    assert all(isinstance(part, dict) for part in parts)
    assert sum(1 for part in parts if part.get("type") == "image_url") >= 2
    assert any(part.get("type") == "text" for part in parts)

    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant and isinstance(assistant[0]["content"], str)

    # Perceive result should surface parsed boxes from response text
    assert res.boxes and res.boxes[0].top_left.x == 1


def test_image_url_passthrough():
    @perceive()
    def fn():
        return image("https://example.com/sample.png")

    task, issues = inspect_task(fn)
    assert issues == []
    assert task and isinstance(task, dict)
    content = task.get("content", [])
    assert content and content[0].get("content") == "https://example.com/sample.png"

    messages = client_mod._task_to_openai_messages(task)
    assert messages and messages[0]["role"] == "user"
    parts = messages[0]["content"]
    assert isinstance(parts, list)
    img_parts = [p for p in parts if isinstance(p, dict) and p.get("type") == "image_url"]
    assert img_parts and img_parts[0]["image_url"]["url"] == "https://example.com/sample.png"


def test_reasoning_hint_enables_reasoning_payload(monkeypatch):
    recorder = _serve(monkeypatch, "Thoughts then answer", reasoning="Because...")

    @perceive(expects="think", model="isaac-0.2-2b-preview", provider="perceptron")
    def make_request():
        return text("Why did the robot stop working?")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    _assert_chat_request(recorder.last, url=PERCEPTRON_CHAT_URL, authorization="Bearer test-key")
    payload = recorder.last_body
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_reasoning_stripped_for_isaac_model(monkeypatch):
    recorder = _serve(monkeypatch, "No reasoning", key_env="FAL_KEY")

    @perceive(reasoning=True, model="isaac-0.1")
    def make_request():
        return text("Hi there")

    with cfg(provider="fal", base_url=BASE_URL):
        res = make_request()

    _assert_chat_request(recorder.last, url=FAL_CHAT_URL, authorization="Key test-key")
    payload = recorder.last_body
    assert "reasoning" not in payload
    assert any(issue.get("code") == "reasoning_not_supported" for issue in res.errors)


def test_reasoning_true_keeps_reasoning_in_payload(monkeypatch):
    recorder = _serve(monkeypatch, "Answer", reasoning="Because...")

    @perceive(reasoning=True, model="isaac-0.2-2b-preview", provider="perceptron")
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", base_url=BASE_URL):
        res = make_request()

    payload = recorder.last_body
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    assert all(issue.get("code") != "reasoning_not_supported" for issue in res.errors)
    assert res.reasoning == "Because..."
    assert res.text == "Answer"


def test_isaac_02_reasoning_true_adds_think_hint(monkeypatch):
    recorder = _serve(monkeypatch, "Answer", reasoning="Because...")

    @perceive(reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    payload = recorder.last_body
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    assert any(
        isinstance(m, dict) and isinstance(m.get("content"), str) and "THINK" in m.get("content") for m in messages
    )


def test_payload_shape_matches_expected(monkeypatch):
    recorder = _serve(monkeypatch, "Answer", reasoning="Because...")

    @perceive(expects="box", reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe the object.")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    _assert_chat_request(recorder.last, url=PERCEPTRON_CHAT_URL, authorization="Bearer test-key")
    payload = recorder.last_body

    expected_messages = [
        {"role": "system", "content": "<hint>BOX THINK</hint>"},
        {"role": "user", "content": "Describe the object."},
    ]

    assert payload.get("model") == "isaac-0.2-1b"
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    assert payload.get("messages") == expected_messages
    # Generation params not sent when using API defaults
    assert "temperature" not in payload
    assert "max_completion_tokens" not in payload


def test_all_generation_params_passed_when_explicit(monkeypatch):
    recorder = _serve(monkeypatch)

    @perceive(
        model="isaac-0.2-1b",
        provider="perceptron",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.9,
        top_k=50,
        frequency_penalty=0.3,
        presence_penalty=0.2,
    )
    def make_request():
        return text("Hello")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    payload = recorder.last_body

    assert payload.get("temperature") == 0.5
    assert payload.get("max_completion_tokens") == 2048
    assert payload.get("top_p") == 0.9
    assert payload.get("top_k") == 50
    assert payload.get("frequency_penalty") == 0.3
    assert payload.get("presence_penalty") == 0.2


def test_model_default_used_when_not_explicit(monkeypatch):
    recorder = _serve(monkeypatch)

    # Do NOT pass model into the decorator; rely on configured default
    @perceive()
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", model="isaac-0.2-1b", base_url=BASE_URL):
        make_request()

    assert recorder.last_body.get("model") == "isaac-0.2-1b"


def test_enable_audio_in_video_sets_vision_config(monkeypatch):
    recorder = _serve(monkeypatch)

    @perceive(enable_audio_in_video=True, model="perceptron-mk1.5", provider="perceptron")
    def make_request():
        return text("What is said in the clip?")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    assert recorder.last_body.get("vision_config") == {"enable_audio_in_video": True}


def test_enable_audio_in_video_false_is_sent_explicitly(monkeypatch):
    recorder = _serve(monkeypatch)

    with cfg(provider="perceptron", base_url=BASE_URL):
        perceive(text("Describe."), enable_audio_in_video=False, model="perceptron-mk1.5", provider="perceptron")

    assert recorder.last_body.get("vision_config") == {"enable_audio_in_video": False}


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high"])
def test_reasoning_effort_is_sent_top_level(monkeypatch, effort):
    recorder = _serve(monkeypatch)

    @perceive(reasoning_effort=effort, model="perceptron-mk1.5", provider="perceptron")
    def make_request():
        return text("Count the cars.")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    payload = recorder.last_body
    assert payload["reasoning_effort"] == effort
    # The tier is independent of the boolean flag's THINK hint.
    assert "reasoning" not in payload
    assert all(message.get("role") != "system" for message in payload["messages"])


def test_reasoning_effort_is_normalized_before_sending(monkeypatch):
    recorder = _serve(monkeypatch)

    with cfg(provider="perceptron", base_url=BASE_URL):
        perceive(text("Describe."), reasoning_effort=" High ", model="perceptron-mk1.5", provider="perceptron")

    assert recorder.last_body["reasoning_effort"] == "high"


def test_reasoning_effort_absent_by_default(monkeypatch):
    recorder = _serve(monkeypatch)

    with cfg(provider="perceptron", base_url=BASE_URL):
        perceive(text("Describe."), model="perceptron-mk1.5", provider="perceptron")

    assert "reasoning_effort" not in recorder.last_body


def test_reasoning_effort_outside_the_tiers_fails_before_any_request(monkeypatch):
    recorder = _serve(monkeypatch)

    with cfg(provider="perceptron", base_url=BASE_URL), pytest.raises(BadRequestError) as excinfo:
        perceive(text("Describe."), reasoning_effort="extreme", model="perceptron-mk1.5", provider="perceptron")

    assert excinfo.value.code == INVALID_REASONING_EFFORT
    assert "extreme" in str(excinfo.value)
    assert recorder.requests == []


def test_hint_tokens_are_sorted_and_deduped(monkeypatch):
    recorder = _serve(monkeypatch)

    @perceive(expects="box", reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    messages = recorder.last_body.get("messages") or []
    assert messages and messages[0]["content"].startswith("<hint>BOX THINK</hint>")
    assert messages[0]["content"].count("<hint") == 1


def test_manual_think_hint_does_not_get_double_injected(monkeypatch):
    """Ensure a user-supplied THINK hint isn't duplicated by client injection."""

    recorder = _serve(monkeypatch, "Answer", reasoning="Because...")

    @perceive(model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("<hint>THINK</hint> Explain.")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    messages = recorder.last_body.get("messages") or []
    # Should still have only one hint, and it should be THINK (no duplicate THINK THINK)
    assert messages and messages[0]["content"].count("<hint") == 1
    assert messages[0]["content"].startswith("<hint>THINK</hint>")


def test_expects_think_exact_hint(monkeypatch):
    """Test that expects='think' produces exactly <hint>THINK</hint>."""
    recorder = _serve(monkeypatch, "Answer", reasoning="Because...")

    @perceive(expects="think", model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Explain.")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    payload = recorder.last_body
    messages = payload.get("messages") or []
    # Should have exactly <hint>THINK</hint> as a system message
    assert messages[0] == {"role": "system", "content": "<hint>THINK</hint>"}
    assert messages[1] == {"role": "user", "content": "Explain."}
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_perceptron_reasoning_hint_is_system_message(monkeypatch):
    """Regression: the perceptron AI gateway only honors `<hint>...</hint>`
    when it arrives as a system-role message. The old SDK prepended the hint
    inside the user message, which the gateway ignored — `reasoning=True`
    appeared to work but `result.reasoning` always came back None.
    """
    recorder = _serve(monkeypatch, "Final answer.", reasoning="Step-by-step thinking.")

    @perceive(reasoning=True, model="perceptron-mk1", provider="perceptron")
    def make_request():
        return text("Why?")

    with cfg(provider="perceptron", base_url=BASE_URL):
        res = make_request()

    payload = recorder.last_body
    messages = payload.get("messages") or []
    # The fix: <hint>THINK</hint> arrives as a system-role message, not in user content.
    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert len(system_msgs) == 1, f"Expected exactly one system message, got {messages!r}"
    assert "<hint>" in str(system_msgs[0].get("content") or "")
    assert "THINK" in str(system_msgs[0].get("content") or "")
    # Old non-functional fields are absent.
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    # User message has no hint prefix.
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert all("<hint>" not in str(m.get("content") or "") for m in user_msgs)
    # And the SDK still extracts reasoning_content from the response.
    assert res.reasoning == "Step-by-step thinking."


def test_perceptron_expects_geometry_emits_system_hint(monkeypatch):
    """Regression: each non-text `expects` value must surface as `<hint>X</hint>`
    in a system-role message so the perceptron gateway emits structured tags.
    """
    recorder = _serve(monkeypatch, "stub")

    for sent, expects_value in enumerate(("box", "point", "polygon", "clip"), start=1):

        @perceive(expects=expects_value, model="perceptron-mk1", provider="perceptron")
        def make_request():
            return text("stub prompt")

        with cfg(provider="perceptron", base_url=BASE_URL):
            make_request()

        assert len(recorder.requests) == sent, f"expects={expects_value!r}: expected one request per call"
        payload = recorder.last_body
        messages = payload.get("messages") or []
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) == 1, (
            f"expects={expects_value!r}: expected exactly one system message, got {messages!r}"
        )
        sys_content = str(system_msgs[0].get("content") or "")
        assert expects_value.upper() in sys_content, (
            f"expects={expects_value!r}: expected {expects_value.upper()!r} inside system hint, "
            f"got system content={sys_content!r}"
        )
        assert "vision_config" not in payload


def test_perceptron_expects_text_no_system_hint(monkeypatch):
    """Sanity: expects='text' with reasoning=False emits no hint at all."""
    recorder = _serve(monkeypatch, "stub")

    @perceive(expects="text", model="perceptron-mk1", provider="perceptron")
    def make_request():
        return text("stub prompt")

    with cfg(provider="perceptron", base_url=BASE_URL):
        make_request()

    messages = recorder.last_body.get("messages") or []
    # No system message at all — nothing to hint.
    assert all(m.get("role") != "system" for m in messages), f"Expected no system message, got {messages!r}"


def test_non_perceptron_provider_keeps_top_level_reasoning(monkeypatch):
    """Other providers (nebius/modal/fal) keep the original top-level reasoning
    field; no system hint is emitted by the perceptron-specific path.
    """
    recorder = _serve(monkeypatch, "stub", key_env="FAL_KEY")

    @perceive(expects="box", model="isaac-0.1", provider="fal")
    def make_request():
        return text("stub prompt")

    with cfg(provider="fal", base_url=BASE_URL):
        make_request()

    _assert_chat_request(recorder.last, url=FAL_CHAT_URL, authorization="Key test-key")
    assert "vision_config" not in recorder.last_body
