from _image_fixtures import PNG_BYTES

from perceptron import agent, box, image, inspect_task, perceive, text
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.pointing.parser import PointParser
from perceptron.pointing.types import SinglePoint, bbox


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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Affirmative <point_box> (1,2) (3,4) </point_box>",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(expects="box")
    def make_request(img):
        im1 = image(img)
        im2 = image(img)
        demo_tag = PointParser.serialize(bbox(10, 12, 20, 24, mention="target"))
        return im1 + im2 + text("Locate the object.") + box(1, 1, 4, 4, image=im2) + agent(demo_tag)

    with cfg(provider="fal", base_url="https://mock.api"):
        res = make_request(PNG_BYTES)

    payload = captured["payload"]
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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Thoughts then answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            captured["headers"] = headers
            captured["url"] = url
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="think", model="isaac-0.2-2b-preview", provider="perceptron")
    def make_request():
        return text("Why did the robot stop working?")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_reasoning_stripped_for_isaac_model(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "No reasoning",
                            "reasoning_content": None,
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            captured["headers"] = headers
            captured["url"] = url
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(reasoning=True, model="isaac-0.1")
    def make_request():
        return text("Hi there")

    with cfg(provider="fal", base_url="https://mock.api"):
        res = make_request()

    payload = captured.get("payload", {})
    assert "reasoning" not in payload
    assert any(issue.get("code") == "reasoning_not_supported" for issue in res.errors)


def test_reasoning_true_keeps_reasoning_in_payload(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(reasoning=True, model="isaac-0.2-2b-preview", provider="perceptron")
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        res = make_request()

    payload = captured.get("payload", {})
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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    messages = payload.get("messages") or []
    assert any(isinstance(m, dict) and isinstance(m.get("content"), str) and "THINK" in m.get("content") for m in messages)


def test_payload_shape_matches_expected(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            captured["url"] = url
            captured["headers"] = headers
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="box", reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe the object.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload") or {}

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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

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

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload") or {}

    assert payload.get("temperature") == 0.5
    assert payload.get("max_completion_tokens") == 2048
    assert payload.get("top_p") == 0.9
    assert payload.get("top_k") == 50
    assert payload.get("frequency_penalty") == 0.3
    assert payload.get("presence_penalty") == 0.2


def test_model_default_used_when_not_explicit(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    # Do NOT pass model into the decorator; rely on configured default
    @perceive()
    def make_request():
        return text("Hi there")

    with cfg(provider="perceptron", model="isaac-0.2-1b", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload") or {}
    assert payload.get("model") == "isaac-0.2-1b"


def test_isaac_02_focus_true_adds_tools_hint(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe the scene.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    assert any(isinstance(m, dict) and isinstance(m.get("content"), str) and "TOOLS" in m.get("content") for m in messages)


def test_focus_not_added_for_isaac_01(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(focus=True, model="isaac-0.1", provider="fal")
    def make_request():
        return text("Describe the scene.")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # TOOLS should NOT be in the hint for Isaac 0.1 (doesn't support focus)
    assert not any(isinstance(m, dict) and isinstance(m.get("content"), str) and "TOOLS" in m.get("content") for m in messages)


def test_focus_and_reasoning_combined_hint(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(focus=True, reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe the scene.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Both THINK and TOOLS should be in the hint (alphabetically sorted)
    assert any(isinstance(m, dict) and isinstance(m.get("content"), str) and "<hint>THINK TOOLS</hint>" in m.get("content") for m in messages)
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_focus_box_reasoning_combined_hint(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Because...",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="box", focus=True, reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Find the object.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # BOX, THINK, and TOOLS should all be in the hint (alphabetically sorted)
    assert any(isinstance(m, dict) and isinstance(m.get("content"), str) and "<hint>BOX THINK TOOLS</hint>" in m.get("content") for m in messages)


def test_focus_only_hint_exact_format(monkeypatch):
    """Test that focus=True alone produces exactly <hint>TOOLS</hint>."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have exactly <hint>TOOLS</hint> as a system message
    assert messages[0] == {"role": "system", "content": "<hint>TOOLS</hint>"}
    assert messages[1] == {"role": "user", "content": "Describe."}


def test_focus_false_no_tools_hint(monkeypatch):
    """Test that focus=False does not add TOOLS hint."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(focus=False, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should NOT have TOOLS in the hint
    assert not any("TOOLS" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_none_no_tools_hint(monkeypatch):
    """Test that focus=None (default) does not add TOOLS hint."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    # No focus parameter specified (defaults to None)
    @perceive(model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should NOT have TOOLS in the hint
    assert not any("TOOLS" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_with_point_expectation(monkeypatch):
    """Test that focus works with expects='point'."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "<point>(50, 50)</point>"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="point", focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Find the center.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have <hint>POINT TOOLS</hint> (sorted alphabetically)
    assert any("<hint>POINT TOOLS</hint>" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_with_polygon_expectation(monkeypatch):
    """Test that focus works with expects='polygon'."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "<polygon>(0,0)(100,0)(100,100)(0,100)</polygon>"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="polygon", focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Outline the region.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have <hint>POLYGON TOOLS</hint> (sorted alphabetically)
    assert any("<hint>POLYGON TOOLS</hint>" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_direct_invocation(monkeypatch):
    """Test that focus works with direct perceive invocation (not decorator)."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        # Direct invocation with nodes
        perceive(text("Describe the scene."), focus=True, model="isaac-0.2-1b", provider="perceptron")

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    assert any("<hint>TOOLS</hint>" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_with_expects_think(monkeypatch):
    """Test that focus=True with expects='think' produces <hint>THINK TOOLS</hint>."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer", "reasoning_content": "Because..."}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="think", focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Why is the sky blue?")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have <hint>THINK TOOLS</hint> (sorted alphabetically)
    assert any("<hint>THINK TOOLS</hint>" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    # reasoning should be enabled when expects="think"
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_focus_true_expects_think_exact_format(monkeypatch):
    """Test exact message format with focus=True and expects='think'."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer", "reasoning_content": "Because..."}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="think", focus=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Explain.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have <hint>THINK TOOLS</hint> in the content
    assert messages and messages[0]["content"].startswith("<hint>THINK TOOLS</hint>")
    # Ensure only one hint tag is present (no duplication)
    assert messages[0]["content"].count("<hint") == 1


def test_hint_tokens_are_sorted_and_deduped(monkeypatch):
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="box", focus=True, reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Describe")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    messages = (captured.get("payload", {}).get("messages") or [])
    assert messages and messages[0]["content"].startswith("<hint>BOX THINK TOOLS</hint>")
    assert messages[0]["content"].count("<hint") == 1


def test_manual_think_hint_does_not_get_double_injected(monkeypatch):
    """Ensure a user-supplied THINK hint isn't duplicated by client injection."""

    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer", "reasoning_content": "Because..."}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("<hint>THINK</hint> Explain.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    messages = (captured.get("payload", {}).get("messages") or [])
    # Should still have only one hint, and it should be THINK (no duplicate THINK THINK)
    assert messages and messages[0]["content"].count("<hint") == 1
    assert messages[0]["content"].startswith("<hint>THINK</hint>")


def test_expects_think_without_focus(monkeypatch):
    """Test that expects='think' without focus only produces <hint>THINK</hint>."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer", "reasoning_content": "Because..."}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="think", model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Explain.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have exactly <hint>THINK</hint> as a system message (no TOOLS)
    assert messages[0] == {"role": "system", "content": "<hint>THINK</hint>"}
    assert messages[1] == {"role": "user", "content": "Explain."}
    assert "TOOLS" not in messages[0]["content"]
    assert "reasoning" not in payload
    assert "vision_config" not in payload


def test_focus_and_expects_think_on_isaac_01_no_hint(monkeypatch):
    """Test that focus with expects='think' on Isaac 0.1 doesn't add TOOLS (unsupported)."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(expects="think", focus=True, model="isaac-0.1", provider="fal")
    def make_request():
        return text("Explain.")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # TOOLS should NOT be present (Isaac 0.1 doesn't support focus)
    # THINK hint may still be present but reasoning won't work
    assert not any("TOOLS" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))


def test_focus_expects_think_reasoning_explicit_true(monkeypatch):
    """Test focus + expects='think' + reasoning=True all together."""
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Answer", "reasoning_content": "Because..."}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="think", focus=True, reasoning=True, model="isaac-0.2-1b", provider="perceptron")
    def make_request():
        return text("Think carefully.")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        res = make_request()

    payload = captured.get("payload", {})
    messages = payload.get("messages") or []
    # Should have <hint>THINK TOOLS</hint>
    assert any("<hint>THINK TOOLS</hint>" in m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    messages = payload.get("messages") or []
    assert any(m.get("role") == "system" and "THINK" in str(m.get("content") or "") for m in messages), (
        f"Expected a system message containing THINK hint, got messages={messages!r}"
    )
    assert "reasoning" not in payload
    assert "vision_config" not in payload
    assert res.reasoning == "Because..."
    assert res.text == "Answer"


def test_perceptron_reasoning_hint_is_system_message(monkeypatch):
    """Regression: the perceptron AI gateway only honors `<hint>...</hint>`
    when it arrives as a system-role message. The old SDK prepended the hint
    inside the user message, which the gateway ignored — `reasoning=True`
    appeared to work but `result.reasoning` always came back None.
    """
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Final answer.",
                            "reasoning_content": "Step-by-step thinking.",
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(reasoning=True, model="isaac-0.3-max", provider="perceptron")
    def make_request():
        return text("Why?")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        res = make_request()

    payload = captured["payload"]
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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "stub",
                            "reasoning_content": None,
                        }
                    }
                ]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    for expects_value in ("box", "point", "polygon", "clip"):
        captured.clear()

        @perceive(expects=expects_value, model="isaac-0.3-max", provider="perceptron")
        def make_request():
            return text("stub prompt")

        with cfg(provider="perceptron", base_url="https://mock.api"):
            make_request()

        payload = captured["payload"]
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
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "stub", "reasoning_content": None}}]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")

    @perceive(expects="text", model="isaac-0.3-max", provider="perceptron")
    def make_request():
        return text("stub prompt")

    with cfg(provider="perceptron", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    messages = payload.get("messages") or []
    # No system message at all — nothing to hint.
    assert all(m.get("role") != "system" for m in messages), f"Expected no system message, got {messages!r}"


def test_non_perceptron_provider_keeps_top_level_reasoning(monkeypatch):
    """Other providers (nebius/modal/fal) keep the original top-level reasoning
    field; no system hint is emitted by the perceptron-specific path.
    """
    captured: dict[str, dict] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "stub", "reasoning_content": None}}]
            }

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

        def stream(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _Client())
    monkeypatch.setenv("FAL_KEY", "test-key")

    @perceive(expects="box", model="isaac-0.1", provider="fal")
    def make_request():
        return text("stub prompt")

    with cfg(provider="fal", base_url="https://mock.api"):
        make_request()

    payload = captured["payload"]
    assert "vision_config" not in payload
