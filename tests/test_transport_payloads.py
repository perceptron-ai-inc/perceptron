from perceptron import agent, box, image, inspect_task, perceive, text
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.pointing.parser import PointParser
from perceptron.pointing.types import SinglePoint, bbox

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 16


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
    assert res.points and res.points[0].top_left.x == 1


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
