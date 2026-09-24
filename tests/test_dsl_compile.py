import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import box, image, inspect_task, perceive, text

FAL_URL = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"


@pytest.fixture
def http(monkeypatch):
    """Direct `perceive` calls on fal (``FAL_KEY`` is the only key), answered through `httpx.MockTransport`."""
    for key in ("PERCEPTRON_API_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FAL_KEY", "test")
    return install(monkeypatch, lambda request: json_response(completion("")))


def _sent_parts(http):
    """The content parts of the only request's single user message, a POST to fal; the call's client was closed."""
    assert [(request.method, str(request.url)) for request in http.requests] == [("POST", FAL_URL)]
    assert [client.is_closed for client in http.clients] == [True]
    [message] = http.last_body["messages"]
    assert message["role"] == "user"
    return message["content"]


@perceive(max_tokens=32)
def describe_region(img):
    im = image(img)
    return im + text("What is in this box?") + box(1, 2, 3, 4, image=im)


def test_compile_task_no_execute():
    # Provide a tiny PNG header as bytes; width/height may be missing
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    task, issues = inspect_task(describe_region, png_bytes)
    assert issues == []
    assert task and isinstance(task, dict)
    content = task.get("content", [])
    # Should contain text and image entries
    kinds = [c.get("type") for c in content]
    assert "image" in kinds and "text" in kinds


def test_perceive_direct_sequence_executes(http):
    seq = image(PNG_BYTES) + text("Describe the scene.")

    res = perceive(seq, expects="text")

    kinds = [part["type"] for part in _sent_parts(http)]
    assert kinds.count("image_url") == 1
    assert kinds.count("text") >= 1
    assert res.text == ""


def test_perceive_direct_list_normalization(http):
    nodes = [image(PNG_BYTES), text("Who is in the frame?")]

    perceive(nodes, expects="text")

    content = _sent_parts(http)
    assert content and content[0]["type"] == "image_url"
    assert {"type": "text", "text": "Who is in the frame?"} in content


def test_perceive_direct_nested_iterables(http):
    nested = [image(PNG_BYTES), [text("First"), (text("Second"),)]]

    perceive(nested, expects="text")

    content = _sent_parts(http)
    kinds = [part["type"] for part in content]
    assert kinds[:2] == ["image_url", "text"]
    assert [part["text"] for part in content if part["type"] == "text"] == ["First", "Second"]


def test_perceive_direct_invalid_payload_type():
    with pytest.raises(TypeError):
        perceive("describe this")


@pytest.mark.parametrize(
    ("expects", "allow_multiple"),
    [
        ("text", False),
        ("point", False),
        ("box", True),
        ("polygon", True),
    ],
)
def test_perceive_direct_structured_matrix(http, expects, allow_multiple):
    perceive(image(PNG_BYTES) + text("Label"), expects=expects, allow_multiple=allow_multiple)

    # `expects` reaches the request as its hint (none for text).
    hints = [part["text"] for part in _sent_parts(http) if part["type"] == "text" and "<hint>" in part["text"]]
    assert hints == ([] if expects == "text" else [f"<hint>{expects.upper()}</hint>"])
    # `allow_multiple`/`max_outputs` stay accepted by perceive but never changed the request, so they are not forwarded
    # (the real `Client.generate` rejects unknown keyword arguments) and not sent.
    body = http.last_body
    assert "allow_multiple" not in body
    assert "max_outputs" not in body
    assert "n" not in body
