import pytest

from perceptron import box, image, inspect_task, perceive, text


class _StubClient:
    last_task = None
    last_kwargs = None

    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        pass

    def generate(self, task, **kwargs):  # pylint: disable=unused-argument
        type(self).last_task = task
        type(self).last_kwargs = kwargs
        return {"text": "", "points": None, "parsed": None, "raw": task}


def _patch_direct_client(monkeypatch):
    _StubClient.last_task = None
    _StubClient.last_kwargs = None
    monkeypatch.setenv("FAL_KEY", "test")
    monkeypatch.setattr("perceptron.dsl.perceive.Client", _StubClient)
    return _StubClient


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


def test_perceive_direct_sequence_executes(monkeypatch):
    stub = _patch_direct_client(monkeypatch)
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"1" * 10
    seq = image(png_bytes) + text("Describe the scene.")

    res = perceive(seq, expects="text")

    assert stub.last_task is not None
    kinds = [entry.get("type") for entry in stub.last_task.get("content", [])]
    assert kinds.count("image") == 1
    assert kinds.count("text") >= 1
    assert res.text == ""


def test_perceive_direct_list_normalization(monkeypatch):
    stub = _patch_direct_client(monkeypatch)
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"2" * 10
    nodes = [image(png_bytes), text("Who is in the frame?")]

    perceive(nodes, expects="text")

    assert stub.last_task is not None
    content = stub.last_task.get("content", [])
    assert content and content[0]["type"] == "image"
    assert any(item.get("type") == "text" for item in content)


def test_perceive_direct_nested_iterables(monkeypatch):
    stub = _patch_direct_client(monkeypatch)
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"3" * 10
    nested = [image(png_bytes), [text("First"), (text("Second"),)]]

    perceive(nested, expects="text")

    assert stub.last_task is not None
    kinds = [item["type"] for item in stub.last_task.get("content", [])]
    assert kinds[:2] == ["image", "text"]
    assert kinds.count("text") == 2


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
def test_perceive_direct_structured_matrix(monkeypatch, expects, allow_multiple):
    stub = _patch_direct_client(monkeypatch)
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"4" * 10

    perceive(image(png_bytes) + text("Label"), expects=expects, allow_multiple=allow_multiple)

    assert stub.last_kwargs is not None
    assert stub.last_kwargs.get("expects") == expects
    assert stub.last_kwargs.get("allow_multiple") == allow_multiple
