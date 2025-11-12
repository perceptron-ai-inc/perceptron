from perceptron import caption, ocr
from perceptron import client as client_mod
from perceptron import config as cfg

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 12


def _echo_task(self, task, **kwargs):  # pylint: disable=unused-argument
    return {"text": "", "points": None, "parsed": None, "raw": task}


def test_caption_highlevel_compile_only(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _echo_task)
    with cfg(api_key="test-key", provider="fal"):
        res = caption(PNG_BYTES, style="concise")
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") == "box"
    content = res.raw.get("content", [])
    assert any(entry.get("content") == "<hint>BOX</hint>" for entry in content)
    assert res.errors == []


def test_caption_highlevel_text_expectation(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _echo_task)
    with cfg(api_key="test-key", provider="fal"):
        res = caption(PNG_BYTES, expects="text")
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") is None
    content = res.raw.get("content", [])
    assert all(entry.get("content") != "<hint>BOX</hint>" for entry in content)
    assert res.errors == []


def test_caption_style_validation():
    try:
        caption(PNG_BYTES, style="unknown")
    except Exception as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("expected caption() to reject invalid style")


def test_ocr_boxes_compile_only(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _echo_task)
    with cfg(api_key="test-key", provider="fal"):
        res = ocr(PNG_BYTES)
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") is None
    assert res.errors == []


def test_ocr_plain_text_compile_only(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _echo_task)
    with cfg(api_key="test-key", provider="fal"):
        res = ocr(PNG_BYTES)
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") is None
    assert res.errors == []
