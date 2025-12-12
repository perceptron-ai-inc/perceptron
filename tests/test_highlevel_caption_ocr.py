from perceptron import caption, json_schema_format, ocr
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
    assert all("<hint>" not in (entry.get("content") or "") for entry in content)
    assert res.errors == []


def test_caption_hints_skipped_for_qwen(monkeypatch):
    """Qwen model has skip_structured_hints=True, so no hints should be present."""
    monkeypatch.setattr(client_mod.Client, "generate", _echo_task)
    with cfg(api_key="test-key", provider="perceptron", model="qwen3-vl-235b-a22b-thinking"):
        res = caption(PNG_BYTES, style="concise")
    assert res.raw and isinstance(res.raw, dict)
    content = res.raw.get("content", [])
    assert all("<hint>" not in (entry.get("content") or "") for entry in content)
    assert any(issue.get("code") == "reasoning_required_for_model" for issue in res.errors)


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


def test_caption_response_format_propagates(monkeypatch):
    """Test that response_format passed to caption() reaches the HTTP payload."""
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"description": "test"}'}}]}

    class _MockClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(client_mod, "_http_client", lambda timeout: _MockClient())

    schema = {"type": "object", "properties": {"description": {"type": "string"}}}
    with cfg(api_key="test-key", provider="fal", base_url="https://mock.api"):
        caption(PNG_BYTES, style="concise", response_format=json_schema_format(schema))

    assert "payload" in captured
    payload = captured["payload"]
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
