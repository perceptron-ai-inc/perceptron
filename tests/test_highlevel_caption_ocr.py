import json

import pytest
from _http_mock import completion, install, json_response
from _image_fixtures import PNG_BYTES

from perceptron import caption, image, json_schema_format, ocr
from perceptron import config as cfg

FAL_URL = "https://fal.run/perceptron/isaac-01/openai/v1/chat/completions"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def http(monkeypatch):
    """The API behind `httpx.MockTransport`, answering with an empty completion."""
    return install(monkeypatch, lambda request: json_response(completion("")))


def _texts(http) -> list[str]:
    """Every text the last request sent: string contents and text parts, in order."""
    texts = []
    for message in http.last_body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(part["text"] for part in content if part["type"] == "text")
    return texts


def test_caption_highlevel_compile_only(http):
    with cfg(api_key="test-key", provider="fal"):
        res = caption(image(PNG_BYTES), style="concise")
    assert str(http.last.url) == FAL_URL
    # style="concise" expects boxes: the BOX hint is sent once, in the user turn with the image (fal).
    [message] = http.last_body["messages"]
    assert message["role"] == "user"
    assert {"type": "text", "text": "<hint>BOX</hint>"} in message["content"]
    assert _texts(http).count("<hint>BOX</hint>") == 1
    assert res.raw == completion("")
    assert res.errors == []


def test_caption_highlevel_text_expectation(http):
    with cfg(api_key="test-key", provider="fal"):
        res = caption(image(PNG_BYTES), expects="text")
    assert str(http.last.url) == FAL_URL
    assert all("<hint>" not in entry for entry in _texts(http))
    assert res.raw == completion("")
    assert res.errors == []


def test_caption_style_validation():
    try:
        caption(image(PNG_BYTES), style="unknown")
    except Exception as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("expected caption() to reject invalid style")


def test_ocr_boxes_compile_only(http):
    with cfg(api_key="test-key", provider="fal"):
        res = ocr(image(PNG_BYTES))
    assert str(http.last.url) == FAL_URL
    assert all("<hint>" not in entry for entry in _texts(http))
    assert res.raw == completion("")
    assert res.errors == []


def test_ocr_plain_text_compile_only(http):
    with cfg(api_key="test-key", provider="fal"):
        res = ocr(image(PNG_BYTES))
    assert str(http.last.url) == FAL_URL
    assert all("<hint>" not in entry for entry in _texts(http))
    assert res.raw == completion("")
    assert res.errors == []


def test_caption_response_format_propagates(monkeypatch):
    """Test that response_format passed to caption() reaches the HTTP payload."""
    answer = {"choices": [{"message": {"content": '{"description": "test"}'}}]}
    http = install(monkeypatch, lambda request: json_response(answer))

    schema = {"type": "object", "properties": {"description": {"type": "string"}}}
    with cfg(api_key="test-key", provider="fal", base_url="https://mock.api"):
        res = caption(image(PNG_BYTES), style="concise", response_format=json_schema_format(schema))

    assert len(http.requests) == 1
    assert str(http.last.url) == "https://mock.api/perceptron/isaac-01/openai/v1/chat/completions"
    payload = http.last_body
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert json.loads(res.text) == {"description": "test"}
