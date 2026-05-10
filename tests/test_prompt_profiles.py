from __future__ import annotations

import pytest

from perceptron import caption, detect, image, ocr, ocr_html, ocr_markdown
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.client import _PROVIDER_CONFIG, _select_model
from perceptron.errors import BadRequestError

from _image_fixtures import PNG_BYTES  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    # Ensure deterministic baseline for provider resolution.
    monkeypatch.delenv("PERCEPTRON_API_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)


@pytest.fixture(autouse=True)
def _stub_client(monkeypatch):
    def _echo(self, task, **kwargs):  # pylint: disable=unused-argument
        return {"text": "", "points": None, "parsed": None, "raw": task}

    monkeypatch.setattr(client_mod.Client, "generate", _echo)


def _collect_text(content, *, role: str) -> list[str]:
    return [entry["content"] for entry in content if entry.get("type") == "text" and entry.get("role") == role]


def test_caption_defaults_to_isaac_prompt_on_fal():
    with cfg(api_key="test-key", provider="fal"):
        res = caption(image(PNG_BYTES), style="concise")

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    assert not system_messages  # Isaac profile omits a system instruction
    user_messages = _collect_text(content, role="user")
    assert "Provide a concise, human-friendly caption for the upcoming image." in user_messages


def test_select_model_accepts_isaac_0_3_max_for_perceptron():
    perceptron_cfg = {"name": "perceptron", **_PROVIDER_CONFIG["perceptron"]}
    resolved = _select_model(perceptron_cfg, "isaac-0.3-max")
    assert resolved == "isaac-0.3-max"


def test_select_model_rejects_isaac_0_3_max_for_fal():
    fal_cfg = {"name": "fal", **_PROVIDER_CONFIG["fal"]}
    with pytest.raises(BadRequestError):
        _select_model(fal_cfg, "isaac-0.3-max")


def test_isaac_0_3_max_models_entry_supports_reasoning_and_focus():
    perceptron_cfg = _PROVIDER_CONFIG["perceptron"]
    entry = perceptron_cfg["models"]["isaac-0.3-max"]
    assert entry["reasoning"] is True
    assert entry["focus"] is True
    assert entry["skip_structured_hints"] is False


def test_config_context_propagates_default_model():
    categories = ["plate/dish"]
    with cfg(api_key="test-key", provider="perceptron", model="isaac-0.2-2b-preview"):
        res = detect(image(PNG_BYTES), classes=categories)

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    assert any("Your goal is to segment out the following categories: plate/dish" in msg for msg in system_messages)


def test_env_default_model_applies(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_MODEL", "isaac-0.2-2b-preview")

    categories = ["plate/dish"]
    with cfg(api_key="test-key", provider="perceptron"):
        res = detect(image(PNG_BYTES), classes=categories)

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    assert any("Your goal is to segment out the following categories: plate/dish" in msg for msg in system_messages)


def test_incompatible_default_model_raises():
    categories = ["plate/dish"]
    with pytest.raises(BadRequestError), cfg(api_key="test-key", provider="fal", model="isaac-0.2-2b-preview"):
        detect(image(PNG_BYTES), classes=categories)


def test_isaac_markdown_ocr_prompt():
    with cfg(api_key="test-key", provider="fal"):
        res = ocr_markdown(image(PNG_BYTES))

    content = res.raw.get("content", [])
    user_messages = _collect_text(content, role="user")
    assert "Transcribe every readable word in the image using Markdown formatting with headings, lists, tables, and other structural elements as appropriate." in user_messages


