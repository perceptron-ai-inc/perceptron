from __future__ import annotations

import pytest

from perceptron import caption, detect, ocr, ocr_html, ocr_markdown
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.client import _PROVIDER_CONFIG, _select_model
from perceptron.errors import BadRequestError

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 12


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


def test_caption_uses_qwen_prompt_text():
    with cfg(api_key="test-key", provider="perceptron"):
        res = caption(PNG_BYTES, style="concise", model="qwen3-vl-235b-a22b-thinking")

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    assert system_messages == []
    user_messages = _collect_text(content, role="user")
    assert any(
        msg == "Describe the primary subjects, their actions, and visible context in one vivid sentence."
        for msg in user_messages
    )


def test_caption_defaults_to_isaac_prompt_on_fal():
    with cfg(api_key="test-key", provider="fal"):
        res = caption(PNG_BYTES, style="concise")

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    assert not system_messages  # Isaac profile omits a system instruction
    user_messages = _collect_text(content, role="user")
    assert "Provide a concise, human-friendly caption for the upcoming image." in user_messages


def test_select_model_rejects_qwen_for_fal():
    fal_cfg = {"name": "fal", **_PROVIDER_CONFIG["fal"]}
    with pytest.raises(BadRequestError):
        _select_model(fal_cfg, "qwen3-vl-235b-a22b-thinking")


def test_select_model_accepts_qwen_for_perceptron():
    perceptron_cfg = {"name": "perceptron", **_PROVIDER_CONFIG["perceptron"]}
    resolved = _select_model(perceptron_cfg, "qwen3-vl-235b-a22b-thinking")
    assert resolved == "qwen3-vl-235b-a22b-thinking"


def test_detect_qwen_prompt_reports_json_bbox():
    categories = [
        "plate/dish",
        "scallop",
        "wine bottle",
        "tv",
        "bowl",
        "spoon",
        "air conditioner",
        "coconut drink",
        "cup",
        "chopsticks",
        "person",
    ]
    with cfg(api_key="test-key", provider="perceptron"):
        res = detect(PNG_BYTES, classes=categories, model="qwen3-vl-235b-a22b-thinking")

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    expected = (
        'Locate every instance that belongs to the following categories: "'
        + ", ".join(categories)
        + '". Report bbox coordinates in JSON format.'
    )
    assert expected in system_messages


def test_config_context_propagates_default_model():
    categories = ["plate/dish"]
    with cfg(api_key="test-key", provider="perceptron", model="qwen3-vl-235b-a22b-thinking"):
        res = detect(PNG_BYTES, classes=categories)

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    expected = (
        'Locate every instance that belongs to the following categories: "plate/dish". '
        "Report bbox coordinates in JSON format."
    )
    assert expected in system_messages


def test_env_default_model_applies(monkeypatch):
    monkeypatch.setenv("PERCEPTRON_MODEL", "qwen3-vl-235b-a22b-thinking")

    categories = ["plate/dish"]
    with cfg(api_key="test-key", provider="perceptron"):
        res = detect(PNG_BYTES, classes=categories)

    content = res.raw.get("content", [])
    system_messages = _collect_text(content, role="system")
    expected = (
        'Locate every instance that belongs to the following categories: "plate/dish". '
        "Report bbox coordinates in JSON format."
    )
    assert expected in system_messages


def test_incompatible_default_model_raises():
    categories = ["plate/dish"]
    with pytest.raises(BadRequestError), cfg(api_key="test-key", provider="fal", model="qwen3-vl-235b-a22b-thinking"):
        detect(PNG_BYTES, classes=categories)


def test_qwen_plain_ocr_prompt():
    with cfg(api_key="test-key", provider="perceptron"):
        res = ocr(PNG_BYTES, model="qwen3-vl-235b-a22b-thinking")

    content = res.raw.get("content", [])
    user_messages = _collect_text(content, role="user")
    assert "Read all the text in the image." in user_messages


def test_qwen_markdown_ocr_prompt():
    with cfg(api_key="test-key", provider="perceptron"):
        res = ocr_markdown(PNG_BYTES, model="qwen3-vl-235b-a22b-thinking")

    content = res.raw.get("content", [])
    user_messages = _collect_text(content, role="user")
    assert "qwenvl markdown" in user_messages


def test_qwen_html_ocr_prompt():
    with cfg(api_key="test-key", provider="perceptron"):
        res = ocr_html(PNG_BYTES, model="qwen3-vl-235b-a22b-thinking")

    content = res.raw.get("content", [])
    user_messages = _collect_text(content, role="user")
    assert "qwenvl html" in user_messages
