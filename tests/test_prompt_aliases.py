"""Prompt profiles for Mk1.5: the model alias, and `video_frames()` media using the video prompts."""

from __future__ import annotations

import pytest

from perceptron import caption, detect, video_frames
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.prompting import prompt_registry, resolve_prompt_profile

FRAMES = [("https://example.com/f0.jpg", 0), ("https://example.com/f1.jpg", 500)]


@pytest.fixture(autouse=True)
def _echo_client(monkeypatch):
    for key in ("PERCEPTRON_API_KEY", "FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(client_mod.Client, "generate", lambda self, task, **kwargs: {"text": "", "raw": task})


def _texts(task: dict, role: str) -> list[str]:
    return [e["content"] for e in task["content"] if e.get("type") == "text" and e.get("role") == role]


@pytest.mark.parametrize("model", ["perceptron-mk1.5", "Perceptron-Mk1.5"])
def test_mk15_resolves_to_the_default_profile(model):
    registry = prompt_registry()
    assert registry._aliases["perceptron-mk1.5"] == registry._default_key
    assert resolve_prompt_profile(model) is resolve_prompt_profile(None)


def test_video_frames_get_the_video_prompts():
    frames = video_frames(FRAMES)
    profile = resolve_prompt_profile("perceptron-mk1.5")
    assert profile.caption.style_prompts["concise"].get(frames) == profile.caption.style_prompts["concise"].video

    with cfg(api_key="k", provider="perceptron", model="perceptron-mk1.5"):
        captioned = caption(frames, expects="text")
        detected = detect(frames, classes=["person"])
    assert "Provide a concise, human-friendly caption for the upcoming video." in _texts(captioned.raw, "user")
    assert any("Make sure to track the objects." in msg for msg in _texts(detected.raw, "system"))
