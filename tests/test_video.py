"""Tests for video support in the DSL and transport layer."""

import pytest

from perceptron import client as client_mod
from perceptron import inspect_task, perceive, text, video
from perceptron.dsl.nodes import Video
from perceptron.dsl.nodes import video as video_fn
from perceptron.errors import AuthError


def test_video_node_creation():
    """Test that video() creates a Video node."""
    node = video_fn("path/to/video.mp4")
    assert isinstance(node, Video)
    assert node.obj == "path/to/video.mp4"


def test_video_url_passthrough():
    """Test that video URLs are passed through without upload."""

    @perceive()
    def fn():
        return video("https://example.com/sample.mp4")

    task, issues = inspect_task(fn)
    assert issues == []
    assert task and isinstance(task, dict)
    content = task.get("content", [])
    assert len(content) == 1
    assert content[0].get("type") == "video"
    assert content[0].get("content") == "https://example.com/sample.mp4"


def test_video_url_passthrough_http():
    """Test that HTTP video URLs are also passed through."""

    @perceive()
    def fn():
        return video("http://example.com/sample.mp4")

    task, issues = inspect_task(fn)
    assert issues == []
    content = task.get("content", [])
    assert content[0].get("content") == "http://example.com/sample.mp4"


def test_video_task_to_openai_messages():
    """Test that video content is converted to OpenAI message format."""
    task = {
        "content": [
            {"type": "video", "role": "user", "content": "https://example.com/video.mp4", "metadata": {}},
            {"type": "text", "role": "user", "content": "What is in this video?"},
        ],
        "expects": None,
    }

    messages = client_mod._task_to_openai_messages(task)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    parts = messages[0]["content"]
    assert isinstance(parts, list)
    assert len(parts) == 2

    video_part = parts[0]
    assert video_part["type"] == "video_url"
    assert video_part["video_url"]["url"] == "https://example.com/video.mp4"

    text_part = parts[1]
    assert text_part["type"] == "text"
    assert text_part["text"] == "What is in this video?"


def test_video_combined_with_text():
    """Test video node combined with text in a perceive function."""

    @perceive()
    def fn():
        return video("https://example.com/video.mp4") + text("Describe this video")

    task, issues = inspect_task(fn)
    assert issues == []
    content = task.get("content", [])
    assert len(content) == 2
    assert content[0]["type"] == "video"
    assert content[1]["type"] == "text"


def test_video_node_with_local_file_requires_upload(tmp_path):
    """Test that local video files trigger the upload path."""
    # Create a minimal MP4 file (just the magic bytes for detection)
    video_file = tmp_path / "test.mp4"
    # MP4 files start with ftyp box
    video_file.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x00\x00isom")

    @perceive()
    def fn():
        return video(str(video_file))

    # This will fail because we don't have API credentials,
    # but we can verify it tries to upload (not pass through)
    with pytest.raises(AuthError):
        inspect_task(fn)
