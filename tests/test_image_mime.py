"""Wire-format MIME tests for base64 image payloads."""

from __future__ import annotations

import pytest
from _image_fixtures import JPEG_BYTES, PNG_BYTES, WEBP_BYTES

from perceptron import image, text
from perceptron.client import _task_to_openai_messages
from perceptron.dsl.perceive import _compile
from perceptron.errors import BadRequestError


def _image_url(seq) -> str:
    task, _ = _compile(seq, expects=None, strict=False)
    msgs = _task_to_openai_messages(task)
    parts = msgs[0]["content"]
    return next(p for p in parts if p["type"] == "image_url")["image_url"]["url"]


def test_png_bytes_emit_image_png_mime():
    url = _image_url(image(PNG_BYTES) + text("hi"))
    assert url.startswith("data:image/png;base64,")


def test_jpeg_bytes_emit_image_jpeg_mime():
    url = _image_url(image(JPEG_BYTES) + text("hi"))
    assert url.startswith("data:image/jpeg;base64,")


def test_webp_bytes_emit_image_webp_mime():
    url = _image_url(image(WEBP_BYTES) + text("hi"))
    assert url.startswith("data:image/webp;base64,")


def test_https_image_url_passthrough_unchanged():
    """URL inputs should still pass through verbatim (no data URL synthesis)."""

    url = _image_url(image("https://example.com/photo.jpg") + text("hi"))
    assert url == "https://example.com/photo.jpg"


def test_undecodable_bytes_raise():
    """Bytes PIL can't decode (corrupt or unsupported) raise instead of being
    silently labeled as PNG. The wire protocol only supports png/jpeg/webp."""

    garbage = b"this is not an image"
    with pytest.raises(BadRequestError):
        _image_url(image(garbage) + text("hi"))
