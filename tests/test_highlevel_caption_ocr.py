from perceptron import caption, ocr

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 12


def test_caption_highlevel_compile_only():
    res = caption(PNG_BYTES, style="concise")
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") == "box"
    content = res.raw.get("content", [])
    assert any(entry.get("content") == '<hint>BOX</hint>' for entry in content)


def test_caption_style_validation():
    try:
        caption(PNG_BYTES, style="unknown")
    except Exception as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("expected caption() to reject invalid style")


def test_ocr_boxes_compile_only():
    res = ocr(PNG_BYTES)
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") is None


def test_ocr_plain_text_compile_only():
    res = ocr(PNG_BYTES)
    assert res.raw and isinstance(res.raw, dict)
    assert res.raw.get("expects") is None
