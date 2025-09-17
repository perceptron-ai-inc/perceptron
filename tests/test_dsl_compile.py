from perceptron import perceive, image, text, box


@perceive(max_tokens=32)
def describe_region(img):
    im = image(img)
    return im + text("What is in this box?") + box(1, 2, 3, 4, image=im)


def test_compile_task_no_execute():
    # Provide a tiny PNG header as bytes; width/height may be missing
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    res = describe_region(png_bytes)
    assert res.raw and isinstance(res.raw, dict)
    content = res.raw.get("content", [])
    # Should contain text and image entries
    kinds = [c.get("type") for c in content]
    assert "image" in kinds and "text" in kinds
