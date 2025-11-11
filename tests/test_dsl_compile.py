from perceptron import box, image, inspect_task, perceive, text


@perceive(max_tokens=32)
def describe_region(img):
    im = image(img)
    return im + text("What is in this box?") + box(1, 2, 3, 4, image=im)


def test_compile_task_no_execute():
    # Provide a tiny PNG header as bytes; width/height may be missing
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    task, issues = inspect_task(describe_region, png_bytes)
    assert issues == []
    assert task and isinstance(task, dict)
    content = task.get("content", [])
    # Should contain text and image entries
    kinds = [c.get("type") for c in content]
    assert "image" in kinds and "text" in kinds


def test_perceive_direct_sequence_compile_only():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"1" * 10
    seq = image(png_bytes) + text("Describe the scene.")
    with cfg(api_key=None, provider=None):
        res = perceive(seq, expects="text")
    assert res.raw and isinstance(res.raw, dict)
    kinds = [c.get("type") for c in res.raw.get("content", [])]
    assert kinds.count("image") == 1
    assert kinds.count("text") >= 1


def test_perceive_direct_list_normalization():
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"2" * 10
    nodes = [image(png_bytes), text("Who is in the frame?")]
    with cfg(api_key=None, provider=None):
        res = perceive(nodes, expects="text")
    assert res.raw and isinstance(res.raw, dict)
    content = res.raw.get("content", [])
    assert content and content[0]["type"] == "image"
    assert any(item.get("type") == "text" for item in content)
