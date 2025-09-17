import asyncio

from perceptron import perceive, async_perceive, inspect_task
from perceptron.dsl.nodes import image, text

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 12


def test_inspect_task_sync():
    @perceive(expects="point")
    def mark(img):
        return image(img) + text("mark point")

    task, issues = inspect_task(mark, PNG_BYTES)
    assert task["expects"] == "point"
    assert not issues
    kinds = [entry["type"] for entry in task["content"]]
    assert kinds[0] == "image"
    assert any(entry.get("content") == "mark point" for entry in task["content"] if entry["type"] == "text")


def test_inspect_task_async():
    @async_perceive()
    async def describe(img):
        await asyncio.sleep(0)
        return image(img) + text("describe")

    task_coro = inspect_task(describe, PNG_BYTES)
    task, issues = asyncio.run(task_coro)
    kinds = [entry["type"] for entry in task["content"]]
    assert kinds[0] == "image"
    assert not issues


def test_inspect_task_async_stream():
    @async_perceive(stream=True)
    def stream(img):
        return image(img) + text("stream")

    task_coro = inspect_task(stream, PNG_BYTES)
    task, issues = asyncio.run(task_coro)
    kinds = [entry["type"] for entry in task["content"]]
    assert kinds[0] == "image"
    assert not issues


def test_inspect_task_invalid():
    def not_decorated(img):
        return image(img)

    try:
        inspect_task(not_decorated, PNG_BYTES)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-perceive function")
