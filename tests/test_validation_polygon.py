import pytest

from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron import image, perceive, polygon
from perceptron.errors import ExpectationError

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None


class _Stub:
    @staticmethod
    def generate(task, **kwargs):
        return {"text": "", "raw": {}}


@pytest.mark.parametrize("vertex", [(1001, 6), (-1, 6)])
def test_polygon_oob_non_strict(monkeypatch, vertex):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        # One vertex off the normalized 0-1000 grid
        return im + polygon([(2, 2), (6, 2), vertex], image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    assert [e.get("code") for e in res.errors] == ["bounds_out_of_range"]


def test_polygon_oob_strict(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + polygon([(2, 2), (6, 2), (1001, 6)], image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn()


def test_polygon_in_grid_on_a_small_image_is_valid(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + polygon([(100, 100), (900, 100), (500, 1000)], image=im)

    with cfg(api_key="test-key", provider="fal"):
        assert fn().errors == []


def test_polygon_needs_three_vertices(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)

    @perceive()
    def fn():
        im = image("https://example.com/a.png")
        return im + polygon([(1, 1), (2, 2)], image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    assert [e.get("code") for e in res.errors] == ["invalid_polygon"]

    @perceive(strict=True)
    def fn_strict():
        im = image("https://example.com/a.png")
        return im + polygon([(1, 1), (2, 2)], image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError) as excinfo:
        fn_strict()
    assert excinfo.value.code == "invalid_polygon"
