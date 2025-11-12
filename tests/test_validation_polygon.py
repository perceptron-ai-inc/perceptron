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


def test_polygon_oob_non_strict(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        # One vertex out-of-bounds
        return im + polygon([(2, 2), (6, 2), (20, 6)], image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)


def test_polygon_oob_strict(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + polygon([(2, 2), (6, 2), (20, 6)], image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn()
