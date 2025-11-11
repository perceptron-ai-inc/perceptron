import pytest

from perceptron import box, image, perceive, point
from perceptron import client as client_mod
from perceptron import config as cfg
from perceptron.errors import AnchorError, ExpectationError

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None


class _Stub:
    @staticmethod
    def generate(task, **kwargs):
        # return minimal response to avoid network
        return {"text": "", "raw": {}}


def test_anchoring_single_image_implicit_no_issue(monkeypatch):
    # Monkeypatch client to avoid HTTP
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        # implicit anchor to the single image present
        return im + point(9, 9)  # out-of-bounds; will be caught below in bounds test

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    # For anchoring only: no anchor_missing issue expected with a single image
    assert not any(e.get("code") == "anchor_missing" for e in res.errors)


def test_anchoring_multi_image_missing_anchor(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im1 = image(PILImage.new("RGB", (8, 8)))
        im2 = image(PILImage.new("RGB", (8, 8)))
        # missing image= in multi-image context → issue
        return im1 + im2 + point(1, 1)

    with cfg(api_key="test-key", provider="fal"):
        res = fn_non_strict()
    assert any(e.get("code") == "anchor_missing" for e in res.errors)

    @perceive(strict=True)
    def fn_strict():
        im1 = image(PILImage.new("RGB", (8, 8)))
        im2 = image(PILImage.new("RGB", (8, 8)))
        return im1 + im2 + point(1, 1)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(AnchorError):
        fn_strict()


def test_bounds_point_out_of_bounds(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(9, 9, image=im)  # OOB

    with cfg(api_key="test-key", provider="fal"):
        res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)

    @perceive(strict=True)
    def fn_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(9, 9, image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn_strict()


def test_bounds_box_out_of_bounds(monkeypatch):
    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + box(0, 0, 10, 10, image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)

    @perceive(strict=True)
    def fn_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + box(0, 0, 10, 10, image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn_strict()
