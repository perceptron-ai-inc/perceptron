import pytest

from perceptron import perceive, image, point, box


class _Stub:
    @staticmethod
    def generate(task, **kwargs):
        # return minimal response to avoid network
        return {"text": "", "raw": {}}


def test_anchoring_single_image_implicit_no_issue(monkeypatch):
    # Monkeypatch client to avoid HTTP
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)

    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception:
        pytest.skip("PIL not available")

    @perceive()
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        # implicit anchor to the single image present
        return im + point(9, 9)  # out-of-bounds; will be caught below in bounds test

    res = fn()
    # For anchoring only: no anchor_missing issue expected with a single image
    assert not any(e.get("code") == "anchor_missing" for e in res.errors)


def test_anchoring_multi_image_missing_anchor(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)

    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im1 = image(PILImage.new("RGB", (8, 8)))
        im2 = image(PILImage.new("RGB", (8, 8)))
        # missing image= in multi-image context → issue
        return im1 + im2 + point(1, 1)

    res = fn_non_strict()
    assert any(e.get("code") == "anchor_missing" for e in res.errors)

    @perceive(strict=True)
    def fn_strict():
        im1 = image(PILImage.new("RGB", (8, 8)))
        im2 = image(PILImage.new("RGB", (8, 8)))
        return im1 + im2 + point(1, 1)

    with pytest.raises(Exception):  # AnchorError maps to Exception hierarchy
        fn_strict()


def test_bounds_point_out_of_bounds(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)

    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(9, 9, image=im)  # OOB

    res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)

    @perceive(strict=True)
    def fn_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(9, 9, image=im)

    with pytest.raises(Exception):
        fn_strict()


def test_bounds_box_out_of_bounds(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _Stub.generate)

    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + box(0, 0, 10, 10, image=im)

    res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)
