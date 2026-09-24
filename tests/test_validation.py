import pytest
from _http_mock import completion, install, json_response

from perceptron import box, image, perceive, point
from perceptron import config as cfg
from perceptron.errors import AnchorError, ExpectationError

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def http(monkeypatch):
    """An empty answer from the API (`httpx.MockTransport`), so only the compile issues are in `errors`."""
    return install(monkeypatch, lambda request: json_response(completion("")))


def test_anchoring_single_image_implicit_no_issue(http):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        # implicit anchor to the single image present
        return im + point(9, 9)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    # For anchoring only: no anchor_missing issue expected with a single image
    assert not any(e.get("code") == "anchor_missing" for e in res.errors)
    assert len(http.requests) == 1


def test_anchoring_multi_image_missing_anchor(http):
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
    assert len(http.requests) == 1  # a non-strict issue is reported, and the request still goes out

    @perceive(strict=True)
    def fn_strict():
        im1 = image(PILImage.new("RGB", (8, 8)))
        im2 = image(PILImage.new("RGB", (8, 8)))
        return im1 + im2 + point(1, 1)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(AnchorError):
        fn_strict()
    assert len(http.requests) == 1  # strict raises before sending


# Coordinates live on the normalized 0-1000 grid, whatever the image's pixel size (8x8 here).


def test_bounds_point_out_of_range(http):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(1001, 5, image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)
    assert len(http.requests) == 1  # a non-strict issue is reported, and the request still goes out

    @perceive(strict=True)
    def fn_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(1001, 5, image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn_strict()
    assert len(http.requests) == 1  # strict raises before sending


@pytest.mark.parametrize("coords", [(0, 0, 1001, 10), (800, 800, 100, 100)], ids=["out_of_range", "unordered"])
def test_bounds_box_out_of_range_or_unordered(http, coords):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive()
    def fn_non_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + box(*coords, image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn_non_strict()
    assert any(e.get("code") == "bounds_out_of_range" for e in res.errors)
    assert len(http.requests) == 1  # a non-strict issue is reported, and the request still goes out

    @perceive(strict=True)
    def fn_strict():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + box(*coords, image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn_strict()
    assert len(http.requests) == 1  # strict raises before sending


def test_grid_coordinates_on_a_small_image_are_valid(http):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + point(500, 500, image=im) + box(0, 0, 1000, 1000, image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    assert res.errors == []
    assert len(http.requests) == 1
