import pytest
from _http_mock import completion, install, json_response

from perceptron import config as cfg
from perceptron import image, perceive, polygon
from perceptron.errors import ExpectationError

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


@pytest.mark.parametrize("vertex", [(1001, 6), (-1, 6)])
def test_polygon_oob_non_strict(http, vertex):
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
    assert len(http.requests) == 1  # a non-strict issue is reported, and the request still goes out


def test_polygon_oob_strict(http):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + polygon([(2, 2), (6, 2), (1001, 6)], image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError):
        fn()
    assert not http.requests  # strict raises before sending


def test_polygon_in_grid_on_a_small_image_is_valid(http):
    if PILImage is None:
        pytest.skip("PIL not available")

    @perceive(strict=True)
    def fn():
        im = image(PILImage.new("RGB", (8, 8)))
        return im + polygon([(100, 100), (900, 100), (500, 1000)], image=im)

    with cfg(api_key="test-key", provider="fal"):
        assert fn().errors == []
    assert len(http.requests) == 1


def test_polygon_needs_three_vertices(http):
    @perceive()
    def fn():
        im = image("https://example.com/a.png")
        return im + polygon([(1, 1), (2, 2)], image=im)

    with cfg(api_key="test-key", provider="fal"):
        res = fn()
    assert [e.get("code") for e in res.errors] == ["invalid_polygon"]
    assert len(http.requests) == 1

    @perceive(strict=True)
    def fn_strict():
        im = image("https://example.com/a.png")
        return im + polygon([(1, 1), (2, 2)], image=im)

    with cfg(api_key="test-key", provider="fal"), pytest.raises(ExpectationError) as excinfo:
        fn_strict()
    assert excinfo.value.code == "invalid_polygon"
    assert len(http.requests) == 1  # strict raises before sending
