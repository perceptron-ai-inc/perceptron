import json

from typer.testing import CliRunner

from perceptron import PerceiveResult
from perceptron.cli import app
from perceptron.pointing.types import BoundingBox, SinglePoint

runner = CliRunner()


class _StubResult(PerceiveResult):
    def __init__(self, text: str):
        super().__init__(text=text, points=None, parsed=None, usage=None, errors=[], raw={"text": text})


def test_caption_command(monkeypatch, tmp_path):
    image_path = tmp_path / "img.bin"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr("perceptron.cli.caption_image", lambda *a, **k: _StubResult("hello"))

    result = runner.invoke(app, ["caption", str(image_path)])
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_caption_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")
    (tmp_path / "notes.txt").write_text("ignore me")

    def _fake_caption(data, **kwargs):
        if data == b"image-one":
            return _StubResult("caption-one")
        if data == b"image-two":
            return _StubResult("caption-two")
        raise AssertionError("unexpected payload")

    monkeypatch.setattr("perceptron.cli.caption_image", _fake_caption)

    result = runner.invoke(app, ["caption", str(tmp_path)])
    assert result.exit_code == 0
    output_path = tmp_path / "captions.json"
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data == {"one.png": "caption-one", "two.jpg": "caption-two"}
    assert "captions.json" in result.stdout


def test_caption_command_directory_stream_not_supported(tmp_path):
    (tmp_path / "one.png").write_bytes(b"image-one")

    result = runner.invoke(app, ["caption", str(tmp_path), "--stream"])
    assert result.exit_code != 0
    assert "Streaming output is not supported" in result.stdout


def test_ocr_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")

    def _fake_ocr(data, *, prompt=None):
        if data == b"image-one":
            return _StubResult("ocr-one")
        if data == b"image-two":
            return _StubResult("ocr-two")
        raise AssertionError("unexpected payload")

    monkeypatch.setattr("perceptron.cli.ocr_image", _fake_ocr)

    result = runner.invoke(app, ["ocr", str(tmp_path), "--prompt", "read everything"])
    assert result.exit_code == 0
    data = json.loads((tmp_path / "ocr.json").read_text())
    assert data == {"one.png": "ocr-one", "two.jpg": "ocr-two"}


def test_ocr_command(monkeypatch):
    monkeypatch.setattr("perceptron.cli.ocr_image", lambda *a, **k: _StubResult("ocr text"))
    result = runner.invoke(app, ["ocr", "https://example.com/img"])
    assert result.exit_code == 0
    assert "ocr text" in result.stdout


def test_detect_command(monkeypatch):
    res = _StubResult("detected")
    res.points = []
    monkeypatch.setattr("perceptron.cli.detect_image", lambda *a, **k: res)
    result = runner.invoke(app, ["detect", "/tmp/img.png", "--classes", "person,bike"])
    assert result.exit_code == 0
    assert "detected" in result.stdout


def test_detect_command_directory(monkeypatch, tmp_path):
    img1 = tmp_path / "one.png"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")

    def _fake_detect(data, *, classes=None):
        assert classes == ["person"]
        res = _StubResult("detected-one" if data == b"image-one" else "detected-two")
        res.points = [
            BoundingBox(
                top_left=SinglePoint(1, 2, mention="person"),
                bottom_right=SinglePoint(3, 4),
                mention="person",
            )
        ]
        return res

    monkeypatch.setattr("perceptron.cli.detect_image", _fake_detect)

    result = runner.invoke(app, ["detect", str(tmp_path), "--classes", "person"])
    assert result.exit_code == 0
    output_path = tmp_path / "detections.json"
    data = json.loads(output_path.read_text())
    assert set(data.keys()) == {"one.png", "two.jpg"}
    assert data["one.png"]["text"] == "detected-one"
    points = data["one.png"].get("points")
    assert points and points[0]["type"] == "box"
    assert points[0]["top_left"]["x"] == 1
    assert points[0]["top_left"]["mention"] == "person"


def test_chat_command(monkeypatch):
    class _StubClient:
        def generate(self, task, **kwargs):
            return {"text": "hello", "raw": {}}

    monkeypatch.setattr("perceptron.cli.Client", _StubClient)
    result = runner.invoke(app, ["chat", "hi there", "--system", "You are kind."])
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_config_command():
    result = runner.invoke(app, ["config", "--provider", "fal", "--api-key", "abc"])
    assert result.exit_code == 0
    assert "PERCEPTRON_PROVIDER=fal" in result.stdout
    assert "PERCEPTRON_API_KEY=abc" in result.stdout
