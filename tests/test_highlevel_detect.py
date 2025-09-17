import json

from perceptron import detect, annotate_image, detect_from_coco
from perceptron.pointing.types import bbox, SinglePoint, collection
from perceptron import config as cfg
from perceptron.highlevel import CocoDetectResult


class _StubClient:
    def generate(self, task, **kwargs):
        return {"text": "", "raw": {"choices": []}}

    def stream(self, task, **kwargs):
        yield {"type": "text.delta", "chunk": "hello"}
        yield {"type": "final", "result": {"text": "done", "points": [], "errors": []}}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload


def test_detect_compile_only(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _StubClient.generate)

    res = detect(b"\x89PNG\r\n\x1a\n" + b"0" * 10, classes=["person"], max_tokens=16)
    assert res.raw and isinstance(res.raw, dict)
    roles = [item.get("role") for item in res.raw.get("content", [])]
    assert roles and roles[0] == "system"
    assert any(err.get("code") == "credentials_missing" for err in res.errors)


def test_detect_with_examples(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _StubClient.generate)

    example = annotate_image(
        b"\x89PNG\r\n\x1a\n" + b"0" * 12,
        [bbox(1, 2, 3, 4, mention="car")],
    )
    res = detect(b"\x89PNG\r\n\x1a\n" + b"1" * 12, classes=["car"], examples=[example])
    content = res.raw.get("content", [])
    # Should include example turns before target image
    assistants = [item for item in content if item.get("role") == "assistant"]
    assert assistants and "<point_box" in assistants[0]["content"]


def test_detect_with_collection_examples(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "generate", _StubClient.generate)

    example = annotate_image(
        b"\x89PNG\r\n\x1a\n" + b"3" * 12,
        [
            collection(
                [
                    bbox(1, 2, 3, 4),
                    SinglePoint(5, 6),
                ],
                mention="group",
            )
        ],
    )

    res = detect(b"\x89PNG\r\n\x1a\n" + b"4" * 12, classes=["group"], examples=[example])
    content = res.raw.get("content", [])
    assistants = [item for item in content if item.get("role") == "assistant"]
    assert assistants and "<collection" in assistants[0]["content"]


def test_detect_canonicalizes_collection_order():
    example = annotate_image(
        b"img",
        {"car": [bbox(1, 2, 3, 4, mention="car")], "person": [bbox(5, 6, 7, 8, mention="person")]},
    )

    with cfg(provider=None):
        res = detect(b"target", classes=["person", "car"], examples=[example])

    assistant = next(item for item in res.raw["content"] if item.get("role") == "assistant")
    content = assistant["content"]
    assert content.index('mention="person"') < content.index('mention="car"')


def test_detect_sorts_collection_children():
    example = annotate_image(
        b"img",
        [
            collection(
                [
                    bbox(50, 60, 70, 80, mention="late"),
                    bbox(10, 20, 30, 40, mention="early"),
                ],
                mention="group",
            )
        ],
    )

    with cfg(provider=None):
        res = detect(b"target", classes=["group"], examples=[example])

    assistant = next(item for item in res.raw["content"] if item.get("role") == "assistant")
    content = assistant["content"]
    first_idx = content.index('(10,20) (30,40)')
    second_idx = content.index('(50,60) (70,80)')
    assert first_idx < second_idx


def test_annotate_image_sorts_annotations():
    example = annotate_image(
        b"img",
        [
            bbox(50, 60, 70, 80),
            bbox(10, 20, 30, 40),
            SinglePoint(5, 5),
            SinglePoint(1, 1),
        ],
    )

    boxes = example["boxes"]
    assert boxes[0].top_left.x == 10
    assert boxes[0].top_left.y == 20
    points = example["points"]
    assert (points[0].x, points[0].y) == (1, 1)


def test_annotate_image_sorts_mapping_collections():
    example = annotate_image(
        b"img",
        {
            "z": [bbox(10, 10, 20, 20)],
            "a": [bbox(1, 1, 5, 5)],
        },
    )

    collections = example["collections"]
    mentions = [coll.mention for coll in collections]
    assert mentions == ["a", "z"]


def test_prompt_collection_canonicalization():
    example = {
        "image": b"img",
        "collections": [collection([bbox(5, 5, 10, 10)], mention="group")],
        "prompt": "context <collection mention=\"group\"> <point_box> (20,20) (30,30) </point_box> <point_box> (10,10) (15,15) </point_box> </collection>",
    }

    with cfg(provider=None):
        res = detect(b"target", classes=["group"], examples=[example])

    assistant = next(item for item in res.raw["content"] if item.get("role") == "assistant")
    prompt_text = next(item for item in res.raw["content"] if item.get("role") == "user" and "context" in item.get("content", ""))["content"]
    assert prompt_text.index('(10,10) (15,15)') < prompt_text.index('(20,20) (30,30)')


def test_detect_stream(monkeypatch):
    from perceptron import client as client_mod

    monkeypatch.setattr(client_mod.Client, "stream", _StubClient.stream)

    events = list(detect(b"\x89PNG\r\n\x1a\n" + b"2" * 12, classes=None, stream=True))
    assert events[0]["type"] == "text.delta"
    assert events[-1]["type"] == "final"


def test_detect_flattens_collection_response(monkeypatch):
    from perceptron import client as client_mod, config as cfg

    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<collection mention="dog"> '
                        "<point_box> (10,20) (30,40) </point_box> "
                        '<point_box mention="named"> (1,2) (3,4) </point_box> '
                        "</collection>"
                    )
                }
            }
        ]
    }

    def _mock_post(url, headers=None, data=None, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")
    monkeypatch.setattr(client_mod.requests, "post", _mock_post)

    with cfg(provider="fal", base_url="https://unit.test", api_key="test-key"):
        res = detect(b"\x89PNG\r\n\x1a\n" + b"3" * 12, classes=["dog"])

    assert res.text and "<collection" in res.text
    assert res.points and len(res.points) == 2
    assert res.points[0].mention == "dog"
    assert res.points[1].mention == "named"


def test_detect_from_coco(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "train" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "img1.png"
    image_path.write_bytes(b"image-one")

    annotations = {
        "images": [{"id": 1, "file_name": "train/images/img1.png"}],
        "annotations": [],
        "categories": [{"id": 1, "name": "cell"}],
    }
    ann_path = dataset / "train" / "_annotations.coco.json"
    ann_path.write_text(json.dumps(annotations))

    class _StubResult:
        def __init__(self, text: str):
            self.text = text
            self.errors = []
            self.points = None
            self.raw = {"text": text}

    def _fake_detect(image_bytes, *, classes, stream=False, examples=None, **kwargs):
        assert classes == ["cell"]
        assert stream is False
        assert image_bytes == b"image-one"
        assert examples is None
        return _StubResult("detected")

    monkeypatch.setattr("perceptron.highlevel.detect", _fake_detect)

    results = detect_from_coco(dataset, split="train")
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, CocoDetectResult)
    assert result.image_path == image_path
    assert result.coco_image["id"] == 1
    assert result.result.text == "detected"


def test_detect_from_coco_shots(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "train" / "images"
    image_dir.mkdir(parents=True)
    img1 = image_dir / "img1.png"
    img2 = image_dir / "img2.png"
    img1.write_bytes(b"image-one")
    img2.write_bytes(b"image-two")

    annotations = {
        "images": [
            {"id": 1, "file_name": "train/images/img1.png"},
            {"id": 2, "file_name": "train/images/img2.png"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [1, 1, 3, 3]},
        ],
        "categories": [
            {"id": 1, "name": "cell"},
            {"id": 2, "name": "artifact"},
        ],
    }
    ann_path = dataset / "train" / "annotations_train.json"
    ann_path.write_text(json.dumps(annotations))

    captured_examples = []

    class _StubResult:
        def __init__(self):
            self.text = "detected"
            self.errors = []
            self.points = None
            self.raw = {"text": "detected"}

    def _fake_detect(image_bytes, *, classes, stream=False, examples=None, **kwargs):
        assert classes == ["cell", "artifact"]
        assert stream is False
        assert examples is not None
        captured_examples.append(examples)
        return _StubResult()

    monkeypatch.setattr("perceptron.highlevel.detect", _fake_detect)

    detect_from_coco(dataset, split="train", shots=2)

    assert captured_examples, "detect should receive examples"
    sample = captured_examples[0]
    assert len(sample) == 2
    mentions = {box.mention for example in sample for box in example.get("boxes", []) if getattr(box, "mention", None)}
    assert mentions == {"cell", "artifact"}
