import json

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from cookbook.utils import cookbook_asset
from perceptron import annotate_image, detect, detect_from_coco, image
from perceptron import config as cfg
from perceptron.highlevel import CocoDetectResult
from perceptron.pointing.types import SinglePoint, bbox, collection


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def http(monkeypatch):
    """The API behind `httpx.MockTransport`, answering with an empty completion."""
    return install(monkeypatch, lambda request: json_response(completion("")))


def _texts(http, role: str) -> list[str]:
    """The text of each ``role`` message in the last request (string content, or its text parts joined)."""
    texts = []
    for message in http.last_body["messages"]:
        if message["role"] != role:
            continue
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.append("".join(part["text"] for part in content if part["type"] == "text"))
    return texts


def test_detect_compile_only(http):
    with cfg(api_key="test-key", provider="fal"):
        res = detect(image(PNG_BYTES), classes=["person"], max_tokens=16)
    assert res.raw == completion("")
    body = http.last_body
    roles = [message["role"] for message in body["messages"]]
    assert roles and roles[0] == "system"
    assert "person" in body["messages"][0]["content"]
    assert body["max_completion_tokens"] == 16
    assert res.errors == []


def test_detect_with_examples(http):
    example = annotate_image(
        PNG_BYTES,
        [bbox(1, 2, 3, 4, mention="car")],
    )
    with cfg(api_key="test-key", provider="fal"):
        detect(image(PNG_BYTES), classes=["car"], examples=[example])
    # Should include example turns before target image
    roles = [message["role"] for message in http.last_body["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assistants = _texts(http, "assistant")
    assert assistants and "<point_box" in assistants[0]


def test_detect_with_collection_examples(http):
    example = annotate_image(
        PNG_BYTES,
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

    with cfg(api_key="test-key", provider="fal"):
        detect(image(PNG_BYTES), classes=["group"], examples=[example])
    assistants = _texts(http, "assistant")
    assert assistants and "<collection" in assistants[0]


def test_detect_canonicalizes_collection_order(http):
    example = annotate_image(
        PNG_BYTES,
        {
            "car": [bbox(1, 2, 3, 4, mention="car")],
            "person": [bbox(5, 6, 7, 8, mention="person")],
        },
    )

    with cfg(api_key="test-key", provider="fal"):
        detect(image(PNG_BYTES), classes=["person", "car"], examples=[example])

    content = _texts(http, "assistant")[0]
    assert content.index('mention="person"') < content.index('mention="car"')


def test_detect_sorts_collection_children(http):
    example = annotate_image(
        PNG_BYTES,
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

    with cfg(api_key="test-key", provider="fal"):
        detect(image(PNG_BYTES), classes=["group"], examples=[example])

    content = _texts(http, "assistant")[0]
    first_idx = content.index("(10,20) (30,40)")
    second_idx = content.index("(50,60) (70,80)")
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


def test_prompt_collection_canonicalization(http):
    example = {
        "image": PNG_BYTES,
        "collections": [collection([bbox(5, 5, 10, 10)], mention="group")],
        "prompt": 'context <collection mention="group"> <point_box> (20,20) (30,30) </point_box> <point_box> (10,10) (15,15) </point_box> </collection>',
    }

    with cfg(api_key="test-key", provider="fal"):
        detect(image(PNG_BYTES), classes=["group"], examples=[example])

    prompt_text = next(text for text in _texts(http, "user") if "context" in text)
    assert prompt_text.index("(10,10) (15,15)") < prompt_text.index("(20,20) (30,30)")


def test_detect_stream(monkeypatch):
    http = install(
        monkeypatch, lambda request: sse_response([chunk({"content": "hello"}), chunk({}, finish_reason="stop")])
    )

    with cfg(api_key="test-key", provider="fal"):
        events = list(detect(image(PNG_BYTES), classes=None, stream=True))
    assert http.last_body["stream"] is True
    assert events[0]["type"] == "text.delta"
    assert events[0]["chunk"] == "hello"
    assert events[-1]["type"] == "final"
    assert events[-1]["result"]["text"] == "hello"
    assert [client.is_closed for client in http.clients] == [True]  # the stream's end closed the call's client


def test_detect_flattens_collection_response(monkeypatch):
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
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test-key")
    http = install(monkeypatch, lambda request: json_response(payload))

    with cfg(provider="fal", base_url="https://unit.test", api_key="test-key"):
        res = detect(image(PNG_BYTES), classes=["dog"])

    assert [(request.method, str(request.url)) for request in http.requests] == [
        ("POST", "https://unit.test/perceptron/isaac-01/openai/v1/chat/completions")
    ]
    assert http.last.headers["authorization"] == "Key test-key"
    assert "stream" not in http.last_body  # a plain completion, not a stream
    assert res.text and "<collection" in res.text
    assert res.boxes and len(res.boxes) == 2
    assert res.boxes[0].mention == "dog"
    assert res.boxes[1].mention == "named"


def test_detect_flattens_track_response_into_timed_boxes(monkeypatch):
    content = (
        '<collection mention="player" asset_idx="0"> <track> '
        '<point_box t="0.0 seconds"> (10,20) (30,40) </point_box> '
        '<point_box t="0.5 seconds"> (12,20) (32,40) </point_box> </track> </collection>'
    )
    install(monkeypatch, lambda request: json_response({"choices": [{"message": {"content": content}}]}))

    with cfg(provider="fal", base_url="https://unit.test", api_key="test-key"):
        res = detect(image(PNG_BYTES), classes=["player"])

    assert [(b.mention, b.t, b.asset_idx) for b in res.boxes] == [("player", 0.0, 0), ("player", 0.5, 0)]


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

    def _fake_detect(image_obj, *, classes, stream=False, examples=None, **kwargs):
        assert classes == ["cell"]
        assert stream is False
        assert image_obj.obj == b"image-one"
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

    def _fake_detect(image_obj, *, classes, stream=False, examples=None, **kwargs):
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


def test_examples_icl_detection_sequence(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion("ok")))

    cat_example = annotate_image(
        str(cookbook_asset("in-context-learning", "multi", "classA.jpg")),
        {"classA": [bbox(316, 136, 703, 906, mention="classA")]},
    )

    dog_example = annotate_image(
        str(cookbook_asset("in-context-learning", "multi", "classB.webp")),
        {"classB": [bbox(161, 48, 666, 980, mention="classB")]},
    )

    with cfg(provider="perceptron", api_key="test-key"):
        res = detect(
            image(str(cookbook_asset("in-context-learning", "multi", "cat_dog_input.png"))),
            classes=["classA", "classB"],
            examples=[cat_example, dog_example],
        )

    assert res.text == "ok"
    assert [str(request.url) for request in http.requests] == ["https://api.perceptron.inc/v1/chat/completions"]
    assistant_messages = _texts(http, "assistant")
    assert any("classA" in msg and "(316,136) (703,906)" in msg for msg in assistant_messages)
    assert any("classB" in msg and "(161,48) (666,980)" in msg for msg in assistant_messages)
    system_msgs = _texts(http, "system")
    # The categories prompt should appear in some system message; on the
    # perceptron provider the `<hint>BOX</hint>` hint is sent with the system
    # role too, so check across all system messages.
    assert any("classA" in msg and "classB" in msg for msg in system_msgs)
    image_parts = [
        part
        for message in http.last_body["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert len(image_parts) == 3
