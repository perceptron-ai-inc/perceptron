"""DSL tag anchors (`image=`/`asset=`/`asset_idx=`, the shared asset ledger) and coordinate validation on the
normalized 0-1000 grid."""

from __future__ import annotations

import asyncio
import json
import warnings

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, async_perceive, config, detect, inspect_task, perceive
from perceptron import client as client_mod
from perceptron.annotations import annotate_image
from perceptron.dsl.nodes import (
    agent,
    audio,
    box,
    image,
    point,
    polygon,
    text,
    tool_result,
    video,
    video_frames,
)
from perceptron.dsl.perceive import _asset_ledger, _compile
from perceptron.errors import (
    ANCHOR_AMBIGUOUS,
    ANCHOR_MISSING,
    ANCHOR_UNKNOWN,
    BOUNDS_OUT_OF_RANGE,
    INVALID_PARAMETER,
    INVALID_POLYGON,
    AnchorError,
    AuthError,
    BadRequestError,
    ExpectationError,
)
from perceptron.pointing.types import bbox

CALL = {"id": "call_1", "type": "function", "function": {"name": "grab", "arguments": "{}"}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _tags(*nodes, strict=False):
    """The tag markup written for ``nodes`` (in order), and the compile issues."""
    seq = nodes[0] if len(nodes) == 1 else sum(nodes[1:], nodes[0])
    task, issues = _compile(seq, expects=None, strict=strict)
    tags = [e["content"] for e in task["content"] if e["type"] == "text" and e["content"].startswith("<")]
    return tags, [issue["code"] for issue in issues]


def _img():
    return image(PNG_BYTES)


# ---------------------------------------------------------------------------
# When asset_idx is written
# ---------------------------------------------------------------------------


def test_single_asset_markup_is_unchanged():
    im = _img()
    for nodes in (
        (im, box(1, 2, 3, 4, image=im, mention="cup")),
        (im, box(1, 2, 3, 4, asset=im, mention="cup")),
        (im, box(1, 2, 3, 4, mention="cup")),
        (box(1, 2, 3, 4, mention="cup"), im),  # the only asset may come after the tag
    ):
        assert _tags(*nodes) == (['<point_box mention="cup"> (1,2) (3,4) </point_box>'], [])


def test_temporal_tags_write_explicit_seconds():
    clip = video("https://x/v.mp4")
    assert _tags(clip, point(1, 2, t=1.5), point(3, 4, t=2)) == (
        ['<point t="1.5 seconds"> (1,2) </point>', '<point t="2.0 seconds"> (3,4) </point>'],
        [],
    )


def test_temporal_tags_keep_frame_accurate_times():
    clip = video("https://x/v.mp4")
    assert _tags(clip, point(1, 2, t=0.033), box(1, 2, 3, 4, t=1 / 30), polygon([(1, 1), (2, 1), (2, 2)], t=0.15)) == (
        [
            '<point t="0.033 seconds"> (1,2) </point>',
            '<point_box t="0.033333 seconds"> (1,2) (3,4) </point_box>',
            '<polygon t="0.15 seconds"> (1,1) (2,1) (2,2) </polygon>',
        ],
        [],
    )


def test_multi_asset_prompts_write_the_anchor_index():
    ref, tgt = _img(), _img()
    tags, issues = _tags(
        ref,
        box(100, 150, 300, 350, image=ref, mention="target object"),
        tgt,
        text("Find it in asset 1."),
        box(10, 10, 20, 20, asset=tgt, t=0.5),
    )
    assert tags == [
        '<point_box mention="target object" asset_idx="0"> (100,150) (300,350) </point_box>',
        '<point_box t="0.5 seconds" asset_idx="1"> (10,10) (20,20) </point_box>',
    ]
    assert issues == []


def test_anchor_to_an_asset_after_the_tag():
    a, b = _img(), _img()
    assert _tags(a, point(1, 2, image=b), b) == (['<point asset_idx="1"> (1,2) </point>'], [])


def test_audio_video_and_frames_each_take_one_index():
    clip = video_frames([("https://x/f0.jpg", 0), ("https://x/f1.jpg", 40), ("https://x/f2.jpg", 80)])
    sound, im, film = audio("https://x/a.wav"), _img(), video("https://x/v.mp4")

    tags, issues = _tags(
        sound, clip, im, film, point(1, 2, image=im), point(3, 4, asset=clip), point(5, 6, asset=sound)
    )

    assert tags == [
        '<point asset_idx="2"> (1,2) </point>',
        '<point asset_idx="1"> (3,4) </point>',
        '<point asset_idx="0"> (5,6) </point>',
    ]
    assert issues == []


def test_tool_result_images_take_an_index():
    a, returned, c = _img(), _img(), image("https://x/c.png")

    tags, issues = _tags(
        a,
        text("Grab the photo."),
        agent(None, tool_calls=[CALL]),
        tool_result("call_1", "here it is", returned),
        c,
        point(1, 2, asset=c),
        point(3, 4, image=returned),
    )

    assert tags == ['<point asset_idx="2"> (1,2) </point>', '<point asset_idx="1"> (3,4) </point>']
    assert issues == []


def test_raw_asset_idx_is_always_written():
    im = _img()
    assert _tags(im, point(1, 2, asset_idx=0)) == (['<point asset_idx="0"> (1,2) </point>'], [])
    assert _tags(im, _img(), point(1, 2, asset_idx=1)) == (['<point asset_idx="1"> (1,2) </point>'], [])
    # Out of range: still written, and reported.
    assert _tags(im, point(1, 2, asset_idx=1)) == (['<point asset_idx="1"> (1,2) </point>'], ["anchor_unknown"])
    with pytest.raises(AnchorError) as excinfo:
        _tags(im, point(1, 2, asset_idx=1), strict=True)
    assert excinfo.value.code == "anchor_unknown"


def test_explicit_zero_is_written_in_multi_asset_prompts():
    a, b = _img(), _img()
    assert _tags(a, b, point(1, 2, image=a)) == (['<point asset_idx="0"> (1,2) </point>'], [])


def test_the_request_carries_the_selector(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))
    a, b = _img(), _img()

    sound = audio("https://x/a.wav")
    res = perceive(
        a + sound + b + text("Same object?") + box(1, 2, 3, 4, image=b), provider="perceptron", expects="box"
    )

    parts = http.last_body["messages"][-1]["content"]
    assert parts[-1] == {"type": "text", "text": '<point_box asset_idx="2"> (1,2) (3,4) </point_box>'}
    assert res.asset_count == 3  # the compile ledger and the lowered request agree


def test_raw_asset_idx_in_create_content_lists(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    Client().chat.completions.create(
        messages=[{"role": "user", "content": [_img(), _img(), point(1, 2, asset_idx=1), "What is here?"]}]
    )

    assert http.last_body["messages"][0]["content"][2] == {
        "type": "text",
        "text": '<point asset_idx="1"> (1,2) </point>',
    }


def test_detect_examples_do_not_gain_selectors(monkeypatch):
    seen = {}

    def _generate(self, task, **kwargs):
        seen["task"] = task
        return {"text": ""}

    monkeypatch.setattr(client_mod.Client, "generate", _generate)
    examples = [annotate_image(PNG_BYTES, [bbox(1, 2, 3, 4, mention="cup")]) for _ in range(2)]

    detect(image(PNG_BYTES), classes=["cup"], examples=examples, provider="perceptron")  # three images

    assistant = [e["content"] for e in seen["task"]["content"] if e.get("role") == "assistant"]
    assert len(assistant) == 2 and all("<point_box" in c and "asset_idx" not in c for c in assistant)


def test_detect_examples_keep_authored_times(monkeypatch):
    seen = {}

    def _generate(self, task, **kwargs):
        seen["task"] = task
        return {"text": ""}

    monkeypatch.setattr(client_mod.Client, "generate", _generate)
    waypoints = [{"x": 1, "y": 1, "t": 0.0}, {"x": 2, "y": 2, "t": 0.033}, {"x": 3, "y": 3, "t": 0.067}]
    example = annotate_image(PNG_BYTES, [{"type": "track", "mention": "cup", "points": waypoints}])

    detect(image(PNG_BYTES), classes=["cup"], examples=[example], provider="perceptron")

    (assistant,) = [e["content"] for e in seen["task"]["content"] if e.get("role") == "assistant"]
    assert (
        '<track mention="cup"> <point t="0.0 seconds"> (1,1) </point> <point t="0.033 seconds"> (2,2) </point> '
        '<point t="0.067 seconds"> (3,3) </point> </track>'
    ) in assistant


# ---------------------------------------------------------------------------
# Anchoring issues
# ---------------------------------------------------------------------------


def test_missing_anchor():
    assert _tags(text("Mark it"), point(1, 2)) == (["<point> (1,2) </point>"], ["anchor_missing"])
    assert _tags(_img(), _img(), point(1, 2)) == (["<point> (1,2) </point>"], ["anchor_missing"])

    _, issues = _compile(text("Mark it") + point(1, 2), expects=None, strict=False)
    assert issues[0]["message"] == "Tag has no media asset to refer to"
    _, issues = _compile(_img() + _img() + point(1, 2), expects=None, strict=False)
    assert issues[0]["message"] == "Tag missing image=/asset= in a multi-asset prompt"

    with pytest.raises(AnchorError) as excinfo:
        _tags(_img(), _img(), point(1, 2), strict=True)
    assert excinfo.value.code == "anchor_missing"


def test_unknown_anchor():
    stray = image("/nonexistent/never-read.png")  # not in the prompt, so never read (no pixel lookup)

    assert _tags(_img(), point(1, 2, image=stray)) == (["<point> (1,2) </point>"], ["anchor_unknown"])
    assert _tags(_img(), _img(), point(1, 2, asset=stray)) == (["<point> (1,2) </point>"], ["anchor_unknown"])
    # Equal but distinct nodes do not match: anchors are looked up by identity.
    assert _tags(image("https://x/a.png"), point(1, 2, image=image("https://x/a.png")))[1] == ["anchor_unknown"]
    with pytest.raises(AnchorError) as excinfo:
        _tags(_img(), point(1, 2, image=stray), strict=True)
    assert excinfo.value.code == "anchor_unknown"


def test_ambiguous_anchor_uses_the_nearest_earlier_use():
    im, other = _img(), _img()

    # Latest use before the tag.
    assert _tags(im, other, im, point(1, 2, image=im)) == (
        ['<point asset_idx="2"> (1,2) </point>'],
        ["anchor_ambiguous"],
    )
    assert _tags(im, im, other, point(1, 2, image=im)) == (
        ['<point asset_idx="1"> (1,2) </point>'],
        ["anchor_ambiguous"],
    )
    # No use before the tag: the first one after it.
    assert _tags(point(1, 2, image=im), other, im, im) == (
        ['<point asset_idx="1"> (1,2) </point>'],
        ["anchor_ambiguous"],
    )
    with pytest.raises(AnchorError) as excinfo:
        _tags(im, im, point(1, 2, image=im), strict=True)
    assert excinfo.value.code == "anchor_ambiguous"


def test_non_media_anchor():
    assert _tags(_img(), point(1, 2, image=text("x"))) == (["<point> (1,2) </point>"], ["anchor_missing"])


def test_anchor_argument_errors():
    im = _img()
    with pytest.raises(TypeError, match=r"^point\(\) takes image= or asset=, not both$"):
        point(1, 2, image=im, asset=im)
    with pytest.raises(TypeError, match=r"^box\(\) takes image=/asset= or asset_idx=, not both$"):
        box(1, 2, 3, 4, asset=im, asset_idx=0)
    for bad in (-1, True, 1.0, "0"):
        with pytest.raises(BadRequestError) as excinfo:
            polygon([(1, 1), (2, 2), (3, 1)], asset_idx=bad)
        assert excinfo.value.code == INVALID_PARAMETER and excinfo.value.param == "asset_idx"


@pytest.mark.parametrize("bad", [-1, -0.5, float("nan"), float("inf"), True, "abc", [1.5]])
def test_invalid_times_raise(bad):
    for make in (
        lambda: point(1, 2, t=bad),
        lambda: box(1, 2, 3, 4, t=bad),
        lambda: polygon([(1, 1), (2, 2), (3, 1)], t=bad),
    ):
        with pytest.raises(BadRequestError, match=r"t must be a finite number of seconds >= 0") as excinfo:
            make()
        assert excinfo.value.code == INVALID_PARAMETER and excinfo.value.param == "t"


def test_inspect_task_reports_anchor_issues():
    @perceive(expects="point")
    def prompt():
        a, b = _img(), _img()
        return a + b + point(1, 2) + point(3, 4, image=b)

    task, issues = inspect_task(prompt)

    assert [issue["code"] for issue in issues] == ["anchor_missing"]
    tags = [e["content"] for e in task["content"] if e["type"] == "text"]
    assert tags == ["<point> (1,2) </point>", '<point asset_idx="1"> (3,4) </point>']


def test_streamed_final_result_carries_the_compile_issues(monkeypatch):
    # The non-stream result lists compile issues first in `errors`; so does the final event of every stream.
    def handler(request):
        if json.loads(request.content).get("stream"):
            return sse_response([chunk({"content": "Done."}), chunk({}, finish_reason="stop")])
        return json_response(completion("Done."))

    install(monkeypatch, handler)
    a, b = _img(), _img()
    nodes = a + b + point(10, 20) + text("What is marked?")

    result = perceive(nodes, provider="perceptron")
    events = list(perceive(nodes, provider="perceptron", stream=True))

    @async_perceive(provider="perceptron", stream=True)
    def ask():
        return nodes

    async def _collect():
        return [event async for event in ask()]

    async_events = asyncio.run(_collect())

    assert [e["code"] for e in result.errors] == ["anchor_missing"]
    for stream_events in (events, async_events):
        assert stream_events[-1]["type"] == "final"
        assert stream_events[-1]["result"]["errors"] == result.errors


def test_a_key_set_in_code_without_a_provider_uses_the_perceptron_api(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_API_KEY")
    http = install(monkeypatch, lambda request: json_response(completion()))

    with config(api_key="k"):
        perceive(text("hi"))

    assert str(http.last.url) == "https://api.perceptron.inc/v1/chat/completions"
    assert http.last.headers["authorization"] == "Bearer k"
    assert http.last_body["model"] == "perceptron-mk1.5"


def test_no_key_raises_credentials_missing_before_any_request(monkeypatch):
    monkeypatch.delenv("PERCEPTRON_API_KEY")
    http = install(monkeypatch, lambda request: json_response(completion()))

    with pytest.raises(AuthError) as excinfo:
        perceive(text("hi"))

    assert excinfo.value.code == "credentials_missing"
    assert "No API key for provider 'perceptron'. Set PERCEPTRON_API_KEY or configure(api_key=...)." in str(
        excinfo.value
    )
    assert http.requests == []


# ---------------------------------------------------------------------------
# Coordinates: integers on the normalized 0-1000 grid, independent of pixel size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "markup"),
    [
        (point(0, 0), "<point> (0,0) </point>"),
        (point(1000, 1000), "<point> (1000,1000) </point>"),
        (point(500.0, 20), "<point> (500,20) </point>"),  # integral floats become ints
        (box(0, 0, 1000, 1000), "<point_box> (0,0) (1000,1000) </point_box>"),
        (box(5, 5, 5, 5), "<point_box> (5,5) (5,5) </point_box>"),  # degenerate boxes are allowed
        (polygon([(0, 0), (1000, 0), (500, 1000.0)]), "<polygon> (0,0) (1000,0) (500,1000) </polygon>"),
    ],
)
def test_valid_coordinates(tag, markup):
    assert _tags(_img(), tag, strict=True) == ([markup], [])


@pytest.mark.parametrize(
    ("tag", "codes"),
    [
        (point(-1, 0), ["bounds_out_of_range"]),
        (point(1001, 0), ["bounds_out_of_range"]),
        (point(0, 1001), ["bounds_out_of_range"]),
        (point(10.5, 3), ["bounds_out_of_range"]),
        (point(True, 0), ["bounds_out_of_range"]),
        (point("5", 0), ["bounds_out_of_range"]),
        (point(float("nan"), 0), ["bounds_out_of_range"]),
        (box(0, 0, 1001, 10), ["bounds_out_of_range"]),
        (box(800, 800, 100, 100), ["bounds_out_of_range"]),  # corners out of order
        (box(0, 10, 10, 5), ["bounds_out_of_range"]),
        (box(-1, 0, 2000, 5), ["bounds_out_of_range", "bounds_out_of_range"]),
        (polygon([(1, 1), (2000, 2), (3, 1)]), ["bounds_out_of_range"]),
        (polygon([(1, 1), (2, 2)]), ["invalid_polygon"]),
        (polygon([(1, 1), (-2, 2)]), ["invalid_polygon", "bounds_out_of_range"]),
    ],
)
def test_invalid_coordinates(tag, codes):
    assert _tags(_img(), tag)[1] == codes
    with pytest.raises(ExpectationError) as excinfo:
        _tags(_img(), tag, strict=True)
    assert excinfo.value.code == codes[0]


def test_coordinate_issue_messages():
    _, issues = _compile(_img() + point(1001, 5) + box(800, 800, 100, 100), expects=None, strict=False)
    assert [issue["message"] for issue in issues] == [
        "point (1001,5) outside the 0-1000 normalized grid (coordinates are integers 0-1000)",
        "box (800,800) (100,100) needs x1 <= x2 and y1 <= y2 (top-left corner first)",
    ]


# ---------------------------------------------------------------------------
# Tags in chat.completions.create() message content: one asset_idx space for the whole request
# ---------------------------------------------------------------------------


def _create(monkeypatch, *messages):
    http = install(monkeypatch, lambda request: json_response(completion()))
    result = Client().chat.completions.create(messages=list(messages))
    return http.last_body["messages"], result


def _texts(message):
    return [part["text"] for part in message["content"] if part["type"] == "text"]


def test_create_anchors_tags_given_as_their_own_items(monkeypatch):
    x, y = _img(), image("https://x/y.png")

    sent, _ = _create(monkeypatch, {"role": "user", "content": [x, y, point(1, 2, image=x), box(1, 2, 3, 4, asset=y)]})

    assert _texts(sent[0]) == [
        '<point asset_idx="0"> (1,2) </point>',
        '<point_box asset_idx="1"> (1,2) (3,4) </point_box>',
    ]


def test_create_counts_the_media_of_every_message(monkeypatch):
    earlier, x, y = image("https://x/a.png"), _img(), image("https://x/y.png")

    sent, result = _create(
        monkeypatch,
        {"role": "user", "content": [earlier, "Describe it."]},
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": [x + y + point(1, 2, image=x) + point(3, 4, image=y)]},
    )

    assert _texts(sent[2]) == ['<point asset_idx="1"> (1,2) </point>', '<point asset_idx="2"> (3,4) </point>']
    assert result.asset_count == 3


def test_create_anchors_across_messages_and_counts_part_dicts(monkeypatch):
    x = _img()
    url_part = {"type": "image_url", "image_url": {"url": "https://x/a.png"}}

    sent, result = _create(
        monkeypatch,
        {"role": "user", "content": (url_part, {"type": "text", "text": "The shelf."})},  # sent as given
        {"role": "user", "content": [x, "Which one shows the cup?"]},
        {"role": "assistant", "content": "The second one."},
        {"role": "user", "content": ["Is this it?", box(1, 2, 3, 4, image=x, mention="cup")]},
    )

    assert _texts(sent[3])[1] == '<point_box mention="cup" asset_idx="1"> (1,2) (3,4) </point_box>'
    assert result.asset_count == 2


def test_create_single_asset_requests_keep_their_markup(monkeypatch):
    x = _img()

    sent, _ = _create(
        monkeypatch,
        {"role": "user", "content": [x, "Find the cup."]},
        {"role": "assistant", "content": "Found it."},
        {"role": "user", "content": ["Is this it?", box(1, 2, 3, 4, image=x, mention="cup"), point(5, 6)]},
    )

    assert _texts(sent[2])[1:] == ['<point_box mention="cup"> (1,2) (3,4) </point_box>', "<point> (5,6) </point>"]


def _unanchored():
    return [_img(), _img(), point(1, 2)]


def _stray_anchor():
    return [_img(), point(1, 2, image=image("https://x/elsewhere.png"))]


def _reused_node_off_the_grid():
    x = _img()
    return [x, x, point(5000, 1, image=x)]  # anchor_ambiguous, which only warns, then bounds_out_of_range


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (_unanchored, ANCHOR_MISSING),
        (lambda: ["No media here.", point(1, 2)], ANCHOR_MISSING),
        (_stray_anchor, ANCHOR_UNKNOWN),
        (lambda: [_img(), point(1, 2, asset_idx=1)], ANCHOR_UNKNOWN),
        (_reused_node_off_the_grid, BOUNDS_OUT_OF_RANGE),
        (lambda: [_img(), point(5000, -3)], BOUNDS_OUT_OF_RANGE),
        (lambda: [_img(), box(800, 800, 100, 100)], BOUNDS_OUT_OF_RANGE),
        (lambda: [_img(), polygon([(1, 1), (2, 2)])], INVALID_POLYGON),
    ],
)
def test_create_raises_for_tag_issues(monkeypatch, content, code):
    http = install(monkeypatch, lambda request: json_response(completion()))
    items = content()

    with pytest.raises(BadRequestError) as excinfo:
        Client().chat.completions.create(messages=[{"role": "user", "content": items}])

    where = f"messages[0].content[{len(items) - 1}]"
    assert excinfo.value.code == code
    assert excinfo.value.param == where
    assert str(excinfo.value).startswith(f"{where}: ")
    assert not http.requests


def _replayed_conversation(x):
    """A conversation replayed with the same image node in two turns, each with a tag anchored to it."""
    return [
        {"role": "user", "content": [x, "Find the cup.", box(1, 2, 3, 4, image=x, mention="cup")]},
        {"role": "assistant", "content": "Found it."},
        {"role": "user", "content": [x, "Is it still here?", point(5, 6, image=x)]},
    ]


_AMBIGUOUS_MESSAGE = "image=/asset= references a media node used 2 times in this prompt; anchored to asset_idx {}"


def _ambiguous_warnings(caught):
    return [(w.category, str(w.message), w.filename) for w in caught if "used 2 times" in str(w.message)]


def test_create_warns_for_a_reused_media_node_and_sends_the_request(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Client().chat.completions.create(messages=_replayed_conversation(_img()))

    # Each tag names the latest use of the node before it.
    assert _ambiguous_warnings(caught) == [
        (UserWarning, f"messages[0].content[2]: {_AMBIGUOUS_MESSAGE.format(0)}", __file__),
        (UserWarning, f"messages[2].content[2]: {_AMBIGUOUS_MESSAGE.format(1)}", __file__),
    ]
    sent = http.last_body["messages"]
    assert _texts(sent[0])[1] == '<point_box mention="cup" asset_idx="0"> (1,2) (3,4) </point_box>'
    assert _texts(sent[2])[1] == '<point asset_idx="1"> (5,6) </point>'


def test_async_create_warns_for_a_reused_media_node_and_sends_the_request(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))

    async def _run():
        await AsyncClient().chat.completions.create(messages=_replayed_conversation(_img()))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(_run())

    assert [(category, filename) for category, _, filename in _ambiguous_warnings(caught)] == [
        (UserWarning, __file__),
        (UserWarning, __file__),
    ]
    sent = json.loads(http.last.content)["messages"]
    assert _texts(sent[0])[1] == '<point_box mention="cup" asset_idx="0"> (1,2) (3,4) </point_box>'
    assert _texts(sent[2])[1] == '<point asset_idx="1"> (5,6) </point>'


def test_create_tag_issues_inside_sequences_name_the_item(monkeypatch):
    http = install(monkeypatch, lambda request: json_response(completion()))
    x = _img()

    with pytest.raises(BadRequestError) as excinfo:
        Client().chat.completions.create(
            messages=[
                {"role": "user", "content": [x, "Look."]},
                {"role": "user", "content": ["Here:", _img() + point(1, 2)]},  # two assets in the request
            ]
        )

    assert (excinfo.value.code, excinfo.value.param) == (ANCHOR_MISSING, "messages[1].content[1]")
    assert not http.requests


def test_compile_against_a_request_ledger():
    x, other = _img(), {"type": "image_url", "image_url": {"url": "https://x/o.png"}}
    ledger = _asset_ledger([other, "text", x, x + text("again")])  # x is asset 1 and asset 2
    assert ledger.count == 3

    def tag(first_asset):
        task, issues = _compile(
            point(1, 2, image=x), expects=None, strict=False, ledger=ledger, first_asset=first_asset
        )
        return task["content"][0]["content"], [issue["code"] for issue in issues]

    # The nearest use before the tag's place in the request, else the first use after it.
    assert tag(3) == ('<point asset_idx="2"> (1,2) </point>', [ANCHOR_AMBIGUOUS])
    assert tag(2) == ('<point asset_idx="1"> (1,2) </point>', [ANCHOR_AMBIGUOUS])
    assert tag(0) == ('<point asset_idx="1"> (1,2) </point>', [ANCHOR_AMBIGUOUS])
