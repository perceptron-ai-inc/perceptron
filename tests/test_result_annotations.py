"""Annotations in results and streams: flattened buckets with context, tracks, lenient/strict parsing, streamed
`points.delta` with context, and `asset_idx` resolution against the request's assets (DESIGN §8, §12.5, §12.7, §12.9)."""

import asyncio

import pytest
from _http_mock import chunk, completion, install, json_response, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, image, perceive, text
from perceptron import client as client_mod
from perceptron.chat import ChatCompletionMessage
from perceptron.dsl.perceive import PerceiveResult
from perceptron.errors import ParseError
from perceptron.pointing.parser import _scan_leaves, collect_annotations, scan_leaves
from perceptron.pointing.types import Clip, ClipTimestamp, Collection, Track, bbox, clip, pt

TASK = {"content": [{"type": "text", "role": "user", "content": "Find the players."}]}

# A collection holding two tracks and a direct child, then a top-level box.
TRACKS = (
    'Players: <collection mention="player" asset_idx="1">'
    ' <track> <point_box t="0.0 seconds"> (10,20) (30,40) </point_box>'
    ' <point_box t="0.5 seconds"> (12,20) (32,40) </point_box> </track>'
    ' <track mention="goalie"> <point_box t="0.0 seconds"> (50,60) (70,80) </point_box> </track>'
    " <point_box> (1,2) (3,4) </point_box>"
    ' </collection> and a ball <point_box mention="ball"> (5,6) (7,8) </point_box>.'
)
TRACKS_BOXES = [
    bbox(10, 20, 30, 40, mention="player", t=0.0, asset_idx=1),
    bbox(12, 20, 32, 40, mention="player", t=0.5, asset_idx=1),
    bbox(50, 60, 70, 80, mention="goalie", t=0.0, asset_idx=1),
    bbox(1, 2, 3, 4, mention="player", asset_idx=1),
    bbox(5, 6, 7, 8, mention="ball"),  # no selector in the markup: none is inferred
]
TRACKS_TRACKS = [
    Track([bbox(10, 20, 30, 40, t=0.0), bbox(12, 20, 32, 40, t=0.5)], "player", asset_idx=1),
    Track([bbox(50, 60, 70, 80, t=0.0)], "goalie", asset_idx=1),
]
MALFORMED = "A <point_box> (1,2) </point_box> and B <point_box> (3,4) (5,6) </point_box>"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _json(monkeypatch, content, **kwargs):
    return install(monkeypatch, lambda request: json_response(completion(content, **kwargs)))


def _stream_text(monkeypatch, pieces, *, finish_reason="stop"):
    events = [chunk({"content": piece}) for piece in pieces]
    if finish_reason is not None:
        events.append(chunk({}, finish_reason=finish_reason))
    return install(monkeypatch, lambda request: sse_response(events))


def _split(content, size):
    return [content[i : i + size] for i in range(0, len(content), size)]


def _client(**overrides):
    return Client(provider="perceptron", **overrides)


def _collect(async_iterable):
    async def _run():
        return [event async for event in async_iterable]

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# generate(): buckets, tracks, lenient and strict parsing
# ---------------------------------------------------------------------------


def test_generate_flattens_tracks_and_collections_with_their_context(monkeypatch):
    _json(monkeypatch, TRACKS)

    result = _client().generate(TASK, expects="box")

    assert result["boxes"] == TRACKS_BOXES
    assert result["tracks"] == TRACKS_TRACKS
    assert [segment["kind"] for segment in result["parsed"]] == ["text", "collection", "text", "box", "text"]
    assert result["errors"] == []
    assert "points" not in result and "polygons" not in result and "clips" not in result
    # The tree holds only the containers' asset_idx (DESIGN §12.10): no copied mention on waypoints.
    collection = result["parsed"][1]["value"]
    assert collection.points[0].points[0] == bbox(10, 20, 30, 40, t=0.0, asset_idx=1)


def test_tracks_hold_every_kind_while_the_bucket_holds_the_expected_one(monkeypatch):
    content = (
        '<track mention="ball"> <point t="0.0 seconds"> (1,2) </point> </track>'
        ' <track mention="car"> <point_box t="1.0 seconds"> (1,2) (3,4) </point_box> </track>'
    )
    _json(monkeypatch, content)

    result = _client().generate(TASK, expects="point")

    assert result["points"] == [pt(1, 2, mention="ball", t=0.0)]
    assert [(track.mention, len(track.points)) for track in result["tracks"]] == [("ball", 1), ("car", 1)]


def test_clip_results_inherit_collection_context_and_hold_every_track(monkeypatch):
    content = (
        '<collection mention="goals" asset_idx="0"> <clip t="1 seconds" />'
        ' <track> <point t="1.0 seconds"> (1,2) </point> </track> </collection>'
        ' <track mention="bad"> <point t="0 seconds"> (1,2) </point> <point_box t="1 seconds"> (1,2) (3,4) </point_box>'
        " </track>"
    )
    _json(monkeypatch, content)

    result = _client().generate(TASK, expects="clip")

    assert result["clips"] == [Clip(ClipTimestamp(at=1.0), "goals", asset_idx=0)]
    # DESIGN §8: every track for any structured expectation. The clip parse leaves spatial markup as text, so the
    # tracks come from a lenient parse whose problems (the mixed-geometry track) are not reported.
    assert result["tracks"] == [Track([pt(1, 2, t=1.0)], "goals", asset_idx=0)]
    assert [segment["kind"] for segment in result["parsed"]] == ["text", "clip", "text"]
    assert result["errors"] == []

    _stream_text(monkeypatch, _split(content, 6))
    final = list(_client().stream(TASK, expects="clip", parse_points=True))[-1]["result"]
    assert final["clips"] == result["clips"] and final["tracks"] == result["tracks"] and final["errors"] == []


def test_malformed_markup_keeps_the_valid_annotations(monkeypatch):
    _json(monkeypatch, MALFORMED)

    result = _client().generate(TASK, expects="box")

    assert result["text"] == MALFORMED
    assert result["boxes"] == [bbox(3, 4, 5, 6)]
    [error] = result["errors"]
    assert error["code"] == "invalid_box_coords"
    assert error["span"] == {"start": 2, "end": 32}


def test_truncated_container_is_kept_incomplete_and_reported(monkeypatch):
    cut = TRACKS[: TRACKS.index(" </collection>")]
    _json(monkeypatch, cut, finish_reason="length")

    result = _client().generate(TASK, expects="box")

    assert result["boxes"] == TRACKS_BOXES[:4]
    assert result["tracks"] == TRACKS_TRACKS
    collection = result["parsed"][1]["value"]
    assert isinstance(collection, Collection) and collection.complete is False
    assert [error["code"] for error in result["errors"]] == ["incomplete_annotation"]
    assert result["complete"] is False


def test_strict_generate_raises_parse_errors(monkeypatch):
    _json(monkeypatch, MALFORMED)
    with pytest.raises(ParseError) as excinfo:
        _client().generate(TASK, expects="box", strict=True)
    assert excinfo.value.code == "invalid_box_coords"

    _json(monkeypatch, TRACKS[: TRACKS.index(" </collection>")], finish_reason="length")
    with pytest.raises(ParseError) as excinfo:
        _client().generate(TASK, expects="box", strict=True)
    assert excinfo.value.code == "incomplete_annotation"

    _json(monkeypatch, TRACKS)
    assert _client().generate(TASK, expects="box", strict=True)["boxes"] == TRACKS_BOXES

    _json(monkeypatch, MALFORMED)
    with pytest.raises(ParseError):
        asyncio.run(AsyncClient(provider="perceptron").generate(TASK, expects="box", strict=True))


def test_strict_generate_error_carries_the_request_id_and_the_answer(monkeypatch):
    # Like the strict stream's error event: the x-trace-id (attribute and details) and the answer as `partial`.
    payload = completion(MALFORMED, reasoning="Looking.")
    install(monkeypatch, lambda request: json_response(payload, headers={"x-trace-id": "trace-s"}))
    expected_partial = {"text": MALFORMED, "reasoning": "Looking.", "tool_calls": None, "finish_reason": "stop"}

    with pytest.raises(ParseError) as excinfo:
        _client().generate(TASK, expects="box", strict=True)
    err = excinfo.value
    assert err.code == "invalid_box_coords"
    assert err.request_id == "trace-s" and err.details["request_id"] == "trace-s"
    assert err.details["span"] == {"start": 2, "end": 32}
    assert err.partial == expected_partial

    with pytest.raises(ParseError) as excinfo:
        asyncio.run(AsyncClient(provider="perceptron").generate(TASK, expects="box", strict=True))
    assert excinfo.value.request_id == "trace-s" and excinfo.value.partial == expected_partial

    with pytest.raises(ParseError) as excinfo:
        perceive(text("Find boxes"), expects="box", provider="perceptron", strict=True)
    assert excinfo.value.details["request_id"] == "trace-s" and excinfo.value.partial == expected_partial

    # Without the header there is no request id to report.
    _json(monkeypatch, MALFORMED)
    with pytest.raises(ParseError) as excinfo:
        _client().generate(TASK, expects="box", strict=True)
    assert excinfo.value.request_id is None and "request_id" not in excinfo.value.details


def test_perceive_strict_raises_and_lenient_maps_tracks(monkeypatch):
    _json(monkeypatch, MALFORMED)
    with pytest.raises(ParseError):
        perceive(text("Find boxes"), expects="box", provider="perceptron", strict=True)

    _json(monkeypatch, TRACKS)
    res = perceive(text("Find boxes"), expects="box", provider="perceptron")
    assert res.boxes == TRACKS_BOXES and res.tracks == TRACKS_TRACKS and res.errors == []


# ---------------------------------------------------------------------------
# stream(): points.delta with context, finalize, strict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 7])
def test_stream_emits_each_leaf_once_with_context_while_containers_are_open(monkeypatch, size):
    _stream_text(monkeypatch, _split(TRACKS, size))

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    answer = ""
    deltas = []
    for event in events:
        if event["type"] == "text.delta":
            answer += event["chunk"]
        elif event["type"] == "points.delta":
            deltas.append((answer, event))
    expected = scan_leaves(TRACKS, expects="box")
    assert [event["span"] for _, event in deltas] == [leaf["span"] for leaf in expected]
    assert [event["points"] for _, event in deltas] == [[leaf["value"]] for leaf in expected]
    assert [event["points"][0] for _, event in deltas] == TRACKS_BOXES
    assert [event["context"] for _, event in deltas] == [
        {"mention": "player", "t": 0.0, "asset_idx": 1, "container": "track"},
        {"mention": "player", "t": 0.5, "asset_idx": 1, "container": "track"},
        {"mention": "goalie", "t": 0.0, "asset_idx": 1, "container": "track"},
        {"mention": "player", "t": None, "asset_idx": 1, "container": "collection"},
        {"mention": "ball", "t": None, "asset_idx": None, "container": None},  # top level: nothing inherited
    ]
    # Each waypoint arrives as soon as it closes, before its track (and the collection) closes.
    first_answer, first = deltas[0]
    assert first["context"] == {"mention": "player", "t": 0.0, "asset_idx": 1, "container": "track"}
    assert "</track>" not in first_answer
    assert first["span"]["end"] <= len(first_answer) < first["span"]["end"] + size  # the chunk that closed it

    assert events[-1]["type"] == "final"
    result = events[-1]["result"]
    assert result["boxes"] == TRACKS_BOXES and result["tracks"] == TRACKS_TRACKS and result["errors"] == []


def test_stream_clip_deltas_carry_collection_context(monkeypatch):
    content = '<collection mention="goals" asset_idx="0"> <clip t="1 seconds" /> <clip t="2 seconds 3 seconds" />'
    _stream_text(monkeypatch, _split(content + " </collection>", 5))

    events = list(_client().stream(TASK, expects="clip", parse_points=True))

    deltas = [event for event in events if event["type"] == "points.delta"]
    assert [event["points"] for event in deltas] == [
        [clip(1.0, mention="goals", asset_idx=0)],
        [clip(2.0, 3.0, mention="goals", asset_idx=0)],
    ]
    assert all(
        event["context"] == {"mention": "goals", "t": None, "asset_idx": 0, "container": "collection"}
        for event in deltas
    )
    assert events[-1]["result"]["clips"] == [event["points"][0] for event in deltas]


def test_stream_unclosed_containers_are_reported_not_closed(monkeypatch):
    cut = TRACKS[: TRACKS.index(" </track> <track")]
    _stream_text(monkeypatch, _split(cut, 7), finish_reason="length")

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    assert [event["points"] for event in events if event["type"] == "points.delta"] == [
        [TRACKS_BOXES[0]],
        [TRACKS_BOXES[1]],
    ]
    assert events[-1]["type"] == "final"
    result = events[-1]["result"]
    assert result["text"] == cut
    assert result["boxes"] == TRACKS_BOXES[:2]
    assert [track.complete for track in result["tracks"]] == [False]
    assert result["parsed"][1]["value"].complete is False
    # One entry for the outermost unclosed container, spanning what arrived.
    assert result["errors"] == [
        {
            "code": "incomplete_annotation",
            "message": "<collection> is not closed (the output may be truncated); completed children were kept",
            "span": {"start": TRACKS.index("<collection"), "end": len(cut)},
        }
    ]


def test_stream_keeps_emitting_after_malformed_markup(monkeypatch):
    _stream_text(monkeypatch, _split(MALFORMED, 4))

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    assert [event["points"] for event in events if event["type"] == "points.delta"] == [[bbox(3, 4, 5, 6)]]
    result = events[-1]["result"]
    assert result["boxes"] == [bbox(3, 4, 5, 6)]
    assert [error["code"] for error in result["errors"]] == ["invalid_box_coords"]


@pytest.mark.parametrize(
    ("container", "code"),
    [
        ('<collection asset_idx="abc"> <point_box> (1,2) (3,4) </point_box> </collection>', "invalid_asset_idx"),
        ('<collection t="soon"> <point_box> (1,2) (3,4) </point_box> </collection>', "invalid_time"),
        (
            '<track asset_idx="2"> <point_box asset_idx="1" t="0 seconds"> (1,2) (3,4) </point_box> </track>',
            "invalid_track_asset",
        ),
        ("<collection> <point> (1,2) <point_box> (1,2) (3,4) </point_box> </collection>", "unclosed_tag"),
    ],
)
def test_stream_skips_leaves_of_an_invalid_container(monkeypatch, container, code):
    # Once a container is invalid (the collector drops it whole) its leaves are not streamed.
    content = container + " <point_box> (5,6) (7,8) </point_box>"
    _stream_text(monkeypatch, _split(content, 1))

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    deltas = [event["points"] for event in events if event["type"] == "points.delta"]
    assert deltas == [[bbox(5, 6, 7, 8)]]
    result = events[-1]["result"]
    assert result["boxes"] == [bbox(5, 6, 7, 8)]
    assert [error["code"] for error in result["errors"]] == [code]


def test_stream_deltas_are_provisional_when_later_markup_invalidates_the_container(monkeypatch):
    # A leaf emitted while its collection was valid cannot be retracted; `final` is authoritative.
    content = (
        '<collection mention="x"> <point_box> (1,2) (3,4) </point_box>'
        " <collection> <point_box> (5,6) (7,8) </point_box> </collection> </collection>"
    )
    _stream_text(monkeypatch, _split(content, 1))

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    assert [event["points"] for event in events if event["type"] == "points.delta"] == [[bbox(1, 2, 3, 4, mention="x")]]
    result = events[-1]["result"]
    assert result["boxes"] == []
    assert [error["code"] for error in result["errors"]] == ["invalid_collection_child"]


def test_stream_rescans_only_on_chunks_that_can_close_a_leaf(monkeypatch):
    # Every leaf and clip ends with `>`, so chunks without one never trigger a rescan of the answer; and once the
    # collection closed, rescans start after it rather than at the start of the answer.
    starts = []

    def counting_scan(text, expects, start=0):
        starts.append(start)
        return _scan_leaves(text, expects, start)

    monkeypatch.setattr(client_mod, "_scan_leaves", counting_scan)
    pieces = _split(TRACKS, 3)
    _stream_text(monkeypatch, pieces)

    events = list(_client().stream(TASK, expects="box", parse_points=True))

    assert len(starts) == sum(">" in piece for piece in pieces) < len(pieces)
    collection_end = TRACKS.index("</collection>") + len("</collection>")
    assert sorted(set(starts)) == [0, collection_end]
    assert [event["points"][0] for event in events if event["type"] == "points.delta"] == TRACKS_BOXES


def test_strict_stream_ends_with_a_parse_error_event_and_no_final(monkeypatch):
    _stream_text(monkeypatch, _split(MALFORMED, 9))

    events = list(_client().stream(TASK, expects="box", parse_points=True, strict=True))

    assert "final" not in [event["type"] for event in events]
    error = events[-1]
    assert error["type"] == "error" and error["code"] == "invalid_box_coords"
    assert error["details"]["span"] == {"start": 2, "end": 32}
    assert error["partial"] == {"text": MALFORMED, "reasoning": None, "tool_calls": None, "finish_reason": "stop"}
    # Deltas for the well-formed box were still emitted before the end.
    assert sum(event["type"] == "points.delta" for event in events) == 1

    _stream_text(monkeypatch, [TRACKS[:40], TRACKS[40:100]], finish_reason="length")
    events = list(_client().stream(TASK, expects="box", strict=True))
    assert events[-1]["type"] == "error" and events[-1]["code"] == "incomplete_annotation"

    _stream_text(monkeypatch, [TRACKS])
    assert list(_client().stream(TASK, expects="box", strict=True))[-1]["type"] == "final"


def test_strict_stream_error_event_carries_the_trace_id(monkeypatch):
    events = [chunk({"content": MALFORMED}), chunk({}, finish_reason="stop")]
    install(monkeypatch, lambda request: sse_response(events, headers={"x-trace-id": "t"}))

    error = list(_client().stream(TASK, expects="box", strict=True))[-1]

    assert error["type"] == "error" and error["code"] == "invalid_box_coords"
    assert error["request_id"] == "t" and error["details"]["request_id"] == "t"
    assert error["partial"]["text"] == MALFORMED

    error = _collect(AsyncClient(provider="perceptron").stream(TASK, expects="box", strict=True))[-1]
    assert error["request_id"] == "t" and error["details"]["request_id"] == "t"


def test_strict_perceive_and_async_streams(monkeypatch):
    _stream_text(monkeypatch, [MALFORMED])
    events = list(perceive(text("Find boxes"), expects="box", provider="perceptron", stream=True, strict=True))
    assert [event["type"] for event in events][-1] == "error" and events[-1]["code"] == "invalid_box_coords"
    assert "final" not in [event["type"] for event in events]

    _stream_text(monkeypatch, [MALFORMED])
    events = _collect(AsyncClient(provider="perceptron").stream(TASK, expects="box", strict=True))
    assert events[-1]["type"] == "error" and events[-1]["code"] == "invalid_box_coords"
    assert events[-1]["partial"]["text"] == MALFORMED


def test_stream_deltas_stop_at_the_buffer_limit(monkeypatch):
    first = "<point_box> (1,2) (3,4) </point_box>"
    _stream_text(monkeypatch, [first, " more text", " <point_box> (5,6) (7,8) </point_box>"])

    events = list(_client(max_buffer_bytes=len(first)).stream(TASK, expects="box", parse_points=True))

    assert [event["points"] for event in events if event["type"] == "points.delta"] == [[bbox(1, 2, 3, 4)]]
    result = events[-1]["result"]
    assert "boxes" not in result
    assert [error["code"] for error in result["errors"]] == ["stream_buffer_overflow"]


# ---------------------------------------------------------------------------
# asset_idx resolution (§12.9: a missing selector means the last asset)
# ---------------------------------------------------------------------------

TWO_ASSET_ANSWER = (
    "<point_box> (1,2) (3,4) </point_box>"
    ' <point_box asset_idx="0"> (5,6) (7,8) </point_box>'
    ' <collection asset_idx="0"> <point_box> (1,1) (2,2) </point_box> </collection>'
    ' <point_box asset_idx="5"> (1,1) (2,2) </point_box>'
)


def test_perceive_result_resolves_asset_idx_against_its_assets(monkeypatch):
    _json(monkeypatch, TWO_ASSET_ANSWER)

    res = perceive(image(PNG_BYTES) + image(PNG_BYTES) + text("Find it"), expects="box", provider="perceptron")

    assert res.asset_count == 2
    assert [box.asset_idx for box in res.boxes] == [None, 0, 0, 5]  # the markup's values; nothing inferred
    assert [res.resolve_asset_idx(box) for box in res.boxes[:3]] == [1, 0, 0]
    with pytest.raises(ValueError, match="out of range"):
        res.resolve_asset_idx(res.boxes[3])
    assert res.boxes[0].asset_idx is None  # resolution never writes back


def test_perceive_result_without_assets_resolves_to_none():
    res = PerceiveResult(None, None, None, None, None, None, None, None, [], None)
    assert res.resolve_asset_idx(pt(1, 2)) is None
    assert res.resolve_asset_idx(pt(1, 2, asset_idx=3)) == 3
    assert (
        PerceiveResult(None, None, None, None, None, None, None, None, [], None, asset_count=1).resolve_asset_idx(
            pt(1, 2)
        )
        == 0
    )


def test_chat_completion_annotations_and_asset_resolution(monkeypatch):
    install(monkeypatch, lambda request: json_response(completion(TWO_ASSET_ANSWER)))
    client = Client()

    result = client.chat.completions.create(
        messages=[{"role": "user", "content": [image(PNG_BYTES), image(PNG_BYTES), "Find it"]}]
    )

    assert result.asset_count == 2
    boxes = result.annotations(expects="box").boxes
    assert [result.resolve_asset_idx(box) for box in boxes[:3]] == [1, 0, 0]
    with pytest.raises(ValueError):
        result.resolve_asset_idx(boxes[3])


def test_streamed_chat_completion_resolves_asset_idx_against_its_assets(monkeypatch):
    # The stream's completion takes `asset_count` from the request it sent (through the accumulator).
    _stream_text(monkeypatch, _split(TWO_ASSET_ANSWER, 16))
    messages = [{"role": "user", "content": [image(PNG_BYTES), image(PNG_BYTES), "Find it"]}]

    completion_ = Client().chat.completions.create(messages=messages, stream=True).get_final_completion()

    assert completion_.asset_count == 2
    boxes = completion_.annotations(expects="box").boxes
    assert boxes[0].asset_idx is None
    assert completion_.resolve_asset_idx(boxes[0]) == 1  # no selector: the last asset
    assert [completion_.resolve_asset_idx(box) for box in boxes[1:3]] == [0, 0]

    async def _async_final():
        stream = await AsyncClient().chat.completions.create(messages=messages, stream=True)
        return await stream.get_final_completion()

    async_completion = asyncio.run(_async_final())
    assert async_completion.asset_count == 2
    assert async_completion.resolve_asset_idx(async_completion.annotations(expects="box").boxes[0]) == 1


# ---------------------------------------------------------------------------
# Message-level annotations()
# ---------------------------------------------------------------------------


def test_message_annotations_are_lenient_and_flattened():
    message = ChatCompletionMessage(role="assistant", content=TRACKS + ' <clip t="1 seconds" />')

    found = message.annotations()

    assert found.boxes == TRACKS_BOXES and found.tracks == TRACKS_TRACKS
    assert found.clips == [clip(1.0)]
    assert found.errors == []
    assert message.annotations(expects="clip").boxes == []
    assert message.annotations(expects="box").clips == []


def test_message_annotations_errors_strict_and_empty_content():
    message = ChatCompletionMessage(role="assistant", content=MALFORMED)
    assert message.annotations().boxes == [bbox(3, 4, 5, 6)]
    assert [error["code"] for error in message.annotations().errors] == ["invalid_box_coords"]
    with pytest.raises(ParseError):
        message.annotations(strict=True)

    empty = ChatCompletionMessage(role="assistant", content=None).annotations()
    assert (empty.points, empty.boxes, empty.tracks, empty.parsed, empty.errors) == ([], [], [], [], [])


@pytest.mark.parametrize("expects", ["boxes", "bbox", "BOX", "text", ""])
def test_annotations_reject_unknown_expects(monkeypatch, expects):
    # No silent drops: an unknown family would otherwise return empty buckets without an error.
    message = ChatCompletionMessage(role="assistant", content="<point_box> (1,2) (3,4) </point_box>")
    with pytest.raises(ValueError, match="expects must be one of 'point', 'box', 'polygon', 'clip' or None"):
        message.annotations(expects=expects)

    install(monkeypatch, lambda request: json_response(completion(message.content)))
    result = Client().chat.completions.create(messages=[{"role": "user", "content": "Find it"}])
    with pytest.raises(ValueError, match=repr(expects)):
        result.annotations(expects=expects)
    assert result.annotations(expects="box").boxes == [bbox(1, 2, 3, 4)]


def test_completion_annotations_match_the_collector(monkeypatch):
    install(monkeypatch, lambda request: json_response(completion(TRACKS)))

    result = Client().chat.completions.create(messages=[{"role": "user", "content": "Find the players."}])

    assert result.annotations() == collect_annotations(TRACKS)
    assert result.message.annotations(expects="box") == collect_annotations(TRACKS, expects="box")
