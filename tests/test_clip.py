"""Clip annotation type, parser, and `expects="clip"` end-to-end coverage."""

from __future__ import annotations

from perceptron import Clip, ClipTimestamp, clip, extract_clips


# ---- Type construction -----------------------------------------------------


def test_clip_factory_moment():
    c = clip(1.5)
    assert c.timestamp == ClipTimestamp(at=1.5)
    assert c.timestamp.until is None
    assert c.mention is None


def test_clip_factory_range_with_mention():
    c = clip(10.0, 20.0, mention="action")
    assert c.timestamp == ClipTimestamp(at=10.0, until=20.0)
    assert c.mention == "action"


def test_clip_timestamp_repr_distinguishes_moment_and_range():
    moment_repr = repr(ClipTimestamp(at=1.5))
    range_repr = repr(ClipTimestamp(at=1.5, until=2.5))
    assert "until" not in moment_repr
    assert "until=2.5" in range_repr


# ---- Parser ----------------------------------------------------------------


def test_extract_clip_moment():
    text = '<clip mention="intro" t=1.5/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=1.5), mention="intro")]


def test_extract_clip_quoted_with_unit():
    """Trailing ``seconds`` token is ignored — first numeric token wins."""

    text = '<clip mention="outro" t="2.5 seconds"/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=2.5), mention="outro")]


def test_extract_clip_range():
    text = '<clip mention="action" t="10 20"/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=10.0, until=20.0), mention="action")]


def test_extract_clip_range_with_units():
    text = '<clip mention="scene" t="30 seconds 45 seconds"/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=30.0, until=45.0), mention="scene")]


def test_extract_clip_no_mention():
    text = "<clip t=4.2/>"
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=4.2), mention=None)]


def test_extract_clip_skips_missing_timestamp():
    text = '<clip mention="bare"/>'
    assert extract_clips(text) == []


def test_extract_clip_with_gt_in_mention():
    """Quoted ``>`` characters inside mention shouldn't terminate the regex early."""

    text = '<clip mention="a > b" t=1.0/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=1.0), mention="a > b")]


def test_extract_clip_with_self_close_in_mention():
    """Quoted ``/>`` inside mention shouldn't be confused with the tag terminator."""

    text = '<clip mention="x/>" t=2.0/>'
    assert extract_clips(text) == [Clip(timestamp=ClipTimestamp(at=2.0), mention="x/>")]


def test_extract_clip_in_collection_inherits_mention():
    text = (
        '<collection mention="parent">'
        "<clip t=1.0/>"
        '<clip mention="child" t=2.0/>'
        "</collection>"
    )
    clips = extract_clips(text)
    assert len(clips) == 2
    assert clips[0].mention == "parent"  # inherited
    assert clips[1].mention == "child"  # explicit wins


def test_extract_clip_mixed_standalone_and_collection():
    text = (
        '<clip mention="solo" t=0.5/>'
        '<collection mention="bundle"><clip t=1.0/></collection>'
        '<clip mention="trailing" t=3.0/>'
    )
    mentions = {c.mention for c in extract_clips(text)}
    assert mentions == {"solo", "bundle", "trailing"}


# ---- High-level expects="clip" --------------------------------------------


def test_caption_clip_populates_result_clips(monkeypatch):
    """End-to-end: caption(..., expects="clip") populates result.clips from model output."""
    from perceptron import caption, config, video
    from perceptron import client as client_mod

    response_text = '<clip mention="intro" t=1.5/> <clip mention="outro" t="2.0 3.0"/>'

    def _stub_generate(self, task, **kwargs):  # pylint: disable=unused-argument
        return self._build_result(
            {"choices": [{"message": {"content": response_text}}]},
            kwargs.get("expects"),
        )

    monkeypatch.setattr(client_mod.Client, "generate", _stub_generate)
    with config(api_key="test-key", provider="fal"):
        res = caption(video("https://x.com/v.mp4"), expects="clip")

    assert res.clips is not None
    assert len(res.clips) == 2
    assert res.clips[0] == Clip(timestamp=ClipTimestamp(at=1.5), mention="intro")
    assert res.clips[1] == Clip(timestamp=ClipTimestamp(at=2.0, until=3.0), mention="outro")
    # Spatial buckets remain unset for clip expectations.
    assert res.points is None
    assert res.boxes is None
    assert res.polygons is None


def test_clip_expects_validates():
    from perceptron.expectations import resolve_structured_expectation

    structured, allow_multiple = resolve_structured_expectation("clip", context="test")
    assert structured == "clip"
    assert allow_multiple is True


def test_clip_hint_emitted_for_clip_expectation():
    """``<hint>CLIP</hint>`` should be appended when expects="clip" is set."""
    from perceptron.client import _build_hint_content

    hint = _build_hint_content("clip", include_reasoning=False, include_focus=False)
    assert hint == "<hint>CLIP</hint>"
