"""Task-entry lowering (`perceptron._lowering`)."""

import pytest

from perceptron import _lowering
from perceptron import client as client_mod
from perceptron.errors import (
    INVALID_DATA_URL,
    INVALID_FILE_ID,
    INVALID_TOOL_RESULT,
    UNSUPPORTED_ENTRY_TYPE,
    UNSUPPORTED_PROVIDER_FEATURE,
    BadRequestError,
)

FILE_ID = "file-abcdefghijklmnopqrstuv"
BASE_URL = "https://api.perceptron.inc/v1/"


def _text(content, role="user"):
    return {"type": "text", "role": role, "content": content}


def test_client_wrapper_delegates():
    task = {"content": [_text("a"), _text("b")]}
    assert client_mod._task_to_openai_messages(task) == [{"role": "user", "content": "ab"}]


@pytest.mark.parametrize(
    ("entry", "part"),
    [
        (_text("hi"), {"type": "text", "text": "hi"}),
        (_text(None), {"type": "text", "text": ""}),
        (
            {"type": "image", "content": "QUJD", "format": "png"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ),
        (
            {"type": "image", "content": "https://x/y.png"},
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ),
        (
            {"type": "image", "content": "data:image/jpeg;base64,/9j/"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}},
        ),
        ({"type": "image", "file_id": FILE_ID}, {"type": "image_file_id", "image_file_id": {"file_id": FILE_ID}}),
        (
            {"type": "video", "content": "https://x/v.mp4", "url": True},
            {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}},
        ),
        (
            {"type": "video", "content": "AAAA", "format": "mp4"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ),
        ({"type": "video", "file_id": FILE_ID}, {"type": "video_file_id", "video_file_id": {"file_id": FILE_ID}}),
        (
            {"type": "audio", "content": "UklG", "format": "wav"},
            {"type": "input_audio", "input_audio": {"data": "UklG", "format": "wav"}},
        ),
        (
            {"type": "audio", "content": "data:audio/flac;base64,ZkxhQw=="},
            {"type": "audio_url", "audio_url": {"url": "data:audio/flac;base64,ZkxhQw=="}},
        ),
        ({"type": "audio", "file_id": FILE_ID}, {"type": "audio_file_id", "audio_file_id": {"file_id": FILE_ID}}),
    ],
)
def test_entry_to_part(entry, part):
    assert _lowering.entry_to_part(entry) == part


@pytest.mark.parametrize("file_id", ["file-abc123", "file-abcdefghijklmnopqrstu!", 123, "abcdefghijklmnopqrstuvwxyz"])
def test_invalid_file_id(file_id):
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.entry_to_part({"type": "image", "file_id": file_id})
    assert excinfo.value.code == INVALID_FILE_ID


@pytest.mark.parametrize(
    ("kind", "url"),
    [
        ("image", "data:video/mp4;base64,AAAA"),  # wrong family
        ("image", "data:image/png,rawtext"),  # not base64
        ("audio", "data:;base64,AAAA"),
    ],
)
def test_invalid_data_url(kind, url):
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.entry_to_part({"type": kind, "content": url})
    assert excinfo.value.code == INVALID_DATA_URL


def test_media_without_format_keeps_the_original_errors():
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.entry_to_part({"type": "image", "content": "QUJD"})
    assert excinfo.value.code == "invalid_image_format"


def test_unknown_entry_type_raises():
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.task_to_messages({"content": [{"type": "hologram", "content": "x"}]})
    assert excinfo.value.code == UNSUPPORTED_ENTRY_TYPE


def test_video_frames_with_urls_and_file_ids():
    entry = {
        "type": "video_frames",
        "role": "user",
        "frames": [
            {"url": "data:image/jpeg;base64,/9j/", "timestamp_ms": 0},
            {"file_id": FILE_ID, "timestamp_ms": 500},
        ],
    }

    part = _lowering.entry_to_part(entry, base_url=BASE_URL, provider_name="perceptron")

    assert part == {
        "type": "video_frames",
        "video_frames": {
            "frames": [
                {"image_url": {"url": "data:image/jpeg;base64,/9j/"}, "timestamp_ms": 0},
                {"image_url": {"url": f"https://api.perceptron.inc/v1/files/{FILE_ID}/content"}, "timestamp_ms": 500},
            ]
        },
    }


def test_file_id_frames_need_provider_perceptron():
    entry = {"type": "video_frames", "frames": [{"file_id": FILE_ID, "timestamp_ms": 0}]}
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.entry_to_part(entry, base_url="https://fal.run", provider_name="fal")
    assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_media_file_ids_need_provider_perceptron(kind):
    entry = {"type": kind, "file_id": FILE_ID}
    for provider in ("fal", "FAL"):
        with pytest.raises(BadRequestError) as excinfo:
            _lowering.entry_to_part(entry, provider_name=provider)
        assert excinfo.value.code == UNSUPPORTED_PROVIDER_FEATURE
    part_type = f"{kind}_file_id"
    expected = {"type": part_type, part_type: {"file_id": FILE_ID}}
    assert _lowering.entry_to_part(entry, provider_name="perceptron") == expected
    assert _lowering.entry_to_part(entry) == expected  # no provider named: not checked


def test_video_frames_timestamp_must_be_an_int():
    entry = {"type": "video_frames", "frames": [{"url": "https://x/f.jpg", "timestamp_ms": 0.5}]}
    with pytest.raises(BadRequestError):
        _lowering.entry_to_part(entry)


def test_merge_rules_keep_todays_wire_bytes():
    task = {
        "content": [
            _text("sys", role="system"),
            _text("Describe the image."),
            {"type": "image", "role": "user", "content": "QUJD", "format": "png"},
            _text("Be concise."),
            _text("ok", role="agent"),
            _text("more", role="assistant"),
            _text("next"),
        ]
    }

    assert _lowering.task_to_messages(task) == [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "text", "text": "Be concise."},
            ],
        },
        {"role": "assistant", "content": "okmore"},
        {"role": "user", "content": "next"},
    ]


def test_video_frames_file_id_and_data_url_media_merge_into_the_user_message():
    task = {
        "content": [
            _text("Compare"),
            {"type": "image", "role": "user", "file_id": FILE_ID},
            {"type": "video_frames", "role": "user", "frames": [{"url": "https://x/a.jpg", "timestamp_ms": 0}]},
            {"type": "audio", "role": "user", "content": "data:audio/wav;base64,UklG"},
        ]
    }

    messages = _lowering.task_to_messages(task)

    assert len(messages) == 1
    assert [p["type"] for p in messages[0]["content"]] == ["text", "image_file_id", "video_frames", "audio_url"]


def test_assistant_turn_and_tool_results_are_standalone():
    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
    task = {
        "content": [
            _text("Weather?"),
            {
                "type": "assistant_turn",
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
                "reasoning_content": "r",
            },
            {"type": "tool_result", "role": "tool", "tool_call_id": "call_1", "content": [_text("18C", role="tool")]},
            {
                "type": "tool_result",
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [_text("radar"), {"type": "image", "content": "https://x/r.png"}],
            },
            {"type": "tool_result", "role": "tool", "tool_call_id": "call_3", "content": []},
            _text("thanks"),
        ]
    }

    assert _lowering.task_to_messages(task) == [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "reasoning_content": "r", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [
                {"type": "text", "text": "radar"},
                {"type": "image_url", "image_url": {"url": "https://x/r.png"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": ""},
        {"role": "user", "content": "thanks"},
    ]


def test_consecutive_assistant_turns_never_merge():
    task = {
        "content": [
            {"type": "assistant_turn", "content": "a"},
            {"type": "assistant_turn", "content": "b"},
        ]
    }
    assert _lowering.task_to_messages(task) == [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_tool_results_allow_only_text_and_images(kind):
    task = {
        "content": [{"type": "tool_result", "tool_call_id": "c", "content": [{"type": kind, "content": "https://x"}]}]
    }
    with pytest.raises(BadRequestError) as excinfo:
        _lowering.task_to_messages(task)
    assert excinfo.value.code == INVALID_TOOL_RESULT


def test_tool_result_needs_a_call_id():
    with pytest.raises(BadRequestError):
        _lowering.task_to_messages({"content": [{"type": "tool_result", "content": []}]})


def test_count_assets_counts_every_media_part_once():
    messages = [
        {"role": "system", "content": "s"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "t"},
                {"type": "image_url", "image_url": {"url": "u"}},
                {"type": "video_frames", "video_frames": {"frames": [{}, {}, {}]}},
                {"type": "input_audio", "input_audio": {}},
                {"type": "audio_file_id", "audio_file_id": {}},
            ],
        },
        {"role": "assistant", "content": None},
        {"role": "tool", "tool_call_id": "c", "content": [{"type": "image_file_id", "image_file_id": {}}]},
        {"role": "user", "content": [{"type": "video_url", "video_url": {}}, {"type": "image_file_id"}]},
    ]

    assert _lowering.count_assets(messages) == 7
    assert _lowering.count_assets([]) == 0
