"""Lower compiled task entries to chat-completions wire parts and messages.

A leaf module (it imports nothing from ``client``, ``chat`` or ``dsl``), so the legacy task path and the message API's
DSL nodes share one entry-to-part function.

Task entries are the dicts ``dsl.perceive._compile`` emits: ``text``, ``image``, ``video``, ``audio`` (base64 with a
``format``, an http(s) or ``data:`` URL, or a ``file_id``), ``video_frames``, and the standalone ``assistant_turn`` and
``tool_result`` entries.
"""

from __future__ import annotations

import re
from typing import Any

from ._providers import _SELECT_PERCEPTRON, PERCEPTRON_PROVIDER
from .errors import (
    INVALID_DATA_URL,
    INVALID_FILE_ID,
    INVALID_MEDIA,
    INVALID_TOOL_RESULT,
    INVALID_VIDEO_FRAMES,
    UNSUPPORTED_ENTRY_TYPE,
    UNSUPPORTED_PROVIDER_FEATURE,
    BadRequestError,
)

FILE_ID_PATTERN = re.compile(r"^file-[A-Za-z0-9]{22}$")
_DATA_URL_PATTERN = re.compile(r"^data:(image|video|audio)/[^;,]+;base64,")

# Wire part types that each take one `asset_idx` (a `video_frames` group counts once).
MEDIA_PART_TYPES = frozenset(
    {
        "image_url",
        "image_file_id",
        "video_url",
        "video_file_id",
        "video_frames",
        "input_audio",
        "audio_url",
        "audio_file_id",
    }
)

# Raised for base64 media without a known format (messages kept from the original lowering).
_MISSING_FORMAT = {
    "image": (
        "Could not determine image format from input. The wire protocol supports png, jpeg, and webp; convert the "
        "image to one of these formats before passing it to the SDK.",
        "invalid_image_format",
    ),
    "video": (
        "Could not determine video format from input. The wire protocol supports mp4 and webm.",
        "invalid_video_format",
    ),
    "audio": (
        "Could not determine audio format from input. The wire protocol supports wav, mp3, and flac.",
        "invalid_audio_format",
    ),
}

_TOOL_RESULT_ENTRY_TYPES = ("text", "image")


def validate_file_id(file_id: Any) -> str:
    """Return ``file_id`` when it is an uploaded-file id (``file-`` + 22 letters/digits), else raise."""
    if not isinstance(file_id, str) or not FILE_ID_PATTERN.match(file_id):
        raise BadRequestError(
            f"Invalid file id {file_id!r}: expected 'file-' followed by 22 letters or digits.", code=INVALID_FILE_ID
        )
    return file_id


def validate_data_url(url: str, family: str) -> str:
    """Return ``url`` when it is a base64 ``data:{family}/...`` URL, else raise ``invalid_data_url``."""
    match = _DATA_URL_PATTERN.match(url)
    if match is None or match.group(1) != family:
        raise BadRequestError(
            f"Invalid {family} data URL {url[:40]!r}...: expected 'data:{family}/<subtype>;base64,<data>'.",
            code=INVALID_DATA_URL,
        )
    return url


def _is_url_payload(item: dict[str, Any]) -> bool:
    """True when the payload should pass through as a URL rather than a data URL."""

    if item.get("url"):
        return True
    payload = item.get("content")
    # Hand-written entries without `url` (DSL entries set it); URL schemes are case-insensitive.
    return isinstance(payload, str) and payload[:8].lower().startswith(("http://", "https://"))


def _media_part(kind: str, entry: dict[str, Any], *, provider_name: str | None = None) -> dict[str, Any]:
    payload = entry.get("content")
    file_id = entry.get("file_id")
    if file_id is not None:
        if payload is not None:
            raise BadRequestError(f"A {kind} entry takes content or file_id, not both.", code=INVALID_MEDIA)
        if provider_name is not None and provider_name.lower() != PERCEPTRON_PROVIDER:
            raise BadRequestError(
                f"Uploaded files exist only on the Perceptron API, so {kind}(file_id=...) needs provider 'perceptron' "
                f"(the provider is '{provider_name}'). Select it with {_SELECT_PERCEPTRON}.",
                code=UNSUPPORTED_PROVIDER_FEATURE,
            )
        part_type = f"{kind}_file_id"
        return {"type": part_type, part_type: {"file_id": validate_file_id(file_id)}}
    if not isinstance(payload, str) or not payload:
        raise BadRequestError(f"A {kind} entry needs content (base64 or a URL) or a file_id.", code=INVALID_MEDIA)
    url_type = f"{kind}_url"
    if payload.startswith("data:"):
        return {"type": url_type, url_type: {"url": validate_data_url(payload, kind)}}
    if _is_url_payload(entry):
        return {"type": url_type, url_type: {"url": payload}}
    fmt = entry.get("format")
    if fmt is None:
        message, code = _MISSING_FORMAT[kind]
        raise BadRequestError(message, code=code)
    if kind == "audio":
        return {"type": "input_audio", "input_audio": {"data": payload, "format": fmt}}
    return {"type": url_type, url_type: {"url": f"data:{kind}/{fmt};base64,{payload}"}}


def _video_frames_part(entry: dict[str, Any], *, base_url: str | None, provider_name: str | None) -> dict[str, Any]:
    frames = entry.get("frames")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise BadRequestError("A video_frames entry needs a non-empty 'frames' list.", code=INVALID_VIDEO_FRAMES)
    wire_frames = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise BadRequestError(f"video_frames frame {i} must be a dict.", code=INVALID_VIDEO_FRAMES)
        timestamp_ms = frame.get("timestamp_ms")
        if not isinstance(timestamp_ms, int) or isinstance(timestamp_ms, bool) or timestamp_ms < 0:
            raise BadRequestError(
                f"video_frames frame {i} needs an integer timestamp_ms >= 0; got {timestamp_ms!r}.",
                code=INVALID_VIDEO_FRAMES,
            )
        if frame.get("file_id") is not None:
            file_id = validate_file_id(frame["file_id"])
            if provider_name != PERCEPTRON_PROVIDER or not base_url:
                raise BadRequestError(
                    "Uploaded-file frames need provider 'perceptron' (they are sent as its files-content URLs).",
                    code=UNSUPPORTED_PROVIDER_FEATURE,
                )
            url = base_url.rstrip("/") + f"/files/{file_id}/content"
        else:
            url = frame.get("url")
            if not isinstance(url, str) or not url:
                raise BadRequestError(f"video_frames frame {i} needs a url or a file_id.", code=INVALID_VIDEO_FRAMES)
            if url.startswith("data:"):
                validate_data_url(url, "image")
        wire_frames.append({"image_url": {"url": url}, "timestamp_ms": timestamp_ms})
    return {"type": "video_frames", "video_frames": {"frames": wire_frames}}


def entry_to_part(entry: dict[str, Any], *, base_url: str | None = None, provider_name: str | None = None) -> dict:
    """Lower one task entry to a wire content part.

    ``base_url``/``provider_name`` are needed only for uploaded-file ids: ``video_frames`` frames given as file ids
    become ``{base_url}/files/{id}/content`` and require provider ``perceptron``, and an image/video/audio ``file_id``
    raises for any other named provider (None skips that check).
    """
    entry_type = entry.get("type") if isinstance(entry, dict) else None
    if entry_type == "text":
        text = entry.get("content")
        return {"type": "text", "text": "" if text is None else text}
    if entry_type in ("image", "video", "audio"):
        return _media_part(entry_type, entry, provider_name=provider_name)
    if entry_type == "video_frames":
        return _video_frames_part(entry, base_url=base_url, provider_name=provider_name)
    raise BadRequestError(
        f"Task entry type {entry_type!r} cannot be used as a content part.", code=UNSUPPORTED_ENTRY_TYPE
    )


def _assistant_turn_message(entry: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": entry.get("content")}
    if entry.get("reasoning_content") is not None:
        message["reasoning_content"] = entry["reasoning_content"]
    if entry.get("tool_calls"):
        message["tool_calls"] = list(entry["tool_calls"])
    return message


def _tool_result_message(entry: dict[str, Any], *, base_url: str | None, provider_name: str | None) -> dict[str, Any]:
    tool_call_id = entry.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise BadRequestError("A tool_result entry needs a non-empty tool_call_id.", code=INVALID_TOOL_RESULT)
    parts = []
    for item in entry.get("content") or []:
        item_type = item.get("type") if isinstance(item, dict) else type(item).__name__
        if item_type not in _TOOL_RESULT_ENTRY_TYPES:
            raise BadRequestError(
                f"Tool results may contain only text and images; got {item_type!r}.", code=INVALID_TOOL_RESULT
            )
        parts.append(entry_to_part(item, base_url=base_url, provider_name=provider_name))
    if not parts:
        content: Any = ""
    elif len(parts) == 1 and parts[0]["type"] == "text":
        content = parts[0]["text"]
    else:
        content = parts
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def task_to_messages(
    task: dict, *, base_url: str | None = None, provider_name: str | None = None
) -> list[dict[str, Any]]:
    """Lower a compiled task to chat messages.

    Consecutive text/media entries with the same role merge into one message, and a text-only message collapses to a
    string (the original wire bytes). ``assistant_turn`` and ``tool_result`` entries are standalone messages, never
    merged. Unknown entry types raise.
    """
    messages: list[dict[str, Any]] = []
    current_role: str | None = None
    current_parts: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal current_role, current_parts
        if current_role is not None:
            if all(part.get("type") == "text" for part in current_parts):
                text = "".join(part.get("text", "") for part in current_parts)
                messages.append({"role": current_role, "content": text})
            else:
                messages.append({"role": current_role, "content": list(current_parts)})
        current_role = None
        current_parts = []

    for entry in task.get("content") or []:
        entry_type = entry.get("type") if isinstance(entry, dict) else None
        if entry_type == "assistant_turn":
            _flush()
            messages.append(_assistant_turn_message(entry))
            continue
        if entry_type == "tool_result":
            _flush()
            messages.append(_tool_result_message(entry, base_url=base_url, provider_name=provider_name))
            continue
        part = entry_to_part(entry, base_url=base_url, provider_name=provider_name)
        role = entry.get("role") or "user"
        if role == "agent":
            role = "assistant"
        if current_role not in {role, None}:
            _flush()
        current_role = role
        current_parts.append(part)
    _flush()
    return messages


def count_assets(messages: list[dict[str, Any]] | None) -> int:
    """Media assets in ``messages`` (every message, in order), i.e. the ``asset_idx`` space the model numbers.

    Counts image, video (a ``video_frames`` group once) and audio parts, including images in tool results.
    """
    total = 0
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, (list, tuple)):  # a message dict's tuple content is sent as a list
            total += sum(1 for part in content if isinstance(part, dict) and part.get("type") in MEDIA_PART_TYPES)
    return total
