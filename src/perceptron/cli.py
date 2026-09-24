"""Perceptron CLI utilities built with Typer + Rich."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

from . import audio as audio_node
from . import caption as caption_image
from . import detect as detect_image
from . import image as image_node
from . import ocr as ocr_image
from . import question as question_image
from . import video as video_node
from .dsl.nodes import Audio as AudioNode
from .errors import SDKError
from .highlevel import default_caption_expects
from .pointing.types import BoundingBox, Clip, Collection, Polygon, SinglePoint, Track

console = Console()
app = typer.Typer(help="Interact with the Perceptron SDK and models.")

# The image formats the API accepts (png, jpeg, webp); directory mode reads only these files.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_VIDEO_EXTENSIONS = {".mp4", ".webm"}

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

_OUTPUT_FILENAMES = {
    "caption": "captions.json",
    "ocr": "ocr.json",
    "detect": "detections.json",
}


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class ExpectationType(str, Enum):
    TEXT = "text"
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    CLIP = "clip"
    THINK = "think"


class ReasoningEffort(str, Enum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Options shared by the model commands (factories, so every command gets its own OptionInfo).
def _model_option():
    return typer.Option(None, "--model", help="Model id, e.g. perceptron-mk1.5 (default: the provider's default).")


def _provider_option():
    return typer.Option(
        None,
        "--provider",
        help=(
            "Provider: perceptron or fal (default: PERCEPTRON_PROVIDER, else perceptron; fal only when FAL_KEY is set "
            "and PERCEPTRON_API_KEY is not)."
        ),
    )


def _reasoning_effort_option():
    return typer.Option(
        None,
        "--reasoning-effort",
        case_sensitive=False,
        help="How much the model reasons before answering (none, minimal, low, medium, or high).",
    )


def _audio_in_video_option():
    return typer.Option(
        None,
        "--audio-in-video/--no-audio-in-video",
        help="Process (or explicitly skip) the soundtrack of video input; unset leaves the server default.",
    )


def _generation_kwargs(
    *,
    audio_in_video: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Keyword arguments a command forwards to its helper; only flags the user set are included."""
    gen_kwargs: dict[str, Any] = {}
    if audio_in_video is not None:
        gen_kwargs["enable_audio_in_video"] = audio_in_video
    if reasoning_effort is not None:
        gen_kwargs["reasoning_effort"] = reasoning_effort.value
    if model is not None:
        gen_kwargs["model"] = model
    if provider is not None:
        gen_kwargs["provider"] = provider
    return gen_kwargs


def _is_url(media: str) -> bool:
    """True for an http(s) or ``data:`` URL, which is passed through (and never looked up on disk)."""
    return media.startswith(("http://", "https://", "data:"))


def _is_directory(media: str) -> bool:
    return not _is_url(media) and Path(media).is_dir()


def _resolve_media(media: str) -> str | bytes:
    """Resolve a media argument to a URL string or local-file bytes."""

    if _is_url(media):
        return media
    path = Path(media)
    if path.is_dir():
        raise ValueError(f"Expected media file, received directory: {media}")
    if path.exists():
        return path.read_bytes()
    return media


def _has_extension(media: str, extensions: set[str]) -> bool:
    """True if the path or URL (ignoring query/fragment) ends with one of ``extensions``."""

    base = media.lower().split("?", 1)[0].split("#", 1)[0]
    return any(base.endswith(ext) for ext in extensions)


def _looks_like_video(media: str) -> bool:
    return _has_extension(media, _VIDEO_EXTENSIONS)


def _looks_like_audio(media: str) -> bool:
    return _has_extension(media, _AUDIO_EXTENSIONS)


def _make_media_node(media_input: str, media_data: str | bytes):
    """Wrap resolved media data in `image()`, `video()`, or `audio()`: by MIME type for a ``data:`` URL, else by the
    input's extension."""

    if media_input.startswith("data:"):
        family = media_input[len("data:") :].split("/", 1)[0].lower()
        return {"video": video_node, "audio": audio_node}.get(family, image_node)(media_data)
    if _looks_like_video(media_input):
        return video_node(media_data)
    if _looks_like_audio(media_input):
        return audio_node(media_data)
    return image_node(media_data)


def _media_node_or_exit(media: str, output_format: OutputFormat, *, image_only: bool = False):
    """The media node for a command's argument. Invalid media (an ``SDKError`` such as ``invalid_data_url``) is reported
    like a request error (exit status 1)."""
    try:
        data = _resolve_media(media)
        return image_node(data) if image_only else _make_media_node(media, data)
    except SDKError as exc:
        _exit_with_error(exc, output_format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _iter_image_files(directory: Path) -> Iterable[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
            yield entry


# ---------------------------------------------------------------------------
# JSON payloads
# ---------------------------------------------------------------------------


def _serialize_annotation(annotation: Any) -> Any:
    """An annotation (point, box, polygon, collection, clip, track) or tool call as a JSON-ready dict."""
    to_dict = getattr(annotation, "to_dict", None)
    return to_dict() if callable(to_dict) else annotation


_BUCKET_BY_EXPECTS = {"point": "points", "box": "boxes", "polygon": "polygons", "clip": "clips"}


def _serialize_points(points: list[Any] | None) -> list[Any] | None:
    if not points:
        return None
    return [_serialize_annotation(point) for point in points]


def _bucket_for_expects(result: Any, expects: str | None) -> tuple[str, list[Any]] | None:
    """Return the (name, list) of the bucket matching ``expects``, if it has data."""

    bucket_name = _BUCKET_BY_EXPECTS.get(expects)
    if bucket_name is None:
        return None
    items = getattr(result, bucket_name, None)
    return (bucket_name, items) if items else None


def _serialize_parsed(
    parsed: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not parsed:
        return None
    serialized: list[dict[str, Any]] = []
    for segment in parsed:
        if not isinstance(segment, dict):
            serialized.append({"kind": "unknown", "value": str(segment)})
            continue
        seg_copy = dict(segment)
        if "value" in seg_copy:
            seg_copy["value"] = _serialize_annotation(seg_copy["value"])
        serialized.append(seg_copy)
    return serialized


def _normalize_usage(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        return usage
    to_dict = getattr(usage, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(usage, "_asdict"):
        return dict(usage._asdict())  # type: ignore[attr-defined]
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return {}


def _add_metadata(payload: dict[str, Any], result: Any) -> None:
    """Add ``tracks``, ``tool_calls``, ``finish_reason``, ``usage`` and ``request_id`` when the result has them."""
    tracks = _serialize_points(getattr(result, "tracks", None))
    if tracks:
        payload["tracks"] = tracks
    tool_calls = _serialize_points(getattr(result, "tool_calls", None))
    if tool_calls:
        payload["tool_calls"] = tool_calls
    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    usage = getattr(result, "usage", None)
    if usage:
        payload["usage"] = _normalize_usage(usage)
    request_id = getattr(result, "request_id", None)
    if request_id is not None:
        payload["request_id"] = request_id


def _result_payload(result: Any, *, include_raw: bool, expects: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    text_value = getattr(result, "text", None)
    if text_value is not None:
        payload["text"] = text_value
    reasoning = getattr(result, "reasoning", None)
    if reasoning:
        payload["reasoning"] = reasoning
    populated = _bucket_for_expects(result, expects)
    if populated is not None:
        bucket_name, items = populated
        payload[bucket_name] = _serialize_points(items)
    serialized_parsed = _serialize_parsed(getattr(result, "parsed", None))
    if serialized_parsed is not None:
        payload["parsed"] = serialized_parsed
    _add_metadata(payload, result)
    errors = getattr(result, "errors", None) or []
    payload["errors"] = errors
    if include_raw:
        raw = getattr(result, "raw", None)
        if raw is not None:
            payload["raw"] = raw
    return payload


def _process_directory(
    directory: Path,
    *,
    command_name: str,
    stream: bool,
    show_raw: bool,
    runner: Callable[[bytes], Any],
    payload_factory: Callable[[Any], Any],
):
    if stream:
        # Emit a friendly error to stdout for CLI tests, then exit non-zero.
        console.print(
            Panel(
                Text(f"Streaming output is not supported when processing a directory for '{command_name}'."),
                title=command_name.capitalize(),
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    image_files = list(_iter_image_files(directory))
    if not image_files:
        console.print(
            Panel(
                Text(f"No image files ({', '.join(sorted(_IMAGE_EXTENSIONS))}) found in {directory}"),
                title=command_name.capitalize(),
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    outputs: dict[str, Any] = {}
    errors: list[tuple[str, dict[str, Any]]] = []

    for image_path in image_files:
        try:
            image_bytes = image_path.read_bytes()
        except Exception as exc:
            console.print(
                Panel(
                    Text(str(exc)),
                    title=Text(f"Error reading: {image_path.name}"),
                    border_style="red",
                )
            )
            continue

        try:
            result = runner(image_bytes)
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(Panel(Text(str(exc)), title=Text(f"Error: {image_path.name}"), border_style="red"))
            continue

        outputs[image_path.name] = payload_factory(result)

        if getattr(result, "errors", None):
            for err in result.errors:
                errors.append((image_path.name, err))

        if show_raw and getattr(result, "raw", None):
            console.print(Panel(Pretty(result.raw), title=Text(f"Raw: {image_path.name}"), border_style="cyan"))

    if not outputs:
        console.print(
            Panel(
                Text(f"No successful {command_name} results produced in {directory}"),
                title=command_name.capitalize(),
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    output_filename = _OUTPUT_FILENAMES.get(command_name, f"{command_name}.json")
    output_path = directory / output_filename
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    console.print(
        Panel(
            Text(f"Wrote {command_name} results for {len(outputs)} file(s) to {output_path}"),
            title=command_name.capitalize(),
            border_style="green",
        )
    )

    if errors:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("file")
        table.add_column("code")
        table.add_column("message")
        for filename, err in errors:
            table.add_row(Text(filename), Text(str(err.get("code"))), Text(str(err.get("message"))))
        console.print(Panel(table, title="Errors", border_style="red"))


def _caption_payload(result: Any, *, expects: str | None = None) -> Any:
    text_value = getattr(result, "text", None) or ""
    payload: dict[str, Any] = {"text": text_value}
    populated = _bucket_for_expects(result, expects)
    if populated is not None:
        bucket_name, items = populated
        payload[bucket_name] = _serialize_points(items)
    tracks = _serialize_points(getattr(result, "tracks", None))
    if tracks:
        payload["tracks"] = tracks
    return payload if len(payload) > 1 else text_value


def _ocr_payload(result: Any) -> str:
    return getattr(result, "text", None) or ""


def _detect_payload(result: Any) -> dict[str, Any]:
    # Detect always emits boxes.
    payload: dict[str, Any] = {"text": getattr(result, "text", None) or ""}
    populated = _bucket_for_expects(result, "box")
    if populated is not None:
        bucket_name, items = populated
        payload[bucket_name] = _serialize_points(items)
    parsed = _serialize_parsed(getattr(result, "parsed", None))
    if parsed:
        payload["parsed"] = parsed
    _add_metadata(payload, result)
    return payload


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _print_errors(errors):
    """The errors as a table (text is printed literally), followed by the request id of any that carry one."""
    if not errors:
        return
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("code")
    table.add_column("message")
    for err in errors:
        table.add_row(Text(str(err.get("code"))), Text(str(err.get("message"))))
    request_ids = dict.fromkeys(err["request_id"] for err in errors if err.get("request_id"))
    body = Group(table, *(Text(f"request id: {request_id}") for request_id in request_ids)) if request_ids else table
    console.print(Panel(body, title="Errors", border_style="red"))


def _error_fields(source: Any) -> dict[str, Any]:
    """The JSON-ready fields of an ``SDKError`` or a stream ``error`` event (None values dropped).

    ``details["task"]`` (the compiled prompt a credentials error carries, with encoded media) is left out.
    """
    if isinstance(source, SDKError):
        fields = {
            "message": str(source),
            "code": source.code,
            "error_type": source.error_type,
            "param": source.param,
            "status": source.status_code,
            "request_id": source.request_id,
            "details": source.details,
        }
    else:
        fields = {key: source.get(key) for key in ("message", "code", "error_type", "param", "status", "request_id")}
        fields["details"] = source.get("details")
    details = fields.get("details")
    fields["details"] = {k: v for k, v in details.items() if k != "task"} if isinstance(details, dict) else None
    return {key: value for key, value in fields.items() if value not in (None, {})}


def _exit_with_error(exc: SDKError, output_format: OutputFormat) -> NoReturn:
    """Report an SDK error (code, message, request id, details) and exit with status 1."""
    fields = _error_fields(exc)
    if output_format is OutputFormat.JSON:
        console.print_json(data={"error": json.loads(json.dumps(fields, default=str))})
    else:
        lines = [f"[{fields.get('code') or type(exc).__name__}] {fields.get('message') or ''}".rstrip()]
        if fields.get("request_id"):
            lines.append(f"request id: {fields['request_id']}")
        console.print(Panel(Text("\n".join(lines)), title=f"Error: {type(exc).__name__}", border_style="red"))
    raise typer.Exit(code=1)


def _call(output_format: OutputFormat, helper: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a helper, turning an ``SDKError`` into a readable report and exit status 1 instead of a traceback."""
    try:
        return helper(*args, **kwargs)
    except SDKError as exc:
        _exit_with_error(exc, output_format)


def _seconds(value: Any) -> str:
    return f"{value:.2f}s"


def _describe_point(point: Any) -> tuple[str, str, str]:
    """Return a ``(kind, coords, mention)`` tuple describing an annotation for tables and streaming displays.

    ``coords`` ends with the spatial ``t`` and ``@asset N`` (the ``asset_idx``) when the annotation has them.
    """

    if isinstance(point, BoundingBox):
        kind, coords = (
            "box",
            f"({point.top_left.x},{point.top_left.y}) → ({point.bottom_right.x},{point.bottom_right.y})",
        )
    elif isinstance(point, SinglePoint):
        kind, coords = "point", f"({point.x},{point.y})"
    elif isinstance(point, Polygon):
        kind, coords = "polygon", ", ".join(f"({p.x},{p.y})" for p in point.hull[:4])
        if len(point.hull) > 4:
            coords += ", …"
    elif isinstance(point, Collection):
        kind, coords = "collection", f"{len(point.points)} items"
    elif isinstance(point, Clip):
        ts = point.timestamp
        kind = "clip"
        coords = f"@{ts.at:.2f}s" if ts.until is None else f"{_seconds(ts.at)} → {_seconds(ts.until)}"
    elif isinstance(point, Track):
        count = len(point.points)
        waypoint_kind = _describe_point(point.points[0])[0] if point.points else "empty"
        kind, coords = "track", f"{count} {waypoint_kind} waypoint{'' if count == 1 else 's'}"
        times = [p.t for p in point.points if p.t is not None]
        if times:
            coords += f", {_seconds(min(times))} → {_seconds(max(times))}"
    else:
        return (type(point).__name__, str(point), getattr(point, "mention", "") or "")
    t = getattr(point, "t", None)  # spatial leaves (and legacy collections); clips and tracks have none
    if t is not None:
        coords += f" t={_seconds(t)}"
    if getattr(point, "complete", True) is False:
        coords += " (incomplete)"
    if point.asset_idx is not None:
        coords += f" @asset {point.asset_idx}"
    return (kind, coords, point.mention or "")


def _build_points_table(points: Iterable[Any], *, title: str = "Points") -> Table:
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("type")
    table.add_column("coords")
    table.add_column("mention")
    for point in points:
        kind, coords, mention = _describe_point(point)
        table.add_row(Text(kind), Text(coords), Text(mention))
    return table


def _dedupe_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any]] = set()
    unique: list[dict[str, Any]] = []
    for err in errors:
        code = err.get("code")
        message = err.get("message")
        key = (code, message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(err)
    return unique


def _coerce_result_dict(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": result.get("text"),
        "reasoning": result.get("reasoning"),
        "points": result.get("points"),
        "boxes": result.get("boxes"),
        "polygons": result.get("polygons"),
        "clips": result.get("clips"),
        "tracks": result.get("tracks"),
        "parsed": result.get("parsed"),
        "tool_calls": result.get("tool_calls"),
        "finish_reason": result.get("finish_reason"),
        "usage": result.get("usage"),
        "request_id": result.get("request_id"),
        "errors": result.get("errors") or [],
        "raw": result.get("raw"),
    }


def _coerce_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):  # avoid True -> 1
            return 1 if value else 0
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _resolve_usage_tokens(
    usage: dict[str, Any] | None,
) -> tuple[int | None, int | None]:
    if not usage:
        return (None, None)
    usage_map = _normalize_usage(usage)
    prompt_keys = ["prompt_tokens", "input_tokens", "prompt"]
    completion_keys = ["completion_tokens", "output_tokens", "completion"]
    tokens_in = None
    tokens_out = None
    for key in prompt_keys:
        tokens_in = _coerce_int(usage_map.get(key))
        if tokens_in is not None:
            break
    for key in completion_keys:
        tokens_out = _coerce_int(usage_map.get(key))
        if tokens_out is not None:
            break
    return (tokens_in, tokens_out)


def _prompt_token_details(usage: Any) -> str:
    """``" (audio N, cached M)"`` from ``prompt_tokens_details``; empty when neither was reported."""
    details = _normalize_usage(usage).get("prompt_tokens_details") if usage else None
    if not isinstance(details, dict):
        return ""
    parts = [
        f"{name} {details[f'{name}_tokens']}"
        for name in ("audio", "cached")
        if details.get(f"{name}_tokens") is not None
    ]
    return f" ({', '.join(parts)})" if parts else ""


def _status_line(result: Any) -> Text | None:
    """``finish_reason`` and token usage of a result, as one dim line (None when the result has neither)."""
    parts: list[str] = []
    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is not None:
        parts.append(f"finish_reason {finish_reason}")
    usage = getattr(result, "usage", None)
    tokens_in, tokens_out = _resolve_usage_tokens(usage)
    if tokens_in is not None:
        parts.append(f"tokens in {tokens_in}{_prompt_token_details(usage)}")
    if tokens_out is not None:
        parts.append(f"tokens out {tokens_out}")
    return Text(" | ".join(parts), style="dim") if parts else None


def _stream_render(
    events: Iterable[dict[str, Any]],
    *,
    title: str,
    output_format: OutputFormat,
    show_raw: bool,
    show_points_table: bool,
    expects: str | None = None,
) -> None:
    """Render streaming events inside a live-updating panel.

    Handles ``text.delta``, ``reasoning.delta``, ``points.delta`` (with its ``context``), ``tool_call.delta``, the
    ``final`` result (``finish_reason``, ``usage``, ``tracks``) and the terminal ``error`` event (``code``, ``details``,
    ``partial``). A stream that ends in an ``error`` event exits with status 1 after the output is printed.
    """

    bucket_name = _BUCKET_BY_EXPECTS.get(expects)

    text_buffer: list[str] = []
    reasoning_buffer: list[str] = []
    annotations_buffer: list[Any] = []
    containers: list[str | None] = []  # the container ("track"/"collection") each streamed annotation arrived in
    tool_call_buffer: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    usage_info: dict[str, Any] | None = None
    finish_reason: str | None = None
    failed = False

    start_ts = time.perf_counter()
    first_token_delta: float | None = None
    last_token_ts: float | None = None
    latency_samples: list[float] = []
    delta_event_count = 0
    end_ts: float | None = None

    def current_panel() -> Panel:
        body: list[Any] = []
        text_content = "".join(text_buffer)
        text_render = Text(text_content or "<waiting for response…>")
        if not text_content:
            text_render.stylize("dim")
        body.append(text_render)
        if annotations_buffer:
            if show_points_table:
                body.append(_build_points_table(annotations_buffer))
            else:
                summary = Text()
                for idx, annotation in enumerate(annotations_buffer, 1):
                    kind, coords, mention = _describe_point(annotation)
                    line = f"{idx}. {kind}: {coords}"
                    if mention:
                        line += f" ({mention})"
                    container = containers[idx - 1] if idx - 1 < len(containers) else None
                    if container:
                        line += f" [in {container}]"
                    summary.append(line + "\n")
                body.append(summary)
        if tool_call_buffer:
            calls = Text()
            for call in (tool_call_buffer[i] for i in sorted(tool_call_buffer)):
                calls.append(f"tool call {call['name'] or '?'}({call['arguments']})\n", style="magenta")
            body.append(calls)
        # metrics summary
        now = time.perf_counter()
        metrics_parts: list[str] = []
        if first_token_delta is not None:
            metrics_parts.append(f"TTFT {first_token_delta * 1000:.0f} ms")
        else:
            metrics_parts.append("TTFT —")

        usage_tokens_in, usage_tokens_out = _resolve_usage_tokens(usage_info)
        if usage_tokens_out is not None and first_token_delta is not None:
            reference = end_ts or now
            effective = max(reference - (start_ts + first_token_delta), 0.0)
            avg_latency = effective * 1000 / max(usage_tokens_out, 1)
            metrics_parts.append(f"Avg {avg_latency:.0f} ms/token")
        elif latency_samples:
            avg_latency = sum(latency_samples) / len(latency_samples) * 1000
            metrics_parts.append(f"Avg {avg_latency:.0f} ms/chunk")
        else:
            metrics_parts.append("Avg —")

        tokens_in_display = "—" if usage_tokens_in is None else f"{usage_tokens_in}{_prompt_token_details(usage_info)}"
        if usage_tokens_out is not None:
            tokens_out_display = str(usage_tokens_out)
        else:
            tokens_out_display = f"~{delta_event_count}" if delta_event_count else "—"
        metrics_parts.append(f"Tokens in {tokens_in_display}")
        metrics_parts.append(f"Tokens out {tokens_out_display}")
        if finish_reason is not None:
            metrics_parts.append(f"Finish {finish_reason}")

        metrics_text = Text(" | ".join(metrics_parts), style="dim")
        body.append(metrics_text)

        content = body[0] if len(body) == 1 else Group(*body)
        return Panel(content, title=title, border_style="red" if failed else "cyan")

    live_panel = current_panel()
    # With --format json the live panel is only progress: clear it so stdout ends up holding just the JSON.
    transient = output_format is OutputFormat.JSON
    with Live(live_panel, console=console, refresh_per_second=12, transient=transient) as live:
        for event in events:
            event_type = event.get("type")
            if event_type == "text.delta":
                chunk = event.get("chunk") or ""
                text_buffer.append(chunk)
                delta_event_count += 1
                now = time.perf_counter()
                if first_token_delta is None:
                    first_token_delta = now - start_ts
                if last_token_ts is not None:
                    latency_samples.append(now - last_token_ts)
                last_token_ts = now
            elif event_type == "reasoning.delta":
                reasoning_buffer.append(event.get("chunk") or "")
            elif event_type == "points.delta":
                pts = event.get("points") or []
                container = (event.get("context") or {}).get("container")
                annotations_buffer.extend(pts)
                containers.extend([container] * len(pts))
            elif event_type == "tool_call.delta":
                call = tool_call_buffer.setdefault(event.get("index") or 0, {"id": None, "name": None, "arguments": ""})
                call["id"] = call["id"] or event.get("id")
                call["name"] = call["name"] or event.get("name")
                call["arguments"] += event.get("arguments") or ""
            elif event_type == "error":
                failed = True
                error = _error_fields(event)
                error.setdefault("code", "stream_error")
                error.setdefault("message", "unknown error")
                errors.append(error)
                partial = event.get("partial") or {}
                if not text_buffer and partial.get("text"):
                    text_buffer = [partial["text"]]
                if not reasoning_buffer and partial.get("reasoning"):
                    reasoning_buffer = [partial["reasoning"]]
                finish_reason = partial.get("finish_reason") or finish_reason
                end_ts = time.perf_counter()
                live.update(current_panel())
                break
            elif event_type == "final":
                final_result = dict(event.get("result") or {})
                end_ts = time.perf_counter()
                if final_result.get("text") is not None:
                    text_buffer = [final_result.get("text") or ""]
                if bucket_name and final_result.get(bucket_name) is not None:
                    annotations_buffer = list(final_result.get(bucket_name) or [])
                    containers = []
                final_errs = final_result.get("errors") or []
                if final_errs:
                    errors.extend(final_errs)
                if final_result.get("usage"):
                    usage_info = _normalize_usage(final_result.get("usage"))
                finish_reason = final_result.get("finish_reason")
            live.update(current_panel())

    if end_ts is None:
        end_ts = time.perf_counter()

    streamed_calls = [
        {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
        for call in (tool_call_buffer[i] for i in sorted(tool_call_buffer))
    ]
    if final_result is None:
        final_result = {
            "text": "".join(text_buffer) or None,
            "reasoning": "".join(reasoning_buffer) or None,
            "parsed": None,
            "tool_calls": streamed_calls or None,
            "finish_reason": finish_reason,
            "usage": usage_info,
            "errors": _dedupe_errors(errors),
            "raw": None,
        }
        if bucket_name:
            final_result[bucket_name] = annotations_buffer or None
    else:
        # ensure buffers win if final result lacked data
        if final_result.get("text") is None:
            final_result["text"] = "".join(text_buffer) or None
        if final_result.get("reasoning") is None:
            final_result["reasoning"] = "".join(reasoning_buffer) or None
        if bucket_name and not final_result.get(bucket_name) and annotations_buffer:
            final_result[bucket_name] = annotations_buffer
        if not final_result.get("tool_calls") and streamed_calls:
            final_result["tool_calls"] = streamed_calls
        merged_errors = list(errors) if errors else []
        final_errs = final_result.get("errors") or []
        if final_errs:
            merged_errors.extend(final_errs)
        final_result["errors"] = _dedupe_errors(merged_errors)
        if not final_result.get("usage") and usage_info:
            final_result["usage"] = usage_info

    coerced = _coerce_result_dict(final_result)
    result_ns = SimpleNamespace(**coerced)

    if output_format is OutputFormat.JSON:
        console.print_json(data=_result_payload(result_ns, include_raw=show_raw, expects=expects))
    else:
        if coerced["errors"]:
            _print_errors(coerced["errors"])
        if show_raw and coerced.get("raw") is not None:
            console.print(coerced["raw"])
    if failed:
        raise typer.Exit(code=1)


def _render_result(
    result: Any,
    *,
    title: str,
    output_format: OutputFormat,
    show_raw: bool,
    expects: str | None = None,
):
    if output_format is OutputFormat.JSON:
        payload = _result_payload(result, include_raw=show_raw, expects=expects)
        console.print_json(data=payload)
        return

    console.print(Panel(Text(result.text or "<no text>"), title=title, border_style="green"))
    populated = _bucket_for_expects(result, expects)
    annotations = [*(populated[1] if populated is not None else []), *(getattr(result, "tracks", None) or [])]
    if annotations:
        console.print(_build_points_table(annotations, title="Clips" if expects == "clip" else "Detections"))
    status = _status_line(result)
    if status is not None:
        console.print(status)
    _print_errors(getattr(result, "errors", []))
    if show_raw and getattr(result, "raw", None):
        console.print(result.raw)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def config(
    provider: str | None = typer.Option(None, help="Provider to export: perceptron (the Perceptron API) or fal."),
    api_key: str | None = typer.Option(None, help="API key to export (as FAL_KEY with --provider fal)."),
    base_url: str | None = typer.Option(None, help="Optional custom base URL (include /v1 for provider perceptron)."),
    model: str | None = typer.Option(None, help="Default model to export, e.g. perceptron-mk1.5."),
):
    """Print shell `export` lines for your settings (nothing is saved)."""

    exports: list[str] = []
    if provider:
        exports.append(f"export PERCEPTRON_PROVIDER={provider}")
    if api_key:
        # fal reads only FAL_KEY; PERCEPTRON_API_KEY is never sent to it.
        key_env = "FAL_KEY" if (provider or "").lower() == "fal" else "PERCEPTRON_API_KEY"
        exports.append(f"export {key_env}={api_key}")
    if base_url:
        exports.append(f"export PERCEPTRON_BASE_URL={base_url}")
    if model:
        exports.append(f"export PERCEPTRON_MODEL={model}")

    if not exports:
        exports = [
            "export PERCEPTRON_API_KEY=<your-key>",
            "export PERCEPTRON_BASE_URL=<optional-base-url>",
        ]

    console.print(Panel(Text("\n".join(exports)), title="Add these to your shell", border_style="cyan"))
    notes = ["Nothing is saved: run these lines in your shell or add them to your shell profile."]
    if provider is None:
        notes.append(
            "Without PERCEPTRON_PROVIDER, the SDK and this CLI use the Perceptron API. Provider 'fal' is selected only "
            "when FAL_KEY is set and PERCEPTRON_API_KEY is not (fal never receives PERCEPTRON_API_KEY)."
        )
    for note in notes:
        console.print(Text(note, style="dim"))


@app.command()
def caption(
    media: str = typer.Argument(..., help="Image, video, or audio path or URL (or a directory of images)."),
    style: str = typer.Option("concise", help="Captioning style."),
    stream: bool = typer.Option(False, help="Stream incremental output."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    audio_in_video: bool | None = _audio_in_video_option(),
    reasoning_effort: ReasoningEffort | None = _reasoning_effort_option(),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    expects: ExpectationType | None = typer.Option(
        None,
        "--expects",
        case_sensitive=False,
        help="Expected output structure (text, point, box, polygon, clip, or think). Defaults to box for images, text for video and audio.",
    ),
    model: str | None = _model_option(),
    provider: str | None = _provider_option(),
):
    """Generate captions using the high-level helper."""

    gen_kwargs = _generation_kwargs(
        audio_in_video=audio_in_video, reasoning_effort=reasoning_effort, model=model, provider=provider
    )
    if _is_directory(media):
        # Directory mode only reads images, so the default stays box.
        directory_expects = expects.value if expects is not None else ExpectationType.BOX.value
        _process_directory(
            Path(media),
            command_name="caption",
            stream=stream,
            show_raw=show_raw,
            runner=lambda data: caption_image(image_node(data), style=style, expects=directory_expects, **gen_kwargs),
            payload_factory=lambda result: _caption_payload(result, expects=directory_expects),
        )
        return

    node = _media_node_or_exit(media, output_format)
    expects_value = expects.value if expects is not None else default_caption_expects(node)
    show_points_table = expects_value == ExpectationType.BOX.value

    if stream:
        _stream_render(
            _call(output_format, caption_image, node, style=style, expects=expects_value, stream=True, **gen_kwargs),
            title="Caption",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=show_points_table,
            expects=expects_value,
        )
        return

    res = _call(output_format, caption_image, node, style=style, expects=expects_value, **gen_kwargs)
    _render_result(
        res,
        title="Caption",
        output_format=output_format,
        show_raw=show_raw,
        expects=expects_value,
    )


@app.command()
def ocr(
    image: str = typer.Argument(..., help="Image path or URL (or a directory of images)."),
    prompt: str | None = typer.Option(None, help="Optional instruction override."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    reasoning_effort: ReasoningEffort | None = _reasoning_effort_option(),
    model: str | None = _model_option(),
    provider: str | None = _provider_option(),
):
    """Run OCR via the high-level helper. Image inputs only."""

    gen_kwargs = _generation_kwargs(reasoning_effort=reasoning_effort, model=model, provider=provider)
    if _is_directory(image):
        _process_directory(
            Path(image),
            command_name="ocr",
            stream=False,
            show_raw=show_raw,
            runner=lambda data: ocr_image(image_node(data), prompt=prompt, **gen_kwargs),
            payload_factory=_ocr_payload,
        )
        return

    node = _media_node_or_exit(image, output_format, image_only=True)
    res = _call(output_format, ocr_image, node, prompt=prompt, **gen_kwargs)
    _render_result(
        res,
        title="OCR",
        output_format=output_format,
        show_raw=show_raw,
    )


@app.command()
def detect(
    media: str = typer.Argument(..., help="Image or video path or URL (or a directory of images)."),
    classes: str | None = typer.Option(None, help="Comma-separated class list."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    stream: bool = typer.Option(False, help="Stream incremental output."),
    audio_in_video: bool | None = _audio_in_video_option(),
    reasoning_effort: ReasoningEffort | None = _reasoning_effort_option(),
    model: str | None = _model_option(),
    provider: str | None = _provider_option(),
):
    """Run detection via the high-level helper. Image and video inputs."""

    class_list = [c.strip() for c in classes.split(",")] if classes else None
    gen_kwargs = _generation_kwargs(
        audio_in_video=audio_in_video, reasoning_effort=reasoning_effort, model=model, provider=provider
    )
    if _is_directory(media):
        _process_directory(
            Path(media),
            command_name="detect",
            stream=False,
            show_raw=show_raw,
            runner=lambda data: detect_image(image_node(data), classes=class_list, **gen_kwargs),
            payload_factory=_detect_payload,
        )
        return

    node = _media_node_or_exit(media, output_format)
    if isinstance(node, AudioNode):
        raise typer.BadParameter("detect takes an image or a video, not audio.")
    if stream:
        _stream_render(
            _call(output_format, detect_image, node, classes=class_list, stream=True, **gen_kwargs),
            title="Detect",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=True,
            expects="box",
        )
        return
    res = _call(output_format, detect_image, node, classes=class_list, **gen_kwargs)
    _render_result(
        res,
        title="Detect",
        output_format=output_format,
        show_raw=show_raw,
        expects="box",
    )


@app.command()
def question(
    media: str = typer.Argument(..., help="Image, video, or audio path or URL."),
    prompt: str = typer.Argument(..., help="Question to answer about the media."),
    expects: ExpectationType = typer.Option(
        ExpectationType.TEXT,
        "--expects",
        case_sensitive=False,
        help="Expected output structure (text, point, box, polygon, clip, or think).",
    ),
    stream: bool = typer.Option(False, help="Stream incremental output."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    audio_in_video: bool | None = _audio_in_video_option(),
    reasoning_effort: ReasoningEffort | None = _reasoning_effort_option(),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    model: str | None = _model_option(),
    provider: str | None = _provider_option(),
):
    """Answer a question about an image, video, or audio clip."""

    if _is_directory(media):
        raise typer.BadParameter("Directory mode is not supported for 'question'.")

    node = _media_node_or_exit(media, output_format)
    expects_value = expects.value
    gen_kwargs = _generation_kwargs(
        audio_in_video=audio_in_video, reasoning_effort=reasoning_effort, model=model, provider=provider
    )

    if stream:
        _stream_render(
            _call(output_format, question_image, node, prompt, expects=expects_value, stream=True, **gen_kwargs),
            title="Question",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=expects is ExpectationType.BOX,
            expects=expects_value,
        )
        return

    res = _call(output_format, question_image, node, prompt, expects=expects_value, **gen_kwargs)
    _render_result(
        res,
        title="Question",
        output_format=output_format,
        show_raw=show_raw,
        expects=expects_value,
    )


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
