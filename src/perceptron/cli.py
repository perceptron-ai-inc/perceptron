"""Perceptron CLI utilities built with Typer + Rich."""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import caption as caption_image
from . import detect as detect_image
from . import ocr as ocr_image
from . import question as question_image
from .pointing.types import BoundingBox, Collection, Polygon, SinglePoint

console = Console()
app = typer.Typer(help="Interact with the Perceptron SDK and models.")

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
}

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


def _resolve_image(image: str) -> str | bytes:
    if image.startswith(("http://", "https://")):
        return image
    path = Path(image)
    if path.is_dir():
        raise ValueError(f"Expected image file, received directory: {image}")
    if path.exists():
        return path.read_bytes()
    return image


def _iter_image_files(directory: Path) -> Iterable[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTENSIONS:
            yield entry


def _serialize_single_point(point: SinglePoint) -> Dict[str, Any]:
    data: Dict[str, Any] = {"x": point.x, "y": point.y}
    if point.mention is not None:
        data["mention"] = point.mention
    if point.t is not None:
        data["t"] = point.t
    return data


def _serialize_annotation(annotation: Any) -> Any:
    if isinstance(annotation, SinglePoint):
        return {"type": "point", **_serialize_single_point(annotation)}
    if isinstance(annotation, BoundingBox):
        data: Dict[str, Any] = {
            "type": "box",
            "top_left": _serialize_single_point(annotation.top_left),
            "bottom_right": _serialize_single_point(annotation.bottom_right),
        }
        if annotation.mention is not None:
            data["mention"] = annotation.mention
        if annotation.t is not None:
            data["t"] = annotation.t
        return data
    if isinstance(annotation, Polygon):
        data = {
            "type": "polygon",
            "points": [_serialize_single_point(pt) for pt in annotation.hull],
        }
        if annotation.mention is not None:
            data["mention"] = annotation.mention
        if annotation.t is not None:
            data["t"] = annotation.t
        return data
    if isinstance(annotation, Collection):
        data = {
            "type": "collection",
            "points": [_serialize_annotation(pt) for pt in annotation.points],
        }
        if annotation.mention is not None:
            data["mention"] = annotation.mention
        if annotation.t is not None:
            data["t"] = annotation.t
        return data
    return annotation


def _serialize_points(points: Optional[List[Any]]) -> Optional[List[Any]]:
    if not points:
        return None
    return [_serialize_annotation(point) for point in points]


def _serialize_parsed(parsed: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not parsed:
        return None
    serialized: List[Dict[str, Any]] = []
    for segment in parsed:
        if not isinstance(segment, dict):
            serialized.append({"kind": "unknown", "value": str(segment)})
            continue
        seg_copy = dict(segment)
        kind = seg_copy.get("kind")
        if kind in {"point", "box", "polygon", "collection"} and "value" in seg_copy:
            try:
                seg_copy["value"] = _serialize_annotation(seg_copy["value"])
            except Exception:
                seg_copy["value"] = str(seg_copy.get("value"))
        serialized.append(seg_copy)
    return serialized


def _result_payload(result: Any, *, include_raw: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    text_value = getattr(result, "text", None)
    if text_value is not None:
        payload["text"] = text_value
    points = _serialize_points(getattr(result, "points", None))
    if points is not None:
        payload["points"] = points
    parsed = getattr(result, "parsed", None)
    serialized_parsed = _serialize_parsed(parsed)
    if serialized_parsed is not None:
        payload["parsed"] = serialized_parsed
    usage = getattr(result, "usage", None)
    if usage:
        payload["usage"] = usage
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
                f"Streaming output is not supported when processing a directory for '{command_name}'.",
                title=command_name.capitalize(),
                border_style="red",
            )
        )
        raise typer.Exit(code=2)

    image_files = list(_iter_image_files(directory))
    if not image_files:
        console.print(
            Panel(
                f"No image files found in {directory}",
                title=command_name.capitalize(),
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    outputs: Dict[str, Any] = {}
    errors: List[tuple[str, Dict[str, Any]]] = []

    for image_path in image_files:
        try:
            image_bytes = image_path.read_bytes()
        except Exception as exc:
            console.print(
                Panel(str(exc), title=f"Error reading: {image_path.name}", border_style="red")
            )
            continue

        try:
            result = runner(image_bytes)
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(Panel(str(exc), title=f"Error: {image_path.name}", border_style="red"))
            continue

        outputs[image_path.name] = payload_factory(result)

        if getattr(result, "errors", None):
            for err in result.errors:
                errors.append((image_path.name, err))

        if show_raw and getattr(result, "raw", None):
            console.print(Panel(result.raw, title=f"Raw: {image_path.name}", border_style="cyan"))

    if not outputs:
        console.print(
            Panel(
                f"No successful {command_name} results produced in {directory}",
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
            f"Wrote {command_name} results for {len(outputs)} file(s) to {output_path}",
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
            table.add_row(filename, str(err.get("code")), str(err.get("message")))
        console.print(Panel(table, title="Errors", border_style="red"))


def _caption_payload(result: Any) -> Any:
    text_value = getattr(result, "text", None) or ""
    points = _serialize_points(getattr(result, "points", None))
    if points:
        return {"text": text_value, "points": points}
    return text_value


def _ocr_payload(result: Any) -> str:
    return getattr(result, "text", None) or ""


def _detect_payload(result: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": getattr(result, "text", None) or ""}
    points = _serialize_points(getattr(result, "points", None))
    if points is not None:
        payload["points"] = points
    parsed = getattr(result, "parsed", None)
    if parsed:
        payload["parsed"] = parsed
    usage = getattr(result, "usage", None)
    if usage:
        payload["usage"] = usage
    return payload


def _print_errors(errors):
    if not errors:
        return
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("code")
    table.add_column("message")
    for err in errors:
        table.add_row(str(err.get("code")), str(err.get("message")))
    console.print(Panel(table, title="Errors", border_style="red"))


def _describe_point(point: Any) -> tuple[str, str, str]:
    """Return a tuple describing the point for streaming displays."""

    if isinstance(point, BoundingBox):
        coords = f"({point.top_left.x},{point.top_left.y}) → ({point.bottom_right.x},{point.bottom_right.y})"
        return ("box", coords, point.mention or "")
    if isinstance(point, SinglePoint):
        coords = f"({point.x},{point.y})"
        return ("point", coords, point.mention or "")
    if isinstance(point, Polygon):
        coords = ", ".join(f"({p.x},{p.y})" for p in point.hull[:4])
        if len(point.hull) > 4:
            coords += ", …"
        return ("polygon", coords, point.mention or "")
    if isinstance(point, Collection):
        return ("collection", f"{len(point.points)} items", point.mention or "")
    return (type(point).__name__, str(point), getattr(point, "mention", "") or "")


def _build_points_table(points: Iterable[Any]) -> Table:
    table = Table(title="Points", show_header=True, header_style="bold blue")
    table.add_column("type")
    table.add_column("coords")
    table.add_column("mention")
    for point in points:
        kind, coords, mention = _describe_point(point)
        table.add_row(kind, coords, mention)
    return table


def _dedupe_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple[Any, Any]] = set()
    unique: List[Dict[str, Any]] = []
    for err in errors:
        code = err.get("code")
        message = err.get("message")
        key = (code, message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(err)
    return unique


def _coerce_result_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": result.get("text"),
        "points": result.get("points"),
        "parsed": result.get("parsed"),
        "usage": result.get("usage"),
        "errors": result.get("errors") or [],
        "raw": result.get("raw"),
    }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):  # avoid True -> 1
            return 1 if value else 0
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_usage(usage: Any) -> Dict[str, Any]:
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "_asdict"):
        return dict(usage._asdict())  # type: ignore[attr-defined]
    if hasattr(usage, "__dict__"):
        return dict(getattr(usage, "__dict__"))
    return {}


def _resolve_usage_tokens(usage: Optional[Dict[str, Any]]) -> tuple[Optional[int], Optional[int]]:
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


def _stream_render(
    events: Iterable[Dict[str, Any]],
    *,
    title: str,
    output_format: OutputFormat,
    show_raw: bool,
    show_points_table: bool,
) -> None:
    """Render streaming events inside a live-updating panel."""

    text_buffer: List[str] = []
    points_buffer: List[Any] = []
    errors: List[Dict[str, Any]] = []
    final_result: Dict[str, Any] | None = None
    usage_info: Optional[Dict[str, Any]] = None

    start_ts = time.perf_counter()
    first_token_delta: Optional[float] = None
    last_token_ts: Optional[float] = None
    latency_samples: List[float] = []
    delta_event_count = 0
    end_ts: Optional[float] = None

    def current_panel() -> Panel:
        body: List[Any] = []
        text_content = "".join(text_buffer)
        text_render = Text(text_content or "<waiting for response…>")
        if not text_content:
            text_render.stylize("dim")
        body.append(text_render)
        if points_buffer:
            if show_points_table:
                body.append(_build_points_table(points_buffer))
            else:
                summary = Text()
                for idx, point in enumerate(points_buffer, 1):
                    kind, coords, mention = _describe_point(point)
                    line = f"{idx}. {kind}: {coords}"
                    if mention:
                        line += f" ({mention})"
                    summary.append(line + "\n")
                body.append(summary)
        # metrics summary
        now = time.perf_counter()
        metrics_parts: List[str] = []
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

        tokens_in_display = (
            str(usage_tokens_in)
            if usage_tokens_in is not None
            else ("—" if usage_info is None else "—")
        )
        if usage_tokens_out is not None:
            tokens_out_display = str(usage_tokens_out)
        else:
            tokens_out_display = f"~{delta_event_count}" if delta_event_count else "—"
        metrics_parts.append(f"Tokens in {tokens_in_display}")
        metrics_parts.append(f"Tokens out {tokens_out_display}")

        metrics_text = Text(" | ".join(metrics_parts), style="dim")
        body.append(metrics_text)

        content = body[0] if len(body) == 1 else Group(*body)
        return Panel(content, title=title, border_style="cyan")

    live_panel = current_panel()
    with Live(live_panel, console=console, refresh_per_second=12) as live:
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
            elif event_type == "points.delta":
                pts = event.get("points") or []
                if pts:
                    points_buffer.extend(pts)
            elif event_type == "error":
                message = str(event.get("message") or "unknown error")
                errors.append({"code": "stream_error", "message": message})
                end_ts = time.perf_counter()
                break
            elif event_type == "final":
                final_result = event.get("result") or {}
                end_ts = time.perf_counter()
                if final_result.get("text") is not None:
                    text_buffer = [final_result.get("text") or ""]
                if final_result.get("points") is not None:
                    points_buffer = list(final_result.get("points") or [])
                final_errs = final_result.get("errors") or []
                if final_errs:
                    errors.extend(final_errs)
                if final_result.get("usage"):
                    usage_info = _normalize_usage(final_result.get("usage"))
            live.update(current_panel())

    if end_ts is None:
        end_ts = time.perf_counter()

    if final_result is None:
        final_result = {
            "text": "".join(text_buffer) or None,
            "points": points_buffer or None,
            "parsed": None,
            "usage": usage_info,
            "errors": _dedupe_errors(errors),
            "raw": None,
        }
    else:
        # ensure buffers win if final result lacked data
        if final_result.get("text") is None:
            final_result["text"] = "".join(text_buffer) or None
        if not final_result.get("points") and points_buffer:
            final_result["points"] = points_buffer
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
        console.print_json(data=_result_payload(result_ns, include_raw=show_raw))
    else:
        if coerced["errors"]:
            _print_errors(coerced["errors"])
        if show_raw and coerced.get("raw") is not None:
            console.print(coerced["raw"])

def _render_result(
    result: Any,
    *,
    title: str,
    output_format: OutputFormat,
    show_raw: bool,
    show_points_table: bool = False,
):
    if output_format is OutputFormat.JSON:
        payload = _result_payload(result, include_raw=show_raw)
        console.print_json(data=payload)
        return

    points_serialized = _serialize_points(getattr(result, "points", None))
    console.print(Panel(result.text or "<no text>", title=title, border_style="green"))
    if show_points_table and getattr(result, "points", None):
        table = Table(title="Detections", show_header=True, header_style="bold blue")
        table.add_column("Bounding Box")
        table.add_column("Mention")
        for point in result.points or []:
            table.add_row(str(point), getattr(point, "mention", ""))
        console.print(table)
    elif points_serialized:
        console.print_json(data={"points": points_serialized})
    _print_errors(getattr(result, "errors", []))
    if show_raw and getattr(result, "raw", None):
        console.print(result.raw)


@app.command()
def config(
    provider: Optional[str] = typer.Option(None, help="Default provider identifier."),
    api_key: Optional[str] = typer.Option(None, help="API key to export."),
    base_url: Optional[str] = typer.Option(None, help="Optional custom base URL."),
):
    """Show shell commands to export credentials."""

    exports: List[str] = []
    if provider:
        exports.append(f"export PERCEPTRON_PROVIDER={provider}")
    if api_key:
        exports.append(f"export PERCEPTRON_API_KEY={api_key}")
    if base_url:
        exports.append(f"export PERCEPTRON_BASE_URL={base_url}")

    if not exports:
        exports = [
            "export PERCEPTRON_PROVIDER=<provider>",
            "export PERCEPTRON_API_KEY=<your-key>",
            "export PERCEPTRON_BASE_URL=<optional-base-url>",
        ]

    console.print(Panel("\n".join(exports), title="Add these to your shell", border_style="cyan"))


@app.command()
def caption(
    image: str = typer.Argument(..., help="Image path or URL."),
    style: str = typer.Option("concise", help="Captioning style."),
    stream: bool = typer.Option(False, help="Stream incremental output."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    expects: ExpectationType = typer.Option(
        ExpectationType.BOX,
        "--expects",
        case_sensitive=False,
        help="Expected output structure (text, point, box, or polygon).",
    ),
):
    """Generate captions using the high-level helper."""

    path = Path(image)
    if path.is_dir():
        _process_directory(
            path,
            command_name="caption",
            stream=stream,
            show_raw=show_raw,
            runner=lambda data: caption_image(data, style=style, expects=expects.value),
            payload_factory=_caption_payload,
        )
        return

    try:
        img = _resolve_image(image)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    expects_value = expects.value

    if stream:
        _stream_render(
            caption_image(img, style=style, expects=expects_value, stream=True),
            title="Caption",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=expects is ExpectationType.BOX,
        )
        return

    res = caption_image(img, style=style, expects=expects_value)
    _render_result(
        res,
        title="Caption",
        output_format=output_format,
        show_raw=show_raw,
        show_points_table=expects is ExpectationType.BOX,
    )


@app.command()
def ocr(
    image: str = typer.Argument(..., help="Image path or URL."),
    prompt: Optional[str] = typer.Option(None, help="Optional instruction override."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
):
    """Run OCR via the high-level helper."""

    path = Path(image)
    if path.is_dir():
        _process_directory(
            path,
            command_name="ocr",
            stream=False,
            show_raw=show_raw,
            runner=lambda data: ocr_image(data, prompt=prompt),
            payload_factory=_ocr_payload,
        )
        return

    img = _resolve_image(image)
    res = ocr_image(img, prompt=prompt)
    _render_result(
        res,
        title="OCR",
        output_format=output_format,
        show_raw=show_raw,
    )


@app.command()
def detect(
    image: str = typer.Argument(..., help="Image path or URL."),
    classes: Optional[str] = typer.Option(None, help="Comma-separated class list."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
    stream: bool = typer.Option(False, help="Stream incremental output."),
):
    """Run detection via the high-level helper."""

    class_list = [c.strip() for c in classes.split(",")] if classes else None
    path = Path(image)
    if path.is_dir():
        _process_directory(
            path,
            command_name="detect",
            stream=False,
            show_raw=show_raw,
            runner=lambda data: detect_image(data, classes=class_list),
            payload_factory=_detect_payload,
        )
        return

    img = _resolve_image(image)
    if stream:
        _stream_render(
            detect_image(img, classes=class_list, stream=True),
            title="Detect",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=True,
        )
        return
    res = detect_image(img, classes=class_list)
    _render_result(
        res,
        title="Detect",
        output_format=output_format,
        show_raw=show_raw,
        show_points_table=True,
    )


@app.command()
def question(
    image: str = typer.Argument(..., help="Image path or URL."),
    prompt: str = typer.Argument(..., help="Question to answer about the image."),
    expects: ExpectationType = typer.Option(
        ExpectationType.TEXT,
        "--expects",
        case_sensitive=False,
        help="Expected output structure (text, point, box, or polygon).",
    ),
    stream: bool = typer.Option(False, help="Stream incremental output."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (text or json).",
    ),
):
    """Answer a question about an image."""

    path = Path(image)
    if path.is_dir():
        raise typer.BadParameter("Directory mode is not supported for 'question'.")

    try:
        img = _resolve_image(image)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    expects_value = expects.value

    if stream:
        _stream_render(
            question_image(img, prompt, expects=expects_value, stream=True),
            title="Question",
            output_format=output_format,
            show_raw=show_raw,
            show_points_table=expects is ExpectationType.BOX,
        )
        return

    res = question_image(img, prompt, expects=expects_value)
    show_points = expects in {ExpectationType.POINT, ExpectationType.BOX, ExpectationType.POLYGON}
    _render_result(
        res,
        title="Question",
        output_format=output_format,
        show_raw=show_raw,
        show_points_table=show_points and expects is ExpectationType.BOX,
    )
def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
