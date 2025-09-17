"""Perceptron CLI utilities built with Typer + Rich."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import caption as caption_image
from . import detect as detect_image
from . import ocr as ocr_image
from .client import Client
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
        raise typer.BadParameter(
            f"Streaming output is not supported when processing a directory for '{command_name}'."
        )

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


def _caption_payload(result: Any) -> str:
    return getattr(result, "text", None) or ""


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
):
    """Generate captions using the high-level helper."""

    path = Path(image)
    if path.is_dir():
        _process_directory(
            path,
            command_name="caption",
            stream=stream,
            show_raw=show_raw,
            runner=lambda data: caption_image(data, style=style),
            payload_factory=_caption_payload,
        )
        return

    try:
        img = _resolve_image(image)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if stream:
        console.print(Panel("Streaming caption", title="Caption", border_style="green"))
        for event in caption_image(img, style=style, stream=True):
            console.print(event)
        return

    res = caption_image(img, style=style)
    console.print(Panel(res.text or "<no text>", title="Caption", border_style="green"))
    _print_errors(res.errors)
    if show_raw:
        console.print(res.raw)


@app.command()
def ocr(
    image: str = typer.Argument(..., help="Image path or URL."),
    prompt: Optional[str] = typer.Option(None, help="Optional instruction override."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
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
    console.print(Panel(res.text or "<no text>", title="OCR", border_style="green"))
    _print_errors(res.errors)
    if show_raw:
        console.print(res.raw)


@app.command()
def detect(
    image: str = typer.Argument(..., help="Image path or URL."),
    classes: Optional[str] = typer.Option(None, help="Comma-separated class list."),
    show_raw: bool = typer.Option(False, help="Display raw response JSON."),
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
    res = detect_image(img, classes=class_list)
    console.print(Panel(res.text or "<no text>", title="Detect", border_style="green"))
    if res.points:
        table = Table(title="Detections", show_header=True, header_style="bold blue")
        table.add_column("Bounding Box")
        table.add_column("Mention")
        for point in res.points:
            table.add_row(str(point), getattr(point, "mention", ""))
        console.print(table)
    _print_errors(res.errors)
    if show_raw:
        console.print(res.raw)


@app.command()
def chat(
    user: str = typer.Argument(..., help="User message."),
    system: Optional[str] = typer.Option(None, help="System instruction."),
    provider: Optional[str] = typer.Option(None, help="Override provider for this call."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    stream: bool = typer.Option(False, help="Stream incremental output."),
):
    """Send a low-level chat-style request."""

    task_content = []
    if system:
        task_content.append({"type": "text", "role": "system", "content": system})
    task_content.append({"type": "text", "role": "user", "content": user})
    task = {"content": task_content}
    client = Client()

    if stream:
        console.print(Panel("Streaming chat", title="Chat", border_style="cyan"))
        for event in client.stream(task, provider=provider, temperature=temperature):
            console.print(event)
        return

    res = client.generate(task, provider=provider, temperature=temperature)
    console.print(Panel(res.get("text") or "<no text>", title="Chat", border_style="cyan"))
    if res.get("raw"):
        console.print(res["raw"])


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
