"""FastMCP server exposing Perceptron helpers."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - triggered when optional extra missing
    raise SystemExit(
        "perceptron-mcp requires the optional MCP dependencies.\n"
        'Install them via `pip install perceptron[mcp]` or `uv pip install "perceptron[mcp]"`.'
    ) from exc

from perceptron import BadRequestError, PerceiveResult, caption, configure, detect, ocr, question
from perceptron.pointing.types import BoundingBox, Collection, Polygon, SinglePoint

app = FastMCP(
    "perceptron",
    instructions="Expose Perceptron caption/detect/OCR/question helpers to MCP clients.",
)


def _serialize_pointlike(point: Any) -> Any:
    serialized: Any
    if isinstance(point, SinglePoint):
        serialized = {
            "type": "point",
            "x": point.x,
            "y": point.y,
            "mention": point.mention,
            "t": point.t,
        }
    elif isinstance(point, BoundingBox):
        serialized = {
            "type": "box",
            "top_left": _serialize_pointlike(point.top_left),
            "bottom_right": _serialize_pointlike(point.bottom_right),
            "mention": point.mention,
            "t": point.t,
        }
    elif isinstance(point, Polygon):
        serialized = {
            "type": "polygon",
            "hull": [_serialize_pointlike(p) for p in point.hull],
            "mention": point.mention,
            "t": point.t,
        }
    elif isinstance(point, Collection):
        serialized = {
            "type": "collection",
            "points": [_serialize_pointlike(p) for p in point.points],
            "mention": point.mention,
            "t": point.t,
        }
    elif isinstance(point, list):
        serialized = [_serialize_pointlike(p) for p in point]
    else:
        serialized = point
    return serialized


def _format_result(result: PerceiveResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "points": [_serialize_pointlike(p) for p in result.points or []] if result.points is not None else None,
        "parsed": result.parsed,
        "errors": result.errors,
    }


def _ensure_non_empty_payload(payload: dict[str, Any]) -> dict[str, Any]:
    updates = {key: value for key, value in payload.items() if value is not None}
    if not updates:
        raise BadRequestError("At least one configuration field must be provided.")
    return updates


def _execute_parallel(
    func: Any,
    images: Sequence[str],
    *args: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute a perceptron function in parallel for multiple images."""
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_index = {executor.submit(func, img, *args, **kwargs): idx for idx, img in enumerate(images)}

        # Collect results in order
        indexed_results: dict[int, dict[str, Any]] = {}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                indexed_results[idx] = _format_result(result)
            except Exception as exc:
                # Include errors in the result
                indexed_results[idx] = {
                    "text": None,
                    "points": None,
                    "parsed": None,
                    "errors": [{"message": str(exc), "type": type(exc).__name__}],
                }

        # Return in original order
        results = [indexed_results[i] for i in range(len(images))]

    return results


@app.tool(description="Update the global Perceptron SDK defaults (provider, model, API key, etc.).")
def set_defaults(  # noqa: PLR0913
    provider: Annotated[str | None, "Provider identifier, e.g., 'perceptron' or 'fal'."] = None,
    api_key: Annotated[str | None, "API key used by the selected provider."] = None,
    base_url: Annotated[str | None, "Override base URL for on-prem or staging clusters."] = None,
    model: Annotated[str | None, "Default model name; leave unset to use provider defaults."] = None,
    timeout: Annotated[float | None, "Request timeout in seconds."] = None,
    max_tokens: Annotated[int | None, "Maximum tokens returned per completion."] = None,
    temperature: Annotated[float | None, "Sampling temperature (0 = deterministic)."] = None,
) -> dict[str, Any]:
    """Override Perceptron configure() defaults for subsequent tool invocations."""

    updates = _ensure_non_empty_payload(
        {
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    configure(**updates)
    return {"updated": sorted(updates.keys())}


@app.tool(description="Generate a caption for one or more images (paths, URLs, or bytes-like objects).")
def caption_image(
    image: Annotated[str | Sequence[str], "Local path, http(s) URL, or list of paths/URLs to images."],
    *,
    style: Annotated[str, "Prompt style defined by the active model profile (e.g., concise, detailed)."] = "concise",
    expects: Annotated[str, "Output structure: text (default) or point/box/polygon for grounded captions."] = "text",
    model: Annotated[str | None, "Override the model for this call."] = None,
    provider: Annotated[str | None, "Override the provider for this call."] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Generate captions for one or more images. Returns a list of results when multiple images are provided."""

    # Handle single image
    if isinstance(image, str):
        result = caption(image, style=style, expects=expects, model=model, provider=provider)
        return _format_result(result)

    # Handle multiple images with parallel execution
    return _execute_parallel(
        caption,
        image,
        style=style,
        expects=expects,
        model=model,
        provider=provider,
    )


@app.tool(description="Detect custom classes in one or more images and return normalized bounding boxes.")
def detect_objects(
    image: Annotated[str | Sequence[str], "Local path, http(s) URL, or list of paths/URLs to images."],
    *,
    classes: Annotated[
        Sequence[str] | None,
        "Optional class labels to bias the detector; omit for open-set detection.",
    ] = None,
    max_outputs: Annotated[int | None, "Cap the number of returned boxes; defaults to model settings."] = None,
    model: Annotated[str | None, "Override the model for this call."] = None,
    provider: Annotated[str | None, "Override the provider for this call."] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run grounded detection on one or more images. Returns a list of results when multiple images are provided."""

    # Handle single image
    if isinstance(image, str):
        result = detect(image, classes=classes, max_outputs=max_outputs, model=model, provider=provider)
        return _format_result(result)

    # Handle multiple images with parallel execution
    return _execute_parallel(
        detect,
        image,
        classes=classes,
        max_outputs=max_outputs,
        model=model,
        provider=provider,
    )


@app.tool(description="Extract text from one or more images via OCR, optionally with a custom prompt.")
def ocr_extract(
    image: Annotated[str | Sequence[str], "Local path, http(s) URL, or list of paths/URLs to images."],
    *,
    prompt: Annotated[str | None, "Override the default OCR prompt (plain text, markdown, etc.)."] = None,
    model: Annotated[str | None, "Override the model for this call."] = None,
    provider: Annotated[str | None, "Override the provider for this call."] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run OCR on one or more images. Returns a list of results when multiple images are provided."""

    # Handle single image
    if isinstance(image, str):
        result = ocr(image, prompt=prompt, model=model, provider=provider)
        return _format_result(result)

    # Handle multiple images with parallel execution
    return _execute_parallel(
        ocr,
        image,
        prompt=prompt,
        model=model,
        provider=provider,
    )


@app.tool(description="Ask a question about one or more images, optionally requesting grounded answers.")
def ask_image(
    image: Annotated[str | Sequence[str], "Local path, http(s) URL, or list of paths/URLs to images."],
    question_text: Annotated[str, "Natural-language question to pose about the scene(s)."],
    *,
    expects: Annotated[str, "Output structure: text, point, box, or polygon when you need grounded answers."] = "text",
    model: Annotated[str | None, "Override the model for this call."] = None,
    provider: Annotated[str | None, "Override the provider for this call."] = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Ask questions about one or more images. Returns a list of results when multiple images are provided."""

    # Handle single image
    if isinstance(image, str):
        result = question(image, question_text, expects=expects, model=model, provider=provider)
        return _format_result(result)

    # Handle multiple images with parallel execution
    return _execute_parallel(
        question,
        image,
        question_text,
        expects=expects,
        model=model,
        provider=provider,
    )


def main() -> None:
    """Entrypoint for the `perceptron-mcp` console script."""

    app.run()


if __name__ == "__main__":
    main()
