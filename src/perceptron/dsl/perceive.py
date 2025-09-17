"""DSL compiler and `@perceive` decorator.

Compiles typed nodes (text/image/point/box/polygon) into a Task shape and
optionally executes it via the Client. Performs compile-time validation of
anchoring and bounds, returning issues (non-strict) or raising (strict).

PerceiveResult
- text: final text (if executed)
- points: list of parsed pointing objects if `expects` set and present
- parsed: ordered segments mixing text and all tags with spans
- errors: semantic/validation issues from compilation/streaming
- raw: provider response or compiled Task for compile-only
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import inspect
from typing import Any, Callable
import os
from urllib.parse import urlparse

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

import requests

from ..config import settings
from ..client import Client, AsyncClient
from ..errors import ExpectationError, AnchorError
from .nodes import (
    DSLNode,
    Sequence,
    Text,
    System,
    Agent,
    Image as ImageNode,
    PointTag as PointTagNode,
    BoxTag as BoxTagNode,
    PolygonTag as PolygonTagNode,
)
from ..pointing.types import SinglePoint, BoundingBox, Polygon
from ..pointing.parser import PointParser_serialize


def _encode_bytes(data: bytes) -> tuple[str, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if PILImage is not None:
        try:
            with PILImage.open(BytesIO(data)) as im:
                meta["width"], meta["height"] = im.size
        except Exception:
            pass
    b64 = base64.b64encode(data).decode("ascii")
    return b64, meta


def _to_b64_image(obj: Any) -> tuple[str, dict]:
    """Return base64 string and metadata with width/height.

    Accepts: Path/str (path or http/https URL), bytes, file-like, PIL.Image.Image, numpy.ndarray (H×W×C, uint8)
    """
    meta: dict[str, Any] = {}
    if isinstance(obj, (str, Path)):
        if isinstance(obj, str):
            parsed = urlparse(obj)
            if parsed.scheme in {"http", "https"}:
                return obj, {}
        p = Path(obj)
        with open(p, "rb") as f:
            data = f.read()
        if PILImage is not None:
            try:
                with PILImage.open(p) as im:
                    meta["width"], meta["height"] = im.size
            except Exception:
                pass
        b64 = base64.b64encode(data).decode("ascii")
        return b64, meta
    if isinstance(obj, bytes):
        b64, meta = _encode_bytes(obj)
        return b64, meta
    if PILImage is not None and isinstance(obj, PILImage.Image):  # type: ignore[attr-defined]
        meta["width"], meta["height"] = obj.size
        buf = BytesIO()
        obj.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, meta
    if np is not None and isinstance(obj, np.ndarray):  # type: ignore[arg-type]
        h, w = obj.shape[:2]
        meta["width"], meta["height"] = int(w), int(h)
        if PILImage is None:
            raise RuntimeError("Pillow is required to encode numpy arrays to PNG")
        im = PILImage.fromarray(obj)
        buf = BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, meta
    raise TypeError(f"Unsupported image object: {type(obj)}")


def _compile(nodes: DSLNode | Sequence, *, expects: str | None, strict: bool) -> tuple[dict, list[dict]]:
    """Compile DSL nodes into a Task JSON and return (task, issues)."""
    seq = nodes if isinstance(nodes, Sequence) else Sequence([nodes])
    content: list[dict[str, Any]] = []
    image_nodes: list[ImageNode] = [n for n in seq.nodes if isinstance(n, ImageNode)]
    total_images = len(image_nodes)
    image_dims: dict[int, tuple[int | None, int | None]] = {}
    last_image_seen: ImageNode | None = None
    issues: list[dict] = []

    def resolve_dims(img_node: ImageNode | None) -> tuple[int | None, int | None] | None:
        if img_node is None:
            return None
        dims = image_dims.get(id(img_node))
        if dims is not None:
            return dims
        try:
            _, meta = _to_b64_image(img_node.obj)
            dims = (meta.get("width"), meta.get("height"))
            image_dims[id(img_node)] = dims
            return dims
        except Exception:
            return (None, None)

    for node in seq.nodes:
        if isinstance(node, Text):
            content.append({"type": "text", "role": "user", "content": node.content})
        elif isinstance(node, System):
            content.append({"type": "text", "role": "system", "content": node.content})
        elif isinstance(node, Agent):
            content.append({"type": "text", "role": "assistant", "content": node.content})
        elif isinstance(node, ImageNode):
            b64, meta = _to_b64_image(node.obj)
            image_dims[id(node)] = (meta.get("width"), meta.get("height"))
            last_image_seen = node
            content.append(
                {
                    "type": "image",
                    "role": "user",
                    "content": b64,
                    "metadata": {"width": meta.get("width"), "height": meta.get("height")},
                }
            )
        elif isinstance(node, (PointTagNode, BoxTagNode, PolygonTagNode)):
            ref: ImageNode | None = node.image
            dims = None
            if ref is None:
                if total_images == 1 and last_image_seen is not None:
                    ref = last_image_seen
                    dims = resolve_dims(ref)
                else:
                    issue = {"code": "anchor_missing", "message": "Tag missing image= in multi-image context"}
                    if strict:
                        raise AnchorError(issue["message"])
                    issues.append(issue)
            else:
                if isinstance(ref, ImageNode):
                    dims = resolve_dims(ref)
                else:
                    issue = {"code": "anchor_missing", "message": "image= must reference an image(...) node"}
                    if strict:
                        raise AnchorError(issue["message"])
                    issues.append(issue)

            if isinstance(node, PointTagNode):
                obj = SinglePoint(node.x, node.y, mention=node.mention, t=node.t)
                if dims and all(d is not None for d in dims):
                    w, h = dims
                    if not (0 <= obj.x <= (w - 1) and 0 <= obj.y <= (h - 1)):
                        issue = {
                            "code": "bounds_out_of_range",
                            "message": f"point ({obj.x},{obj.y}) outside image bounds ({w}x{h})",
                        }
                        if strict:
                            raise ExpectationError(issue["message"])
                        issues.append(issue)
                tag = PointParser_serialize(obj)
            elif isinstance(node, BoxTagNode):
                obj = BoundingBox(
                    SinglePoint(node.x1, node.y1), SinglePoint(node.x2, node.y2), mention=node.mention, t=node.t
                )
                if dims and all(d is not None for d in dims):
                    w, h = dims
                    x1, y1 = obj.top_left.x, obj.top_left.y
                    x2, y2 = obj.bottom_right.x, obj.bottom_right.y
                    ok = (0 <= x1 <= x2 <= (w - 1)) and (0 <= y1 <= y2 <= (h - 1))
                    if not ok:
                        issue = {
                            "code": "bounds_out_of_range",
                            "message": f"box coords out of bounds or invalid for image ({w}x{h})",
                        }
                        if strict:
                            raise ExpectationError(issue["message"])
                        issues.append(issue)
                tag = PointParser_serialize(obj)
            else:
                obj = Polygon([SinglePoint(x, y) for (x, y) in node.coords], mention=node.mention, t=node.t)
                if dims and all(d is not None for d in dims):
                    w, h = dims
                    for p in obj.hull:
                        if not (0 <= p.x <= (w - 1) and 0 <= p.y <= (h - 1)):
                            issue = {
                                "code": "bounds_out_of_range",
                                "message": f"polygon contains point ({p.x},{p.y}) outside image bounds ({w}x{h})",
                            }
                            if strict:
                                raise ExpectationError(issue["message"])
                            issues.append(issue)
                tag = PointParser_serialize(obj)
            content.append({"type": "text", "role": "user", "content": tag})
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    task = {"content": content, "expects": expects}
    return task, issues


@dataclass
class PerceiveResult:
    text: str | None
    points: list[Any] | None
    parsed: list[dict] | None
    usage: dict | None
    errors: list[dict]
    raw: Any


def perceive(
    *,
    visual_reasoning: str | None = None,
    expects: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    top_k: int | None = None,
    strict: bool = False,
    allow_multiple: bool = False,
    max_outputs: int | None = 1,
    stream: bool = False,
):
    """Decorator for building Tasks from DSL nodes.

    Executes via the default Client unless compile-only fallback is triggered
    (no provider configured or the selected provider lacks credentials).
    """

    def wrapper(fn: Callable[..., Any]):
        def _call(*args: Any, **kwargs: Any):
            nodes = fn(*args, **kwargs)
            task, issues = _compile(nodes, expects=expects, strict=strict)
            client = Client()
            # Resolve provider but avoid forcing remote execution in tests/local
            env = settings()
            resolved_provider = provider or env.provider
            provider_name = resolved_provider or "fal"

            # Compile-only fallback: if no explicit provider and no API key configured,
            # return the compiled task without executing a request.
            if not stream:
                if resolved_provider is None:
                    # No configured provider → compile-only with credential hint
                    errors_with_hint = [*issues, _credentials_issue(provider_name)]
                    return PerceiveResult(
                        text=None,
                        points=None,
                        parsed=None,
                        usage=None,
                        errors=errors_with_hint,
                        raw=task,
                    )
                if not _has_credentials(provider_name, env):
                    errors_with_hint = [*issues, _credentials_issue(provider_name)]
                    return PerceiveResult(
                        text=None,
                        points=None,
                        parsed=None,
                        usage=None,
                        errors=errors_with_hint,
                        raw=task,
                    )

            if stream:
                # Delegate to client.stream; pass through events
                return client.stream(
                    task,
                    expects=expects,
                    parse_points=expects in {"point", "box", "polygon"},
                    provider=provider_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    allow_multiple=allow_multiple,
                    max_outputs=max_outputs,
                )
            else:
                try:
                    resp = client.generate(
                        task,
                        expects=expects,
                        provider=provider_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        allow_multiple=allow_multiple,
                        max_outputs=max_outputs,
                    )
                except TypeError:
                    # Support tests that monkeypatch Client.generate as a @staticmethod
                    gen = getattr(type(client), "generate", None)
                    if callable(gen):
                        resp = gen(
                            task,
                            expects=expects,
                            provider=provider_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k,
                            allow_multiple=allow_multiple,
                            max_outputs=max_outputs,
                        )
                    else:
                        raise
                text = resp.get("text")
                points = resp.get("points")
                parsed = resp.get("parsed")
                return PerceiveResult(
                    text=text, points=points, parsed=parsed, usage=None, errors=issues, raw=resp.get("raw")
                )

        def _inspect(*args: Any, **kwargs: Any):
            nodes_local = fn(*args, **kwargs)
            return _compile(nodes_local, expects=expects, strict=strict)

        _call.__perceptron_inspector__ = _inspect  # type: ignore[attr-defined]

        return _call

    return wrapper


def async_perceive(
    *,
    visual_reasoning: str | None = None,
    expects: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    top_k: int | None = None,
    strict: bool = False,
    allow_multiple: bool = False,
    max_outputs: int | None = 1,
    stream: bool = False,
):
    """Async counterpart to ``perceive`` using :class:`AsyncClient`."""

    def wrapper(fn: Callable[..., Any]):
        async def _prepare_nodes(*args: Any, **kwargs: Any):
            nodes = fn(*args, **kwargs)
            if inspect.isawaitable(nodes):
                nodes = await nodes
            return _compile(nodes, expects=expects, strict=strict)

        if stream:

            def _call(*args: Any, **kwargs: Any):
                async def _generator():
                    task, _issues = await _prepare_nodes(*args, **kwargs)
                    client = AsyncClient()
                    env = settings()
                    resolved_provider = provider or env.provider
                    provider_name = resolved_provider or "fal"

                    async for event in client.stream(
                        task,
                        expects=expects,
                        parse_points=expects in {"point", "box", "polygon"},
                        provider=provider_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        allow_multiple=allow_multiple,
                        max_outputs=max_outputs,
                    ):
                        yield event

                return _generator()

            async def _inspect_async(*args: Any, **kwargs: Any):
                return await _prepare_nodes(*args, **kwargs)

            _call.__perceptron_inspector__ = _inspect_async  # type: ignore[attr-defined]

            return _call

        async def _call(*args: Any, **kwargs: Any):
            task, issues = await _prepare_nodes(*args, **kwargs)
            client = AsyncClient()
            env = settings()
            resolved_provider = provider or env.provider
            provider_name = resolved_provider or "fal"

            if resolved_provider is None:
                errors_with_hint = [*issues, _credentials_issue(provider_name)]
                return PerceiveResult(text=None, points=None, parsed=None, usage=None, errors=errors_with_hint, raw=task)
            if not _has_credentials(provider_name, env):
                errors_with_hint = [*issues, _credentials_issue(provider_name)]
                return PerceiveResult(text=None, points=None, parsed=None, usage=None, errors=errors_with_hint, raw=task)

            resp = await client.generate(
                task,
                expects=expects,
                provider=provider_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                allow_multiple=allow_multiple,
                max_outputs=max_outputs,
            )
            text = resp.get("text")
            points = resp.get("points")
            parsed = resp.get("parsed")
            return PerceiveResult(text=text, points=points, parsed=parsed, usage=None, errors=issues, raw=resp.get("raw"))

        async def _inspect_async(*args: Any, **kwargs: Any):
            return await _prepare_nodes(*args, **kwargs)

        _call.__perceptron_inspector__ = _inspect_async  # type: ignore[attr-defined]

        return _call

    return wrapper


def inspect_task(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any):
    """Return the compiled Task dict (and issues) for a `perceive`/`async_perceive` function without executing it."""

    inspector = getattr(callable_obj, "__perceptron_inspector__", None)
    if inspector is None:
        raise TypeError("inspect_task expects a function produced by perceive/async_perceive")
    result = inspector(*args, **kwargs)
    return result


__all__ = ["perceive", "async_perceive", "PerceiveResult", "inspect_task"]


def _credentials_issue(provider_name: str) -> dict[str, str]:
    if provider_name == "fal":
        message = (
            "No credentials found for provider 'fal'. Export PERCEPTRON_PROVIDER=fal and "
            "set PERCEPTRON_API_KEY or FAL_KEY (see `perceptron config`)."
        )
    else:
        message = (
            f"No credentials found for provider '{provider_name}'. Set PERCEPTRON_PROVIDER and "
            "the appropriate API key before running."
        )
    return {"code": "credentials_missing", "message": message}


def _has_credentials(provider_name: str, env) -> bool:
    if provider_name == "fal":
        return bool(env.api_key or os.getenv("FAL_KEY") or os.getenv("PERCEPTRON_API_KEY"))
    # Unknown providers default to checking the explicit api_key hook
    return bool(env.api_key)
