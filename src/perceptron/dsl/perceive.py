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
import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..client import _PROVIDER_CONFIG, AsyncClient, Client
from ..config import settings
from ..errors import AnchorError, AuthError, BadRequestError, ExpectationError
from ..pointing.geometry import scale_points_to_pixels
from ..pointing.parser import PointParser_serialize
from ..pointing.types import BoundingBox, Polygon, SinglePoint
from .nodes import (
    Agent,
    DSLNode,
    Sequence,
    System,
    Text,
)
from .nodes import (
    BoxTag as BoxTagNode,
)
from .nodes import (
    Image as ImageNode,
)
from .nodes import (
    PointTag as PointTagNode,
)
from .nodes import (
    PolygonTag as PolygonTagNode,
)

_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",
    b"GIF89a",
    b"BM",  # BMP
    b"II*\x00",  # TIFF (little endian)
    b"MM\x00*",  # TIFF (big endian)
)

_WEBP_SIGNATURE_LENGTH = 12


def _is_webp(data: bytes) -> bool:
    return len(data) >= _WEBP_SIGNATURE_LENGTH and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(sig) for sig in _IMAGE_SIGNATURES) or _is_webp(data)


def _validate_image_bytes(data: bytes, *, origin: str) -> dict[str, Any]:
    """Ensure the payload is an actual bitmap; raise otherwise."""

    meta: dict[str, Any] = {}
    pil_exc: Exception | None = None
    if PILImage is not None:
        try:
            with PILImage.open(BytesIO(data)) as im:
                meta["width"], meta["height"] = im.size
                return meta
        except Exception as exc:  # Let imghdr double-check before failing
            pil_exc = exc
    if not _looks_like_image(data):
        reason = "decoder_failed" if pil_exc is not None else "unknown_format"
        details = {"origin": origin, "reason": reason}
        raise BadRequestError(
            "Image payload is not a decodable bitmap", code="invalid_image", details=details
        ) from pil_exc
    return meta


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

    Accepts: Path/str (path or http/https URL), bytes, file-like, PIL.Image.Image, numpy.ndarray (HxWxC, uint8)
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
        meta = _validate_image_bytes(data, origin=str(p))
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

    def resolve_dims(
        img_node: ImageNode | None,
    ) -> tuple[int | None, int | None] | None:
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
                    "metadata": {
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                    },
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
                    issue = {
                        "code": "anchor_missing",
                        "message": "Tag missing image= in multi-image context",
                    }
                    if strict:
                        raise AnchorError(issue["message"], code=issue.get("code"), details=issue)
                    issues.append(issue)
            elif isinstance(ref, ImageNode):
                dims = resolve_dims(ref)
            else:
                issue = {
                    "code": "anchor_missing",
                    "message": "image= must reference an image(...) node",
                }
                if strict:
                    raise AnchorError(issue["message"], code=issue.get("code"), details=issue)
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
                            raise ExpectationError(issue["message"], code=issue.get("code"), details=issue)
                        issues.append(issue)
                tag = PointParser_serialize(obj)
            elif isinstance(node, BoxTagNode):
                obj = BoundingBox(
                    SinglePoint(node.x1, node.y1),
                    SinglePoint(node.x2, node.y2),
                    mention=node.mention,
                    t=node.t,
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
                            raise ExpectationError(issue["message"], code=issue.get("code"), details=issue)
                        issues.append(issue)
                tag = PointParser_serialize(obj)
            else:
                obj = Polygon(
                    [SinglePoint(x, y) for (x, y) in node.coords],
                    mention=node.mention,
                    t=node.t,
                )
                if dims and all(d is not None for d in dims):
                    w, h = dims
                    for p in obj.hull:
                        if not (0 <= p.x <= (w - 1) and 0 <= p.y <= (h - 1)):
                            issue = {
                                "code": "bounds_out_of_range",
                                "message": f"polygon contains point ({p.x},{p.y}) outside image bounds ({w}x{h})",
                            }
                            if strict:
                                raise ExpectationError(issue["message"], code=issue.get("code"), details=issue)
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

    def points_to_pixels(self, width: int, height: int, *, clamp: bool = True) -> list[Any] | None:
        """Return a pixel-space copy of ``points`` given the image dimensions."""

        return scale_points_to_pixels(self.points, width=width, height=height, clamp=clamp)


def _prepare_client_kwargs(
    *,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    allow_multiple: bool,
    max_outputs: int | None,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
):
    env = settings()
    resolved_provider = provider_override or env.provider
    provider_name = resolved_provider or "fal"
    client_kwargs = {
        "expects": expects,
        "provider": provider_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "allow_multiple": allow_multiple,
        "max_outputs": max_outputs,
    }
    if model_override is not None:
        client_kwargs["model"] = model_override
    return env, resolved_provider, provider_name, client_kwargs


def _maybe_compile_only_result(
    *,
    stream: bool,
    resolved_provider: str | None,
    provider_name: str,
    env,
    issues: list[dict],
    task: dict,
):
    if resolved_provider is None or not _has_credentials(provider_name, env):
        errors_with_hint = [*issues, _credentials_issue(provider_name)]
        issue = errors_with_hint[-1]
        raise AuthError(
            issue["message"],
            code=issue.get("code"),
            details={"task": task, "errors": errors_with_hint, "stream": stream, "provider": provider_name},
        )


def _perceive_result_from_response(resp: dict, issues: list[dict]) -> PerceiveResult:
    text = resp.get("text")
    points = resp.get("points")
    parsed = resp.get("parsed")
    return PerceiveResult(
        text=text,
        points=points,
        parsed=parsed,
        usage=None,
        errors=issues,
        raw=resp.get("raw"),
    )


def _compile_nodes_sync(
    fn: Callable[..., Any], *, expects: str | None, strict: bool, args: tuple[Any, ...], kwargs: dict[str, Any]
):
    nodes = fn(*args, **kwargs)
    return _compile(nodes, expects=expects, strict=strict)


async def _compile_nodes_async(
    fn: Callable[..., Any],
    *,
    expects: str | None,
    strict: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    nodes = fn(*args, **kwargs)
    if inspect.isawaitable(nodes):
        nodes = await nodes
    return _compile(nodes, expects=expects, strict=strict)


def _prepare_execution_context(
    *,
    task: dict,
    issues: list[dict],
    stream: bool,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    allow_multiple: bool,
    max_outputs: int | None,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
):
    env, resolved_provider, provider_name, client_kwargs = _prepare_client_kwargs(
        provider_override=provider_override,
        model_override=model_override,
        expects=expects,
        allow_multiple=allow_multiple,
        max_outputs=max_outputs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    compile_only = _maybe_compile_only_result(
        stream=stream,
        resolved_provider=resolved_provider,
        provider_name=provider_name,
        env=env,
        issues=issues,
        task=task,
    )
    return compile_only, client_kwargs


def _expects_structured(expects: str | None) -> bool:
    return expects in {"point", "box", "polygon"}


def _collect_nodes(value: Any, acc: list[DSLNode]) -> None:
    if isinstance(value, Sequence):
        for node in value.nodes:
            _collect_nodes(node, acc)
        return
    if isinstance(value, DSLNode):
        acc.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_nodes(item, acc)
        return
    raise TypeError(
        "perceive direct invocation expects DSL nodes (text/image/...) or sequences; "
        f"received unsupported type {type(value)!r}"
    )


def _normalize_direct_nodes(values: tuple[Any, ...]) -> DSLNode | Sequence:
    if not values:
        raise TypeError("perceive direct invocation requires at least one DSL node")
    flat: list[DSLNode] = []
    for value in values:
        _collect_nodes(value, flat)
    if not flat:
        raise TypeError("perceive direct invocation did not receive any DSL nodes")
    if len(flat) == 1:
        return flat[0]
    return Sequence(flat)


def _execute_sync_task(
    *,
    task: dict,
    issues: list[dict],
    parse_points: bool,
    stream: bool,
    provider_override: str | None,
    model_override: str | None,
    expects: str | None,
    allow_multiple: bool,
    max_outputs: int | None,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int | None,
):
    compile_only, client_kwargs = _prepare_execution_context(
        task=task,
        issues=issues,
        stream=stream,
        provider_override=provider_override,
        model_override=model_override,
        expects=expects,
        allow_multiple=allow_multiple,
        max_outputs=max_outputs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    if compile_only is not None:
        return compile_only

    client = Client()
    if stream:
        return client.stream(
            task,
            parse_points=parse_points,
            **client_kwargs,
        )

    try:
        resp = client.generate(task, **client_kwargs)
    except TypeError:
        gen = getattr(type(client), "generate", None)
        if callable(gen):
            resp = gen(task, **client_kwargs)
        else:
            raise
    return _perceive_result_from_response(resp, issues)


def perceive(
    *nodes_or_fn: Any,
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
    """Decorator (or direct helper) for building Tasks from DSL nodes.

    When called without nodes it returns a decorator; when passed nodes directly
    it immediately compiles and executes them. Executes via the default Client
    unless compile-only fallback is triggered (no provider configured or the
    selected provider lacks credentials).
    """

    parse_points = _expects_structured(expects)

    def wrapper(fn: Callable[..., Any]):
        def _inspect(*args: Any, **kwargs: Any):
            return _compile_nodes_sync(fn, expects=expects, strict=strict, args=args, kwargs=kwargs)

        def _call(*args: Any, **kwargs: Any):
            task, issues = _inspect(*args, **kwargs)
            return _execute_sync_task(
                task=task,
                issues=issues,
                parse_points=parse_points,
                stream=stream,
                provider_override=provider,
                model_override=model,
                expects=expects,
                allow_multiple=allow_multiple,
                max_outputs=max_outputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )

        _call.__perceptron_inspector__ = _inspect  # type: ignore[attr-defined]

        return _call

    if not nodes_or_fn:
        return wrapper

    if len(nodes_or_fn) == 1 and callable(nodes_or_fn[0]):
        return wrapper(nodes_or_fn[0])

    nodes = _normalize_direct_nodes(nodes_or_fn)
    task, issues = _compile(nodes, expects=expects, strict=strict)
    return _execute_sync_task(
        task=task,
        issues=issues,
        parse_points=parse_points,
        stream=stream,
        provider_override=provider,
        model_override=model,
        expects=expects,
        allow_multiple=allow_multiple,
        max_outputs=max_outputs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )


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

    parse_points = _expects_structured(expects)

    def wrapper(fn: Callable[..., Any]):
        async def _inspect_async(*args: Any, **kwargs: Any):
            return await _compile_nodes_async(fn, expects=expects, strict=strict, args=args, kwargs=kwargs)

        if stream:

            def _call(*args: Any, **kwargs: Any):
                async def _generator():
                    task, issues = await _inspect_async(*args, **kwargs)
                    compile_only, client_kwargs = _prepare_execution_context(
                        task=task,
                        issues=issues,
                        stream=True,
                        provider_override=provider,
                        model_override=model,
                        expects=expects,
                        allow_multiple=allow_multiple,
                        max_outputs=max_outputs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    if compile_only is not None:
                        return
                    client = AsyncClient()
                    async for event in client.stream(
                        task,
                        parse_points=parse_points,
                        **client_kwargs,
                    ):
                        yield event

                return _generator()

            _call.__perceptron_inspector__ = _inspect_async  # type: ignore[attr-defined]

            return _call

        async def _call(*args: Any, **kwargs: Any):
            task, issues = await _inspect_async(*args, **kwargs)
            compile_only, client_kwargs = _prepare_execution_context(
                task=task,
                issues=issues,
                stream=False,
                provider_override=provider,
                model_override=model,
                expects=expects,
                allow_multiple=allow_multiple,
                max_outputs=max_outputs,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )
            if compile_only is not None:
                return compile_only

            client = AsyncClient()
            resp = await client.generate(task, **client_kwargs)
            return _perceive_result_from_response(resp, issues)

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


__all__ = ["PerceiveResult", "async_perceive", "inspect_task", "perceive"]


def _credentials_issue(provider_name: str) -> dict[str, str]:
    if provider_name == "fal":
        message = "No credentials found for provider 'fal'. Set and validate api_key (e.g., PERCEPTRON_API_KEY or FAL_KEY) before running."
    else:
        message = f"No credentials found for provider '{provider_name}'. Set and validate api_key before running."
    return {"code": "credentials_missing", "message": message}


def _has_credentials(provider_name: str, env) -> bool:
    if env.api_key:
        return True

    provider_cfg = _PROVIDER_CONFIG.get(provider_name or "") or {}
    return any(os.getenv(env_key) for env_key in provider_cfg.get("env_keys", []))
