"""HTTP client for executing compiled Tasks against supported providers.

Providers
- fal: Fal-hosted endpoint (OpenAI-compatible)

Additional transports can be registered by extending `_PROVIDER_CONFIG`.

Streaming yields SSE `data:` lines and maps them to:
- text.delta: textual deltas as they arrive
- points.delta: emitted when a full canonical tag closes (based on cumulative parse)
- final: final text, parsed segments, usage, and any parsing issues
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx

from .config import settings
from .errors import (
    AuthError,
    BadRequestError,
    RateLimitError,
    SDKError,
    ServerError,
    TimeoutError,
    TransportError,
)
from .expectations import STRUCTURED_EXPECTATIONS
from .pointing.parser import extract_points, parse_text

# ---------------------------------------------------------------------------
# Response format types for constrained decoding
# ---------------------------------------------------------------------------


class JsonSchemaSpec(TypedDict, total=False):
    """JSON Schema specification object containing name, schema, and optional strict flag."""

    name: str
    schema: dict[str, Any]
    strict: bool


class JsonSchemaFormat(TypedDict, total=False):
    """JSON Schema response format specification."""

    type: str  # Must be "json_schema"
    json_schema: JsonSchemaSpec


class RegexFormat(TypedDict, total=False):
    """Regex response format specification."""

    type: str  # Must be "regex"
    regex: str  # The regex pattern to constrain output


# Union type for response_format parameter
ResponseFormat = JsonSchemaFormat | RegexFormat | dict[str, Any]


@dataclass
class _PreparedInvocation:
    url: str
    headers: dict[str, Any]
    body: dict[str, Any]
    expects: str | None
    provider_cfg: dict[str, Any]


def _build_response_format(
    response_format: ResponseFormat | None,
) -> tuple[str, dict[str, Any] | str] | None:
    """Validate and normalize response_format for the API request.

    Returns:
        None if response_format is None, otherwise a tuple of (field_name, value):
        - For json_schema: ("response_format", {"type": "json_schema", "json_schema": {...}})
        - For regex: ("regex", "pattern_string")
    """
    if response_format is None:
        return None

    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        schema_spec = response_format.get("json_schema")
        if not isinstance(schema_spec, dict):
            raise ValueError("json_schema response_format requires a 'json_schema' dict with 'name' and 'schema'")
        return ("response_format", {"type": "json_schema", "json_schema": schema_spec})

    if fmt_type == "regex":
        regex_pattern = response_format.get("regex")
        if not isinstance(regex_pattern, str):
            raise ValueError("regex response_format requires a 'regex' string pattern")
        return ("regex", regex_pattern)

    raise ValueError(f"Unknown response_format type: {fmt_type!r}. Supported types: 'json_schema', 'regex'")


class _StreamProcessor:
    def __init__(
        self,
        *,
        client_core: _ClientCore,
        expects: str | None,
        parse_points: bool,
        max_buffer_bytes: int | None,
    ) -> None:
        self._client_core = client_core
        self._expects = expects
        self._parse_points = parse_points and expects in {"point", "box", "polygon"}
        self._max_buffer_bytes = max_buffer_bytes
        self._cumulative: str = ""
        self._reasoning: str = ""
        self._emitted_spans: set[tuple[int, int]] = set()
        self._parsing_enabled = True
        self._usage_payload: dict[str, Any] | None = None

    def handle_payload(self, obj: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not isinstance(obj, dict):
            return events

        # Capture usage info
        usage_field = obj.get("usage")
        if isinstance(usage_field, dict) and self._usage_payload is None:
            self._usage_payload = usage_field

        # Extract delta
        try:
            delta = obj["choices"][0]["delta"]
        except (KeyError, IndexError, TypeError):
            return events

        # Process reasoning content
        reasoning = delta.get("reasoning_content")
        if reasoning:
            self._reasoning += reasoning

        # Process answer content
        content = delta.get("content")
        if content:
            self._cumulative += content
            events.append({
                "type": "text.delta",
                "chunk": content,
                "total_chars": len(self._cumulative),
            })

        # Check buffer limits
        if self._parsing_enabled and self._max_buffer_bytes is not None:
            if len(self._cumulative.encode("utf-8")) > self._max_buffer_bytes:
                self._parsing_enabled = False

        # Parse points
        if self._parse_points and self._parsing_enabled:
            events.extend(self._point_events())

        return events

    def finalize(self) -> dict[str, Any]:
        content = self._cumulative or None
        result: dict[str, Any] = {"text": content, "raw": None}
        if self._reasoning:
            result["reasoning"] = [self._reasoning]
        expects = self._expects
        parsed_segments: list[dict[str, Any]] | None = None
        if expects in {"point", "box", "polygon"} and self._parsing_enabled and isinstance(content, str):
            parsed_segments = parse_text(content)
            result["points"] = [seg["value"] for seg in parsed_segments if seg["kind"] == expects]
            result["parsed"] = parsed_segments
        issues: list[dict[str, Any]] = []
        if not self._parsing_enabled:
            issues.append(
                {
                    "code": "stream_buffer_overflow",
                    "message": "parsing disabled due to buffer limit",
                }
            )
        return {
            "type": "final",
            "result": {
                "text": result.get("text"),
                "points": result.get("points"),
                "parsed": result.get("parsed"),
                "usage": self._usage_payload,
                "errors": issues,
                "raw": result.get("raw"),
            },
        }

    def _point_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        expects = self._expects
        if expects not in {"point", "box", "polygon"}:
            return events
        try:
            segments = parse_text(self._cumulative)
        except Exception:
            return events
        for seg in segments:
            if seg.get("kind") not in {"point", "box", "polygon"}:
                continue
            span_info = seg.get("span")
            if not isinstance(span_info, dict):
                continue
            span = (span_info.get("start"), span_info.get("end"))
            if None in span:
                continue
            if span in self._emitted_spans or seg.get("kind") != expects:
                continue
            self._emitted_spans.add(span)  # type: ignore[arg-type]
            events.append(
                {
                    "type": "points.delta",
                    "points": [seg.get("value")],
                    "span": span_info,
                }
            )
        return events


def _iter_sse_lines(resp):
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            yield raw_line.decode("utf-8", errors="ignore")
        else:
            yield raw_line


async def _aiter_sse_lines(resp):
    async for line in resp.aiter_lines():
        if not line:
            continue
        yield line


def _process_sse_line(line: str, processor: _StreamProcessor) -> tuple[bool, list[dict[str, Any]]]:
    if not line.startswith("data:"):
        return False, []
    data_line = line[len("data:") :].strip()
    if data_line == "[DONE]":
        return True, []
    try:
        obj = json.loads(data_line)
    except Exception:
        return False, []
    return False, processor.handle_payload(obj)


def _response_json(resp) -> dict[str, Any]:
    if resp.status_code != 200:
        raise _map_http_error(resp)
    return resp.json()


def _task_to_openai_messages(task: dict) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    current_role: str | None = None
    current_content: list[dict[str, Any]] = []
    contains_non_text = False

    def _flush() -> None:
        nonlocal current_role, current_content, contains_non_text
        if current_role is not None:
            if not contains_non_text and all(part.get("type") == "text" for part in current_content):
                text = "".join(part.get("text", "") for part in current_content)
                messages.append({"role": current_role, "content": text})
            else:
                messages.append({"role": current_role, "content": list(current_content)})
        current_role = None
        current_content = []
        contains_non_text = False

    for item in task.get("content", []):
        itype = item.get("type")
        role = item.get("role", "user")
        if role == "agent":
            role = "assistant"
        if itype == "text":
            part = {"type": "text", "text": item.get("content", "")}
            if current_role not in {role, None}:
                _flush()
            current_role = role
            current_content.append(part)
        elif itype == "image":
            payload = item.get("content")
            if payload is None:
                continue
            if isinstance(payload, str) and payload.startswith(("http://", "https://")):
                image_part = {"type": "image_url", "image_url": {"url": payload}}
            else:
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{payload}"},
                }
            if current_role not in {role, None}:
                _flush()
            current_role = role
            current_content.append(image_part)
            contains_non_text = True
        else:
            continue
    _flush()
    return messages


def _model_entry(model_name: str | None, provider_cfg: dict[str, Any] | None) -> dict | None:
    models_cfg = provider_cfg.get("models") if isinstance(provider_cfg, dict) else None
    if isinstance(models_cfg, dict):
        entry = models_cfg.get(model_name)
        if isinstance(entry, dict):
            return entry
    return None


def _model_capabilities(model_name: str | None, provider_cfg: dict[str, Any] | None) -> tuple[bool, bool, bool, bool]:
    entry = _model_entry(model_name, provider_cfg) or {}
    supports_reasoning = bool(entry.get("reasoning", True))
    requires_reasoning = bool(entry.get("only_reasoning", False))
    skip_hints = bool(entry.get("skip_structured_hints", False))
    supports_focus = bool(entry.get("focus", True))
    return supports_reasoning, requires_reasoning, skip_hints, supports_focus


def _build_hint_content(expects: str | None, include_reasoning: bool, include_focus: bool) -> str | None:
    tokens: list[str] = []
    if expects and expects.lower() in STRUCTURED_EXPECTATIONS:
        tokens.append(expects.upper())
    if include_focus:
        tokens.append("TOOLS")
    if include_reasoning:
        tokens.append("THINK")
    if not tokens:
        return None
    return f"<hint>{' '.join(sorted(tokens))}</hint>"


def _inject_expectation_hint(
    task: dict,
    expects: str | None,
    *,
    model_name: str | None,
    provider_cfg: dict[str, Any] | None,
    include_reasoning: bool,
    include_focus: bool = False,
) -> dict:
    _, _, skip_hints, supports_focus = _model_capabilities(model_name, provider_cfg)
    include_focus = include_focus and supports_focus
    if skip_hints:
        content = task.get("content") or []
        filtered = [
            entry
            for entry in content
            if not (
                isinstance(entry, dict)
                and entry.get("type") == "text"
                and isinstance(entry.get("content"), str)
                and "<hint" in entry.get("content", "").lower()
            )
        ]
        new_task = dict(task)
        new_task["content"] = filtered
        return new_task

    hint = _build_hint_content(expects, include_reasoning, include_focus)
    if hint is None:
        return task

    content = task.get("content") or []
    if any(entry.get("content") == hint for entry in content if isinstance(entry, dict)):
        return task
    new_content: list[dict[str, Any]] = []
    inserted = False
    for entry in content:
        if not inserted and entry.get("role") != "system":
            new_content.append({"type": "text", "role": "user", "content": hint})
            inserted = True
        new_content.append(entry)
    if not inserted:
        new_content.append({"type": "text", "role": "user", "content": hint})
    new_task = dict(task)
    new_task["content"] = new_content
    return new_task


def _apply_reasoning_and_hints(
    *,
    task: dict,
    expects: str | None,
    model_name: str | None,
    provider_cfg: dict[str, Any] | None,
    reasoning_flag: bool | None,
    focus_flag: bool | None = None,
) -> tuple[dict, bool, bool]:
    supports, requires, _, supports_focus = _model_capabilities(model_name, provider_cfg)

    final_reasoning = reasoning_flag  # None means "auto"

    # If the model requires reasoning, force it on.
    if requires and final_reasoning is not True:
        final_reasoning = True

    # If caller didn't specify, enable when expects==think or a THINK hint will be injected.
    if final_reasoning is None and expects and expects.lower() == "think":
        final_reasoning = True

    # Disable if model lacks support.
    if final_reasoning is True and not supports:
        final_reasoning = False

    # Focus is allowed only if the model supports it
    final_focus = focus_flag if focus_flag is True and supports_focus else False

    include_reasoning_hint = bool((final_reasoning is True) or requires or (expects and expects.lower() == "think"))
    task_with_hint = _inject_expectation_hint(
        task,
        expects,
        model_name=model_name,
        provider_cfg=provider_cfg,
        include_reasoning=include_reasoning_hint,
        include_focus=final_focus,
    )

    return task_with_hint, final_reasoning, final_focus


def _requires_reasoning(model_name: str | None, provider_cfg: dict[str, Any] | None) -> bool:
    _, requires, _ = _reasoning_capabilities(model_name, provider_cfg)
    return requires


_PROVIDER_CONFIG = {
    "fal": {
        "base_url": "https://fal.run",
        "path": "/perceptron/isaac-01/openai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Key ",
        "env_keys": ["FAL_KEY", "PERCEPTRON_API_KEY"],
        "default_model": "isaac-0.1",
        "supported_models": ["isaac-0.1"],
        "models": {
            "isaac-0.1": {"reasoning": False, "skip_structured_hints": False, "focus": False},
        },
        "stream": True,
    },
    "perceptron": {
        "base_url": "https://api.perceptron.inc/v1",
        "path": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "env_keys": ["PERCEPTRON_API_KEY"],
        "default_model": "isaac-0.1",
        "supported_models": ["isaac-0.1", "isaac-0.2-1b", "isaac-0.2-2b-preview", "qwen3-vl-235b-a22b-thinking"],
        "models": {
            "isaac-0.1": {"reasoning": False, "skip_structured_hints": False, "focus": False},
            "isaac-0.2-1b": {"reasoning": True, "skip_structured_hints": False, "focus": True},
            "isaac-0.2-2b-preview": {"reasoning": True, "skip_structured_hints": False, "focus": True},
            "qwen3-vl-235b-a22b-thinking": {
                "reasoning": True,
                "only_reasoning": True,
                "skip_structured_hints": True,
                "focus": False,
            },
        },
        "stream": True,
    },
}


def _select_model(
    provider_cfg: dict[str, Any],
    requested_model: str | None,
    *,
    provider_name: str | None = None,
) -> str | None:
    model = requested_model or provider_cfg.get("default_model")
    supported = provider_cfg.get("supported_models")
    provider_label = provider_name or provider_cfg.get("name") or "unknown"
    if supported and model and model not in supported:
        allowed = ", ".join(supported)
        raise BadRequestError(f"Model '{model}' is not supported for provider='{provider_label}'. Allowed: {allowed}")
    return model


def _pop_and_resolve_model(provider_cfg: dict[str, Any], gen_kwargs: dict[str, Any]) -> str:
    requested_model = gen_kwargs.pop("model", None)
    resolved = _select_model(provider_cfg, requested_model)
    if resolved:
        return resolved
    default_model = provider_cfg.get("default_model")
    if default_model:
        return default_model
    provider_label = provider_cfg.get("name") or "unknown"
    raise BadRequestError(
        f"No model configured for provider '{provider_label}'. Specify a model explicitly or configure a default."
    )


def _resolve_provider(provider: str | None) -> dict:
    provider = provider or "fal"
    provider_lc = provider.lower() if isinstance(provider, str) else provider
    if provider_lc not in _PROVIDER_CONFIG:
        raise BadRequestError(f"Unsupported provider: {provider}")
    return {"name": provider_lc, **_PROVIDER_CONFIG[provider_lc]}


def _prepare_transport(settings_obj, provider_cfg, task, *, stream=False):
    base_url = settings_obj.base_url or provider_cfg.get("base_url")
    if not base_url:
        raise BadRequestError(f"base_url required for provider={provider_cfg['name']}")
    url = base_url.rstrip("/") + provider_cfg["path"]
    headers = {"Content-Type": "application/json"}
    token = settings_obj.api_key
    for env in provider_cfg.get("env_keys", []):
        token = token or os.getenv(env)
    auth_header = provider_cfg.get("auth_header")
    if auth_header:
        if not token:
            raise AuthError(f"API key required for provider='{provider_cfg['name']}'")
        prefix = provider_cfg.get("auth_prefix", "")
        headers[auth_header] = f"{prefix}{token}"
    if stream and not provider_cfg.get("stream", True):
        raise BadRequestError(f"Streaming is not supported for provider='{provider_cfg['name']}'")
    return task, url, headers, provider_cfg


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _extract_error_metadata(
    data: Any,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    message: str | None = None
    code: str | None = None
    details: dict[str, Any] | None = None

    if isinstance(data, dict):
        nested_error = data.get("error")
        if isinstance(nested_error, dict):
            message = _first_nonempty(
                nested_error.get("message"),
                nested_error.get("detail"),
                nested_error.get("error"),
            )
            code = nested_error.get("code") or nested_error.get("type")
            details = nested_error or None
        elif isinstance(nested_error, str):
            message = _first_nonempty(nested_error)
            details = data or None
        else:
            message = _first_nonempty(
                data.get("message"),
                data.get("detail"),
                data.get("error") if isinstance(data.get("error"), str) else None,
            )
            code = data.get("code")
            details = data or None
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                candidate = _first_nonempty(item.get("message"), item.get("detail"))
                if candidate:
                    message = candidate
                    code = item.get("code")
                    details = item
                    break
            elif isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    message = candidate
                    break
    elif isinstance(data, str):
        message = data.strip() or None

    return message, code, details


def _map_http_error(resp) -> SDKError:
    try:
        data = resp.json()
    except Exception:
        data = None

    message, code, details = _extract_error_metadata(data)
    try:
        fallback_text = resp.text
    except Exception:
        fallback_text = ""
    fallback_text = fallback_text.strip()
    detail_payload = details if isinstance(details, dict) and details else None

    if resp.status_code == 400:
        msg = message or fallback_text or "bad request"
        return BadRequestError(msg, code=code, details=detail_payload)

    if resp.status_code in (401, 403):
        auth_msg = message or fallback_text or "authentication failed"
        return AuthError(auth_msg, code=code or "auth_error", details=detail_payload)

    if resp.status_code == 404:
        msg = message or fallback_text or "not found"
        return BadRequestError(msg, code=code, details=detail_payload)

    if resp.status_code == 429:
        retry_after = None
        try:
            retry_after = float(resp.headers.get("Retry-After", "0"))
        except Exception:
            retry_after = None
        msg = message or fallback_text or "rate limited"
        return RateLimitError(msg, retry_after=retry_after, details=detail_payload)

    if 400 <= resp.status_code < 500:
        msg = message or fallback_text or "bad request"
        return BadRequestError(msg, code=code, details=detail_payload)

    msg = message or fallback_text or f"server error: {resp.status_code}"
    return ServerError(msg, code=code, details=detail_payload)


def _http_client(timeout: float):
    return httpx.Client(timeout=timeout, http2=True)


class _ClientCore:
    def __init__(self, **overrides: Any) -> None:
        self._settings = settings()
        for k, v in overrides.items():
            if hasattr(self._settings, k):
                setattr(self._settings, k, v)

    def _prepare_invocation(
        self,
        task: dict,
        *,
        expects: str | None,
        stream: bool,
        gen_kwargs: dict[str, Any],
    ) -> _PreparedInvocation:
        s = self._settings
        local_kwargs = dict(gen_kwargs)
        reasoning_flag = local_kwargs.pop("reasoning", None)
        focus_flag = local_kwargs.pop("focus", None)
        provider_cfg = _resolve_provider(local_kwargs.pop("provider", None) or s.provider)
        temperature = local_kwargs.pop("temperature", s.temperature)
        max_tokens = local_kwargs.pop("max_tokens", s.max_tokens)
        top_p = local_kwargs.pop("top_p", s.top_p)
        top_k = local_kwargs.pop("top_k", s.top_k)
        frequency_penalty = local_kwargs.pop("frequency_penalty", s.frequency_penalty)
        presence_penalty = local_kwargs.pop("presence_penalty", s.presence_penalty)
        response_format = local_kwargs.pop("response_format", None)

        if "model" not in local_kwargs and s.model is not None:
            local_kwargs["model"] = s.model
        model = _pop_and_resolve_model(provider_cfg, local_kwargs)
        task_with_hint, reasoning_flag, focus_flag = _apply_reasoning_and_hints(
            task=task,
            expects=expects,
            model_name=model,
            provider_cfg=provider_cfg,
            reasoning_flag=reasoning_flag,
            focus_flag=focus_flag,
        )
        prepared_task, url, headers, resolved_cfg = _prepare_transport(s, provider_cfg, task_with_hint, stream=stream)
        messages = _task_to_openai_messages(prepared_task)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if reasoning_flag:
            body["reasoning"] = True
        if stream:
            body["stream"] = True

        # Add constrained decoding field (json_schema → response_format, regex → regex)
        format_result = _build_response_format(response_format)
        if format_result is not None:
            field_name, field_value = format_result
            body[field_name] = field_value

        return _PreparedInvocation(url=url, headers=headers, body=body, expects=expects, provider_cfg=resolved_cfg)

    def _build_result(self, data: dict[str, Any], expects: str | None) -> dict[str, Any]:
        message = data.get("choices", [{}])[0].get("message", {})

        reasoning_content = message.get("reasoning_content")
        content = message.get("content")

        result: dict[str, Any] = {"text": content, "raw": data}
        if reasoning_content:
            result["reasoning"] = [reasoning_content]
        if expects in {"point", "box", "polygon"} and isinstance(content, str):
            kind = "point" if expects == "point" else ("box" if expects == "box" else "polygon")
            result["points"] = extract_points(content, expected=kind)
            result["parsed"] = parse_text(content)
        return result


class Client(_ClientCore):
    def generate(self, task: dict, *, expects: str | None = None, **gen_kwargs: Any) -> dict:
        invocation = self._prepare_invocation(task, expects=expects, stream=False, gen_kwargs=gen_kwargs)
        try:
            with _http_client(self._settings.timeout) as session:
                resp = session.post(invocation.url, headers=invocation.headers, json=invocation.body)
        except httpx.TimeoutException as e:  # pragma: no cover - error path
            raise TimeoutError("request timed out") from e
        except httpx.HTTPError as e:  # pragma: no cover
            raise TransportError(str(e)) from e
        data = _response_json(resp)
        return self._build_result(data, invocation.expects)

    def stream(
        self,
        task: dict,
        *,
        expects: str | None = None,
        parse_points: bool = False,
        **gen_kwargs: Any,
    ):
        try:
            invocation = self._prepare_invocation(task, expects=expects, stream=True, gen_kwargs=gen_kwargs)
        except SDKError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        processor = _StreamProcessor(
            client_core=self,
            expects=expects,
            parse_points=parse_points,
            max_buffer_bytes=self._settings.max_buffer_bytes,
        )

        try:
            with _http_client(self._settings.timeout) as session:
                with session.stream("POST", invocation.url, headers=invocation.headers, json=invocation.body) as resp:
                    if resp.status_code != 200:
                        err = _map_http_error(resp)
                        yield {"type": "error", "message": str(err)}
                        return
                    for line in _iter_sse_lines(resp):
                        done, events = _process_sse_line(line, processor)
                        yield from events
                        if done:
                            break
        except httpx.TimeoutException:
            yield {"type": "error", "message": "timeout"}
            return
        except httpx.HTTPError as e:
            yield {"type": "error", "message": str(e)}
            return

        yield processor.finalize()


class AsyncClient(_ClientCore):
    """Asynchronous variant using httpx.AsyncClient."""

    async def generate(self, task: dict, *, expects: str | None = None, **gen_kwargs: Any) -> dict:
        invocation = self._prepare_invocation(task, expects=expects, stream=False, gen_kwargs=gen_kwargs)
        try:
            async with httpx.AsyncClient(timeout=self._settings.timeout) as session:
                resp = await session.post(
                    invocation.url,
                    headers=invocation.headers,
                    content=json.dumps(invocation.body),
                )
        except httpx.TimeoutException as e:  # pragma: no cover - error path
            raise TimeoutError("request timed out") from e
        except httpx.HTTPError as e:  # pragma: no cover
            raise TransportError(str(e)) from e
        data = _response_json(resp)
        return self._build_result(data, invocation.expects)

    def stream(
        self,
        task: dict,
        *,
        expects: str | None = None,
        parse_points: bool = False,
        **gen_kwargs: Any,
    ):
        async def _run_async_stream():
            try:
                invocation = self._prepare_invocation(task, expects=expects, stream=True, gen_kwargs=gen_kwargs)
            except SDKError as exc:
                yield {"type": "error", "message": str(exc)}
                return

            processor = _StreamProcessor(
                client_core=self,
                expects=expects,
                parse_points=parse_points,
                max_buffer_bytes=self._settings.max_buffer_bytes,
            )

            try:
                async with httpx.AsyncClient(timeout=self._settings.timeout) as session:
                    async with session.stream(
                        "POST",
                        invocation.url,
                        headers=invocation.headers,
                        content=json.dumps(invocation.body),
                    ) as resp:
                        if resp.status_code != 200:
                            err = _map_http_error(resp)
                            yield {"type": "error", "message": str(err)}
                            return
                        async for line in _aiter_sse_lines(resp):
                            done, events = _process_sse_line(line, processor)
                            for event in events:
                                yield event
                            if done:
                                break
            except httpx.TimeoutException:
                yield {"type": "error", "message": "timeout"}
                return
            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

            yield processor.finalize()

        return _run_async_stream()


# ---------------------------------------------------------------------------
# Response format helpers for constrained decoding
# ---------------------------------------------------------------------------


def json_schema_format(
    schema: dict[str, Any],
    *,
    name: str = "response",
    strict: bool | None = None,
) -> JsonSchemaFormat:
    """Create a JSON schema response format for constrained decoding.

    Args:
        schema: JSON Schema object defining the expected output structure.
        name: A name for this schema (used for identification in logs/errors).
        strict: If True, enforce strict schema validation. Defaults to None (provider default).

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> result = client.generate(task, response_format=json_schema_format(schema))
    """
    json_schema_spec: JsonSchemaSpec = {"name": name, "schema": schema}
    if strict is not None:
        json_schema_spec["strict"] = strict
    return {"type": "json_schema", "json_schema": json_schema_spec}


def regex_format(pattern: str) -> RegexFormat:
    """Create a regex response format for constrained decoding.

    Args:
        pattern: A regular expression pattern that the output must match.

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> # Constrain output to a valid email address format
        >>> result = client.generate(task, response_format=regex_format(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"))
    """
    return {"type": "regex", "regex": pattern}


def pydantic_format(
    model: type,
    *,
    name: str | None = None,
    strict: bool | None = None,
) -> JsonSchemaFormat:
    """Create a JSON schema response format from a Pydantic model.

    This is a convenience wrapper that extracts the JSON schema from a Pydantic
    model class and passes it to the constrained decoding engine.

    Args:
        model: A Pydantic model class (subclass of pydantic.BaseModel).
        name: Optional name for the schema. Defaults to the model's class name.
        strict: If True, enforce strict schema validation. Defaults to None (provider default).

    Returns:
        A response_format dict suitable for passing to generate/stream/perceive.

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        ...     email: str | None = None
        >>>
        >>> result = client.generate(task, response_format=pydantic_format(Person))
        >>> person = Person.model_validate_json(result.text)

    Note:
        Requires pydantic to be installed. The model must be a Pydantic v2 model
        (subclass of pydantic.BaseModel with model_json_schema method).
    """
    # Check if it's a Pydantic model
    if not hasattr(model, "model_json_schema"):
        raise TypeError(
            f"Expected a Pydantic model class with model_json_schema method, got {type(model).__name__}. "
            "Make sure you're using Pydantic v2."
        )

    # Extract JSON schema from the Pydantic model
    schema = model.model_json_schema()

    # Use model class name as default schema name
    schema_name = name if name is not None else model.__name__

    return json_schema_format(schema, name=schema_name, strict=strict)
