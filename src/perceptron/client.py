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
from typing import Any

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
from .expectations import expectation_hint_text
from .pointing.parser import extract_points, extract_reasoning, parse_text


@dataclass
class _PreparedInvocation:
    url: str
    headers: dict[str, Any]
    body: dict[str, Any]
    expects: str | None
    provider_cfg: dict[str, Any]


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
        self._emitted_spans: set[tuple[int, int]] = set()
        self._parsing_enabled = True
        self._usage_payload: dict[str, Any] | None = None
        self._reasoning_started = False
        self._answering_started = False

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

        # Process reasoning content (wrap in <think> tags)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self._reasoning_started:
                self._reasoning_started = True
                reasoning = f"<think>{reasoning}"
            self._cumulative += reasoning
            events.append({
                "type": "text.delta",
                "chunk": reasoning,
                "total_chars": len(self._cumulative),
            })

        # Process answer content
        content = delta.get("content")
        if content:
            # Close </think> tag before first content chunk
            if self._reasoning_started and not self._answering_started:
                close_tag = "</think>"
                self._cumulative += close_tag
                events.append({
                    "type": "text.delta",
                    "chunk": close_tag,
                    "total_chars": len(self._cumulative),
                })
                self._answering_started = True

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
        cleaned_text, reasoning_final = self._client_core._clean_text_with_reasoning(self._cumulative)
        result: dict[str, Any] = {"text": cleaned_text, "raw": None}
        if reasoning_final:
            result["reasoning"] = reasoning_final
        expects = self._expects
        parsed_segments: list[dict[str, Any]] | None = None
        if expects in {"point", "box", "polygon"} and self._parsing_enabled and isinstance(cleaned_text, str):
            parsed_segments = parse_text(cleaned_text)
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


def _inject_expectation_hint(task: dict, expects: str | None) -> dict:
    hint = expectation_hint_text(expects)
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


_PROVIDER_CONFIG = {
    "fal": {
        "base_url": "https://fal.run",
        "path": "/perceptron/isaac-01/openai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Key ",
        "env_keys": ["FAL_KEY", "PERCEPTRON_API_KEY"],
        "default_model": "isaac-0.1",
        "supported_models": ["isaac-0.1"],
        "stream": True,
    },
    "perceptron": {
        "base_url": "https://api.perceptron.inc/v1",
        "path": "/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "env_keys": ["PERCEPTRON_API_KEY"],
        "default_model": "isaac-0.1",
        "supported_models": ["isaac-0.1", "qwen3-vl-235b-a22b-thinking"],
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


def _prepare_transport(settings_obj, provider_cfg, task, expects, *, stream=False):
    task = _inject_expectation_hint(task, expects)
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

    @staticmethod
    def _clean_text_with_reasoning(content: Any, payload: dict | None = None) -> tuple[Any, list[str] | None]:
        if not isinstance(content, str):
            return content, None
        extraction = extract_reasoning(content)
        reasoning = extraction.reasoning
        cleaned = extraction.text if extraction.text is not None else content
        if payload:
            try:
                message = payload.get("choices", [{}])[0].get("message")
                if isinstance(message, dict):
                    message["content"] = cleaned
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    elif "reasoning_content" in message:
                        message["reasoning_content"] = None
            except Exception:
                pass
        return cleaned, reasoning

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
        provider_cfg = _resolve_provider(local_kwargs.pop("provider", None) or s.provider)
        temperature = local_kwargs.pop("temperature", s.temperature)
        max_tokens = local_kwargs.pop("max_tokens", s.max_tokens)
        top_p = local_kwargs.pop("top_p", s.top_p)
        top_k = local_kwargs.pop("top_k", s.top_k)

        prepared_task, url, headers, resolved_cfg = _prepare_transport(s, provider_cfg, task, expects, stream=stream)
        if "model" not in local_kwargs and s.model is not None:
            local_kwargs["model"] = s.model
        messages = _task_to_openai_messages(prepared_task)
        model = _pop_and_resolve_model(resolved_cfg, local_kwargs)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "top_p": top_p,
        }
        if top_k is not None:
            body["top_k"] = top_k
        if stream:
            body["stream"] = True
        return _PreparedInvocation(url=url, headers=headers, body=body, expects=expects, provider_cfg=resolved_cfg)

    def _build_result(self, data: dict[str, Any], expects: str | None) -> dict[str, Any]:
        message = data.get("choices", [{}])[0].get("message", {})

        # Get both reasoning and content separately
        reasoning_content = message.get("reasoning_content")
        answer_content = message.get("content")

        # Build full content with <think> tags if reasoning exists
        if reasoning_content and answer_content:
            full_content = f"<think>{reasoning_content}</think>{answer_content}"
        elif reasoning_content:
            full_content = f"<think>{reasoning_content}</think>"
        else:
            full_content = answer_content

        # Clean and extract reasoning
        content, reasoning = self._clean_text_with_reasoning(full_content, data)
        result: dict[str, Any] = {"text": content, "raw": data}
        if reasoning:
            result["reasoning"] = reasoning
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
