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

import os
from typing import Any, Dict, List, Optional

import json
import requests

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None

from .config import settings
from .errors import (
    SDKError,
    TransportError,
    TimeoutError,
    AuthError,
    RateLimitError,
    ServerError,
    BadRequestError,
)
from .pointing.parser import extract_points, parse_text, extract_reasoning


def _task_to_openai_messages(task: dict) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    current_role: Optional[str] = None
    current_content: List[Dict[str, Any]] = []
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
                image_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}
            if current_role not in {role, None}:
                _flush()
            current_role = role
            current_content.append(image_part)
            contains_non_text = True
        else:
            continue
    _flush()
    return messages


def _inject_expectation_hint(task: dict, expects: Optional[str]) -> dict:
    if expects not in {"point", "box", "polygon"}:
        return task
    hint = f"<hint>{expects.upper()}</hint>"
    content = task.get("content") or []
    if any(entry.get("content") == hint for entry in content if isinstance(entry, dict)):
        return task
    new_content: List[Dict[str, Any]] = []
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
        "path": "/perceptron/genesis/openai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Key ",
        "env_keys": ["FAL_KEY", "PERCEPTRON_API_KEY"],
        "default_model": "perceptron",
        "stream": True,
    },
}


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


def _map_http_error(resp: requests.Response) -> SDKError:
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code == 400:
        return BadRequestError(str(data) or resp.text)
    if resp.status_code in (401, 403):
        return AuthError("authentication failed")
    if resp.status_code == 404:
        return BadRequestError("not found")
    if resp.status_code == 429:
        retry_after = None
        try:
            retry_after = float(resp.headers.get("Retry-After", "0"))
        except Exception:
            pass
        return RateLimitError("rate limited", retry_after=retry_after)
    if 400 <= resp.status_code < 500:
        return BadRequestError(str(data) or resp.text)
    return ServerError(f"server error: {resp.status_code}")


class Client:
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

    def generate(self, task: dict, *, expects: Optional[str] = None, **gen_kwargs: Any) -> dict:
        """Execute a Task and return a result dict."""
        s = self._settings
        provider_cfg = _resolve_provider(gen_kwargs.pop("provider", None) or s.provider)

        temperature = gen_kwargs.pop("temperature", s.temperature)
        max_tokens = gen_kwargs.pop("max_tokens", s.max_tokens)
        top_p = gen_kwargs.pop("top_p", s.top_p)
        top_k = gen_kwargs.pop("top_k", s.top_k)

        task, url, headers, provider_cfg = _prepare_transport(s, provider_cfg, task, expects)

        messages = _task_to_openai_messages(task)
        model = gen_kwargs.pop("model", provider_cfg.get("default_model", "gpt-4o"))
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if top_k is not None:
            body["top_k"] = top_k
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=s.timeout)
        except requests.Timeout as e:
            raise TimeoutError("request timed out") from e
        except requests.RequestException as e:
            raise TransportError(str(e)) from e
        if resp.status_code != 200:
            raise _map_http_error(resp)
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        content, reasoning = self._clean_text_with_reasoning(content, data)
        result: dict[str, Any] = {"text": content, "raw": data}
        if reasoning:
            result["reasoning"] = reasoning
        if expects in {"point", "box", "polygon"} and isinstance(content, str):
            kind = "point" if expects == "point" else ("box" if expects == "box" else "polygon")
            result["points"] = extract_points(content, expected=kind)
            result["parsed"] = parse_text(content)
        return result

    def stream(self, task: dict, *, expects: Optional[str] = None, parse_points: bool = False, **gen_kwargs: Any):
        """Yield streaming events: text.delta, points.delta, usage, error, final.

        Notes
        - points.delta is emitted only when a full tag has closed (no partial tags).
        - For collections, by default we emit a single event on the collection closing tag containing all child items.
        """
        s = self._settings
        provider_cfg = _resolve_provider(gen_kwargs.pop("provider", None) or s.provider)
        temperature = gen_kwargs.pop("temperature", s.temperature)
        max_tokens = gen_kwargs.pop("max_tokens", s.max_tokens)
        top_p = gen_kwargs.pop("top_p", s.top_p)
        top_k = gen_kwargs.pop("top_k", s.top_k)

        task = _inject_expectation_hint(task, expects)

        try:
            task, url, headers, provider_cfg = _prepare_transport(s, provider_cfg, task, expects, stream=True)
        except SDKError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        messages = _task_to_openai_messages(task)
        body = {
            "model": gen_kwargs.pop("model", provider_cfg.get("default_model", "gpt-4o")),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        if top_k is not None:
            body["top_k"] = top_k

        cumulative: str = ""
        emitted_spans: set[tuple[int, int]] = set()
        parsing_enabled: bool = True
        usage_payload: dict[str, Any] | None = None

        try:
            with requests.post(url, headers=headers, data=json.dumps(body), timeout=s.timeout, stream=True) as resp:
                if resp.status_code != 200:
                    err = _map_http_error(resp)
                    yield {"type": "error", "message": str(err)}
                    return
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="ignore")
                    else:
                        line = raw_line
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("usage") and usage_payload is None:
                        usage_field = obj.get("usage")
                        if isinstance(usage_field, dict):
                            usage_payload = usage_field
                    try:
                        delta = obj["choices"][0]["delta"].get("content")
                    except Exception:
                        delta = None
                    if delta:
                        cumulative += delta
                        yield {"type": "text.delta", "chunk": delta, "total_chars": len(cumulative)}
                        # Check buffer size
                        if parsing_enabled and s.max_buffer_bytes is not None:
                            if len(cumulative.encode("utf-8")) > s.max_buffer_bytes:
                                parsing_enabled = False
                                # We stop emitting points thereafter; final will include an issue
                        # Incremental points
                        if parse_points and parsing_enabled and expects in {"point", "box", "polygon"}:
                            # parse segments and emit newly completed tags
                            for seg in parse_text(cumulative):
                                if seg["kind"] in {"point", "box", "polygon"}:
                                    span = (seg["span"]["start"], seg["span"]["end"])
                                    if span not in emitted_spans and seg["kind"] == expects:
                                        emitted_spans.add(span)
                                        yield {"type": "points.delta", "points": [seg["value"]], "span": seg["span"]}
                    # else: ignore other delta types
        except requests.Timeout:
            yield {"type": "error", "message": "timeout"}
            return
        except requests.RequestException as e:
            yield {"type": "error", "message": str(e)}
            return
        # final
        cleaned_text, reasoning_final = self._clean_text_with_reasoning(cumulative)
        result: dict[str, Any] = {"text": cleaned_text, "raw": None}
        if reasoning_final:
            result["reasoning"] = reasoning_final
        if expects in {"point", "box", "polygon"} and parsing_enabled and isinstance(cleaned_text, str):
            result["points"] = [seg["value"] for seg in parse_text(cleaned_text) if seg["kind"] == expects]
            result["parsed"] = parse_text(cleaned_text)
        issues: list[dict] = []
        if not parsing_enabled:
            issues.append({"code": "stream_buffer_overflow", "message": "parsing disabled due to buffer limit"})
        if usage_payload:
            result["usage"] = usage_payload
        yield {
            "type": "final",
            "result": {
                "text": result.get("text"),
                "points": result.get("points"),
                "parsed": result.get("parsed"),
                "usage": result.get("usage"),
                "errors": issues,
                "raw": result.get("raw"),
            },
        }


class AsyncClient(Client):
    """Asynchronous variant using httpx.AsyncClient."""

    def __init__(self, **overrides: Any) -> None:
        super().__init__(**overrides)
        if httpx is None:  # pragma: no cover - guarded during tests
            raise ImportError("Install the 'httpx' package to enable AsyncClient support.")

    async def generate(self, task: dict, *, expects: Optional[str] = None, **gen_kwargs: Any) -> dict:
        if httpx is None:  # pragma: no cover
            raise ImportError("httpx is required for AsyncClient.generate")
        s = self._settings
        provider_cfg = _resolve_provider(gen_kwargs.pop("provider", None) or s.provider)

        temperature = gen_kwargs.pop("temperature", s.temperature)
        max_tokens = gen_kwargs.pop("max_tokens", s.max_tokens)
        top_p = gen_kwargs.pop("top_p", s.top_p)
        top_k = gen_kwargs.pop("top_k", s.top_k)

        prepared_task, url, headers, provider_cfg = _prepare_transport(s, provider_cfg, task, expects)

        messages = _task_to_openai_messages(prepared_task)
        model = gen_kwargs.pop("model", provider_cfg.get("default_model", "gpt-4o"))
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if top_k is not None:
            body["top_k"] = top_k

        try:
            async with httpx.AsyncClient(timeout=s.timeout) as session:
                resp = await session.post(url, headers=headers, content=json.dumps(body))
        except httpx.TimeoutException as e:  # pragma: no cover - error path
            raise TimeoutError("request timed out") from e
        except httpx.HTTPError as e:  # pragma: no cover
            raise TransportError(str(e)) from e
        if resp.status_code != 200:
            raise _map_http_error(resp)
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        content, reasoning = self._clean_text_with_reasoning(content, data)
        result: dict[str, Any] = {"text": content, "raw": data}
        if reasoning:
            result["reasoning"] = reasoning
        if expects in {"point", "box", "polygon"} and isinstance(content, str):
            kind = "point" if expects == "point" else ("box" if expects == "box" else "polygon")
            result["points"] = extract_points(content, expected=kind)
            result["parsed"] = parse_text(content)
        return result

    def stream(self, task: dict, *, expects: Optional[str] = None, parse_points: bool = False, **gen_kwargs: Any):
        if httpx is None:  # pragma: no cover
            raise ImportError("httpx is required for AsyncClient.stream")
        s = self._settings
        provider_cfg = _resolve_provider(gen_kwargs.pop("provider", None) or s.provider)
        temperature = gen_kwargs.pop("temperature", s.temperature)
        max_tokens = gen_kwargs.pop("max_tokens", s.max_tokens)
        top_p = gen_kwargs.pop("top_p", s.top_p)
        top_k = gen_kwargs.pop("top_k", s.top_k)

        task_with_hint = _inject_expectation_hint(task, expects)

        async def _run_async_stream():
            try:
                prepared_task, url, headers, resolved_cfg = _prepare_transport(
                    s, provider_cfg, task_with_hint, expects, stream=True
                )
            except SDKError as exc:
                yield {"type": "error", "message": str(exc)}
                return

            messages = _task_to_openai_messages(prepared_task)
            body = {
                "model": gen_kwargs.pop("model", resolved_cfg.get("default_model", "gpt-4o")),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": True,
            }
            if top_k is not None:
                body["top_k"] = top_k

            cumulative: str = ""
            emitted_spans: set[tuple[int, int]] = set()
            parsing_enabled: bool = True
            try:
                async with httpx.AsyncClient(timeout=s.timeout) as session:
                    async with session.stream("POST", url, headers=headers, content=json.dumps(body)) as resp:
                        if resp.status_code != 200:
                            err = _map_http_error(resp)
                            yield {"type": "error", "message": str(err)}
                            return
                        async for line in resp.aiter_lines():
                            if not line:
                                continue
                            if not line.startswith("data:"):
                                continue
                            data_line = line[len("data:") :].strip()
                            if data_line == "[DONE]":
                                break
                            try:
                                obj = json.loads(data_line)
                            except Exception:
                                continue
                            try:
                                delta = obj["choices"][0]["delta"].get("content")
                            except Exception:
                                delta = None
                            if delta:
                                cumulative += delta
                                yield {"type": "text.delta", "chunk": delta, "total_chars": len(cumulative)}
                                if parsing_enabled and s.max_buffer_bytes is not None:
                                    if len(cumulative.encode("utf-8")) > s.max_buffer_bytes:
                                        parsing_enabled = False
                                if parse_points and parsing_enabled and expects in {"point", "box", "polygon"}:
                                    for seg in parse_text(cumulative):
                                        if seg["kind"] in {"point", "box", "polygon"}:
                                            span = (seg["span"]["start"], seg["span"]["end"])
                                            if span not in emitted_spans and seg["kind"] == expects:
                                                emitted_spans.add(span)
                                                yield {
                                                    "type": "points.delta",
                                                    "points": [seg["value"]],
                                                    "span": seg["span"],
                                                }
            except httpx.TimeoutException:
                yield {"type": "error", "message": "timeout"}
                return
            except httpx.HTTPError as e:
                yield {"type": "error", "message": str(e)}
                return

            cleaned_text, reasoning_final = self._clean_text_with_reasoning(cumulative)
            result: dict[str, Any] = {"text": cleaned_text, "raw": None}
            if reasoning_final:
                result["reasoning"] = reasoning_final
            if expects in {"point", "box", "polygon"} and parsing_enabled and isinstance(cleaned_text, str):
                result["points"] = [seg["value"] for seg in parse_text(cleaned_text) if seg["kind"] == expects]
                result["parsed"] = parse_text(cleaned_text)
            issues: list[dict] = []
            if not parsing_enabled:
                issues.append({"code": "stream_buffer_overflow", "message": "parsing disabled due to buffer limit"})
            yield {
                "type": "final",
                "result": {
                    "text": result.get("text"),
                    "points": result.get("points"),
                    "parsed": result.get("parsed"),
                    "usage": None,
                    "errors": issues,
                    "raw": result.get("raw"),
                },
            }

        return _run_async_stream()
