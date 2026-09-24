"""`create(stream=True)`: SSE parsing, accumulation, terminal conditions, and stream lifetime (sync and async)."""

import asyncio
import gc
import json
import warnings

import httpx
import pytest
from _http_mock import Body, FailingBody, chunk, install, json_response, sse_body, sse_response

from perceptron import AsyncClient, Client
from perceptron.chat import (
    AsyncChatCompletionStream,
    ChatCompletionChunk,
    ChatCompletionStream,
    ChatStreamAccumulator,
)
from perceptron.errors import (
    INVALID_STREAM_CHUNK,
    STREAM_TRUNCATED,
    BadRequestError,
    IncompleteStreamError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
)
from perceptron.errors import TimeoutError as SDKTimeoutError

USER = {"role": "user", "content": "Weather in SF and NYC?"}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"audio_tokens": 0}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _serve(monkeypatch, events, *, done=True, headers=None):
    body = Body(sse_body(events, done=done))
    recorder = install(monkeypatch, lambda request: sse_response(None, body=body, headers=headers))
    return recorder, body


def _stream(**kwargs):
    kwargs.setdefault("messages", [USER])
    return Client().chat.completions.create(stream=True, **kwargs)


TEXT_EVENTS = [
    chunk({"role": "assistant", "reasoning_content": "Counting"}),
    chunk({"reasoning_content": " cars"}),
    ": keep-alive",
    chunk({"content": "There are "}),
    chunk({"content": "3 cars."}),
    chunk({}, finish_reason="stop"),
    chunk(choices=False, usage=USAGE),
]


def test_stream_yields_chunks_and_final_completion(monkeypatch):
    recorder, body = _serve(monkeypatch, TEXT_EVENTS, headers={"x-trace-id": "trace-s"})

    stream = _stream()

    assert isinstance(stream, ChatCompletionStream)
    assert recorder.last_body["stream"] is True
    assert recorder.last_body["stream_options"] == {"include_usage": True}
    chunks = list(stream)
    assert all(isinstance(c, ChatCompletionChunk) for c in chunks)
    assert len(chunks) == 6  # the keep-alive is not a chunk
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 15

    final = stream.completion
    assert final is stream.get_final_completion()
    assert final.text == "There are 3 cars."
    assert final.reasoning == "Counting cars"
    assert final.finish_reason == "stop"
    assert final.usage.prompt_tokens_details == {"audio_tokens": 0}
    assert final.request_id == stream.request_id == "trace-s"
    assert (final.id, final.model, final.object) == ("chatcmpl-1", "perceptron-mk1.5", "chat.completion")
    assert final.done and final.complete
    assert final.asset_count == 0
    assert body.closed


def test_caller_stream_options_win(monkeypatch):
    recorder, _ = _serve(monkeypatch, [chunk({}, finish_reason="stop")])
    _stream(stream_options={"include_usage": False}).get_final_completion()
    assert recorder.last_body["stream_options"] == {"include_usage": False}


def test_no_default_stream_options_on_fal(monkeypatch):
    recorder, _ = _serve(monkeypatch, [chunk({}, finish_reason="stop")])
    Client(provider="fal", api_key="fal-key").chat.completions.create(
        messages=[USER], stream=True
    ).get_final_completion()
    assert "stream_options" not in recorder.last_body


def test_usage_last_non_null_wins(monkeypatch):
    events = [
        chunk({"content": "a"}, usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        chunk({}, finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
        chunk(choices=False, usage=USAGE),
        chunk(choices=False),
    ]
    _serve(monkeypatch, events)
    assert _stream().get_final_completion().usage.total_tokens == 15


def test_null_and_missing_deltas_are_tolerated(monkeypatch):
    events = [chunk(None), chunk({"content": "ok"}), chunk(finish_reason="stop", omit_delta=True)]
    _serve(monkeypatch, events)

    final = _stream().get_final_completion()

    assert final.text == "ok"
    assert final.message.role == "assistant"
    assert final.complete


def test_tool_call_fragments_assemble_by_index(monkeypatch):
    events = [
        chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {"index": 1, "id": "call_b", "type": "function", "function": {"name": "lookup", "arguments": ""}}
                ],
            }
        ),
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"ci'},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"sku": 1}'}}]}),
        chunk({"tool_calls": [{"index": 0, "id": "", "function": {"name": "", "arguments": 'ty": "SF"}'}}]}),
        chunk({}, finish_reason="tool_calls"),
    ]
    _serve(monkeypatch, events)

    final = _stream().get_final_completion()

    calls = [(c.id, c.name, c.arguments, c.type) for c in final.tool_calls]
    assert calls == [
        ("call_a", "get_weather", '{"city": "SF"}', "function"),
        ("call_b", "lookup", '{"sku": 1}', "function"),
    ]
    assert final.text is None
    assert final.complete
    assert final.message.to_dict()["content"] is None


def test_legacy_tool_calls_arrive_whole_with_the_finish(monkeypatch):
    calls = [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "w", "arguments": '{"c": "SF"}'}},
        {"index": 1, "id": "call_2", "type": "function", "function": {"name": "w", "arguments": '{"c": "NYC"}'}},
    ]
    _serve(monkeypatch, [chunk({"role": "assistant", "tool_calls": calls}, finish_reason="tool_calls")])

    final = _stream().get_final_completion()

    assert [c.id for c in final.tool_calls] == ["call_1", "call_2"]
    assert final.complete


def test_done_without_finish_reason_is_not_an_error(monkeypatch):
    _serve(monkeypatch, [chunk({"content": "partial"})])

    final = _stream().get_final_completion()

    assert final.text == "partial"
    assert final.finish_reason is None
    assert final.done and not final.complete


def test_length_is_incomplete_but_not_an_error(monkeypatch):
    call = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "w", "arguments": '{"c": '}}
    _serve(monkeypatch, [chunk({"tool_calls": [call]}), chunk({}, finish_reason="length")])

    final = _stream().get_final_completion()

    assert final.finish_reason == "length"
    assert final.tool_calls  # visible, but not executable
    assert not final.complete


@pytest.mark.parametrize(
    ("error", "expected_cls", "code"),
    [
        (
            {"message": "Generation failed.", "type": "server_error", "param": None, "code": "internal_error"},
            ServerError,
            "internal_error",
        ),
        (
            {"message": "bad", "type": "invalid_request_error", "param": "regex", "code": "invalid_regex"},
            BadRequestError,
            "invalid_regex",
        ),
        (
            {"message": "slow", "type": "rate_limit_error", "param": None, "code": "rate_limit_exceeded"},
            RateLimitError,
            "rate_limit",
        ),
        (
            {"message": "quota", "type": "insufficient_quota", "param": None, "code": None},
            QuotaExceededError,
            "insufficient_quota",
        ),
        ({"message": "The server had an error", "type": None, "param": None, "code": None}, ServerError, None),
    ],
)
def test_error_event_raises_the_mapped_error_with_partial(monkeypatch, error, expected_cls, code):
    events = [chunk({"role": "assistant", "content": "The answer is"}), {"error": error}]
    _, body = _serve(monkeypatch, events, done=False, headers={"x-trace-id": "trace-e"})
    stream = _stream()

    received = []
    with pytest.raises(expected_cls) as excinfo:
        for piece in stream:
            received.append(piece)

    err = excinfo.value
    assert type(err) is expected_cls
    assert err.code == code
    assert err.error_type == error["type"]
    assert err.request_id == "trace-e"
    assert err.status_code is None
    assert err.partial.text == "The answer is"
    assert not err.partial.complete
    assert len(received) == 1
    assert body.closed
    with pytest.raises(expected_cls):
        stream.get_final_completion()


def test_eof_without_done_is_truncated(monkeypatch):
    _serve(monkeypatch, [chunk({"content": "cut"}), chunk({}, finish_reason="stop")], done=False)

    with pytest.raises(IncompleteStreamError) as excinfo:
        _stream().get_final_completion()

    assert excinfo.value.code == STREAM_TRUNCATED
    assert excinfo.value.partial.text == "cut"
    assert excinfo.value.partial.finish_reason == "stop"
    assert not excinfo.value.partial.complete


def test_malformed_json_is_a_terminal_error(monkeypatch):
    _serve(monkeypatch, [chunk({"content": "a"}), "data: {not json", chunk({"content": "b"})])

    with pytest.raises(ServerError) as excinfo:
        _stream().get_final_completion()

    assert excinfo.value.code == INVALID_STREAM_CHUNK
    assert excinfo.value.partial.text == "a"


def test_connection_cut_mid_body_is_truncated(monkeypatch):
    class _Cut(httpx.SyncByteStream):
        def __iter__(self):
            yield sse_body([chunk({"content": "a"})], done=False)
            raise httpx.RemoteProtocolError("peer closed connection")

    install(monkeypatch, lambda request: httpx.Response(200, stream=_Cut()))

    with pytest.raises(IncompleteStreamError) as excinfo:
        _stream().get_final_completion()
    assert excinfo.value.code == STREAM_TRUNCATED


class _Chunked(Body):
    """A body delivered ``size`` bytes at a time, as a socket may split it."""

    def __init__(self, data: bytes, size: int) -> None:
        super().__init__(data)
        self._size = size

    def _pieces(self):
        return [self._data[i : i + self._size] for i in range(0, len(self._data), self._size)]

    def __iter__(self):
        yield from self._pieces()

    async def __aiter__(self):
        for piece in self._pieces():
            yield piece


def _stream_surfaces():
    """Final text (or the terminal error code) from each stream surface: message API and legacy, sync and async."""
    task = {"content": [{"type": "text", "role": "user", "content": "hi"}]}

    def _legacy(events):
        last = events[-1]
        return last["result"]["text"] if last["type"] == "final" else last["code"]

    async def _async_message():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        try:
            return (await stream.get_final_completion()).text
        except IncompleteStreamError as exc:
            return exc.code

    async def _async_legacy():
        return _legacy([event async for event in AsyncClient().stream(task)])

    try:
        message = _stream().get_final_completion().text
    except IncompleteStreamError as exc:
        message = exc.code
    return {
        "message": message,
        "async message": asyncio.run(_async_message()),
        "legacy": _legacy(list(Client().stream(task))),
        "async legacy": asyncio.run(_async_legacy()),
    }


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_line_separators_inside_an_event_are_content(monkeypatch, separator):
    # serde_json writes U+2028/U+2029/U+0085 raw; SSE lines end only at CR, LF or CRLF.
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "perceptron")
    raw = json.dumps(chunk({"content": f"one{separator}two"}), ensure_ascii=False)
    data = f"data: {raw}\n\n".encode() + sse_body([chunk({}, finish_reason="stop")])
    install(monkeypatch, lambda request: sse_response(None, body=_Chunked(data, 5)))

    results = _stream_surfaces()

    assert results == dict.fromkeys(results, f"one{separator}two")


def test_eof_mid_line_is_truncated_not_malformed(monkeypatch):
    # A body that ends partway through an event (no newline) lost that event: truncated, like any EOF before [DONE].
    monkeypatch.setenv("PERCEPTRON_PROVIDER", "perceptron")
    data = sse_body([chunk({"content": "a"})], done=False) + b'data: {"id":"chatcmpl-1","choices":[{"index":0,"de'
    install(monkeypatch, lambda request: sse_response(None, body=_Chunked(data, 7)))

    results = _stream_surfaces()

    assert results == dict.fromkeys(results, STREAM_TRUNCATED)


# A timeout or cut connection after the first event: (exception, raised class, code).
MID_BODY_FAILURES = [
    (httpx.RemoteProtocolError("peer closed connection"), IncompleteStreamError, STREAM_TRUNCATED),
    (httpx.ReadTimeout("read timed out"), SDKTimeoutError, None),
]


def _serve_failing(monkeypatch, exc):
    body = FailingBody(sse_body([chunk({"content": "a"})], done=False), exc)
    install(monkeypatch, lambda request: sse_response(None, body=body))
    return body


@pytest.mark.parametrize(("exc", "expected_cls", "code"), MID_BODY_FAILURES)
def test_failure_mid_body_raises_with_partial(monkeypatch, exc, expected_cls, code):
    body = _serve_failing(monkeypatch, exc)

    with pytest.raises(expected_cls) as excinfo:
        _stream().get_final_completion()

    assert excinfo.value.code == code
    assert excinfo.value.__cause__ is exc
    assert excinfo.value.partial.text == "a"
    assert not excinfo.value.partial.complete
    assert body.closed


def test_http_error_raises_before_the_stream_is_returned(monkeypatch):
    error = {"message": "Model 'x' does not exist", "type": "invalid_request_error", "param": None, "code": None}
    install(monkeypatch, lambda request: json_response({"error": error}, 400, headers={"x-trace-id": "t"}))

    with pytest.raises(BadRequestError) as excinfo:
        _stream()

    assert str(excinfo.value) == "Model 'x' does not exist"  # the unread body was read before mapping
    assert excinfo.value.request_id == "t"


def test_context_manager_and_close_release_the_connection(monkeypatch):
    _, body = _serve(monkeypatch, TEXT_EVENTS)
    with _stream() as stream:
        next(stream)
    assert body.closed
    with pytest.raises(IncompleteStreamError):
        stream.get_final_completion()

    _, body = _serve(monkeypatch, TEXT_EVENTS)
    stream = _stream()
    stream.close()
    stream.close()
    assert body.closed
    assert list(stream) == []


# ---------------------------------------------------------------------------
# Dropped streams release their connection (the client's pool stays open)
# ---------------------------------------------------------------------------


class _CountingBody(Body):
    """A response body that counts how often it is closed."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.closes = 0

    def close(self) -> None:
        super().close()
        self.closes += 1

    async def aclose(self) -> None:
        await super().aclose()
        self.closes += 1


def _serve_counted(monkeypatch, events=TEXT_EVENTS, *, done=True):
    """Serve ``events``; returns the response body (counting its closes) and the recorder (with the pooled clients)."""
    body = _CountingBody(sse_body(events, done=done))
    http = install(monkeypatch, lambda request: sse_response(None, body=body))
    return body, http


def _released(body, http) -> bool:
    """The response is closed exactly once, and the one pooled HTTP client stays open for the client's next request."""
    (pool,) = http.clients
    return body.closes == 1 and not pool.is_closed


async def _loop_turns():
    for _ in range(20):
        await asyncio.sleep(0)


def test_a_dropped_unconsumed_stream_releases_its_connection(monkeypatch):
    body, http = _serve_counted(monkeypatch)
    stream = _stream()
    assert not body.closed

    del stream
    gc.collect()

    assert _released(body, http)


def test_a_stream_dropped_after_breaking_out_of_a_loop_releases_its_connection(monkeypatch):
    body, http = _serve_counted(monkeypatch)
    stream = _stream()
    for _ in stream:
        break
    assert not body.closed

    del stream
    gc.collect()

    assert _released(body, http)


def _break_inside_with(stream):
    with stream:
        for _ in stream:
            break


@pytest.mark.parametrize(
    "finish",
    [list, ChatCompletionStream.get_final_completion, ChatCompletionStream.close, _break_inside_with],
    ids=["exhausted", "get_final_completion", "close", "break-inside-with"],
)
def test_normal_paths_close_exactly_once(monkeypatch, finish):
    body, http = _serve_counted(monkeypatch)
    stream = _stream()

    finish(stream)

    assert _released(body, http)
    stream.close()
    del stream
    gc.collect()
    assert body.closes == 1


def test_a_failed_stream_closes_exactly_once(monkeypatch):
    body, http = _serve_counted(monkeypatch, [chunk({"content": "x"})], done=False)
    stream = _stream()

    with pytest.raises(IncompleteStreamError):
        stream.get_final_completion()

    assert _released(body, http)
    del stream
    gc.collect()
    assert body.closes == 1


@pytest.mark.parametrize("iterated", [False, True], ids=["unconsumed", "break-without-async-with"])
def test_an_async_stream_dropped_while_its_loop_runs_is_closed_there(monkeypatch, iterated):
    body, http = _serve_counted(monkeypatch)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        if iterated:
            async for _ in stream:
                break
        assert not body.closed
        del stream
        gc.collect()
        await _loop_turns()
        return _released(body, http)

    assert asyncio.run(_run())
    assert body.closes == 1


def test_async_early_break_inside_async_with_releases_at_once(monkeypatch):
    body, http = _serve_counted(monkeypatch)

    async def _run():
        async with await AsyncClient().chat.completions.create(messages=[USER], stream=True) as stream:
            async for _ in stream:
                break
        return _released(body, http)

    assert asyncio.run(_run())


@pytest.mark.parametrize("iterated", [False, True])
def test_async_close_releases_at_once_and_exactly_once(monkeypatch, iterated):
    body, http = _serve_counted(monkeypatch)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        if iterated:
            await stream.__anext__()
        await stream.close()
        released = _released(body, http)
        await stream.close()
        del stream
        gc.collect()
        await _loop_turns()
        return released

    assert asyncio.run(_run())
    assert body.closes == 1


def test_async_normal_paths_close_exactly_once(monkeypatch):
    body, http = _serve_counted(monkeypatch)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        await stream.get_final_completion()
        released = _released(body, http)
        del stream
        gc.collect()
        await _loop_turns()
        return released

    assert asyncio.run(_run())
    assert body.closes == 1


def test_a_failed_scheduled_close_is_not_reported_as_unretrieved(monkeypatch):
    body, _ = _serve_counted(monkeypatch)
    reported, attempts = [], []

    async def _failing_aclose():
        attempts.append(1)
        raise httpx.ConnectError("gone")

    async def _run():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: reported.append(context))
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        monkeypatch.setattr(body, "aclose", _failing_aclose)
        del stream
        gc.collect()
        await _loop_turns()
        gc.collect()  # a task whose exception was never retrieved reports it when collected

    asyncio.run(_run())

    assert attempts == [1]
    assert reported == []


def test_an_async_stream_dropped_between_runs_of_its_loop_is_closed_on_the_next_run(monkeypatch):
    body, http = _serve_counted(monkeypatch)
    loop = asyncio.new_event_loop()
    try:
        stream = loop.run_until_complete(AsyncClient().chat.completions.create(messages=[USER], stream=True))
        assert not body.closed
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del stream
            gc.collect()  # the loop is idle but open: the close waits for its next run
        assert not [w for w in caught if "AsyncChatCompletionStream" in str(w.message)]

        loop.run_until_complete(_loop_turns())

        assert _released(body, http)
    finally:
        loop.close()


def test_an_async_stream_collected_after_its_loop_closed_warns(monkeypatch):
    body, http = _serve_counted(monkeypatch)

    async def _open():
        return await AsyncClient().chat.completions.create(messages=[USER], stream=True)

    stream = asyncio.run(_open())
    closes = body.closes  # asyncio.run's shutdown of async generators may close it (httpx's stream() is one)
    with pytest.warns(ResourceWarning, match="garbage collected unclosed with no open event loop"):
        del stream
        gc.collect()

    assert body.closes == closes  # the finalizer closes nothing: an async response closes only on its (stopped) loop
    assert not http.clients[0].is_closed  # and a stream never closes the client's pool


LEGACY_TASK = {"content": [{"type": "text", "role": "user", "content": "hi"}]}


def test_a_legacy_stream_dropped_mid_iteration_releases_its_connection(monkeypatch):
    body, http = _serve_counted(monkeypatch)
    events = Client().stream(LEGACY_TASK)
    next(events)
    assert not body.closed

    del events  # the generator is closed as soon as nothing references it

    assert _released(body, http)


def test_an_async_legacy_stream_dropped_mid_iteration_releases_its_connection(monkeypatch):
    body, http = _serve_counted(monkeypatch)

    async def _run():
        events = AsyncClient().stream(LEGACY_TASK)
        await events.__anext__()
        assert not body.closed
        del events  # asyncio closes a dropped async generator on its loop
        await _loop_turns()
        return _released(body, http)

    assert asyncio.run(_run())


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


def test_accumulator_folds_chunks():
    acc = ChatStreamAccumulator(request_id="r", asset_count=2)
    acc.feed(chunk({"role": "assistant", "content": "a"}))
    acc.feed({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]}}]})
    acc.feed({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}}]})
    acc.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})

    before = acc.snapshot()
    assert not before.done and not before.complete
    acc.mark_done()
    after = acc.snapshot()

    assert after.complete
    assert after.text == "a"
    assert after.tool_calls[0].to_dict() == {
        "id": "c",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    assert (after.request_id, after.asset_count, after.raw) == ("r", 2, None)


def test_chunk_to_dict_omits_unset_fields():
    parsed = ChatCompletionChunk.from_dict(
        {"id": "x", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}]}
    )
    assert parsed.to_dict()["choices"] == [
        {"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}
    ]


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_stream(monkeypatch):
    recorder, body = _serve(monkeypatch, TEXT_EVENTS, headers={"x-trace-id": "trace-a"})

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        assert isinstance(stream, AsyncChatCompletionStream)
        chunks = [piece async for piece in stream]
        assert stream.completion is not None
        return chunks, await stream.get_final_completion()

    chunks, final = asyncio.run(_run())

    assert len(chunks) == 6
    assert final.text == "There are 3 cars."
    assert final.complete
    assert final.request_id == "trace-a"
    assert recorder.last_body["stream_options"] == {"include_usage": True}
    assert body.closed


def test_async_stream_error_event(monkeypatch):
    _serve(monkeypatch, [chunk({"content": "x"}), {"error": {"message": "boom", "type": "server_error"}}], done=False)

    async def _run():
        async with await AsyncClient().chat.completions.create(messages=[USER], stream=True) as stream:
            await stream.get_final_completion()

    with pytest.raises(ServerError) as excinfo:
        asyncio.run(_run())
    assert excinfo.value.partial.text == "x"


def test_async_stream_truncated(monkeypatch):
    _serve(monkeypatch, [chunk({"content": "x"})], done=False)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        await stream.get_final_completion()

    with pytest.raises(IncompleteStreamError):
        asyncio.run(_run())


@pytest.mark.parametrize(("exc", "expected_cls", "code"), MID_BODY_FAILURES)
def test_async_failure_mid_body_raises_with_partial(monkeypatch, exc, expected_cls, code):
    body = _serve_failing(monkeypatch, exc)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        await stream.get_final_completion()

    with pytest.raises(expected_cls) as excinfo:
        asyncio.run(_run())

    assert excinfo.value.code == code
    assert excinfo.value.__cause__ is exc
    assert excinfo.value.partial.text == "a"
    assert not excinfo.value.partial.complete
    assert body.closed


def test_async_eof_after_the_finish_without_done_is_truncated(monkeypatch):
    """The truncation error surfaces as itself on the async path, not as a transport error."""
    _, body = _serve(monkeypatch, [chunk({"content": "hi"}), chunk({}, finish_reason="stop")], done=False)

    async def _run():
        stream = await AsyncClient().chat.completions.create(messages=[USER], stream=True)
        return await stream.get_final_completion()

    with pytest.raises(IncompleteStreamError) as excinfo:
        asyncio.run(_run())
    assert excinfo.value.code == STREAM_TRUNCATED
    assert excinfo.value.partial.text == "hi"
    assert excinfo.value.partial.finish_reason == "stop"
    assert not excinfo.value.partial.complete
    assert body.closed
