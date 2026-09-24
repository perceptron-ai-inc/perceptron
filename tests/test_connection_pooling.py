"""One pooled HTTP client per `Client` / `AsyncClient`: reused by every call and surface, closed by `close()` /
`aclose()` / `with`, never closed when the caller supplied it, carrying per-request timeouts, left open by streams, and
closed by the helpers that create a client per call."""

from __future__ import annotations

import asyncio
import gc
import json
import socket
import threading

import httpcore
import httpx
import pytest
from _http_mock import chunk, completion, install, json_response, sse_body, sse_response
from _image_fixtures import PNG_BYTES

from perceptron import AsyncClient, Client, async_perceive, image, perceive, question, settings, text
from perceptron import client as client_mod
from perceptron.errors import STREAM_TRUNCATED, BadRequestError, IncompleteStreamError

USER = {"role": "user", "content": "hi"}
TASK = {"content": [{"type": "text", "role": "user", "content": "hi"}]}
FIRST_EVENT, LAST_EVENTS = [chunk({"content": "Hello"})], [chunk({}, finish_reason="stop")]
FILE = {"object": "file", "id": "file-Q7m2Lx9aPz3Kc8Rt1VbN0s", "bytes": 8, "created_at": 1, "filename": "a.png"}
MODEL = {"id": "perceptron-mk1.5", "object": "model", "created": 1, "owned_by": "perceptron"}
MULTILOOK = {
    "id": "mlcmpl-1",
    "object": "chat.completion.multilook",
    "model": "perceptron-mk1.5",
    "results": [{"prompt_index": 0, "completions": [{"index": 0, "message": {"role": "assistant", "content": "a"}}]}],
}
SURFACE_REQUESTS = 8  # the requests `_every_surface` sends


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("FAL_KEY", "PERCEPTRON_PROVIDER", "PERCEPTRON_MODEL", "PERCEPTRON_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PERCEPTRON_API_KEY", "sk-test")


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/chat/completions/multilook":
        return json_response(MULTILOOK)
    if path == "/v1/chat/completions":
        if json.loads(request.content).get("stream"):
            return sse_response(FIRST_EVENT + LAST_EVENTS)
        return json_response(completion())
    if path == "/v1/models":
        return json_response({"object": "list", "data": [MODEL]})
    if path == "/v1/files" and request.method == "POST":
        return json_response(FILE)
    if path == "/v1/files":
        return json_response({"object": "list", "data": [FILE], "has_more": False})
    raise AssertionError(f"unexpected request: {request.method} {path}")


@pytest.fixture
def http(monkeypatch):
    return install(monkeypatch, _handler)


def _every_surface(client: Client) -> None:
    assert client.generate(TASK)["text"] == "Hello"
    assert list(client.stream(TASK))[-1]["type"] == "final"
    assert client.chat.completions.create(messages=[USER]).text == "Hello"
    with client.chat.completions.create(messages=[USER], stream=True) as stream:
        assert stream.get_final_completion().text == "Hello"
    assert client.chat.completions.multilook(context=[], prompts=["q"]).results[0].ok
    assert client.files.upload(PNG_BYTES).id == FILE["id"]
    assert client.files.list().data[0].id == FILE["id"]
    assert client.models.list()[0].id == MODEL["id"]


async def _aevery_surface(client: AsyncClient) -> None:
    assert (await client.generate(TASK))["text"] == "Hello"
    assert [event async for event in client.stream(TASK)][-1]["type"] == "final"
    assert (await client.chat.completions.create(messages=[USER])).text == "Hello"
    async with await client.chat.completions.create(messages=[USER], stream=True) as stream:
        assert (await stream.get_final_completion()).text == "Hello"
    assert (await client.chat.completions.multilook(context=[], prompts=["q"])).results[0].ok
    assert (await client.files.upload(PNG_BYTES)).id == FILE["id"]
    assert (await client.files.list()).data[0].id == FILE["id"]
    assert (await client.models.list())[0].id == MODEL["id"]


def _count_closes(pool: httpx.Client) -> list[int]:
    calls: list[int] = []
    close = pool.close

    def _close() -> None:
        calls.append(1)
        close()

    pool.close = _close
    return calls


def _count_acloses(pool: httpx.AsyncClient) -> list[int]:
    calls: list[int] = []
    aclose = pool.aclose

    async def _aclose() -> None:
        calls.append(1)
        await aclose()

    pool.aclose = _aclose
    return calls


# ---------------------------------------------------------------------------
# The pooled client
# ---------------------------------------------------------------------------


def test_created_clients_use_http2_and_the_configured_timeout(monkeypatch):
    made = []

    class _Client(httpx.Client):
        def __init__(self, **kwargs):
            made.append(("sync", kwargs))
            super().__init__(**kwargs)

    class _AsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            made.append(("async", kwargs))
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)

    Client(timeout=42.0)._session().close()
    asyncio.run(AsyncClient(timeout=7.0)._session().aclose())

    assert made == [("sync", {"timeout": 42.0, "http2": True}), ("async", {"timeout": 7.0, "http2": True})]


def test_one_http_client_serves_every_call_and_surface(http):
    client = Client()
    assert http.clients == []  # created on first use

    for _ in range(3):
        client.chat.completions.create(messages=[USER])
    _every_surface(client)

    (pool,) = http.clients
    assert isinstance(pool, httpx.Client) and not pool.is_closed
    assert len(http.requests) == 3 + SURFACE_REQUESTS


def test_separate_clients_have_separate_pools(http):
    first, second = Client(), Client()
    first.generate(TASK)
    second.generate(TASK)
    first.generate(TASK)

    assert len(http.clients) == 2 and http.clients[0] is not http.clients[1]


def test_close_closes_the_pool_once_and_the_client_stays_closed(http):
    client = Client()
    client.generate(TASK)
    (pool,) = http.clients
    closes = _count_closes(pool)

    client.close()
    client.close()

    assert pool.is_closed and closes == [1]
    with pytest.raises(RuntimeError, match="Client is closed"):
        client.chat.completions.create(messages=[USER])
    assert len(http.clients) == 1  # a closed client does not build a new pool


def test_a_with_block_closes_the_pool(http):
    with Client() as client:
        _every_surface(client)
        (pool,) = http.clients
        assert not pool.is_closed

    assert pool.is_closed
    with Client():
        pass  # nothing was sent, so nothing was created
    Client().close()
    assert len(http.clients) == 1


def test_a_supplied_http_client_is_used_and_never_closed(http):
    own = http.http_client()

    with Client(http_client=own) as client:
        _every_surface(client)
    Client(http_client=own).close()

    assert http.clients == []  # the SDK created no client of its own
    assert len(http.requests) == SURFACE_REQUESTS
    assert not own.is_closed
    assert own.get("https://api.perceptron.inc/v1/models").status_code == 200  # still the caller's to use
    own.close()


def test_http_client_must_be_the_matching_httpx_client():
    with pytest.raises(TypeError, match=r"http_client must be an httpx\.Client; got AsyncClient"):
        Client(http_client=httpx.AsyncClient())
    with pytest.raises(TypeError, match=r"http_client must be an httpx\.AsyncClient; got Client"):
        AsyncClient(http_client=httpx.Client())
    with pytest.raises(TypeError, match=r"http_client must be an httpx\.Client; got str"):
        Client(http_client="https://api.perceptron.inc")


# ---------------------------------------------------------------------------
# Per-request timeouts
# ---------------------------------------------------------------------------


def test_each_request_carries_its_timeout_through_the_one_pool(http):
    client = Client(timeout=30.0)

    client.chat.completions.create(messages=[USER])
    client.chat.completions.create(messages=[USER], timeout=7.5)
    client.chat.completions.create(messages=[USER], stream=True, timeout=8.0).close()
    client.generate(TASK)
    list(client.stream(TASK))
    client.chat.completions.multilook(context=[], prompts=["q"])
    client.chat.completions.multilook(context=[], prompts=["q"], timeout=9.0)
    client.files.list()
    client.models.list()

    assert http.timeouts == [30.0, 7.5, 8.0, 30.0, 30.0, 305.0, 9.0, 30.0, 30.0]
    (pool,) = http.clients  # a per-call timeout did not build another client
    assert pool.timeout.read == 30.0


def test_the_sdk_timeout_applies_to_a_supplied_http_client(http):
    own = http.http_client(timeout=300.0)

    Client(http_client=own).chat.completions.create(messages=[USER])
    Client(http_client=own, timeout=20.0).chat.completions.create(messages=[USER])
    Client(http_client=own).chat.completions.create(messages=[USER], timeout=3.0)

    assert http.timeouts == [settings().timeout, 20.0, 3.0]
    own.close()


# ---------------------------------------------------------------------------
# Streams close their response, never the pool
# ---------------------------------------------------------------------------


def test_streams_leave_the_pool_open(http):
    client = Client()
    stream = client.chat.completions.create(messages=[USER], stream=True)
    next(stream)
    stream.close()
    dropped = client.chat.completions.create(messages=[USER], stream=True)
    events = client.stream(TASK)
    next(events)
    del dropped, events
    gc.collect()

    (pool,) = http.clients
    assert not pool.is_closed
    assert client.chat.completions.create(messages=[USER]).text == "Hello"


def test_closing_a_client_with_open_streams_does_not_fail(http):
    client = Client()
    stream = client.chat.completions.create(messages=[USER], stream=True)
    next(stream)
    dropped = client.chat.completions.create(messages=[USER], stream=True)
    events = client.stream(TASK)
    next(events)

    client.close()
    stream.close()
    events.close()
    del dropped
    gc.collect()

    assert http.clients[0].is_closed


# ---------------------------------------------------------------------------
# Helpers close the client they create
# ---------------------------------------------------------------------------


def test_perceive_and_the_helpers_close_the_client_of_each_call(http):
    assert perceive(text("hi")).text == "Hello"
    assert question(image(PNG_BYTES), "What is this?").text == "Hello"
    assert list(perceive(text("hi"), stream=True))[-1]["type"] == "final"
    stopped = perceive(text("hi"), stream=True)
    next(stopped)
    stopped.close()
    dropped = question(image(PNG_BYTES), "What is this?", stream=True)
    next(dropped)
    del dropped

    assert len(http.clients) == 5
    assert all(pool.is_closed for pool in http.clients)


def test_a_helper_closes_its_client_when_the_request_fails(monkeypatch):
    http = install(monkeypatch, lambda request: json_response({"error": {"message": "no"}}, 400))

    with pytest.raises(BadRequestError, match="no"):
        perceive(text("hi"))

    assert [pool.is_closed for pool in http.clients] == [True]


def test_async_perceive_closes_the_client_of_each_call(http):
    @async_perceive()
    def ask():
        return text("hi")

    @async_perceive(stream=True)
    def ask_stream():
        return text("hi")

    async def _run():
        assert (await ask()).text == "Hello"
        assert [event async for event in ask_stream()][-1]["type"] == "final"
        stopped = ask_stream()
        await stopped.__anext__()
        await stopped.aclose()

    asyncio.run(_run())

    assert len(http.clients) == 3
    assert all(isinstance(pool, httpx.AsyncClient) and pool.is_closed for pool in http.clients)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_client_pools_every_surface_and_closes_once(http):
    async def _run():
        async with AsyncClient() as client:
            for _ in range(3):
                await client.chat.completions.create(messages=[USER])
            await _aevery_surface(client)
            (pool,) = http.clients
            assert isinstance(pool, httpx.AsyncClient) and not pool.is_closed
            closes = _count_acloses(pool)
        await client.aclose()
        with pytest.raises(RuntimeError, match="AsyncClient is closed"):
            await client.chat.completions.create(messages=[USER])
        return pool, closes

    pool, closes = asyncio.run(_run())

    assert pool.is_closed and closes == [1]
    assert len(http.clients) == 1
    assert len(http.requests) == 3 + SURFACE_REQUESTS


def test_async_supplied_http_client_is_never_closed_and_timeouts_pass_through(http):
    async def _run():
        own = http.async_http_client(timeout=300.0)
        async with AsyncClient(http_client=own, timeout=40.0) as client:
            await _aevery_surface(client)
            await client.chat.completions.create(messages=[USER], timeout=4.0)
        await AsyncClient(http_client=own).aclose()
        still_open = not own.is_closed
        await own.aclose()
        return still_open

    assert asyncio.run(_run())
    assert http.clients == []
    assert http.timeouts == [40.0] * 4 + [305.0] + [40.0] * 3 + [4.0]


def test_async_streams_leave_the_pool_open_and_survive_closing_the_client(http):
    async def _run():
        client = AsyncClient()
        stream = await client.chat.completions.create(messages=[USER], stream=True)
        await stream.__anext__()
        await stream.close()
        events = client.stream(TASK)
        await events.__anext__()
        await events.aclose()
        pool_open_after_streams = not http.clients[0].is_closed

        open_stream = await client.chat.completions.create(messages=[USER], stream=True)
        await open_stream.__anext__()
        await client.aclose()
        await open_stream.close()
        return pool_open_after_streams

    assert asyncio.run(_run())
    assert len(http.clients) == 1 and http.clients[0].is_closed


# ---------------------------------------------------------------------------
# A real connection pool (httpcore over in-process socket pairs, no network)
# ---------------------------------------------------------------------------


class _SocketStream(httpcore.NetworkStream):
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            self._sock.settimeout(timeout)
            return self._sock.recv(max_bytes)
        except OSError as exc:
            raise httpcore.ReadError(exc) from exc

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        try:
            self._sock.settimeout(timeout)
            self._sock.sendall(buffer)
        except OSError as exc:
            raise httpcore.WriteError(exc) from exc

    def close(self) -> None:
        self._sock.close()

    def get_extra_info(self, info: str):
        return None


class _SocketServer(httpcore.NetworkBackend):
    """Each connection is a socket pair served by a thread: JSON completions, and streams that send their first event
    and then wait for ``release``."""

    def __init__(self) -> None:
        self.connects = 0
        self.release = threading.Event()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connects += 1
        ours, theirs = socket.socketpair()
        threading.Thread(target=self._serve, args=(theirs,), daemon=True).start()
        return _SocketStream(ours)

    def _serve(self, sock: socket.socket) -> None:
        with sock, sock.makefile("rb") as reader:
            try:
                while reader.readline():
                    length = 0
                    while (line := reader.readline()) not in (b"\r\n", b""):
                        name, _, value = line.decode().partition(":")
                        length = int(value) if name.lower() == "content-length" else length
                    if json.loads(reader.read(length)).get("stream"):
                        self._send_stream(sock)
                    else:
                        payload = json.dumps(completion()).encode()
                        head = f"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {len(payload)}"
                        sock.sendall(head.encode() + b"\r\n\r\n" + payload)
            except OSError:
                return

    def _send_stream(self, sock: socket.socket) -> None:
        def _chunk(data: bytes) -> bytes:
            return b"%x\r\n%s\r\n" % (len(data), data)

        sock.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        sock.sendall(_chunk(sse_body(FIRST_EVENT, done=False)))
        self.release.wait(timeout=5)
        sock.sendall(_chunk(sse_body(LAST_EVENTS)) + b"0\r\n\r\n")


@pytest.fixture
def socket_server(monkeypatch):
    server = _SocketServer()

    def _pool(timeout):
        transport = httpx.HTTPTransport()
        transport._pool = httpcore.ConnectionPool(network_backend=server, max_connections=1)
        return httpx.Client(transport=transport, timeout=timeout)

    monkeypatch.setattr(client_mod, "_http_client", _pool)
    yield server
    server.release.set()


def _socket_client() -> Client:
    return Client(base_url="http://pool.test/v1", timeout=2.0)  # a short pool timeout if a connection were stuck


def test_a_real_pool_reuses_its_connection_and_gets_it_back_from_streams(socket_server):
    client = _socket_client()
    for _ in range(3):
        assert client.chat.completions.create(messages=[USER]).text == "Hello"
    assert socket_server.connects == 1  # one kept-alive connection

    # The pool holds one connection: a stream that kept its connection would make the next request time out.
    with client.chat.completions.create(messages=[USER], stream=True) as unread:
        pass  # never read, and `unread` stays referenced: only closing its response can release the connection
    assert client.chat.completions.create(messages=[USER]).text == "Hello"
    assert list(unread) == []

    events = client.stream(TASK)
    next(events)
    events.close()
    assert client.chat.completions.create(messages=[USER]).text == "Hello"

    dropped = client.chat.completions.create(messages=[USER], stream=True)
    next(dropped)
    del dropped
    gc.collect()
    assert client.chat.completions.create(messages=[USER]).text == "Hello"

    assert socket_server.connects == 4  # a half-read HTTP/1.1 connection is discarded, not reused
    client.close()


def test_closing_a_client_ends_its_open_streams_with_a_truncation_error(socket_server):
    client = _socket_client()
    stream = client.chat.completions.create(messages=[USER], stream=True)
    next(stream)
    client.close()

    with pytest.raises(IncompleteStreamError) as excinfo:
        next(stream)
    assert excinfo.value.code == STREAM_TRUNCATED
    stream.close()

    client = _socket_client()
    events = client.stream(TASK)
    next(events)
    client.close()

    assert list(events)[-1]["code"] == STREAM_TRUNCATED
