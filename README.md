<p align="center">
  <a href="https://www.perceptron.inc/" target="_blank" rel="noopener">
    <img src="./assets/banner-light.svg" alt="Perceptron" width="680" />
  </a>
</p>

<div align="center">
  <h3>The platform for physical AI</h3>
</div>

<p align="center">
  <a href="https://github.com/perceptron-ai-inc/perceptron/actions/workflows/tests.yml"><img src="https://github.com/perceptron-ai-inc/perceptron/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/github/perceptron-ai-inc/perceptron"><img src="https://codecov.io/github/perceptron-ai-inc/perceptron/graph/badge.svg?token=HW6JASKQJR" alt="codecov"></a>
</p>

**Perceptron is the Python SDK for building with perceptive-language models, from our flagship Perceptron Mk1.5 to edge-ready models like Isaac 0.2 2B (Preview).** Designed for physical AI applications—robotics, manufacturing, logistics, and security—it provides a unified interface for grounded perception: detection, localization, tracking, OCR, and visual, video, and audio Q&A with structured outputs ready for robotics, analytics, and edge deployment. Compose multimodal prompts with a typed DSL, or use the OpenAI-compatible message API with tool calling.

<p align="center">
  <a href="https://www.perceptron.inc/" target="_blank"><strong>Website</strong></a> ·
  <a href="https://docs.perceptron.inc" target="_blank"><strong>Docs</strong></a> ·
  <a href="https://discord.gg/fgBeaACQzE" target="_blank"><strong>Community</strong></a>
</p>

---

## Why Perceptron?

**Grounded, spatial intelligence**
Get precise localization and grounded answers with conversational pointing—every claim is visually cited. Ask "what's broken in this machine?" and get highlighted regions with robust spatial reasoning that handles occlusions, relationships, and object interactions.

**In-context learning for perception**
Show a few annotated examples (defects, safety conditions, custom categories) in your prompt and the model adapts—no YOLO-style fine-tuning or custom detector stacks required. Learn novel tasks from a handful of examples.

**Flagship reasoning across image, video, and audio**
Perceptron Mk1.5, our flagship, brings grounded perception to long-form video—Q&A, temporal clipping, object tracking, and multimodal in-context learning—plus audio understanding and tool calling. Built for cloud workloads where capability outweighs footprint.

**Efficient frontier for edge deployment**
The Isaac family is built for the edge—Isaac 0.1 matches models 50x its size, with Isaac 0.2 2B (Preview) extending the lineup. Both deliver edge-ready latencies and drastically lower serving costs.

**Prompt for anything, control the output type**
Ask for whatever you need in natural language—"find safety violations", "locate damaged components", "identify obstacles"—and choose the output: bounding boxes, points, polygons, clips, tracks, or text.

---

## Installation

Requires Python 3.10+.

```bash
pip install perceptron            # the latest published release (0.3.5)
```

This README describes the SDK on `main`. The features marked **Mk1.5** below (the `perceptron-mk1.5` model, the message API and tool calling, Files, Models, Multilook, `asset_idx`, tracks, `video_frames`, audio input, `reasoning_effort`) and the new result fields (`finish_reason`, `usage`, `tool_calls`, `tracks`, `request_id`) ship in the next release after 0.3.5 (planned as 0.4.0). Until that release is on PyPI, install from GitHub:

```bash
pip install "git+https://github.com/perceptron-ai-inc/perceptron"
```

Optional extras: `pip install "perceptron[torch]"` (tensor utilities, requires PyTorch) and `pip install "perceptron[dev]"` (ruff, pytest, pre-commit). `uv pip install ...` works the same way. The install also provides the `perceptron` command-line tool.

## Configuration

```bash
export PERCEPTRON_API_KEY=sk_live_...
```

or in code:

```python
from perceptron import config, configure

configure(api_key="sk_live_...")

with config(max_tokens=512, timeout=300):
    ...  # temporary overrides inside the block
```

**Providers.** `perceptron` is the Perceptron API (`https://api.perceptron.inc/v1`); its default model is `perceptron-mk1.5`. `fal` serves `isaac-0.1` only.

> **Which provider is used.** One rule applies to every surface (the helpers, `perceive`, `Client.generate`/`stream`, the message API, Files, Models, Multilook, and the CLI): the provider you choose wins (`Client(provider=...)`, `configure(provider=...)`, `PERCEPTRON_PROVIDER`, a per-call `provider=`, or the CLI's `--provider`). Otherwise it is `fal` only when `FAL_KEY` is your only key (it is set, and neither `PERCEPTRON_API_KEY` nor a key set in code with `configure(api_key=...)` or `Client(api_key=...)` is), and the Perceptron API in every other case. So a key you set in code goes to the Perceptron API unless you choose `fal`.
>
> Provider `fal` authenticates with `FAL_KEY`, or with a key you set in code when you choose `fal` (`configure(provider="fal", api_key=...)`, `Client(provider="fal", api_key=...)`); a `PERCEPTRON_API_KEY` is never sent to fal. Files, Models, and Multilook exist only on the Perceptron API: on provider `fal` they raise `BadRequestError` with code `unsupported_provider_feature`. A `base_url` you set (`Client(base_url=...)`, `configure(base_url=...)`, or `PERCEPTRON_BASE_URL`) applies to every surface.
>
> **Upgrading from 0.3.x:** with no provider set, 0.3.x used fal and sent it `PERCEPTRON_API_KEY` (or the key from `configure(api_key=...)`). The Perceptron API is now used unless `FAL_KEY` is your only key. To keep using fal, choose it (`PERCEPTRON_PROVIDER=fal` or `configure(provider="fal")`) and put its key in `FAL_KEY` or `configure(api_key=...)`.

| Setting | Environment variable | Notes |
| --- | --- | --- |
| `provider` | `PERCEPTRON_PROVIDER` | `perceptron` or `fal` (see above) |
| `api_key` | `PERCEPTRON_API_KEY` (provider `fal`: `FAL_KEY`) | a key set in code is used by whichever provider is selected, and selects the Perceptron API when you choose no provider |
| `model` | `PERCEPTRON_MODEL` | default `perceptron-mk1.5` on `perceptron`; also `perceptron-mk1`, `isaac-0.3-fast`, `isaac-0.2-2b-preview`, `isaac-0.2-1b`, `isaac-0.1` |
| `base_url` | `PERCEPTRON_BASE_URL` | replaces the provider's URL on every surface; include `/v1` for provider `perceptron` |
| `timeout` | | seconds per request, default 125 (Multilook waits at least 305) |
| `retries` | | accepted, but the SDK does not retry requests |

A value set with `configure()` or `config()` wins over its environment variable, and `Client(...)` keyword arguments win over both for that client. A per-call `model=` wins over all of them, and so does a per-call `provider=` on the helpers, `perceive`, and `Client.generate`/`stream`. `perceptron-mk1.5-preview` was renamed to `perceptron-mk1.5` (the old id raises `BadRequestError` with code `model_renamed`).

Without an API key for the selected provider, requests raise `AuthError` with code `credentials_missing` before anything is sent; use `inspect_task` (see [Composing tasks](#composing-tasks-with-the-dsl)) to look at a compiled prompt offline. `perceptron config` prints the `export` lines for your shell (it does not save anything).

**Clients and connections.** A `Client` sends all its requests (`generate`/`stream`, the message API, Files, Models, and Multilook) through one connection pool, opened on first use, that keeps HTTP/1.1 connections alive between requests. Reuse one client, and close it when you are done with `client.close()` or a `with` block (`AsyncClient`: `await client.aclose()` or `async with`). An `AsyncClient`'s pool belongs to the event loop it runs on: after that loop ends (a second `asyncio.run`, say), the next request opens a new pool, and using one `AsyncClient` from event loops running at the same time in different threads raises `RuntimeError`, so give each loop its own. The helpers and `perceive` open a client for each call and close it afterwards. To set up HTTP yourself (proxies, connection limits, a custom transport), pass your own `httpx.Client` (`httpx.AsyncClient` for `AsyncClient`) as `http_client=`: the SDK uses it as is and never closes it, and each request still uses the SDK's `timeout`. HTTP/2 (`httpx.Client(http2=True)`) works too, with one caveat: a stream or download you close before its end is not cancelled, so the server keeps sending it, and enough abandoned data stalls later requests on that connection.

```python
import httpx

from perceptron import Client

with Client() as client:  # one connection pool for both requests, closed at the end of the block
    print(client.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}]).text)
    print([model.id for model in client.models.list()])

own = httpx.Client(limits=httpx.Limits(max_connections=20))
client = Client(http_client=own)  # closing `client` leaves `own` open; close it yourself
```

## Quick start

```python
from perceptron import caption, detect, image

result = detect(image("warehouse.jpg"), classes=["forklift", "person", "pallet"])
for box in result.boxes or []:
    print(f"{box.mention}: {box.top_left} → {box.bottom_right}")

description = caption(image("warehouse.jpg"), style="detailed", expects="text")
print(description.text)
```

Helpers take media nodes: wrap paths, URLs, and bytes in `image()`, `video()`, or `audio()`. Coordinates are integers on a normalized 0–1000 grid. Structured results land in the bucket for what you asked for: `result.boxes` for boxes (including after `detect`), `result.points`, `result.polygons`, or `result.clips`.

---

## Core features

### Detection, captioning, OCR, and visual Q&A

```python
from perceptron import caption, detect, image, ocr, question

boxes = detect(image("warehouse.jpg"), classes=["defect", "warning"]).boxes
text = ocr(image("schematic.png"), prompt="Extract all component labels and their values").text
summary = caption(image("product.png"), style="concise", expects="text").text

answer = question(image("scene.jpg"), "Where is the safety equipment?", expects="box")
print(answer.text, answer.boxes, answer.finish_reason, answer.usage)
```

- `caption(media, *, style="concise", expects=None, stream=False, **kwargs)`: describe images, video, or audio. `expects` defaults to `"box"` for images and `"text"` for video and audio.
- `question(media, question_text, *, expects="text", stream=False, **kwargs)`: answer questions, optionally grounded (`point`, `box`, `polygon`, `clip`).
- `detect(media, *, classes=None, examples=None, stream=False, **kwargs)`: grounded detection on images or video.
- `ocr(image, *, prompt=None, stream=False, **kwargs)`, `ocr_markdown(...)`, `ocr_html(...)`: text extraction.
- `detect_from_coco(dataset_dir, *, split=None, classes=None, shots=0, limit=None, **kwargs)`: few-shot prompts from COCO datasets.
- `perceive(nodes, *, expects=None, stream=False, **kwargs)` / `@perceive`: compose any multimodal prompt with the DSL.

The keyword arguments include `model`, `provider`, `reasoning`, `reasoning_effort`, `enable_audio_in_video`, `temperature`, `max_tokens`, `top_p`, `top_k`, `response_format`, `tools`, `tool_choice`, and `stream_options`. Unknown keywords raise `TypeError`. The retired Focus controls (`focus=`, `visual_reasoning=`) raise `TypeError` too.

### Audio and video soundtracks (Mk1.5)

`audio()` takes WAV, MP3, or FLAC (paths and bytes are encoded into the request; HTTP(S) URLs are passed through). A video's soundtrack is ignored unless you pass `enable_audio_in_video=True`.

```python
from perceptron import audio, question, video

transcript = question(audio("call.wav"), "Transcribe the audio verbatim.")
print(transcript.text)

answer = question(audio("voicemail.mp3"), "What does the caller want, and by when?", reasoning_effort="low")
print(answer.text)

scene = question(video("clip.mp4"), "What is said while the door opens?", enable_audio_in_video=True)
print(scene.text)

usage = scene.usage or {}  # the server's usage object, as sent
details = usage.get("prompt_tokens_details") or {}
print(usage.get("prompt_tokens"), usage.get("completion_tokens"), details.get("audio_tokens"), details.get("cached_tokens"))
```

`enable_audio_in_video=False` is sent as an explicit `false`. Fields the server did not report are absent (never filled with 0).

### Reasoning (Mk1.5)

`reasoning_effort` picks how much the model reasons before answering: `none`, `minimal`, `low`, `medium`, or `high` (case-insensitive; anything else raises `BadRequestError`). The reasoning text is in `result.reasoning`.

```python
from perceptron import image, question

result = question(image("scene.jpg"), "How many people are behind the counter?", reasoning_effort="high")
print(result.reasoning)
print(result.text)
```

`reasoning=True` (and `expects="think"`) is separate: it adds the `<hint>THINK</hint>` instruction to the prompt, and `reasoning_effort` is sent as its own request field. Both are sent when you pass both, and the server treats the THINK hint as "reasoning on", so `reasoning=True, reasoning_effort="none"` still reasons. Use `reasoning_effort` alone to pick a tier.

### Streaming

`stream=True` returns an iterator of events: `text.delta` (`chunk`), `reasoning.delta` (`chunk`), `points.delta` (`points` plus `context` with the inherited `mention`, `t`, `asset_idx`, and `container`), `tool_call.delta` (`index`, `id`, `name`, `arguments` fragment), and exactly one terminal event: `final` (with `result`, a dict) or `error`.

```python
from perceptron import detect, image

for event in detect(image("frame.png"), classes=["person"], stream=True):
    if event["type"] == "text.delta":
        print(event["chunk"], end="", flush=True)
    elif event["type"] == "points.delta":
        for box in event["points"]:
            print("\nbox:", box.top_left, box.bottom_right, box.mention)
    elif event["type"] == "final":
        result = event["result"]
        print("\n", result["finish_reason"], len(result["boxes"] or []), "boxes", result["usage"])
    elif event["type"] == "error":
        print("\nfailed:", event["code"], event["message"], event["details"])
```

`event["result"]` of `final` holds `text`, `reasoning`, the bucket for `expects` (`boxes` for `detect`) as annotation objects, `tracks`, `parsed`, `tool_calls`, `finish_reason`, `complete`, `usage`, `request_id`, and `errors`. An `error` event (HTTP error, server error event, truncated stream) carries `code`, `message`, `status`, `request_id`, `details`, and `partial` (text, reasoning, and tool calls received so far); no `final` follows it. Streams to the Perceptron API request usage (`stream_options={"include_usage": True}`) unless you pass `stream_options`.

---

## Message API and tool calling (Mk1.5)

`Client().chat.completions.create(...)` mirrors the OpenAI chat-completions API. Messages are sent as you write them. Inside a message's `content` list, strings and DSL nodes (`image(...)`, `video(...)`, `audio(...)`, `video_frames(...)`, `text(...)`) become content parts.

```python
from perceptron import Client, image

client = Client()  # the Perceptron API (see Configuration)
completion = client.chat.completions.create(
    model="perceptron-mk1.5",
    messages=[{"role": "user", "content": [image("warehouse.jpg"), "Count the pallets."]}],
    reasoning_effort="low",
)
print(completion.text, completion.finish_reason, completion.complete)
print(completion.usage.prompt_tokens, completion.usage.audio_tokens, completion.usage.cached_tokens)
print(completion.annotations().boxes)  # annotation markup in the answer, parsed
```

A complete tool-calling loop. Append the returned message as-is (it keeps `tool_calls` and `reasoning_content`), then one `tool` message per call:

```python
import json

from perceptron import Client, function_tool

client = Client()
tools = [
    function_tool(
        "get_weather",
        description="Current weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
]


def get_weather(city: str) -> dict:
    return {"city": city, "temperature_c": 18, "conditions": "cloudy"}


messages = [{"role": "user", "content": "Should I bring an umbrella in Paris today?"}]
for _ in range(5):  # bound the number of tool rounds
    completion = client.chat.completions.create(model="perceptron-mk1.5", messages=messages, tools=tools)
    messages.append(completion.message)
    if completion.finish_reason != "tool_calls":
        break
    for call in completion.tool_calls:
        output = get_weather(**call.parse_arguments())
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)})
print(completion.text)
```

`tool_choice` is `"auto"` or `"none"`; tools cannot be combined with `regex` or a `json_schema` response format. Invalid combinations raise `BadRequestError` before any request.

Streaming returns chunks; `get_final_completion()` returns the assembled `ChatCompletion` (text, reasoning, tool calls, usage):

```python
from perceptron import Client

client = Client()
with client.chat.completions.create(
    model="perceptron-mk1.5",
    messages=[{"role": "user", "content": "Write a haiku about forklifts."}],
    stream=True,
) as stream:
    for chunk in stream:
        for choice in chunk.choices:
            print(choice.delta.content or "", end="", flush=True)
    completion = stream.get_final_completion()
print("\n", completion.finish_reason, completion.usage)
```

A stream that fails mid-way raises the mapped error (with `.partial`); one that ends without `[DONE]` raises `IncompleteStreamError`. The connection opens when `create()` returns, so use `with`/`async with` (or `close()`) when you may stop early; closing a stream before its end closes its connection, so the server stops generating (a finished stream hands its connection back to the client's pool). A stream dropped unfinished is closed when it is garbage collected (an async one on its event loop, unless that loop has been closed). `AsyncClient` mirrors everything:

```python
import asyncio

from perceptron import AsyncClient


async def main():
    async with AsyncClient() as client:
        completion = await client.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}])
        print(completion.text)

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello again!"}], stream=True
        )
        async with stream:
            final = await stream.get_final_completion()
        print(final.text)


asyncio.run(main())
```

The DSL does tool calling too: `perceive(..., tools=[...])` returns `tool_calls`; replay the turn with `result.as_agent()` and answer each call with `tool_result(call_id, ...)`, which takes text, dicts or lists (sent as JSON), or images:

```python
from perceptron import function_tool, image, perceive, text, tool_result

zoom = function_tool(
    "zoom",
    description="Return a close-up of a region given in 0-1000 coordinates.",
    parameters={
        "type": "object",
        "properties": {name: {"type": "integer"} for name in ("x1", "y1", "x2", "y2")},
        "required": ["x1", "y1", "x2", "y2"],
    },
)
prompt = [image("shelf.jpg"), text("What does the price tag say?")]
result = perceive(prompt, tools=[zoom])
if result.finish_reason == "tool_calls":
    followup = [*prompt, result.as_agent()]
    for call in result.tool_calls:
        followup.append(tool_result(call.id, image("price_tag_crop.png")))  # your tool's output
    result = perceive(followup, tools=[zoom])
print(result.text)
```

## Files, Models, and Multilook (Mk1.5)

These exist only on the Perceptron API: on provider `fal` (chosen, or selected because `FAL_KEY` is your only key) they raise `BadRequestError` with code `unsupported_provider_feature`.

```python
from perceptron import Client, image

client = Client()

uploaded = client.files.upload("warehouse.jpg")  # path, bytes, or file object; up to 128 MiB
completion = client.chat.completions.create(
    messages=[{"role": "user", "content": [image(uploaded), "Count the pallets."]}],  # or image(file_id=uploaded.id)
)
print(completion.text)
for stored in client.files.iter():  # every page
    print(stored.id, stored.filename, stored.bytes)
client.files.delete(uploaded.id)

for model in client.models.list():
    print(model.id)
info = client.models.retrieve("perceptron-mk1.5", extended=True)
print(info.supports("tool_calling"), info.max_output_tokens)

# Multilook: several prompts against one shared context, prefilled once.
response = client.chat.completions.multilook(
    model="perceptron-mk1.5",
    context=[{"role": "user", "content": [image("warehouse.jpg")]}],
    prompts=["How many forklifts are there?", "Is anyone missing a hard hat?"],
)
for result in response.results:
    if result.ok:
        print(result.prompt_index, result.completions[0].text)
    else:
        print(result.prompt_index, "failed:", result.error.message)
print(response.usage.cached_tokens)
```

Files also offer `list`, `retrieve`, `content`, and `download`. The Models API is rate-limited (30 requests per minute), so the SDK never calls it on its own.

## Multiple assets, `asset_idx`, and tracks (Mk1.5)

Every image, video, audio clip, or `video_frames` video in a request is one asset, numbered from 0 in order. Annotations name their asset with `asset_idx`:

- A container's `asset_idx` applies to what it holds: `<collection asset_idx="1">` covers its boxes and tracks, and the flattened buckets (`result.boxes`, ...) carry the inherited value.
- An annotation without `asset_idx` refers to the **last** asset. `result.resolve_asset_idx(annotation)` (or `resolve_asset_idx(annotation, n_assets)`) applies that rule; parsed objects keep `asset_idx=None`, because indices are relative to one request.
- In prompts, anchor tags to an asset with `image=`/`asset=` (or pass `asset_idx=`). The DSL writes `asset_idx` only when the prompt has more than one asset, so single-asset prompts are unchanged.

```python
from perceptron import box, image, perceive, text

before, after = image("shelf_monday.jpg"), image("shelf_friday.jpg")
result = perceive(
    [
        before,
        box(120, 80, 360, 420, image=before, mention="cereal"),
        after,
        text("Box every product that was added on Friday."),
    ],
    expects="box",
)
for found in result.boxes or []:
    print(found.mention, found.asset_idx, result.resolve_asset_idx(found))
```

Video tracking returns `<track>` elements. When `expects` is `point`, `box`, `polygon`, or `clip`, `result.tracks` holds the `Track` objects (`mention`, `asset_idx`, and time-stamped waypoints in `points`), and waypoints of the requested shape also appear in its flat bucket. `collect_annotations` parses the same markup from any text, such as `result.text` of an answer without `expects`:

```python
from perceptron import collect_annotations, resolve_asset_idx

markup = (
    '<collection mention="cars" asset_idx="0"><point_box> (10,10) (50,60) </point_box></collection> '
    '<track mention="ball"><point t="0.5 seconds"> (100,200) </point><point t="1.0 seconds"> (120,210) </point></track>'
)
found = collect_annotations(markup)
print(found.boxes[0].asset_idx)  # 0, from the collection
ball = found.tracks[0]
print(resolve_asset_idx(ball, n_assets=2))  # 1: no asset_idx means the last asset
for waypoint in ball.points:
    print(waypoint.t, waypoint.x, waypoint.y)
```

`parse_annotations` (lenient, with `errors`), `parse_text` (strict), `extract_points`/`extract_clips`/`extract_tracks`, `scan_leaves` (partial streaming text), and `scale_annotations_by_asset` (pixels per asset) are also exported.

## Media inputs: video frames, file ids, and data URLs (Mk1.5)

```python
import base64
from pathlib import Path

from perceptron import image, perceive, text, video_frames

# A video given as timestamped frames (2-256 frames, integer milliseconds that never decrease); one asset.
frames = video_frames([("frame_000.jpg", 0), ("frame_001.jpg", 500), ("frame_002.jpg", 1000)])
result = perceive([frames, text("When does the door open?")], expects="clip")
for moment in result.clips or []:
    print(moment.timestamp.at, moment.timestamp.until, moment.mention)

# data: URLs are passed through; uploaded files are referenced by id.
data_url = "data:image/png;base64," + base64.b64encode(Path("frame.png").read_bytes()).decode()
result = perceive([image(data_url), image(file_id="file-0123456789abcdefghijkl"), text("Compare the two images.")])
print(result.text)
```

`video(...)` and `audio(...)` accept the same forms (`file_id=`, a `File`, a matching `data:video/...` or `data:audio/...` URL).

---

## Results, usage, and errors

`PerceiveResult` (from the helpers and `perceive`) has `text`, `reasoning`, `points`/`boxes`/`polygons`/`clips`, `tracks`, `parsed`, `errors` (malformed markup, reported instead of raised; pass `strict=True` to raise), `finish_reason`, `complete`, `tool_calls`, `usage` (the server's usage dict), `request_id`, `asset_count`, and `raw`.

```python
from perceptron import QuotaExceededError, RateLimitError, SDKError, image, question

try:
    result = question(image("scene.jpg"), "What is shown?")
    print(result.finish_reason, result.complete, result.request_id)
except QuotaExceededError as err:  # a RateLimitError that retrying does not fix
    print("quota exhausted:", err.code)
except RateLimitError as err:
    print("rate limited; retry after", err.retry_after, "seconds")
except SDKError as err:
    print(err.code, err.request_id, err.details)
```

Errors derive from `SDKError` (`.code`, `.details`, `.request_id`, `.status_code`): `AuthError` (and `PermissionDeniedError` for 403), `BadRequestError` (and `NotFoundError`), `RateLimitError` (and `QuotaExceededError`, which is not retryable), `ServerError`, `TimeoutError`, `TransportError` (and `IncompleteStreamError`), `ParseError`, `ExpectationError`, and `AnchorError`.

---

## CLI

```bash
perceptron config --api-key sk_live_...   # prints export lines; nothing is saved

# Caption a single image or a directory (directories write captions.json)
perceptron caption image.jpg
perceptron caption ./images --style detailed

# OCR with a custom prompt
perceptron ocr document.png --prompt "Extract table data"

# Detect objects in an image, a video, or a directory (writes detections.json)
perceptron detect ./frames --classes forklift,person,pallet
perceptron detect clip.mp4 --classes person --format json

# Visual Q&A with grounding, streaming, a model, and a reasoning tier
perceptron question scene.jpg "Where is the safety equipment?" --expects box --stream
perceptron question scene.jpg "How many people are there?" --model perceptron-mk1.5 --reasoning-effort high

# Audio Q&A, and video Q&A that also hears the soundtrack
perceptron question call.wav "Summarize the conversation."
perceptron question clip.mp4 "What is said as the door opens?" --audio-in-video
```

`caption`, `question`, `detect`, and `ocr` take `--model`, `--provider`, `--reasoning-effort`, and `--format text|json`. `caption`, `question`, and `detect` also take `--audio-in-video/--no-audio-in-video`: unset leaves the server default, and `--no-audio-in-video` sends an explicit `false`. JSON output includes the annotations, `tracks`, `parsed`, `finish_reason`, `usage`, and `request_id`. Errors print their code and request id and exit with status 1. Media arguments are paths, http(s) URLs, or `data:` URLs. Directory mode reads `.jpg`, `.jpeg`, `.png`, and `.webp` files, disables streaming, and logs per-file validation issues.

## Advanced usage

### Few-shot detection with COCO datasets

```python
from perceptron import detect_from_coco

results = detect_from_coco(
    "datasets/custom",
    split="train",
    shots=4,  # balanced examples per class
    classes=["defect", "ok"],
)
for sample in results:
    print(f"{sample.image_path.name}: {len(sample.result.boxes or [])} detections")
```

### Coordinate scaling

Outputs use normalized 0–1000 coordinates. Convert them to pixels for rendering or metrics:

```python
from PIL import Image

from perceptron import detect, image, scale_points_to_pixels

result = detect(image("frame.png"), classes=["forklift"])
width, height = Image.open("frame.png").size

pixel_boxes = scale_points_to_pixels(result.boxes, width=width, height=height)
pixel_boxes = result.boxes_to_pixels(width, height)  # the same, as a method

for box in pixel_boxes or []:
    print(f"{box.mention}: [{box.top_left.x}, {box.top_left.y}, {box.bottom_right.x}, {box.bottom_right.y}]")
```

### Composing tasks with the DSL

```python
from perceptron import image, inspect_task, perceive, text


@perceive(expects="box")
def find_safety_equipment(image_path):
    return image(image_path) + text("Locate all safety equipment including helmets, vests, and signs.")


result = find_safety_equipment("warehouse.jpg")
for box in result.boxes or []:
    print(box.mention, box.top_left, box.bottom_right)

task, issues = inspect_task(find_safety_equipment, "warehouse.jpg")  # the compiled prompt; nothing is sent
print(task["content"][-1], issues)
```

Prompt nodes: `text`, `system`, `agent`, `image`, `video`, `audio`, `video_frames`, `tool_result`, `point`, `box`, `polygon`, and `block`, composed with `+` or lists. `pt`, `bbox`, `poly`, `collection`, `clip`, and `track` build annotation objects (for `annotate_image` and `detect(examples=...)`).

## Troubleshooting

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| Requests go to `fal.run`, or `Model 'perceptron-mk1.5' is not supported for provider='fal'` | Provider `fal` is selected: you chose it, or `FAL_KEY` is your only key (no `PERCEPTRON_API_KEY` and no key set in code) | Export `PERCEPTRON_API_KEY`, or choose the Perceptron API with `configure(provider="perceptron")` or `export PERCEPTRON_PROVIDER=perceptron`. |
| `AuthError` with code `credentials_missing` (`No API key for provider ...`) | No API key for the selected provider (`fal` never uses `PERCEPTRON_API_KEY`) | Export `PERCEPTRON_API_KEY` (Perceptron API) or `FAL_KEY` (fal), or call `configure(api_key=...)`. |
| `BadRequestError` with code `unsupported_provider_feature` | Files, Models, Multilook, or an uploaded file's id on provider `fal` | `configure(provider="perceptron")` or `export PERCEPTRON_PROVIDER=perceptron`. |
| `BadRequestError` with code `model_renamed` | `perceptron-mk1.5-preview` was renamed | Use `perceptron-mk1.5`. |
| `TypeError` (`Unknown node type: <class 'str'>` or `... expected Image, Video, VideoFrames, or Audio, got str`) | A helper got a bare path or URL | Wrap it: `image("x.jpg")`, `video(...)`, `audio(...)`. |
| `result.points` is `None` after `detect` | Boxes are in the `boxes` bucket | Read `result.boxes`. |
| `stream_buffer_overflow` warning | Streaming responses exceeded the buffer | Raise `max_buffer_bytes` via `configure(...)` or disable streaming. |
| `No image files (.jpeg, .jpg, .png, .webp) found` in directory mode, or `invalid_image_format` | The API accepts PNG, JPEG, and WebP images only | Convert other images (GIF, BMP, TIFF, HEIC) to PNG or JPEG. |

---

## Development

```bash
git clone https://github.com/perceptron-ai-inc/perceptron.git
cd perceptron
uv pip install -e ".[dev]"
pre-commit install

pytest                          # tests with coverage, README snippets included; offline unless PERCEPTRON_API_KEY is set
pytest -m "not integration"     # always offline (skips the live integration tests)
pre-commit run --all-files      # ruff format + lint
```

CI also builds the wheel, installs it into a fresh virtualenv, and reruns the offline tests against the installed package with `PERCEPTRON_TEST_INSTALLED=1` (which keeps `src/` off `sys.path`).

**Repository structure:**
- `src/perceptron/` – SDK core, client, DSL, providers
- `tests/` – Test suite with coverage reporting
- `cookbook/` – Example notebooks and scripts
- `papers/` – Research publications
- `tools/` – Development utilities

---

## Documentation & Support

- **Full Documentation**: [docs.perceptron.inc](https://docs.perceptron.inc)
- **Research Paper**: [papers/isaac_01.pdf](papers/isaac_01.pdf)
- **Technical Support**: [support@perceptron.inc](mailto:support@perceptron.inc)
- **Commercial Licensing**: [sales@perceptron.inc](mailto:sales@perceptron.inc)
- **Careers**: [join-us@perceptron.inc](mailto:join-us@perceptron.inc)

---

## License

Model weights are released under the Creative Commons Attribution-NonCommercial 4.0 International License. For commercial licensing, contact [sales@perceptron.inc](mailto:sales@perceptron.inc).
