# Isaac 0.1 SDK

Isaac 0.1 is Perceptron's first perceptive-language model: a 2B-parameter system built to understand and act in the physical world. Despite being over 50x smaller than incumbent perception stacks, it matches or outperforms far larger models across grounded reasoning, in-context learning, and fine-grained visual analysis. This repository ships the Python SDK and CLI you need to bring Isaac into production workflows today.

---

## Why Isaac
- **Efficient frontier performance** - attain state-of-the-art perception quality with edge-ready latency, power, and cost.
- **Grounded spatial intelligence** - answer questions with highlighted evidence, resilient to occlusion and clutter.
- **Few-shot adaptability** - teach new categories or defects inline via prompt examples; no custom detector retraining.
- **Fine-grained reading** - robust OCR over dense layouts and small text.
- **Conversational pointing** - every claim stays synchronized with cited visual regions to reduce hallucinations.

> Explore the interactive demo, open weights, and model card on Hugging Face; hosted inference is available via the fal.ai partner endpoint.

---

## Feature Highlights
- **Visual QA, simply trained** - reproducible recipe delivering strong diagram and real-world benchmark scores.
- **Grounded localization** - precise pointing outputs for automation, inspection, and human-in-the-loop review.
- **In-context perception** - prompt the model with a few annotated examples to adapt on the fly.
- **OCR & micro-detail** - dynamic image handling keeps tiny features legible across resolutions.
- **Auditable reasoning** - conversational pointing couples language and vision turn-by-turn.

---

## Quick Performance Snapshot
- **Grounding & localization:** high-precision pointing AP under occlusion and clutter.
- **Visual question answering:** leads the 2B parameter class and competes with much larger systems.
- **In-context object learning:** matches / exceeds fine-tuned YOLO-style detectors without task-specific retraining.

Full benchmark tables, prompts, and methodology will be published in the forthcoming technical report.

---

## Get Started Now
| Resource | Link |
| --- | --- |
| Interactive demo | _Upload images and test Isaac in your browser_ |
| Open weights | Isaac 0.1 (2B) checkpoint on Hugging Face |
| Model card & docs | Hugging Face hub documentation |
| Hosted inference | fal.ai partner endpoint (configure via `PERCEPTRON_PROVIDER=fal`) |
| Python SDK | This repository (`.`) |
| Technical report | Coming soon |

---

## Repository Layout
- `src/perceptron` – core SDK modules and tensorstream extras
- `examples` – runnable usage samples
- `tests` – pytest suite covering the high-level APIs and DSL

---

## Installation
Isaac's SDK is distributed as a single Python package with all dependencies included.

```bash
# Local development from this repo
uv pip install --editable .

# Or install the wheel/source archive directly
uv pip install .
```

The CLI entry point `perceptron` installs automatically.

TensorStream helpers depend on PyTorch; install the optional extra with `pip install .[torch]` (or `perceptron[torch]` after packaging).

---

## Configuration
The SDK is transport-agnostic; the fal.ai endpoint is bundled by default. Additional providers can be registered by extending `perceptron.client._PROVIDER_CONFIG`.

| Variable | Purpose | Example |
| --- | --- | --- |
| `PERCEPTRON_PROVIDER` | Provider identifier (`fal` by default) | `fal` |
| `PERCEPTRON_API_KEY` | API key supplied to the provider | `sk_live_...` |
| `PERCEPTRON_BASE_URL` | Optional override for the provider base URL | `https://fal.run` |
| `FAL_KEY` | Alternative env var consumed when provider=`fal` | `fal_sk_...` |

Programmatic overrides:
```python
from perceptron import configure, config

configure(provider="fal", api_key="sk_live_...", base_url="https://api.example/v1")

with config(max_tokens=512):
    ...  # run caption/detect helpers with temporary overrides
```

CLI helper:
```bash
perceptron config --provider fal --api-key sk_live_...
```

If no provider credentials are found, all helpers return compile-only payloads so you can inspect tasks before executing them remotely.

---

## Python Quickstart
```python
from perceptron import caption, detect

# Captioning
a = caption("/path/to/image.png", style="concise")
print(a.text)

# Streaming grounded detections
for event in detect("local.png", classes=["person", "forklift"], stream=True):
    if event["type"] == "text.delta":
        print("chunk", event["chunk"])
    elif event["type"] == "points.delta":
        print("bbox", event["points"])
    elif event["type"] == "final":
        print("final", event["result"]["points"])
```

### Few-shot detection from COCO datasets
```python
from perceptron import detect_from_coco

runs = detect_from_coco(
    "/datasets/isaac-demo",
    split="train",
    shots=4,                # build balanced ICL examples automatically
    classes=["defect", "ok"],
    max_tokens=256,
)

for sample in runs:
    print(sample.image_path.name)
    for box in sample.result.points or []:
        print(" -", box.mention, box)
```

---

## CLI Usage
Isaac's CLI mirrors the high-level helpers and supports directory batching (JSON summaries emitted alongside input folders).

```bash
# Generate captions
perceptron caption image.jpg
perceptron caption ./line-inspection --style detailed

# OCR with a custom prompt
perceptron ocr schematic.png --prompt "Extract component labels"

# Batched detection (writes detections.json)
perceptron detect ./factory-frames --classes defect,warning

# Chat-style prompt wrapping
perceptron chat "Summarize anomalies" --system "You are a grounded vision assistant"
```

Directory mode disables streaming but logs raw responses and aggregates validation issues per file.

---

## Testing & Quality Gates
- Unit tests: `pytest tests`
- Linting: `ruff check src tests`
- Static typing: `ty src`

---

## High-Level APIs
- `caption(image, *, style="concise", stream=False, **kwargs)`
- `ocr(image, *, prompt=None, stream=False, **kwargs)`
- `detect(image, *, classes=None, examples=None, stream=False, **kwargs)`
- `detect_from_coco(dataset_dir, *, split=None, classes=None, shots=0, limit=None, **kwargs)`

`detect_from_coco` automatically discovers annotations, constructs balanced in-context examples when `shots > 0`, and returns a list of `CocoDetectResult` objects containing both original COCO metadata and inference results.

For advanced workflows, compose tasks directly with the typed DSL (`text`, `system`, `image`, `point`, `box`, `polygon`, `collection`) and decorate with `@perceive` / `@async_perceive`. Use `inspect_task(fn, *args)` to view the compiled payload without executing it.

---

## Troubleshooting
| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| Compile-only result (no text) | Missing provider credentials | Export `FAL_KEY` / `PERCEPTRON_API_KEY` or call `configure(...)` |
| `stream_buffer_overflow` warning | Long streaming responses exceeded buffer | Increase `max_buffer_bytes` via `configure` |
| Empty JSON output in directory mode | No supported image extensions | Ensure files end with `.jpg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`, `.heic`, or `.heif` |
| Bounding-box bounds errors | Inconsistent coordinates or missing `image=` anchors | Validate input annotations and ensure images are attached |

---

## Contacts & Support
- Technical inquiries: [support@perceptron.inc](mailto:support@perceptron.inc)
- Commercial engagements: [sales@perceptron.inc](mailto:sales@perceptron.inc)
- Careers: [join-us@perceptron.inc](mailto:join-us@perceptron.inc)

Isaac 0.1 is just the beginning. We're partnering with enterprises in manufacturing, logistics, and security - and we're already building the next generation of models to push the frontier of physical AI. Let us know what you ship with it.
