# Perceptron SDK

[![Tests](https://github.com/perceptron-ai-inc/perceptron/actions/workflows/tests.yml/badge.svg)](https://github.com/perceptron-ai-inc/perceptron/actions/workflows/tests.yml) [![codecov](https://codecov.io/github/perceptron-ai-inc/perceptron/graph/badge.svg?token=HW6JASKQJR)](https://codecov.io/github/perceptron-ai-inc/perceptron)

<p align="center">
  <a href="https://www.perceptron.inc/" target="_blank" rel="noopener">
    <img src="./assets/banner-light.svg" alt="Perceptron" width="680" />
  </a>
</p>

**Perceptron is the Python SDK for building with perceptive-language models like Isaac 0.1.** Designed for physical AI applications—robotics, manufacturing, logistics, and security—it provides a unified interface for grounded perception: detection, localization, OCR, and visual Q&A with structured outputs ready for robotics, analytics, and edge deployment. Route tasks to specialized models, swap providers per call, and compose complex multimodal flows with a typed DSL. Efficient enough for edge deployment, flexible enough for any real-world task.

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

**Efficient frontier for real-world deployment**
Isaac 0.1 matches models 50x its size while delivering edge-ready latencies and drastically lower serving costs. Perception workloads are continuous and latency-sensitive—Perceptron is built for the efficient frontier where capability meets real-world constraints.

**Prompt for anything, control the output type**
Ask for whatever you need in natural language—"find safety violations", "locate damaged components", "identify obstacles"—and specify the output format: bounding boxes, points, polygons, or text. The flexibility of language models with the structure your application needs.

**Provider-agnostic architecture**
Swap between providers and models without rewriting code. Start with `fal`, add custom endpoints, or route different task types to different backends based on latency, cost, or accuracy requirements.

---

## Installation

```bash
uv pip install perceptron

# Optional: PyTorch helpers for tensor utilities
uv pip install "perceptron[torch]"

# Optional: Dev tools (ruff, pytest, pre-commit)
uv pip install "perceptron[dev]"
```

The CLI entry point `perceptron` is available after install. Works with regular `pip` too if you don't use [`uv`](https://github.com/astral-sh/uv).

## Quick Start

```python
from perceptron import detect, caption

# Detect objects with structured bounding boxes
result = detect(
    "warehouse.jpg",
    classes=["forklift", "person", "pallet"],
    model="perceptron"
)

for box in result.points or []:
    print(f"{box.mention}: ({box.top_left.x}, {box.top_left.y})")

# Generate image captions
desc = caption("scene.png", style="detailed")
print(desc.text)
```

No credentials? The SDK returns compile-only payloads when API keys are missing, letting you inspect requests before sending them.

## Configuration

**Environment variables:**
```bash
export PERCEPTRON_PROVIDER=fal        # or your custom provider
export PERCEPTRON_API_KEY=sk_live_... # or FAL_KEY for fal provider
```

**Programmatic:**
```python
from perceptron import configure

configure(provider="fal", api_key="sk_live_...")
```

**CLI:**
```bash
perceptron config --provider fal --api-key sk_live_...
```

---

## Core Features

### Detection with structured outputs
Get normalized bounding boxes (0-1000 coordinate space) ready for downstream tasks:

```python
from perceptron import detect

result = detect("factory_floor.jpg", classes=["defect", "warning"])

for box in result.points or []:
    print(f"{box.mention}: {box.top_left} → {box.bottom_right}")
```

### Image captioning
```python
from perceptron import caption

result = caption("product.png", style="concise")
print(result.text)  # "A blue widget on a white background"
```

### OCR with custom prompts
```python
from perceptron import ocr

result = ocr("schematic.png", prompt="Extract all component labels and their values")
print(result.text)
```

### Streaming responses
Stream incremental text and coordinate deltas for real-time applications:

```python
from perceptron import detect

for event in detect("frame.png", classes=["person"], stream=True):
    if event["type"] == "text.delta":
        print(event["chunk"], end="", flush=True)
    elif event["type"] == "points.delta":
        print(f"Detection: {event['points']}")
    elif event["type"] == "final":
        result = event["result"]
```

---

## CLI Usage

The CLI provides quick access to core features for batch processing and scripting:

```bash
# Caption single image or directory
perceptron caption image.jpg
perceptron caption ./images --style detailed

# OCR with custom prompt
perceptron ocr document.png --prompt "Extract table data"

# Detect objects (writes detections.json)
perceptron detect ./frames --classes forklift,person,pallet

# Visual Q&A with grounding
perceptron question scene.jpg "Where is the safety equipment?" --expects box
```

## Advanced Usage

### Few-shot detection with COCO datasets
Automatically build balanced in-context examples from annotated datasets:

```python
from perceptron import detect_from_coco

results = detect_from_coco(
    "/datasets/custom",
    split="train",
    shots=4,  # balanced examples per class
    classes=["defect", "ok"]
)

for sample in results:
    print(f"{sample.image_path.name}: {len(sample.result.points or [])} detections")
```

### Coordinate scaling
Outputs use normalized 0-1000 coordinates. Convert to pixels for rendering or metrics:

```python
from PIL import Image
from perceptron import detect

result = detect("frame.png", classes=["forklift"])
width, height = Image.open("frame.png").size

# Scale to pixel space
pixel_boxes = result.points_to_pixels(width, height)

for box in pixel_boxes or []:
    x1, y1 = box.top_left.x, box.top_left.y
    x2, y2 = box.bottom_right.x, box.bottom_right.y
    print(f"{box.mention}: [{x1}, {y1}, {x2}, {y2}]")
```

### Composing tasks with the DSL
For complex workflows, compose multimodal prompts with typed nodes and the `@perceive` decorator:

```python
from perceptron import perceive, image, text

@perceive(expects="box", stream=True)
def find_safety_equipment(image_path):
    return [
        image(image_path),
        text("Locate all safety equipment including helmets, vests, and signs")
    ]

# Use the decorated function
for event in find_safety_equipment("warehouse.jpg"):
    if event["type"] == "final":
        for box in event["result"]["points"]:
            print(f"{box['mention']}: {box['top_left']}")

# Inspect compiled payload without executing
payload = find_safety_equipment.inspect("warehouse.jpg")
print(payload)
```

Available DSL nodes: `image`, `text`, `system`, `point`, `box`, `polygon`, `collection`

## Troubleshooting

**Compile-only results (no text returned)**
Missing API credentials. Set `FAL_KEY` or `PERCEPTRON_API_KEY` environment variables, or call `configure(api_key="...")`.

**Stream buffer overflow warnings**
Long streaming responses exceeded buffer size. Increase via `configure(max_buffer_bytes=...)`.

**Empty detections in directory mode**
No supported image files found. Supported extensions: `.jpg`, `.png`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff`, `.heic`, `.heif`.

**Bounding box coordinate errors**
Validate that input annotations are consistent and images are properly attached to requests.

---

## Development

Clone the repo and install in editable mode with dev dependencies:

```bash
git clone https://github.com/perceptron-ai-inc/perceptron.git
cd perceptron
uv pip install -e ".[dev]"
pre-commit install
```

**Run tests and checks:**
```bash
pytest                          # Run tests with coverage
pre-commit run --all-files      # Run linters and formatters
```

**Repository structure:**
- `src/perceptron/` – SDK core, client, DSL, providers
- `tests/` – Test suite with coverage reporting
- `cookbook/` – Example notebooks and scripts
- `papers/` – Research publications
- `tools/` – Development utilities

Coverage reports are automatically published to Codecov via CI.

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
