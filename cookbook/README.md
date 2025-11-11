# Perceptron Cookbook

Hands-on quickstarts, capability recipes, and end-to-end tutorials for building with the Perceptron SDK. Every notebook pairs with a `.py` script (gitignored by default) so you can run code locally or in Colab with minimal changes.

---

## Quickstarts

| Notebook | What it covers | Colab |
| --- | --- | --- |
| [`quickstart/quickstart_isaac/quickstart_isaac.ipynb`](quickstart/quickstart_isaac/quickstart_isaac.ipynb) | Run the Isaac 0.1 model to localize shipping boxes in a factory scene. | [Open in Colab](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/quickstart/quickstart_isaac/quickstart_isaac.ipynb) |
| [`quickstart/quickstart_qwen/quickstart_qwen.ipynb`](quickstart/quickstart_qwen/quickstart_qwen.ipynb) | Use Qwen 3 VL to detect flowers with a single helper. | [Open in Colab](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/quickstart/quickstart_qwen/quickstart_qwen.ipynb) |

Each quickstart demonstrates:

1. Exporting `PERCEPTRON_API_KEY`.
2. Installing the SDK (`uv pip install --upgrade perceptron pillow`).
3. Configuring the client via `os.environ.get("PERCEPTRON_API_KEY", "<placeholder>")`.
4. Wrapping the call in a reusable `@perceive` helper.

---

## Capability Recipes

| Folder | Scenario | Colab |
| --- | --- | --- |
| [`recipes/capabilities/captioning/captioning.ipynb`](recipes/capabilities/captioning/captioning.ipynb) | Generate concise or grounded captions (with bounding boxes). | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/captioning/captioning.ipynb) |
| [`recipes/capabilities/ocr/ocr.ipynb`](recipes/capabilities/ocr/ocr.ipynb) | Run OCR with custom prompts and parse the output. | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/ocr/ocr.ipynb) |
| [`recipes/capabilities/object-detection/object-detection.ipynb`](recipes/capabilities/object-detection/object-detection.ipynb) | Detect PPE with a `@perceive` helper or the high-level `detect()` API. | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/object-detection/object-detection.ipynb) |
| [`recipes/capabilities/visual-qa/visual-qa.ipynb`](recipes/capabilities/visual-qa/visual-qa.ipynb) | Ask grounded questions and cite regions with bounding boxes. | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/visual-qa/visual-qa.ipynb) |
| [`recipes/capabilities/in-context-learning/in-context-learning.ipynb`](recipes/capabilities/in-context-learning/in-context-learning.ipynb) | Single-image in-context detection (bootstrap exemplar → apply to target). | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/in-context-learning/in-context-learning.ipynb) |
| [`recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb`](recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) | Multi-shot guidance to classify/ground multiple categories at once. | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) |

Each recipe follows a consistent format inspired by the Gemini and OpenAI cookbooks: install cell → configure cell → run helper → visualize results → next steps. Source assets live in [`cookbook/_shared/assets`](./_shared/assets) so notebooks can run offline.

> **When to use `detect()` vs `@perceive`?** Use `detect()` for quick, single-shot helpers. Reach for `@perceive` when you want to embed custom prompts, streaming, or multi-step logic inside your own pipeline.

---

## Tutorials

| Folder | Description | Colab |
| --- | --- | --- |
| [`recipes/tutorials/isaac_0.1_frame_by_frame/isaac_0.1_frame_by_frame.ipynb`](recipes/tutorials/isaac_0.1_frame_by_frame/isaac_0.1_frame_by_frame.ipynb) | Extract frames from `surf.mp4`, run Isaac 0.1 on each frame, and stitch an annotated video. | [Launch](https://colab.research.google.com/github/ericpence/perceptron_repo/blob/main/cookbook/recipes/tutorials/isaac_0.1_frame_by_frame/isaac_0.1_frame_by_frame.ipynb) |

Tutorials include reusable utilities for decoding/encoding video, batching Perceptron requests, and saving artifacts (e.g., `frames/`, `frames_annotated/`, `*_annotated.mp4`), which are gitignored by default.

---

## Running Notebooks Locally

```bash
uv venv .venv
source .venv/bin/activate            # or .venv\\Scripts\\activate on Windows
uv pip install --upgrade perceptron pillow opencv-python tqdm
export PERCEPTRON_API_KEY="..."
uv run jupyter lab  # launch from the repo root so `cookbook.utils` imports succeed
```

- Every notebook/script raises a helpful error if `PERCEPTRON_API_KEY` is missing or still set to the placeholder.
- To execute headlessly, run `uv run jupyter nbconvert --to notebook --execute path/to/notebook.ipynb`.

Or execute the entire suite (requires valid credentials) with:

```bash
PERCEPTRON_API_KEY="..." PERCEPTRON_BASE_URL="https://staging-api.perceptron.build/v1" \
uv run python tools/run_notebooks.py --timeout 900
```

---

## Shared Assets + Legacy Examples

Centrally import the helper instead:

```python
from cookbook.utils import cookbook_asset

scene = cookbook_asset("capabilities", "detection", "ppe_line.webp")
```

Make sure you run Python from the repo root (or add the repo root to `PYTHONPATH`) so `import cookbook.utils` resolves.

> Integration tests and docs share the `_shared/assets/in-context-learning/multi` shots. Load them via `cookbook_asset("in-context-learning", "multi", ...)` to stay in sync.

---

## Repository Layout

```
cookbook/
├── quickstart/                    # Quickstart notebooks (Isaac, Qwen)
├── quickstart_qwen/                 # Qwen quickstart (notebook + script)
├── recipes/
│   ├── capabilities/                # Captioning, OCR, detection, visual QA, ICL
│   └── tutorials/                   # Longer-form guides (e.g., frame-by-frame video)
└── _shared/assets/                  # Images, MP4s, and supporting files
```

All generated frames, annotated images, and derived MP4s are ignored via [`cookbook/.gitignore`](./.gitignore) to keep the repo clean.

---

## Contributing

1. Fork the repo and create a branch.
2. Add or update notebooks/scripts following the quickstart format (install → configure → helper → visualization → next steps).
3. Reference shared assets inside `_shared/assets` so notebooks remain self-contained.
4. Run `jupyter nbconvert --execute` (or the paired script) before opening a PR.

Issues and PRs are welcome—let us know which capabilities or tutorials you’d like to see next!
