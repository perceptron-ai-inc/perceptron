# Perceptron Cookbook

Hands-on quickstarts, capability recipes, and end-to-end tutorials for building with the Perceptron SDK.

**[Prompting Guide](PROMPTING.md)** — Optimal prompts for each primitive with SDK and curl examples.

---

## Quickstarts

| Notebook | What it covers | Colab |
| --- | --- | --- |
| [`quickstart_perceptron`](quickstart/quickstart_perceptron/quickstart_perceptron.ipynb) | Ask Perceptron Mk1 a question about an image and get a natural-language answer. | [Open in Colab](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/quickstart/quickstart_perceptron/quickstart_perceptron.ipynb) |
| [`quickstart_perceptron_video`](quickstart/quickstart_perceptron_video/quickstart_perceptron_video.ipynb) | Ask Perceptron Mk1 a question about a video and get a natural-language answer. | [Open in Colab](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/quickstart/quickstart_perceptron_video/quickstart_perceptron_video.ipynb) |
| [`quickstart_isaac_0_2`](quickstart/quickstart_isaac_0_2/quickstart_isaac_0_2.ipynb) | Ask Isaac 0.2 a question about an image (defaults to `isaac-0.2-1b`); switch to `isaac-0.2-2b-preview` for reasoning. | [Open in Colab](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/quickstart/quickstart_isaac_0_2/quickstart_isaac_0_2.ipynb) |
| [`quickstart_isaac_0_1`](quickstart/quickstart_isaac_0_1/quickstart_isaac_0_1.ipynb) | Ask Isaac 0.1 a question about an image (legacy open-weights model; reasoning not supported). | [Open in Colab](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/quickstart/quickstart_isaac_0_1/quickstart_isaac_0_1.ipynb) |

---

## Capability Recipes

### Perceptron Mk1

Image and video recipes for the flagship `perceptron-mk1` model, using the v0.3.5+ SDK API. Mk1 adds long-form video Q&A, temporal clipping, and multimodal in-context learning on top of the image capabilities.

| Notebook | Scenario | Colab |
| --- | --- | --- |
| [`perceptron-mk1/image-qa`](recipes/capabilities/perceptron-mk1/image-qa.ipynb) | Grounded Q&A with bounding-box citations on a studio scene. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/image-qa.ipynb) |
| [`perceptron-mk1/image-captioning`](recipes/capabilities/perceptron-mk1/image-captioning.ipynb) | Concise and detailed captions, optionally with grounded snippets. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/image-captioning.ipynb) |
| [`perceptron-mk1/object-detection`](recipes/capabilities/perceptron-mk1/object-detection.ipynb) | PPE detection via the `@perceive` helper with `expects="box"`. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/object-detection.ipynb) |
| [`perceptron-mk1/ocr`](recipes/capabilities/perceptron-mk1/ocr.ipynb) | OCR with custom prompts targeting product labels. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/ocr.ipynb) |
| [`perceptron-mk1/in-context-learning-image`](recipes/capabilities/perceptron-mk1/in-context-learning-image.ipynb) | Single-image ICL: bootstrap an exemplar, apply to a new scene. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/in-context-learning-image.ipynb) |
| [`perceptron-mk1/video-qa`](recipes/capabilities/perceptron-mk1/video-qa.ipynb) | Long-form video Q&A with reasoning enabled (robot-assembly walkthrough). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/video-qa.ipynb) |
| [`perceptron-mk1/video-clipping`](recipes/capabilities/perceptron-mk1/video-clipping.ipynb) | Temporal grounding: return start/end timestamps via `expects="clip"`. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/video-clipping.ipynb) |
| [`perceptron-mk1/in-context-learning-video`](recipes/capabilities/perceptron-mk1/in-context-learning-video.ipynb) | Multimodal ICL: example image + intent → query video → clip back. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/in-context-learning-video.ipynb) |
| [`perceptron-mk1/structured-outputs`](recipes/capabilities/perceptron-mk1/structured-outputs.ipynb) | Force model output to match a Pydantic schema or regex pattern. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/perceptron-mk1/structured-outputs.ipynb) |

> **When to use `detect()` vs `@perceive`?** Use `detect()` for quick, single-shot helpers. Reach for `@perceive` when you want to embed custom prompts, streaming, or multi-step logic inside your own pipeline.

### Isaac 0.2

Image-only recipes pinned to `isaac-0.2-1b`. Open-weights edge-tier model with reasoning support on the 2B Preview variant.

| Notebook | Scenario | Colab |
| --- | --- | --- |
| [`isaac-0.2/image-qa`](recipes/capabilities/isaac-0.2/image-qa.ipynb) | Ask grounded questions and cite regions with bounding boxes. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/image-qa.ipynb) |
| [`isaac-0.2/image-captioning`](recipes/capabilities/isaac-0.2/image-captioning.ipynb) | Generate concise or grounded captions (with bounding boxes). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/image-captioning.ipynb) |
| [`isaac-0.2/object-detection`](recipes/capabilities/isaac-0.2/object-detection.ipynb) | Detect PPE with a `@perceive` helper or the high-level `detect()` API. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/object-detection.ipynb) |
| [`isaac-0.2/ocr`](recipes/capabilities/isaac-0.2/ocr.ipynb) | Run OCR with custom prompts and parse the output. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/ocr.ipynb) |
| [`isaac-0.2/in-context-learning-image`](recipes/capabilities/isaac-0.2/in-context-learning-image.ipynb) | Single-image in-context detection (bootstrap exemplar → apply to target). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/in-context-learning-image.ipynb) |
| [`isaac-0.2/multi-image-in-context-learning`](recipes/capabilities/isaac-0.2/multi-image-in-context-learning.ipynb) | Multi-shot guidance to classify/ground multiple categories at once. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.2/multi-image-in-context-learning.ipynb) |
| [`constrained-decoding`](recipes/capabilities/constrained-decoding/constrained-decoding.ipynb) | Structured output with Pydantic schemas or regex patterns (uses `isaac-0.2-1b`). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/constrained-decoding/constrained-decoding.ipynb) |

### Isaac 0.1 (legacy)

Original Isaac recipes pinned to `isaac-0.1`. Kept for existing integrations.

| Notebook | Scenario | Colab |
| --- | --- | --- |
| [`captioning`](recipes/capabilities/captioning/captioning.ipynb) | Generate concise or grounded captions (with bounding boxes). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/captioning/captioning.ipynb) |
| [`ocr`](recipes/capabilities/ocr/ocr.ipynb) | Run OCR with custom prompts and parse the output. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/ocr/ocr.ipynb) |
| [`object-detection`](recipes/capabilities/object-detection/object-detection.ipynb) | Detect PPE with a `@perceive` helper or the high-level `detect()` API. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/object-detection/object-detection.ipynb) |
| [`visual-qa`](recipes/capabilities/visual-qa/visual-qa.ipynb) | Ask grounded questions and cite regions with bounding boxes. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/visual-qa/visual-qa.ipynb) |
| [`in-context-learning`](recipes/capabilities/in-context-learning/in-context-learning.ipynb) | Single-image in-context detection (bootstrap exemplar → apply to target). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/in-context-learning/in-context-learning.ipynb) |
| [`multi-image-in-context-learning`](recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) | Multi-shot guidance to classify/ground multiple categories at once. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) |

---

## Tutorials

> For native video Q&A with Mk1, see the [`perceptron-mk1/video-qa`](recipes/capabilities/perceptron-mk1/video-qa.ipynb) recipe above.

| Notebook | Description | Colab |
| --- | --- | --- |
| [`isaac_frame_by_frame`](recipes/tutorials/isaac_frame_by_frame/isaac_frame_by_frame.ipynb) | Extract frames from a video, run Isaac 0.1 on each frame, and stitch an annotated video. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/tutorials/isaac_frame_by_frame/isaac_frame_by_frame.ipynb) |
