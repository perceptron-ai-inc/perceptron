# Perceptron Cookbook

Hands-on quickstarts, capability recipes, and end-to-end tutorials for building with the Perceptron SDK.

**[Prompting Guide](PROMPTING.md)** — Optimal prompts for each primitive with SDK and curl examples.

---

## Quickstarts

| Notebook | What it covers | Colab |
| --- | --- | --- |
| [`quickstart_isaac`](quickstart/quickstart_isaac/quickstart_isaac.ipynb) | Run the Isaac 0.2 model to localize shipping boxes in a factory scene. | [Open in Colab](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/quickstart/quickstart_isaac/quickstart_isaac.ipynb) |

---

## Capability Recipes

| Notebook | Scenario | Colab |
| --- | --- | --- |
| [`captioning`](recipes/capabilities/captioning/captioning.ipynb) | Generate concise or grounded captions (with bounding boxes). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/captioning/captioning.ipynb) |
| [`ocr`](recipes/capabilities/ocr/ocr.ipynb) | Run OCR with custom prompts and parse the output. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/ocr/ocr.ipynb) |
| [`object-detection`](recipes/capabilities/object-detection/object-detection.ipynb) | Detect PPE with a `@perceive` helper or the high-level `detect()` API. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/object-detection/object-detection.ipynb) |
| [`visual-qa`](recipes/capabilities/visual-qa/visual-qa.ipynb) | Ask grounded questions and cite regions with bounding boxes. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/visual-qa/visual-qa.ipynb) |
| [`in-context-learning`](recipes/capabilities/in-context-learning/in-context-learning.ipynb) | Single-image in-context detection (bootstrap exemplar → apply to target). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/in-context-learning/in-context-learning.ipynb) |
| [`multi-image-in-context-learning`](recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) | Multi-shot guidance to classify/ground multiple categories at once. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/multi-image-in-context-learning/multi-image-in-context-learning.ipynb) |
| [`constrained-decoding`](recipes/capabilities/constrained-decoding/constrained-decoding.ipynb) | Force structured output with Pydantic schemas or regex patterns. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/constrained-decoding/constrained-decoding.ipynb) |

> **When to use `detect()` vs `@perceive`?** Use `detect()` for quick, single-shot helpers. Reach for `@perceive` when you want to embed custom prompts, streaming, or multi-step logic inside your own pipeline.

### Isaac 0.3 Max sibling recipes

The same capabilities, configured for the flagship `isaac-0.3-max` model and using the post-v0.3.0 SDK API. Image-only legacy recipes above remain valid for the 0.x family.

| Notebook | Scenario | Colab |
| --- | --- | --- |
| [`isaac-0.3-max/image-qa`](recipes/capabilities/isaac-0.3-max/image-qa.ipynb) | Grounded Q&A with bounding-box citations on a studio scene. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/image-qa.ipynb) |
| [`isaac-0.3-max/image-captioning`](recipes/capabilities/isaac-0.3-max/image-captioning.ipynb) | Concise and detailed captions, optionally with grounded snippets. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/image-captioning.ipynb) |
| [`isaac-0.3-max/object-detection`](recipes/capabilities/isaac-0.3-max/object-detection.ipynb) | PPE detection via the `@perceive` helper with `expects="box"`. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/object-detection.ipynb) |
| [`isaac-0.3-max/ocr`](recipes/capabilities/isaac-0.3-max/ocr.ipynb) | OCR with custom prompts targeting product labels. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/ocr.ipynb) |
| [`isaac-0.3-max/in-context-learning-image`](recipes/capabilities/isaac-0.3-max/in-context-learning-image.ipynb) | Single-image ICL: bootstrap an exemplar, apply to a new scene. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/in-context-learning-image.ipynb) |
| [`isaac-0.3-max/video-qa`](recipes/capabilities/isaac-0.3-max/video-qa.ipynb) | Long-form video Q&A with reasoning enabled (robot-assembly walkthrough). | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/video-qa.ipynb) |
| [`isaac-0.3-max/video-clipping`](recipes/capabilities/isaac-0.3-max/video-clipping.ipynb) | Temporal grounding: return start/end timestamps via `expects="clip"`. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/video-clipping.ipynb) |
| [`isaac-0.3-max/in-context-learning-video`](recipes/capabilities/isaac-0.3-max/in-context-learning-video.ipynb) | Multimodal ICL: example image + intent → query video → clip back. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/capabilities/isaac-0.3-max/in-context-learning-video.ipynb) |

---

## Tutorials

| Notebook | Description | Colab |
| --- | --- | --- |
| [`isaac_frame_by_frame`](recipes/tutorials/isaac_frame_by_frame/isaac_frame_by_frame.ipynb) | Extract frames from a video, run Isaac 0.1 on each frame, and stitch an annotated video. | [Launch](https://colab.research.google.com/github/perceptron-ai-inc/perceptron/blob/main/cookbook/recipes/tutorials/isaac_frame_by_frame/isaac_frame_by_frame.ipynb) |
