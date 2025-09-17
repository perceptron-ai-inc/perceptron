"""Two-shot detect() example highlighting cups on a desk.

We reuse two differently staged cup photos to build stronger examples before
running on a third workspace scene. Provide --provider/--api-key or set
FAL_KEY (or another compatible key) to execute against a live endpoint;
otherwise the script will just print the compiled Task.
"""

from __future__ import annotations

import argparse
import contextlib
import os

from perceptron import annotate_image, detect, bbox, config as cfg


EXAMPLES = [
    {
        "image": "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?auto=format&fit=crop&w=800&q=80",
        "boxes": [
            bbox(512, 72, 604, 206, mention="cup"),
            bbox(524, 266, 608, 386, mention="cup"),
            bbox(446, 454, 528, 568, mention="cup"),
            bbox(252, 632, 368, 782, mention="cup"),
        ],
        "prompt": "Example 1: mark each coffee cup with <point_box mention=\"cup\">.",
    },
    {
        "image": "https://images.unsplash.com/photo-1470337458703-46ad1756a187?auto=format&fit=crop&w=800&q=80",
        "boxes": [bbox(521, 263, 837, 907, mention="cup")],
        "prompt": "Example 2: highlight the mug in this scene using <point_box> tags.",
    },
]

TARGET_IMAGE = "https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=800&q=80"


def detect_two_shot(
    image_url: str,
    *,
    ex1_path: str | None = None,
    ex1_boxes: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    ex2_path: str | None = None,
    ex2_boxes: list[tuple[str, tuple[int, int, int, int]]] | None = None,
):
    """Run detect() with two ICL examples."""

    if ex1_path and ex1_boxes and ex2_path and ex2_boxes:
        examples = []
        seen_labels: set[str] = set()
        for prompt, path, specs in [
            ("Example 1: return <point_box> tags for each object.", ex1_path, ex1_boxes),
            ("Example 2: repeat the labelling convention.", ex2_path, ex2_boxes),
        ]:
            annotations = []
            for label, (x1, y1, x2, y2) in specs:
                annotations.append(bbox(x1, y1, x2, y2, mention=label))
                seen_labels.add(label)
            examples.append(annotate_image(path, annotations))
        classes = sorted(seen_labels) if seen_labels else None
    else:
        examples = [annotate_image(item["image"], item["boxes"]) for item in EXAMPLES]
        classes = ["cup"]

    return detect(image_url, classes=classes, examples=examples, max_outputs=None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-shot cup detection demo")
    parser.add_argument(
        "--image",
        default=TARGET_IMAGE,
        help="Image URL to run (default: curated Unsplash workspace photo)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("PERCEPTRON_PROVIDER"),
        help="Provider override (e.g. fal)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("PERCEPTRON_API_KEY"),
        help="API key for the chosen provider",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {k: v for k, v in {"provider": args.provider, "api_key": args.api_key}.items() if v}
    ctx = cfg(**overrides) if overrides else contextlib.nullcontext()
    with ctx:
        res = detect_two_shot(args.image)
    print("text:\n", res.text)
    print("\npoints:")
    for box in res.points or []:
        print("  -", box)


if __name__ == "__main__":
    main()
