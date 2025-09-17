"""One-shot detect() example with a real coffee-cup reference image.

Usage (requires a live provider key, e.g. FAL_KEY):

    python detect_icl_one_shot.py --provider fal

This script uses an Unsplash photo annotated with bounding boxes to prime the
model and then asks it to find cups in a different desk scene. If you omit the
provider/api-key overrides the call runs in compile-only mode so you can inspect
the generated Task payload.
"""

from __future__ import annotations

import argparse
import contextlib
import os

from perceptron import annotate_image, detect, bbox, config as cfg


EXAMPLE_IMAGE = "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?auto=format&fit=crop&w=800&q=80"
EXAMPLE_BOXES = [
    bbox(512, 72, 604, 206, mention="cup"),
    bbox(524, 266, 608, 386, mention="cup"),
    bbox(446, 454, 528, 568, mention="cup"),
    bbox(252, 632, 368, 782, mention="cup"),
]
TARGET_IMAGE = "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=800&q=80"


def detect_with_icl(
    image_url: str,
    *,
    example_path: str | None = None,
    example_boxes: list[tuple[int, int, int, int]] | None = None,
    label: str | None = None,
):
    """Run detect() with either curated or caller-provided examples."""

    if example_path and example_boxes:
        annotations = [bbox(x1, y1, x2, y2, mention=label) for x1, y1, x2, y2 in example_boxes]
        examples = [annotate_image(example_path, annotations)]
        classes = [label] if label else None
    else:
        examples = [annotate_image(EXAMPLE_IMAGE, EXAMPLE_BOXES)]
        classes = ["cup"]

    return detect(image_url, classes=classes, examples=examples, max_outputs=None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICL cup detection demo")
    parser.add_argument(
        "--image",
        default=TARGET_IMAGE,
        help="Image URL to analyse (default: curated Unsplash desk photo)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("PERCEPTRON_PROVIDER"),
        help="Provider override (set to 'fal' for live calls)",
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
        res = detect_with_icl(args.image)
    print("text:\n", res.text)
    print("\npoints:")
    for box in res.points or []:
        print("  -", box)


if __name__ == "__main__":
    main()
