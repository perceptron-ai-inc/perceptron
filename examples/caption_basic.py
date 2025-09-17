"""Basic captioning example with optional live execution."""

from __future__ import annotations

import argparse
from perceptron import caption


DEFAULT_IMAGE = "https://pbs.twimg.com/media/G04-cR0WMAAu7mo?format=jpg&name=medium"


def describe(img, *, stream=False):
    """Generate a caption for the given image."""
    if stream:
        return caption(img, stream=True)
    return caption(img)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption an image using the Perceptron SDK")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image URL or path to caption")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print("=== streaming ===")
    for event in describe(args.image, stream=True):
        print(event)

    print("\n=== single call ===")
    res = describe(args.image)
    print(res)
    print("text:", res.text)
    if res.raw and isinstance(res.raw, dict):
        choices = res.raw.get("choices") or []
        reasoning = choices[0]["message"].get("reasoning_content") if choices else None
        if reasoning:
            print("reasoning:")
            print(reasoning)
    print("errors:", res.errors)
    print("compiled keys:", list((res.raw or {}).keys()))
