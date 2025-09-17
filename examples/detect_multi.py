"""Detect multiple object categories on a sample street scene."""

from __future__ import annotations

import argparse
import contextlib
import os

from perceptron import detect, config as cfg

IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
CLASSES = ["bus", "person"]


def find_objects(image_url: str):
    return detect(image_url, classes=CLASSES, max_outputs=None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiclass detection demo")
    parser.add_argument("--image", default=IMAGE_URL, help="Image URL to analyse")
    parser.add_argument("--provider", default=os.getenv("PERCEPTRON_PROVIDER"))
    parser.add_argument("--api-key", default=os.getenv("PERCEPTRON_API_KEY"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {k: v for k, v in {"provider": args.provider, "api_key": args.api_key}.items() if v}
    ctx = cfg(**overrides) if overrides else contextlib.nullcontext()
    with ctx:
        res = find_objects(args.image)
    print("text:\n", res.text)
    print("\npoints:")
    for obj in res.points or []:
        print("  -", obj)
    if res.errors:
        print("\nerrors:", res.errors)


if __name__ == "__main__":
    main()
