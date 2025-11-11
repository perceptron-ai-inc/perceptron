from __future__ import annotations

import hashlib
from pathlib import Path

from cookbook.utils import cookbook_asset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_in_context_learning_assets_present():
    assets = {
        "classA.jpg": cookbook_asset("in-context-learning", "multi", "classA.jpg"),
        "classA_boxed.png": cookbook_asset("in-context-learning", "multi", "classA_boxed.png"),
        "classB.webp": cookbook_asset("in-context-learning", "multi", "classB.webp"),
        "classB_boxed.png": cookbook_asset("in-context-learning", "multi", "classB_boxed.png"),
        "input.png": cookbook_asset("in-context-learning", "multi", "cat_dog_input.png"),
        "input_boxed.png": cookbook_asset("in-context-learning", "multi", "cat_dog_input_boxed.png"),
    }

    hashes = {name: _sha256(path) for name, path in assets.items() if path.exists()}
    assert len(hashes) == len(assets), "Missing one or more required cookbook assets"
