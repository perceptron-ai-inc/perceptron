from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

# PERCEPTRON_TEST_INSTALLED=1 tests the installed package (e.g. a built wheel) instead of the source tree: `src` stays
# off sys.path. The project root is still added for repo helpers such as `cookbook.utils`.
TEST_INSTALLED = os.environ.get("PERCEPTRON_TEST_INSTALLED", "").strip().lower() in {"1", "true", "yes"}

for candidate in (PROJECT_ROOT,) if TEST_INSTALLED else (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
