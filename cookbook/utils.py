from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from functools import lru_cache
from pathlib import Path

__all__ = ["cookbook_asset", "cookbook_assets_root"]


def _candidate_paths() -> Iterable[Path]:
    """Yield directories to probe for the shared assets folder."""

    anchors: list[Path] = [Path.cwd()]
    with suppress(NameError):  # pragma: no cover - __file__ should always exist
        anchors.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for anchor in anchors:
        for candidate in [anchor, *anchor.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


@lru_cache(maxsize=1)
def cookbook_assets_root() -> Path:
    """Return the root of cookbook/_shared/assets, independent of cwd."""

    for candidate in _candidate_paths():
        for root in (candidate, candidate / "cookbook"):
            assets_dir = root / "_shared" / "assets"
            if assets_dir.is_dir():
                return assets_dir
    raise RuntimeError(
        "Unable to locate cookbook/_shared/assets. Run the example from the repo root or ensure "
        "the perceptron checkout is on PYTHONPATH."
    )


def cookbook_asset(*parts: str, ensure_exists: bool = True) -> Path:
    """Resolve a shared asset path given its relative components."""

    if not parts:
        raise ValueError("Provide at least one path component to cookbook_asset().")

    path = cookbook_assets_root().joinpath(*parts)
    if ensure_exists and not path.exists():
        raise FileNotFoundError(f"Missing cookbook asset: {path}")
    return path
