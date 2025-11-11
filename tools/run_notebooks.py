"""Execute every cookbook notebook with nbclient."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

logger = logging.getLogger(__name__)


def discover_notebooks(root: Path) -> list[Path]:
    return sorted(root.rglob("*.ipynb"))


def run_notebook(path: Path, timeout: int) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Execution timeout per notebook in seconds (default: 900)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cookbook"),
        help="Directory to search for notebooks (default: cookbook)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failure instead of running all notebooks.",
    )
    args = parser.parse_args()

    if not os.getenv("PERCEPTRON_API_KEY"):
        parser.error("PERCEPTRON_API_KEY must be set to run the cookbooks against the backend")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    nb_paths = discover_notebooks(args.root)
    if not nb_paths:
        parser.error(f"No notebooks found under {args.root}")

    failures: list[tuple[Path, Exception | str]] = []
    for path in nb_paths:
        logger.info("Executing %s...", path)
        try:
            run_notebook(path, timeout=args.timeout)
        except CellExecutionError as exc:
            logger.error("FAILED: %s\n%s", path, exc)
            failures.append((path, exc))
            if args.fail_fast:
                break
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("FAILED: %s\n%s", path, exc)
            failures.append((path, exc))
            if args.fail_fast:
                break
        else:
            logger.info("SUCCESS: %s", path)

    if failures:
        logger.error("%d notebook(s) failed:", len(failures))
        for path, exc in failures:
            logger.error(" - %s: %s", path, exc)
        return 1

    logger.info("All notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
