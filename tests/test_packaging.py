"""Packaging: the public API imports, and (in CI's wheel job) the installed package is complete.

CI builds the wheel, installs it into a fresh virtualenv and runs the offline suite with
``PERCEPTRON_TEST_INSTALLED=1`` (``conftest.py`` then keeps ``src`` off ``sys.path``) and ``PERCEPTRON_TEST_WHEEL``
pointing at the built wheel; the installed-only checks below are skipped otherwise.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import perceptron

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE = PROJECT_ROOT / "src" / "perceptron"
INSTALLED = os.environ.get("PERCEPTRON_TEST_INSTALLED", "").strip().lower() in {"1", "true", "yes"}
WHEEL = os.environ.get("PERCEPTRON_TEST_WHEEL")

installed_only = pytest.mark.skipif(not INSTALLED, reason="set PERCEPTRON_TEST_INSTALLED=1 to test an installed wheel")


def _source_modules() -> list[str]:
    """Every module file of the source package, relative to ``src/perceptron`` (posix paths)."""
    return sorted(
        path.relative_to(SRC_PACKAGE).as_posix()
        for path in SRC_PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_every_name_in_all_imports():
    assert len(perceptron.__all__) == len(set(perceptron.__all__))
    missing = [name for name in perceptron.__all__ if not hasattr(perceptron, name)]
    assert missing == []
    namespace: dict = {}
    exec("from perceptron import *", namespace)  # works without the optional torch dependency
    assert set(perceptron.__all__) <= set(namespace)


def test_new_public_names_are_the_module_objects():
    from perceptron import chat, errors, files, models, multilook
    from perceptron.dsl import nodes
    from perceptron.pointing import geometry, parser, types

    assert perceptron.ChatCompletion is chat.ChatCompletion
    assert perceptron.function_tool is chat.function_tool
    assert perceptron.Files is files.Files
    assert perceptron.ModelInfo is models.ModelInfo
    assert perceptron.MultilookResponse is multilook.MultilookResponse
    assert perceptron.tool_result is nodes.tool_result
    assert perceptron.video_frames is nodes.video_frames
    assert perceptron.Track is types.Track
    assert perceptron.collect_annotations is parser.collect_annotations
    assert perceptron.resolve_asset_idx is parser.resolve_asset_idx
    assert perceptron.scale_annotations_by_asset is geometry.scale_annotations_by_asset
    assert perceptron.QuotaExceededError is errors.QuotaExceededError
    assert perceptron.ParseError is errors.ParseError


def test_tensorstream_stays_lazy():
    code = (
        "import sys, perceptron; "
        "assert 'perceptron.tensorstream' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'tensorstream' not in perceptron.__all__"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(perceptron.__file__).resolve().parents[1])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=PROJECT_ROOT.parent)


def test_tensorstream_loads_on_attribute_access():
    pytest.importorskip("torch")
    assert perceptron.tensorstream.TensorStream is not None


@pytest.mark.parametrize(
    "module",
    [
        "perceptron.dsl.perceive",
        "perceptron.dsl.nodes",
        "perceptron.pointing.parser",
        "perceptron.pointing.types",
        "perceptron.pointing.geometry",
        "perceptron.chat",
        "perceptron.files",
        "perceptron.models",
        "perceptron.multilook",
        "perceptron.cli",
    ],
)
def test_submodules_import(module):
    importlib.import_module(module)


def test_subpackages_are_regular_packages():
    from perceptron import dsl, pointing

    for package in (dsl, pointing):
        assert package.__file__ is not None
        assert Path(package.__file__).name == "__init__.py"


def test_version_matches_pyproject():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    assert match is not None
    assert perceptron.__version__ == match.group(1)


@installed_only
def test_installed_package_is_the_one_under_test():
    package_dir = Path(perceptron.__file__).resolve().parent
    assert package_dir != SRC_PACKAGE.resolve(), "the source tree was imported instead of the installed package"
    assert importlib.metadata.version("perceptron") == perceptron.__version__


@installed_only
def test_installed_package_has_every_module():
    package_dir = Path(perceptron.__file__).resolve().parent
    missing = [module for module in _source_modules() if not (package_dir / module).is_file()]
    assert missing == []


@pytest.mark.skipif(not WHEEL, reason="set PERCEPTRON_TEST_WHEEL to a built wheel to check its contents")
def test_wheel_contains_every_module_and_the_cli_entry_point():
    with zipfile.ZipFile(WHEEL) as wheel:
        names = set(wheel.namelist())
        missing = [module for module in _source_modules() if f"perceptron/{module}" not in names]
        assert missing == []
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(entry_points) == 1
        assert "perceptron = perceptron.cli:app" in wheel.read(entry_points[0]).decode()
