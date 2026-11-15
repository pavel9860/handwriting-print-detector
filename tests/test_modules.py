"""Every hwdet module imports and every dataset driver compiles."""

import importlib
import pkgutil
import py_compile
from pathlib import Path

import pytest

import hwdet

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name", sorted(m.name for m in pkgutil.walk_packages(hwdet.__path__, "hwdet.") if not m.name.endswith("__main__"))
)
def test_module_imports(name):
    if name in ("hwdet.data.detds", "hwdet.data.ds3k"):
        pytest.importorskip("dspages.multilang", reason="needs the data extra")
    importlib.import_module(name)


@pytest.mark.parametrize(
    "script", sorted((ROOT / "scripts").rglob("*.py")) + sorted((ROOT / "samples").glob("*.py")), ids=lambda p: p.name
)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)
