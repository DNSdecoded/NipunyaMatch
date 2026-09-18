import py_compile
from pathlib import Path

import pytest

UI = sorted(Path("app/ui").rglob("*.py"))


@pytest.mark.parametrize("path", UI, ids=[p.name for p in UI])
def test_ui_file_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)
