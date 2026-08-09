import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "analyze_late_spectral_grid.py"
SPEC = importlib.util.spec_from_file_location("analyze_late_spectral_grid", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_grid_cell() -> None:
    assert MODULE.parse_grid_cell("late_l24_f00_07_delete") == (24, 8)
    assert MODULE.parse_grid_cell("native_rope") is None
