import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_late_spectral_grid.py"
SPEC = importlib.util.spec_from_file_location("make_late_spectral_grid", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grid_is_unique_and_contains_controls() -> None:
    specs = MODULE.grid_specs()
    names = [spec["name"] for spec in specs]
    assert len(specs) == 30
    assert len(names) == len(set(names))
    assert names[0] == "native_rope"
    assert "late_l24_f00_07_delete" in names
    assert "periodic_every4_full_nope" in names


def test_layer_by_width_grid_has_24_cells() -> None:
    specs = MODULE.grid_specs()
    cells = [spec for spec in specs if spec["name"].endswith("_delete")]
    assert len(cells) == 24
