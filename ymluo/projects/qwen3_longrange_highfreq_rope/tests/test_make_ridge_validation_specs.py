import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_ridge_validation_specs.py"
SPEC = importlib.util.spec_from_file_location("make_ridge_validation_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ridge_specs_are_unique_and_cover_depth_band_tradeoff() -> None:
    specs = MODULE.ridge_specs()
    names = [spec["name"] for spec in specs]
    assert len(specs) == 7
    assert len(names) == len(set(names))
    assert "late_l18_f00_03_delete" in names
    assert "late_l30_f00_15_delete" in names
    assert "late_l24_f00_07_scale075" in names
