import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_stability_specs.py"
SPEC = importlib.util.spec_from_file_location("make_stability_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validation_grid_is_fixed_and_unique() -> None:
    specs = MODULE.validation_specs()
    assert len(specs) == 13
    assert specs[0]["name"] == "native_rope"
    assert len({spec["name"] for spec in specs}) == 13


def test_grid_contains_continuous_scales_and_binary_control() -> None:
    specs = {spec["name"]: spec for spec in MODULE.validation_specs()}
    assert specs["l25_g3_f46_a0"]["atoms"][0]["frequency_scale"] == 0.0
    assert specs["l25_g3_f46_a0.5"]["atoms"][0]["frequency_scale"] == 0.5
    assert len(specs["l25_g3_f40_47_a0"]["atoms"][0]["frequency_pairs"]) == 8
