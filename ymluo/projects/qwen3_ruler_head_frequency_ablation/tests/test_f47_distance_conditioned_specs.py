from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_f47_distance_conditioned_specs.py"
SPEC = importlib.util.spec_from_file_location("make_f47_distance_conditioned_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_grid_contains_native_fixed_and_nine_piecewise_specs() -> None:
    specs = MODULE.make_specs()
    assert len(specs) == 11
    assert specs[0]["name"] == "native_rope"
    assert specs[1]["name"] == "l18_23_g4_f47_fixed_a0"
    assert len({spec["name"] for spec in specs}) == len(specs)


def test_all_interventions_target_only_selected_f47_region() -> None:
    for spec in MODULE.make_specs()[1:]:
        assert len(spec["atoms"]) == 1
        atom = spec["atoms"][0]
        assert atom["layers"] == list(range(18, 24))
        assert atom["head_groups"] == [4]
        assert atom["frequency_pairs"] == [47]


def test_piecewise_grid_has_expected_starts_and_scales() -> None:
    atoms = [spec["atoms"][0] for spec in MODULE.make_specs()[2:]]
    assert {atom["position_warp_start"] for atom in atoms} == {8192, 16384, 24576}
    assert {atom["frequency_scale"] for atom in atoms} == {0.0, 0.25, 0.5}
