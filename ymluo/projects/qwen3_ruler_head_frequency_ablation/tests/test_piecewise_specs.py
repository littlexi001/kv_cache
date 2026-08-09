from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_piecewise_warp_specs.py"
SPEC = importlib.util.spec_from_file_location("make_piecewise_warp_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_piecewise_grid_is_unique_and_preserves_native_control() -> None:
    specs = MODULE.piecewise_specs()
    assert len(specs) == 16
    assert specs[0]["name"] == "native_rope"
    assert len({spec["name"] for spec in specs}) == len(specs)


def test_piecewise_grid_uses_long_context_only_starts() -> None:
    starts = {
        int(atom["position_warp_start"])
        for spec in MODULE.piecewise_specs()[1:]
        for atom in spec["atoms"]
    }
    assert starts == {8192, 16384, 24576}
