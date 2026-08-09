import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_progressive_spectral_specs.py"
SPEC = importlib.util.spec_from_file_location("make_progressive_spectral_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_progressive_ridge_expands_nope_band_with_depth() -> None:
    spec = MODULE.progressive_specs()[0]
    atoms = spec["atoms"]
    assert [atom["frequency_pairs"] for atom in atoms] == [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 12)),
        list(range(12, 16)),
    ]
    assert [atom["layers"][0] for atom in atoms] == [18, 21, 24, 30]


def test_progressive_specs_are_unique() -> None:
    names = [spec["name"] for spec in MODULE.progressive_specs()]
    assert len(names) == len(set(names)) == 4


def test_validated_schedule_uses_held_out_safe_boundaries() -> None:
    spec = MODULE.progressive_specs()[-1]
    assert [atom["layers"][0] for atom in spec["atoms"]] == [21, 24, 30]
    assert [atom["frequency_pairs"] for atom in spec["atoms"]] == [
        list(range(0, 8)),
        list(range(8, 12)),
        list(range(12, 16)),
    ]
