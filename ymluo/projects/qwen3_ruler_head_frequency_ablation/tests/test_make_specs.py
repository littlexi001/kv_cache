import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_specs.py"
SPEC = importlib.util.spec_from_file_location("make_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_coarse_grid_has_192_interventions_plus_native() -> None:
    specs = MODULE.coarse_specs()
    assert len(specs) == 193
    assert specs[0]["name"] == "native_rope"
    assert len({spec["name"] for spec in specs}) == len(specs)


def test_coarse_grid_covers_deep_layers_and_all_frequencies() -> None:
    specs = MODULE.coarse_specs()[1:]
    layers = {layer for spec in specs for layer in spec["atoms"][0]["layers"]}
    groups = {spec["atoms"][0]["head_groups"][0] for spec in specs}
    frequencies = {value for spec in specs for value in spec["atoms"][0]["frequency_pairs"]}
    assert layers == set(range(18, 36))
    assert groups == set(range(8))
    assert frequencies == set(range(64))


def test_dense_layer_band_grid_is_complete_and_atomic_in_layer() -> None:
    specs = MODULE.dense_layer_band_specs()
    assert len(specs) == 1 + 18 * 8 * 8
    assert specs[0]["name"] == "native_rope"
    assert len({spec["name"] for spec in specs}) == len(specs)

    interventions = specs[1:]
    coordinates = {
        (
            spec["atoms"][0]["layers"][0],
            spec["atoms"][0]["head_groups"][0],
            tuple(spec["atoms"][0]["frequency_pairs"]),
        )
        for spec in interventions
    }
    expected = {
        (layer, group, tuple(range(start, start + 8)))
        for layer in range(18, 36)
        for group in range(8)
        for start in range(0, 64, 8)
    }
    assert coordinates == expected
    assert all(len(spec["atoms"][0]["layers"]) == 1 for spec in interventions)
