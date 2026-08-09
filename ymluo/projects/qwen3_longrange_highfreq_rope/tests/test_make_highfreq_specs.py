import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_highfreq_specs.py"
SPEC = importlib.util.spec_from_file_location("make_highfreq_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_discovery_specs_are_unique_and_include_controls() -> None:
    specs = MODULE.discovery_specs()
    names = [spec["name"] for spec in specs]
    assert len(specs) == 15
    assert len(names) == len(set(names))
    assert names[0] == "native_rope"
    assert "global_f00_07_delete" in names
    assert "remote8192_f00_07_stop" in names
    assert "deep_f56_63_delete" in names


def test_remote_interventions_preserve_native_scores_below_threshold() -> None:
    remote = next(
        spec for spec in MODULE.discovery_specs()
        if spec["name"] == "remote8192_f00_07_stop"
    )
    atom = remote["atoms"][0]
    assert atom["warp_mode"] == "relative_distance"
    assert atom["position_warp_start"] == 8192
    assert atom["query_tail_tokens"] == 1
    assert atom["frequency_scale"] == 0.0
    assert atom["frequency_pairs"] == list(range(8))
    assert atom["layers"] == list(range(18, 36))
