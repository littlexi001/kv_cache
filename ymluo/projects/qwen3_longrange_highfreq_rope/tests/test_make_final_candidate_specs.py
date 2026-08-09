import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "make_final_candidate_specs.py"
SPEC = importlib.util.spec_from_file_location("make_final_candidate_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_final_specs_are_baseline_and_pareto_candidates() -> None:
    assert [spec["name"] for spec in MODULE.final_specs()] == [
        "native_rope",
        "late_l24_f00_11_delete",
        "late_l30_f00_15_delete",
    ]
