import importlib.util
import sys
from pathlib import Path


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
PATH = SRC / "make_validation_specs.py"
SPEC = importlib.util.spec_from_file_location("make_validation_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validation_specs_include_baseline_controls_and_candidates() -> None:
    assert [spec["name"] for spec in MODULE.validation_specs()] == [
        "native_rope",
        "global_f00_07_delete",
        "late_f00_07_delete",
        "deep_f08_15_delete",
    ]
