import importlib.util
import sys
from pathlib import Path


SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
PATH = SRC / "make_progressive_validation_specs.py"
SPEC = importlib.util.spec_from_file_location("make_progressive_validation_specs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_only_non_degrading_smoke_candidates_are_selected() -> None:
    assert [spec["name"] for spec in MODULE.validation_specs()] == [
        "progressive_late",
        "progressive_conservative",
    ]
