import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "analyze_validation.py"
SPEC = importlib.util.spec_from_file_location("analyze_validation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates() -> None:
    assert MODULE.percentile([0.0, 10.0], 0.25) == 2.5


def test_sign_test_ignores_ties_and_is_two_sided() -> None:
    assert MODULE.exact_two_sided_sign_p(3, 0) == 0.25
    assert MODULE.exact_two_sided_sign_p(0, 0) is None
