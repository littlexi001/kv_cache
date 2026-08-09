from pathlib import Path
import importlib.util


PATH = Path(__file__).parents[1] / "src" / "summarize_training_curve.py"
SPEC = importlib.util.spec_from_file_location("summary", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_module_imports() -> None:
    assert callable(MODULE.main)

