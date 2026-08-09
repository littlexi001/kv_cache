import importlib.util
import sys
import types
from pathlib import Path


# Avoid loading the remote-only model runner while unit-testing pure helpers.
sys.modules.setdefault("head_frequency_intervention", types.SimpleNamespace(HeadFrequencyIntervention=object))
sys.modules.setdefault("run_inference_rnope_ruler", types.SimpleNamespace())
PATH = Path(__file__).parents[1] / "src" / "eval_short_ppl.py"
SPEC = importlib.util.spec_from_file_location("eval_short_ppl", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_token_windows_are_non_overlapping_and_have_next_token() -> None:
    windows = MODULE.token_windows(list(range(12)), length=3, count=3)
    assert windows == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
