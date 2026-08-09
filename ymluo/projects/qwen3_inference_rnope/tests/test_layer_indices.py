from pathlib import Path
import importlib.util


PATH = Path(__file__).parents[1] / "src" / "run_inference_rnope_ruler.py"
SPEC = importlib.util.spec_from_file_location("runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_layer_sets_for_qwen3_8b() -> None:
    assert RUNNER.layer_indices("native_rope", 36) == ()
    assert RUNNER.layer_indices("nope_every4_offset3", 36) == tuple(range(3, 36, 4))
    assert RUNNER.layer_indices("nope_every4_offset0", 36) == tuple(range(0, 36, 4))
    assert RUNNER.layer_indices("nope_alternating_odd", 36) == tuple(range(1, 36, 2))

