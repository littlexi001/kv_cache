from pathlib import Path
import importlib.util
import numpy as np


PATH = Path(__file__).parents[1] / "src" / "train_rnope_qlora.py"
SPEC = importlib.util.spec_from_file_location("train_rnope", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nope_layers() -> None:
    assert MODULE.nope_layers(36) == tuple(range(3, 36, 4))


def test_disjoint_split(tmp_path: Path) -> None:
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(384, dtype=np.int32).reshape(48, 8))
    train = MODULE.PackedTokenDataset(path, 32, "train", 2)
    valid = MODULE.PackedTokenDataset(path, 32, "eval", 2)
    assert len(train) == 10
    assert len(valid) == 2
    assert int(train[len(train) - 1]["input_ids"][-1]) == 319
    assert int(valid[0]["input_ids"][0]) == 320
