#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_rope_aware_kv_repack_benchmark as runtime  # noqa: E402


def make_checkpoint(path: Path, *, selected_tau: float | None, use_text_features: bool) -> None:
    model = runtime.MLP(1, 4, 2)
    payload = {
        "feature_names": ["dummy"],
        "mean": [0.0],
        "std": [1.0],
        "label_to_id": {"k2_compact": 0, "k3_compact": 1},
        "selected_tau": selected_tau,
        "config": {"use_text_features": use_text_features},
        "input_dim": 1,
        "hidden_dim": 4,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_planner(path: Path, threshold: float) -> runtime.RuntimeVariableBudgetPlanner:
    return runtime.RuntimeVariableBudgetPlanner(
        str(path),
        policy="tail_risk",
        tail_threshold=threshold,
        temperature=1.0,
        source_name="auto",
        max_examples_per_task=4,
    )


def test_method_filter() -> None:
    cfg = SimpleNamespace(runtime_methods=("all",))
    assert runtime.method_enabled(cfg, "prompt_rebuild_selected_pages")
    cfg = SimpleNamespace(runtime_methods=("full_kv_cache", "variable_budget_kv_planner"))
    assert runtime.method_enabled(cfg, "variable_budget_kv_planner")
    assert not runtime.method_enabled(cfg, "prompt_rebuild_selected_pages")
    assert runtime.any_method_enabled(cfg, ("prompt_rebuild_selected_pages", "variable_budget_kv_planner"))
    assert not runtime.any_method_enabled(cfg, ("prompt_rebuild_selected_pages", "output_level_risk_kv_planner"))


def test_conformal_tau_and_text_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "planner.pt"
        make_checkpoint(path, selected_tau=0.125, use_text_features=False)
        planner = load_planner(path, -1.0)
        assert abs(planner.tail_threshold - 0.125) < 1e-9
        assert planner.use_text_features is False

        planner_override = load_planner(path, 0.35)
        assert abs(planner_override.tail_threshold - 0.35) < 1e-9

        make_checkpoint(path, selected_tau=None, use_text_features=True)
        planner_no_tau = load_planner(path, -1.0)
        assert abs(planner_no_tau.tail_threshold - (-1.0)) < 1e-9
        assert planner_no_tau.use_text_features is True


def test_variable_budget_floor() -> None:
    assert runtime.apply_variable_budget_floor("k1_compact", [1, 2, 3, 8], 2) == "k2_compact"
    assert runtime.apply_variable_budget_floor("k2_compact", [1, 2, 3, 8], 2) == "k2_compact"
    assert runtime.apply_variable_budget_floor("k3_compact", [1, 2, 3, 8], 2) == "k3_compact"
    assert runtime.apply_variable_budget_floor("full", [1, 2, 3, 8], 2) == "full"
    assert runtime.apply_variable_budget_floor("k1_compact", [1], 2) == "k1_compact"
    assert runtime.apply_variable_budget_floor("k1_compact", [1, 3], 2) == "k3_compact"
    assert runtime.apply_variable_budget_floor("k1_compact", [1, 2], 0) == "k1_compact"


def main() -> None:
    test_method_filter()
    test_conformal_tau_and_text_flag()
    test_variable_budget_floor()
    print("runtime control tests OK")


if __name__ == "__main__":
    main()
